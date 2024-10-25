from functools import partial
from typing import Any
from types import MethodType

import pytorch_lightning as L
import torch
import torch.optim as optim
import torchmetrics
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, nn
from torch.autograd import Function
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import resnet18, ResNet18_Weights
from torch.optim import lr_scheduler

from torch.distributions.beta import Beta

class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x):
        if not self.training:
            return x
        
        if torch.rand(1).item() > self.p:
            return x

        B = x.size(0) # batch size

        mu = x.mean(dim=[2, 3], keepdim=True) # compute instance mean
        var = x.var(dim=[2, 3], keepdim=True) # compute instance variance
        sig = (var + self.eps).sqrt() # compute instance standard deviation
        mu, sig = mu.detach(), sig.detach() # block gradients
        x_normed = (x - mu) / sig # normalize input

        lmda = Beta(self.alpha, self.alpha).sample((B, 1, 1, 1)).to(x.device) # sample instance-wise convex weights

        # if domain label is given:
        if False:
            # in this case, input x = [xˆi, xˆj]
            perm = torch.arange(B-1, -1, -1) # inverse index
            perm_j, perm_i = perm.chunk(2) # separate indices
            perm_j = perm_j[torch.randperm(B // 2)] # shuffling
            perm_i = perm_i[torch.randperm(B // 2)] # shuffling
            perm = torch.cat([perm_j, perm_i], 0) # concatenation
        else:
            perm = torch.randperm(B) # generate shuffling indices

        mu2, sig2 = mu[perm], sig[perm] # shuffling
        mu_mix = mu * lmda + mu2 * (1 - lmda) # generate mixed mean
        sig_mix = sig * lmda + sig2 * (1 - lmda) # generate mixed standard deviation

        return x_normed * sig_mix + mu_mix # denormalize input using the mixed statistics

def res18(cfg):
    if cfg.model.pretrained:
        model = resnet18(ResNet18_Weights.DEFAULT)
    else:
        model = resnet18()
    model.fc = nn.Linear(in_features=512, out_features=cfg.data.num_classes, bias=True)
    
    if cfg.mixstyle.active:
        model.ms = MixStyle(
            p=cfg.mixstyle.p,
            alpha=cfg.mixstyle.alpha,
            eps=cfg.mixstyle.eps
        )
    else:
        model.ms = None

    def _forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.ms(x) if self.ms else x

        x = self.layer2(x)
        x = self.ms(x) if self.ms else x

        x = self.layer3(x)
        x = self.ms(x) if self.ms else x

        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x 
    
    model.forward = MethodType(_forward, model)

    return model

class ResnetClf(L.LightningModule):
    def __init__(self, cfg, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.model = res18(cfg)

        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=cfg.data.num_classes)
        
        self.cfg = cfg

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        X, t, d = batch

        y = self.model(X)
        loss = self.criterion(y, t)
        self.log("train/loss", loss.item())

        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> None:
        X, t, d = batch

        y = self.model(X)
        loss = self.criterion(y, t)
        self.accuracy(y, t)

        self.log("val/loss", loss.item())
        self.log("val/acc", self.accuracy, on_epoch=True)

    def configure_optimizers(self) -> Any:
        optimizer = optim.SGD(params=self.parameters(), lr=self.cfg.param.lr, weight_decay=1e-4, momentum=0.9)
        # scheduler = lr_scheduler.StepLR(optimizer, )

        return [optimizer]#, [scheduler]