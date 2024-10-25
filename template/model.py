from typing import Any

import torch
from torch import nn
import torch.optim as optim
from torchvision.models.resnet import resnet18

import pytorch_lightning as L
from pytorch_lightning.utilities.types import STEP_OUTPUT

class Decoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, output_padding=1, padding=1, stride=2)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, output_padding=1, padding=1, stride=2)
        self.deconv5 = nn.ConvTranspose2d(32, 3, kernel_size=3, output_padding=1, padding=1, stride=2)

    def forward(self, x):
        x = x.view(-1, 512, 1, 1)
        x = nn.functional.relu(self.deconv1(x))
        x = nn.functional.relu(self.deconv2(x))
        x = nn.functional.relu(self.deconv3(x))
        x = nn.functional.relu(self.deconv4(x))
        x = nn.functional.tanh(self.deconv5(x))

        return x

class ResnetClf(L.LightningModule):
    def __init__(self, cfg, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.encoder = resnet18()
        self.encoder.fc = nn.Identity()

        self.decoder = Decoder()

        self.criterion = nn.MSELoss()
        
        self.cfg = cfg

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        X, _ = batch

        Y = self.decoder(self.encoder(X))
        loss = self.criterion(Y, X)
        self.log("train/loss", loss.item(), prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> None:
        X, _ = batch

        Y = self.decoder(self.encoder(X))
        loss = self.criterion(Y, X)

        self.log("val/loss", loss.item())

    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(params=self.parameters(), lr=self.cfg.param.lr)

        return [optimizer]