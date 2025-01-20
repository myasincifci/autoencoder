import math
import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as L

from template.modules import Encoder, Decoder, ConvEncoder, ConvDecoder
    
class VAE(nn.Module):
    def __init__(self, latent_dim, data_shape):
        super().__init__()
        self.encoder = ConvEncoder(data_shape, c_hid=32, latent_dim=latent_dim, act_fn=nn.ReLU)
        self.decoder = ConvDecoder(data_shape, c_hid=32, latent_dim=latent_dim, act_fn=nn.ReLU)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)

        return self.decoder(z), mu, logvar

class VAEModule(L.LightningModule):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = VAE(
            latent_dim=cfg.param.latent_dim ,
            data_shape=cfg.data.shape
        )
        self.cfg = cfg

    def loss_function(self, recon_x, x, mu, logvar):
        # BCE = F.binary_cross_entropy(recon_x, x.view(-1, x.shape[-1]), reduction='sum')
        BCE = F.binary_cross_entropy(recon_x, x.flatten(start_dim=1), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def training_step(self, batch, batch_idx):
        X, *_ = batch
        X = X#.flatten(start_dim=1)

        X_, mu, logvar = self.model(X)
        loss = self.loss_function(X_, X, mu, logvar)

        self.log('train/loss', loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, *_ = batch
        X = X#.flatten(start_dim=1)

        X_, mu, logvar = self.model(X)
        loss = self.loss_function(X_, X, mu, logvar)
        
        self.log('val/loss', loss, prog_bar=True)

    def on_validation_epoch_end(self):
        # Log val image
        val_img, *_ = self.trainer.datamodule.val_dataloader().dataset[0]
        val_img = val_img[None].cuda()#.flatten(start_dim=1)
        val_img_, *_ = self.model(val_img)
        val_img_ = val_img_.reshape(1, *self.cfg.data.shape)
        wandb_logger = self.logger
        wandb_logger.log_image('val/image', [val_img_])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.param.lr)
        
        return optimizer

def main():
    pass

if __name__ == "__main__":
    main()