import math
import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as L

from template.modules.autoencoder import Encoder, Decoder, ConvEncoder, ConvDecoder

class LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTMCell(in_dim, hidden_dim)
        self.lstm2 = nn.LSTMCell(hidden_dim, hidden_dim)
        self.lstm3 = nn.LSTMCell(hidden_dim, hidden_dim)
        self.lstm4 = nn.LSTMCell(hidden_dim, hidden_dim)
        self.lstm5 = nn.LSTMCell(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, in_dim)

        self.hidden_dim = hidden_dim

    def forward(self, input, future = 0):
        B, T, D = input.size()

        outputs = []
        h_t = torch.zeros(B, self.hidden_dim, dtype=torch.float, device=input.device)
        c_t = torch.zeros(B, self.hidden_dim, dtype=torch.float, device=input.device)
        h_t2 = torch.zeros(B, self.hidden_dim, dtype=torch.float, device=input.device)
        c_t2 = torch.zeros(B, self.hidden_dim, dtype=torch.float, device=input.device)
        h_t3 = torch.zeros(B, self.hidden_dim, dtype=torch.float, device=input.device)
        c_t3 = torch.zeros(B, self.hidden_dim, dtype=torch.float, device=input.device)
        h_t4 = torch.zeros(B, self.hidden_dim, dtype=torch.float, device=input.device)
        c_t4 = torch.zeros(B, self.hidden_dim, dtype=torch.float, device=input.device)
        h_t5 = torch.zeros(B, self.hidden_dim, dtype=torch.float, device=input.device)
        c_t5 = torch.zeros(B, self.hidden_dim, dtype=torch.float, device=input.device)

        for input_t in input.split(1, dim=1):
            input_t = input_t.squeeze(1)

            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
            h_t4, c_t4 = self.lstm4(h_t3, (h_t4, c_t4))
            h_t5, c_t5 = self.lstm5(h_t4, (h_t5, c_t5))
            output = self.linear(h_t5)
            output = output[:,None,:]

            outputs += [output]

        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output.squeeze(1), (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
            h_t4, c_t4 = self.lstm4(h_t3, (h_t4, c_t4))
            h_t5, c_t5 = self.lstm5(h_t4, (h_t5, c_t5))
            output = self.linear(h_t5)

            output = output[:,None,:]

            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs

class VAE(nn.Module):
    def __init__(self, latent_dim, data_shape, checkpoint=None):
        super().__init__()
        self.encoder = ConvEncoder(data_shape, c_hid=32, latent_dim=latent_dim, act_fn=nn.GELU)
        self.decoder = ConvDecoder(data_shape, c_hid=32, latent_dim=latent_dim, act_fn=nn.GELU)

        if checkpoint:
            sd = torch.load(checkpoint)['state_dict']
            sd = {k.replace('model.', ''): v for k, v in sd.items()}
            self.load_state_dict(sd)

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        return self.decode(z), mu, logvar
    
class VAELSTM(nn.Module):
    def __init__(self, latent_dim, data_shape, checkpoint=None):
        super().__init__()
        self.vae = VAE(latent_dim, data_shape, checkpoint=checkpoint)
        self.lstm = LSTM(in_dim=latent_dim, hidden_dim=128)
        self.vae.requires_grad_(False)

    def forward(self, x, t):
        B, T, H, W = x.shape

        # Encode all frames
        x = x.flatten(start_dim=0, end_dim=1)[:,None]
        mu, logvar = self.vae.encode(x)
        z = self.vae.reparameterize(mu, logvar)
        
        # Predict frames at t+1
        z = z.view(B, T, -1)

        z_x = z[:,:-1]
        z_y = z[:,1:]

        z_y_pred = self.lstm(z_x)
        
        loss = F.mse_loss(z_y_pred[:,3:], z_y[:,3:])

        y = self.vae.decode(z_y_pred)
        y = y.view(B*(T-1), 1, H, W)
        
        return loss

class VAELSTMModule(L.LightningModule):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = VAELSTM(
            latent_dim=cfg.param.latent_dim ,
            data_shape=cfg.data.shape,
            checkpoint='/home/yasin/repos/autoencoder/lightning_logs/7c632070/checkpoints/vae-epoch=200-val_loss=0.00.ckpt'
        )
        self.cfg = cfg

    def training_step(self, batch, batch_idx):
        X, T = batch

        loss = self.model(X, T)
        self.log('train/loss', loss, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.param.lr)

        return optimizer
        

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
    
    def configure_callbacks(self):
        # Checkpoint for best validation loss
        checkpoint_callback = L.callbacks.ModelCheckpoint(
            monitor='val/loss',
            filename='vae-{epoch:02d}-{val_loss:.2f}',
            save_top_k=1,
            mode='min'
        )
        return [checkpoint_callback]

def main():
    pass

if __name__ == "__main__":
    main()