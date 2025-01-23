import math
import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as L

from template.modules.autoencoder import Encoder, Decoder, ConvEncoder, ConvDecoder

class LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=10):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.lstm_cells = nn.ModuleList([nn.LSTMCell(in_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])
        self.linear = nn.Linear(hidden_dim, in_dim)
        self.hidden_dim = hidden_dim

    def forward(self, input, future=0):
        B, T, D = input.size()
        outputs = []

        h_t = [torch.zeros(B, self.hidden_dim, dtype=torch.float, device=input.device) for _ in range(self.num_layers)]
        c_t = [torch.zeros(B, self.hidden_dim, dtype=torch.float, device=input.device) for _ in range(self.num_layers)]

        for input_t in input.split(1, dim=1):
            input_t = input_t.squeeze(1)
            h_t[0], c_t[0] = self.lstm_cells[0](input_t, (h_t[0], c_t[0]))
            for i in range(1, self.num_layers):
                h_t[i], c_t[i] = self.lstm_cells[i](h_t[i-1], (h_t[i], c_t[i]))
            output = self.linear(h_t[-1])
            output = output[:, None, :]
            outputs += [output]

        for i in range(future):  # if we should predict the future
            h_t[0], c_t[0] = self.lstm_cells[0](output.squeeze(1), (h_t[0], c_t[0]))
            for i in range(1, self.num_layers):
                h_t[i], c_t[i] = self.lstm_cells[i](h_t[i-1], (h_t[i], c_t[i]))
            output = self.linear(h_t[-1])
            output = output[:, None, :]
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
        self.lstm = LSTM(in_dim=2*latent_dim, hidden_dim=512, num_layers=5)
        # self.lstm = nn.LSTM(latent_dim, hidden_size=128, num_layers=5, batch_first=True, proj_size=latent_dim)
        self.vae.requires_grad_(False)

    def forward(self, x, t):
        B, T, H, W = x.shape

        # Encode all frames
        x = x.flatten(start_dim=0, end_dim=1)[:,None]
        mu, logvar = self.vae.encode(x)
        mu_logvar = torch.cat([mu, logvar], dim=1).view(B, T, -1)

        mu_logvar_x = mu_logvar[:,:-1]
        mu_logvar_y = mu_logvar[:,1:]

        mu_logvar_pred = mu_logvar_x + self.lstm(mu_logvar_x)
        loss = F.mse_loss(mu_logvar_pred[:,3:], mu_logvar_y[:,3:])

        z = self.vae.reparameterize(mu_logvar_pred[:,:,:64], mu_logvar_pred[:,:,64:])
        

        y = self.vae.decode(z)
        y = y.view(B*(T-1), 1, H, W)
        
        return loss

    # def forward(self, x, t):
    #     B, T, H, W = x.shape

    #     # Encode all frames
    #     x = x.flatten(start_dim=0, end_dim=1)[:,None]
    #     mu, logvar = self.vae.encode(x)

    #     z = self.vae.reparameterize(mu, logvar)
        
    #     # Predict frames at t+1
    #     z = z.view(B, T, -1)

    #     z_x = z[:,:-1]
    #     z_y = z[:,1:]
    #     z_y_pred = self.lstm(z_x)
    #     loss = F.mse_loss(z_y_pred[:,3:], z_y[:,3:])

    #     y = self.vae.decode(z_y_pred)
    #     y = y.view(B*(T-1), 1, H, W)
        
    #     return loss

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