import torch
from torch import nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc31 = nn.Linear(512, 20)
        self.fc32 = nn.Linear(512, 20)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        
        return self.fc31(h2), self.fc32(h2)
    
class Decoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(20, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, out_dim)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))

        return F.sigmoid(self.fc3(h2))

class ConvEncoder(nn.Module):
    def __init__(self, data_shape, c_hid, latent_dim, act_fn, variational=True):
        super().__init__()
        self.variational = variational
        self.layers = nn.ModuleDict({
            'conv_1': nn.Sequential(
                nn.Conv2d(data_shape[0], c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
                nn.BatchNorm2d(c_hid),
                act_fn(),
            ),
            'conv_2': nn.Sequential(
                nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1), # 32x32 => 16x16
                nn.BatchNorm2d(c_hid),
                act_fn(),
            ),
            'conv_3': nn.Sequential(
                nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
                nn.BatchNorm2d(2*c_hid),
                act_fn(),
            ),
            'conv_4': nn.Sequential(
                nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1), # 32x32 => 16x16
                nn.BatchNorm2d(2*c_hid),
                act_fn(),
            ),
            'conv_5': nn.Sequential(
                nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
                nn.BatchNorm2d(2*c_hid),
                act_fn(),
            ),
            'flatten': nn.Sequential(
                nn.Flatten(),
            )
        })
        self.fc_mu = nn.Linear(64*2*c_hid, latent_dim)
        if variational:
            self.fc_logvar = nn.Linear(64*2*c_hid, latent_dim)

    def forward(self, x):
        for name, layer in self.layers.items():
            x = layer(x)

        if self.variational:
            return self.fc_mu(x), self.fc_logvar(x)
        else:
            return self.fc_mu(x)

class ConvDecoder(nn.Module):
    def __init__(self, data_shape, c_hid, latent_dim, act_fn):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64*2*c_hid)
        self.layers = nn.ModuleDict({
            'deconv_1': nn.Sequential(
                nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
                nn.BatchNorm2d(2*c_hid),
                act_fn(),
            ),
            'deconv_2': nn.Sequential(
                nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
                nn.BatchNorm2d(2*c_hid),
                act_fn(),
            ),
            'deconv_3': nn.Sequential(
                nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
                nn.BatchNorm2d(c_hid),
                act_fn(),
            ),
            'deconv_4': nn.Sequential(
                nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_hid),
                act_fn(),
            ),
            'deconv_5': nn.Sequential(
                nn.ConvTranspose2d(c_hid, data_shape[0], kernel_size=3, output_padding=1, padding=1, stride=2),
                nn.Sigmoid(),
            ),
            'flatten': nn.Sequential(
                nn.Flatten(),
            )
        })

    def forward(self, x):
        x = self.fc(x).view(-1, 64, 8, 8)
        for name, layer in self.layers.items():
            x = layer(x)

        return x

def main():
    x = torch.rand((1, 1, 64, 64))
    model = ConvEncoder((1, 64, 64), c_hid=32, latent_dim=20, act_fn=nn.ReLU)

    y = model(x)

    a=1

if __name__ == '__main__':
    main()