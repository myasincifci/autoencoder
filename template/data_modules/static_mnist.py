import torch
from torch.utils.data import Dataset
from template.data_modules.MovingMNIST.MovingMNIST import MovingMNIST

class StaticMNIST(Dataset):
    def __init__(self, train=True, transform=None):
        super().__init__()
        self.moving_mnist = MovingMNIST(
            '/home/yasin/repos/autoencoder/template/data_modules/MovingMNIST', 
            train=train,
            transform=transform
        )

        # N: samples, T: timesteps in each sample 
        self.N, self.T = len(self.moving_mnist), self.moving_mnist[0][0].size(0)

    def __len__(self):
        return self.N * self.T

    def __getitem__(self, index):
        n, t = divmod(index, self.T)
        x = self.moving_mnist[n][0][t]

        return x[None], x[None]

def main():
    from torchvision.transforms import v2 as T
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    ds = StaticMNIST(transform=T.ToTensor())
    print(len(ds))

    loader = DataLoader(ds, batch_size=64, num_workers=4, shuffle=False)

    mean = 0.0
    std = 0.0
    num_pixels = 0

    # Iterate through the dataset
    for images, _ in tqdm(loader):
        batch_size, channels, height, width = images.size()
        num_pixels += batch_size * height * width

        # Compute batch mean and variance
        mean += images.mean(dim=[0, 2, 3]) * batch_size * height * width
        std += images.std(dim=[0, 2, 3])**2 * batch_size * height * width

    # Final mean and std calculation
    mean /= num_pixels
    std = torch.sqrt(std / num_pixels)

    print(f"Mean: {mean}")
    print(f"Std: {std}")

    # Mean: tensor([0.0493])
    # Std: tensor([0.2002])

    a=1

if __name__ == '__main__':
    main()