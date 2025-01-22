from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS

from template.data_modules.MovingMNIST.MovingMNIST import MovingMNIST


class MovingMnistDM(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.data_dir = cfg.data.path
        self.batch_size = cfg.param.batch_size

        self.train_transform = T.Compose(
            [
                T.ToTensor(),
            ]
        )

        self.val_transform = T.Compose(
            [
                T.ToTensor(),
            ]
        )

        self.train_set = MovingMNIST(
            "./template/data_modules/MovingMNIST",
            transform=self.train_transform,
            target_transform=self.train_transform,
        )
        self.val_set = MovingMNIST(
            "./template/data_modules/MovingMNIST",
            train=False,
            transform=self.val_transform,
            target_transform=self.val_transform,
        )

        self.cfg = cfg

    def setup(self, stage: str) -> None:
        if stage == "fit":
            pass

        elif stage == "test":
            pass

        elif stage == "predict":
            pass

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )
