import os
from typing import List

from torch import tensor
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
import pytorch_lightning as pl
from lightly.transforms.byol_transform import (BYOLTransform,
                                               BYOLView1Transform,
                                               BYOLView2Transform)
from lightly.transforms.utils import IMAGENET_NORMALIZE
from data_modules.pacs_h5_dataset import get_pacs_loo
from utils import DomainMapper

class PacsDM(pl.LightningDataModule):
    def __init__(self, cfg, leave_out: List=None) -> None:
        super().__init__()
        self.data_dir = cfg.data.path
        self.batch_size = cfg.param.batch_size

        self.transform = T.Compose([
            
            T.Normalize(
                mean=[0.6400, 0.6076, 0.5604],
                std=[0.3090, 0.3109, 0.3374],
            ),
        ])

        self.train_set, self.test_set = get_pacs_loo(
            root=cfg.data.path,
            leave_out=leave_out,
            train_tf=self.transform,
            test_tf=self.transform
        )

        self.cfg = cfg
        self.num_classes = self.train_set.n_classes

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            pass
            
        elif stage == 'test':
            pass
        
        elif stage == 'predict':
            pass

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=8,
            pin_memory=True,
        )
    
    def val_dataloader(self) -> TRAIN_DATALOADERS:    
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=8,
            pin_memory=True
        )
    
def main():
    pass

if __name__ == '__main__':
    main()