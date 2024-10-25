from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torchvision import transforms as T
import pytorch_lightning as pl
from lightly.transforms.byol_transform import (BYOLTransform,
                                               BYOLView1Transform,
                                               BYOLView2Transform)
from lightly.transforms.utils import IMAGENET_NORMALIZE
from wilds import get_dataset
from wilds.common.grouper import CombinatorialGrouper
from utils import DomainMapper

class CamelyonDM(pl.LightningDataModule):
    def __init__(self, cfg, unlabeled=False) -> None:
        super().__init__()
        self.data_dir = cfg.data_path
        self.unlabeled = cfg.unlabeled
        self.batch_size = cfg.param.batch_size

        self.train_transform = BYOLTransform(
            view_1_transform=T.Compose([
                BYOLView1Transform(input_size=96, gaussian_blur=0.0),
            ]),
            view_2_transform=T.Compose([
                BYOLView2Transform(input_size=96, gaussian_blur=0.0),
            ])
        )

        self.val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=IMAGENET_NORMALIZE["mean"],
                std=IMAGENET_NORMALIZE["std"],
            ),
        ])

        self.labeled_dataset = get_dataset(
            dataset='camelyon17',
            download=True, 
            root_dir=self.data_dir, 
            unlabeled=False
        )

        if self.unlabeled:
            self.unlabeled_dataset = get_dataset(
                dataset='camelyon17',
                download=True, 
                root_dir=self.data_dir, 
                unlabeled=True
            )

        self.grouper = CombinatorialGrouper(self.labeled_dataset, ['hospital'])

        self.cfg = cfg
        self.domain_mapper = DomainMapper().setup(
            self.labeled_dataset.get_subset("train").metadata_array[:, 0]
        )

        self.num_classes = self.labeled_dataset.n_classes

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            train_set_labeled = self.labeled_dataset.get_subset(
                    "train", 
                    transform=self.train_transform
            )

            if self.unlabeled:
                self.train_set = self.unlabeled_dataset.get_subset(
                    "train_unlabeled", 
                    transform=self.train_transform
                )
            else:
                self.train_set = train_set_labeled

            self.val_set_id = self.labeled_dataset.get_subset(
                "id_val", 
                transform=self.val_transform
            )

            self.val_set = self.labeled_dataset.get_subset(
                "val", 
                transform=self.val_transform
            )

            self.test_set = self.labeled_dataset.get_subset(
                "test",
                transform=self.val_transform
            )

            ################

            self.train_set_knn = self.labeled_dataset.get_subset(
                "train", 
                frac=4096/len(train_set_labeled), 
                transform=self.val_transform
            )

            self.val_set_knn_id = self.labeled_dataset.get_subset(
                "id_val",
                frac=1024/len(self.val_set_id), 
                transform=self.val_transform
            )

            self.val_set_knn = self.labeled_dataset.get_subset(
                "val", 
                frac=1024/len(self.val_set), 
                transform=self.val_transform
            )

            self.test_set_knn = self.labeled_dataset.get_subset(
                "test",
                frac=1024/len(self.test_set),
                transform=self.val_transform
            )

            # self.domain_mapper = self.domain_mapper.setup(
            #     train_set_labeled.metadata_array[:, 0]
            # )
            
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
            pin_memory=True
        )
    
    def val_dataloader(self) -> TRAIN_DATALOADERS:    
        train_loader_knn = DataLoader(
            self.train_set_knn,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=8,
            pin_memory=True
        )

        val_loader_knn_id = DataLoader(
            self.val_set_knn_id,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=8,
            pin_memory=True
        )

        val_loader_knn = DataLoader(
            self.val_set_knn,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=8,
            pin_memory=True
        )

        test_loader_knn = DataLoader(
            self.test_set_knn,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=8,
            pin_memory=True
        )

        return [
            train_loader_knn,
            val_loader_knn_id,
            val_loader_knn,
            test_loader_knn
        ]