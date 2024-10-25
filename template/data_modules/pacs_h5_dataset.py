import os
from typing import List, Any

import random
import pathlib
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image
import h5py

class PACSDataset(Dataset):
    def __init__(self, root: str, leave_out: List=None, transform=None) -> None:
        self.classes = {
            'dog': 0, 
            'giraffe': 1, 
            'guitar': 2, 
            'house': 3, 
            'person': 4, 
            'horse': 5, 
            'elephant': 6
        }

        self.domains = {
            'sketch': 0, 
            'cartoon': 1, 
            'art_painting': 2, 
            'photo': 3
        }

        self.n_classes = len(self.classes)

        full_path = os.path.join(root, 'PACS.hdf5')
        self.f = h5py.File(full_path, 'r')
        self.D = self.f['D'][:]
        self.T = self.f['T'][:]

        indices = torch.arange(len(self.f['X']))
        if leave_out:
            holdout = [self.domains[l] for l in leave_out]

            index_mask = torch.ones_like(indices)
            for i in holdout:
                index_mask = index_mask & (self.D != i)
            index_mask = index_mask.to(torch.bool)

            self.valid_indices = indices[index_mask]
        else:
            self.valid_indices = indices

        self.transform = transform
        
    def __len__(self) -> int:
        return self.valid_indices.__len__()
    
    def __getitem__(self, index) -> Any:
        index_ = self.valid_indices[index].item()
        
        image = torch.from_numpy(self.f['X'][index_])

        if self.transform:
            image = self.transform(image)

        domain = int(self.f['D'][index_])
        cls = int(self.f['T'][index_])

        return image, cls, domain
    
    def get_stats(self):
        X = torch.from_numpy(self.f['X'][self.valid_indices])
        mean = torch.mean(X, dim=(0,2,3))
        std = torch.std(X, dim=(0,2,3))

        print(mean)
        print(std)

        return mean, std
    
def get_pacs_loo(root, leave_out=None, train_tf=None, test_tf=None):
    domains = {'sketch', 'cartoon', 'art_painting', 'photo'}
    
    train_set = PACSDataset(root=root, leave_out=leave_out, transform=train_tf)
    test_set = PACSDataset(root=root, leave_out=list(domains-set(leave_out)), transform=test_tf)

    return train_set, test_set

def main():
    train_set, test_set = get_pacs_loo(
        '/data',
        leave_out=['sketch'],
    )

    train_set.get_stats()

    a=1

if __name__ == '__main__':
    main()