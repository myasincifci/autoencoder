from typing import Iterator, List, Dict, OrderedDict

import torch
from torch.utils.data import Sampler

class DeterministicSampler(Sampler):
    def __init__(self, data_source, initial_seed) -> None:
        self.data = data_source
        self.seed = initial_seed

    def __len__(self) -> int:
        return len(self.data)
    
    def len(self):
        return self.__len__()
    
    def __iter__(self) -> Iterator:
        print(self.seed)
        p = torch.randperm(self.len(), generator=torch.Generator().manual_seed(self.seed))
        # self.seed += 1
        r = torch.arange(self.len())[p]
        yield from r.tolist()

class DomainMapper():
    def setup(self, domains: List):
        self.unique_domains = domains.unique()
        self.map_dict: Dict = {self.unique_domains[i].item():i for i in range(len(self.unique_domains))}
        self.unmap_dict: Dict = dict((v, k) for k, v in self.map_dict.items())

        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.map(x)

    def map(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor([self.map_dict[v.item()] for v in x])
    
    def unmap(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor([self.unmap_dict[v.item()] for v in x])
    
def get_backbone_from_ckpt(ckpt_path: str) -> torch.nn.Module:
    state_dict = torch.load(ckpt_path)["state_dict"]
    state_dict = OrderedDict([
        (".".join(name.split(".")[1:]), param) for name, param in state_dict.items() if name.startswith("backbone")
    ])

    return state_dict