from typing import Dict, List
from .utils.ml_utils import SNN_Block
import torch.nn as nn
import torch

class OmicsReverseTokenizer(nn.Module):
    def __init__(self, mapping_dict: Dict[str, List], input_dim: int, hidden: List[int] = []):
        super(OmicsReverseTokenizer, self).__init__()
        self.mapping_dict = mapping_dict
        layers_size = {k: [input_dim] + hidden + [len(v)] for k, v in mapping_dict.items()}
        self.token_reverse = nn.ModuleDict()

        for key, sizes in layers_size.items():
            layers = []
            if len(sizes) == 2:
                layers.append(nn.Linear(sizes[0], sizes[1]))
            else:
                for i in range(len(sizes) - 2):
                    layers.append(SNN_Block(sizes[i], sizes[i + 1]))
                layers.append(nn.Linear(sizes[-2], sizes[-1]))
            self.token_reverse[key] = nn.Sequential(*layers)
    
    def forward(self, x: Dict[str, torch.tensor]) -> torch.tensor:
        return {k: self.token_reverse[k](x[k]) for k in x.keys()}
    