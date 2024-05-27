
from typing import List
from torch.nn import Module
from torch.utils.data import DataLoader

import torch

class Standalone:
    def __init__(
            self,
            model: Module,
            train_loader: DataLoader,
            test_loaders: List[DataLoader],
            device: torch.device | None = None,
    ):
        pass
