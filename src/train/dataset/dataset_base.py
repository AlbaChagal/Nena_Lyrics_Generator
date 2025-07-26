from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class DatasetBase(Dataset):
    def __init__(self,
                 data: Optional[List[int]],
                 block_size: int,
                 random_state: np.random.RandomState,
                 mask_prob: float = 0.5) -> None:

        self.data: Optional[List[int]] = data
        self.block_size: int = block_size
        self.random_state: np.random.RandomState = random_state
        self.mask_prob: float = mask_prob
        self.mask_token_idx: Optional[int] = None

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.mask_token_idx is not None
        chunk: List[int] = self.data[idx: idx + self.block_size + 1]
        x: torch.Tensor = torch.tensor(chunk[:-1], dtype=torch.long)
        y: torch.Tensor = torch.tensor(chunk[1:], dtype=torch.long)

        x_masked = x.clone()

        for i in range(len(x_masked)):
            if self.random_state.random() < self.mask_prob:
                x_masked[i] = self.mask_token_idx
        return x_masked, y, x
