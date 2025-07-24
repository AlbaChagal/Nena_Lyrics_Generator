from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset


class DatasetBase(Dataset):
    def __init__(self, data: Optional[List[int]], block_size: int) -> None:

        self.data: Optional[List[int]] = data
        self.block_size: int = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk: List[int] = self.data[idx: idx + self.block_size + 1]
        x: torch.Tensor = torch.tensor(chunk[:-1], dtype=torch.long)
        y: torch.Tensor = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
