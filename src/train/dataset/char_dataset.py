from typing import List, Dict, Optional

import numpy as np

from src.global_constants import mask_character
from src.train.dataset.dataset_base import DatasetBase


class CharDataset(DatasetBase):
    def __init__(self,
                 text: str,
                 block_size: int,
                 random_state: np.random.RandomState,
                 mask_token_idx: Optional[int] = None):
        super(CharDataset, self).__init__(data=None, block_size=block_size, random_state=random_state)

        self.chars: List[str] = sorted(list(set(text)))
        self.chars.append(mask_character)
        self.vocab_size: int = len(self.chars)

        self.word2idx: Dict[str, int] = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2word: Dict[int, str] = {i: ch for ch, i in self.word2idx.items()}
        self.mask_token_idx = mask_token_idx if mask_token_idx is not None else self.chars.index(mask_character)
        self.data: List[int] = [self.word2idx[c] for c in text]

    @property
    def vocab(self):
        return self.chars
