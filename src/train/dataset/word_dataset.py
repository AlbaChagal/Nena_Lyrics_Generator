from typing import List, Optional

import numpy as np

from src.global_constants import mask_character
from src.train.dataset.dataset_base import DatasetBase


class WordDataset(DatasetBase):
    def __init__(self,
                 token_ids: List[int],
                 vocab: List[str],
                 block_size: int,
                 random_state: np.random.RandomState,
                 mask_token_idx: Optional[int] = None) -> None:
        super(WordDataset, self).__init__(data=token_ids, block_size=block_size, random_state=random_state)
        self.vocab: List[str] = vocab
        self.vocab.append(mask_character)
        self.vocab_size: int = len(vocab)
        self.mask_token_idx = mask_token_idx if mask_token_idx is not None else self.vocab.index(mask_character)

    @property
    def chars(self):
        return self.vocab
