from typing import List

from src.train.dataset.data_set_base import DatasetBase


class WordDataset(DatasetBase):
    def __init__(self, token_ids: List[int], vocab: List[str], block_size: int) -> None:
        super(WordDataset, self).__init__(data=token_ids, block_size=block_size)
        self.vocab: List[str] = vocab
        self.vocab_size: int = len(vocab)

    @property
    def chars(self):
        return self.vocab
