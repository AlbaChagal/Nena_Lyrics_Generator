from typing import List, Dict

from src.train.dataset.data_set_base import DatasetBase


class CharDataset(DatasetBase):
    def __init__(self, text: str, block_size: int):
        super(CharDataset, self).__init__(data=None, block_size=block_size)

        self.chars: List[str] = sorted(list(set(text)))
        self.vocab_size: int = len(self.chars)

        self.word2idx: Dict[str, int] = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2word: Dict[int, str] = {i: ch for ch, i in self.word2idx.items()}

        self.data: List[int] = [self.word2idx[c] for c in text]

    @property
    def vocab(self):
        return self.chars
