import os
from typing import List, Tuple, Optional, Dict

import torch
from torch.utils.data import Dataset

from src.train.dataset.tokenizer import Tokenizer


class TitleToLyricsDataset(Dataset):
    def __init__(self,
                 inputs_path: str,
                 tokenizer: Tokenizer,
                 word2idx: Dict[str, int],
                 idx2word: Dict[int, str],
                 max_length: int = 512,
                 is_debug: bool = False) -> None:
        self.inputs_path: str = inputs_path
        self.pairs: Optional[List[Tuple[str, str]]] = None
        self.tokenizer: Tokenizer = tokenizer
        self.max_length: int = max_length
        self.word2idx: Optional[Dict[str, int]] = word2idx
        self.idx2word: Optional[Dict[int, str]] = idx2word
        self.is_debug: bool = is_debug

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        title, lyrics = self.pairs[idx]

        input_text: str = f"<BOS>{title}<SEP>"
        target_text: str = f"{lyrics}<EOS>"

        input_tokens: List[str] = self.tokenizer.tokenize(input_text)
        target_tokens: List[str] = self.tokenizer.tokenize(target_text)

        input_ids: List[int] = [
            self.word2idx.get(tok, self.word2idx["<UNK>"])
            for tok in input_tokens
        ]
        target_ids: List[int] = [
            self.word2idx.get(tok, self.word2idx["<UNK>"])
            for tok in target_tokens
        ]

        if self.is_debug == True:
            print(f"[TitleToLyricsDataset] Input tokens: {input_tokens[:10]}...")  # Debugging
            print(f"[TitleToLyricsDataset] Target tokens: {target_tokens[:10]}...")  # Debugging

        # Optionally truncate
        input_ids = input_ids[:self.max_length]
        target_ids = target_ids[:self.max_length]

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)
    
    @property
    def vocab(self) -> List[str]:
        assert self.word2idx is not None, "word2idx is not initialized"
        return list(self.word2idx.keys())

    def load_title_lyrics_pairs(self) -> None:
        pairs: List[Tuple[str, str]] = []
        for fname in os.listdir(self.inputs_path):
            if fname.endswith(".txt"):
                title: str = fname.replace(".txt", "").replace("_", " ")
                path: str = os.path.join(self.inputs_path, fname)
                with open(path, encoding="utf-8") as f:
                    lyrics: str = f.read().strip()
                    if lyrics:
                        pairs.append((title, lyrics))
        self.pairs = pairs
