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
        self.word2idx: Dict[str, int] = word2idx
        self.idx2word: Dict[int, str] = idx2word
        self.is_debug: bool = is_debug

    def __len__(self) -> int:
        # Return the total number of (title, lyrics) pairs times the average number of prefixes per song
        # For efficiency, just return a large number (approximate total number of training steps)
        if self.pairs is None:
            return 0
        avg_lyrics_len = int(sum(len(self.tokenizer.tokenize(lyrics)) for _, lyrics in self.pairs) / max(1, len(self.pairs)))
        return len(self.pairs) * avg_lyrics_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Sample a random (title, lyrics) pair
        import random
        if self.pairs is None or len(self.pairs) == 0:
            raise IndexError("No data in TitleToLyricsDataset")
        song_idx = idx % len(self.pairs)
        title, lyrics = self.pairs[song_idx]

        # Tokenize lyrics (word-level)
        lyrics_tokens: List[str] = self.tokenizer.tokenize(lyrics)
        # N = number of words to use as prefix (0 to len(lyrics_tokens)-1)
        if len(lyrics_tokens) > 1:
            N = random.randint(0, len(lyrics_tokens)-1)
        else:
            N = 0
        prefix_tokens = lyrics_tokens[:N]
        # Input: <BOS>title<SEP> + first N words
        input_tokens: List[str] = self.tokenizer.tokenize(f"<BOS>{title}<SEP>") + prefix_tokens
        # Target: <BOS>title<SEP> + first N words + next word
        if N < len(lyrics_tokens):
            next_word = lyrics_tokens[N]
            target_tokens = input_tokens + [next_word]
        else:
            target_tokens = input_tokens
        # Always add <EOS> to target
        target_tokens = target_tokens + ["<EOS>"]

        if self.word2idx is None:
            raise ValueError("word2idx is None in TitleToLyricsDataset. It must be set before using __getitem__.")
        unk_idx = self.word2idx["<UNK>"] if "<UNK>" in self.word2idx else 0
        input_ids: List[int] = [self.word2idx.get(tok, unk_idx) for tok in input_tokens]
        target_ids: List[int] = [self.word2idx.get(tok, unk_idx) for tok in target_tokens]

        if self.is_debug:
            print(f"[TitleToLyricsDataset] Input tokens: {input_tokens[:10]}...")
            print(f"[TitleToLyricsDataset] Target tokens: {target_tokens[:10]}...")

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
                        # Ensure every line ends with '\n'
                        lines = lyrics.splitlines()
                        lines = [line if line.endswith("\n") else line + "\n" for line in lines]
                        lyrics_with_newlines = "".join(lines)
                        pairs.append((title, lyrics_with_newlines))
        self.pairs = pairs
