from collections import Counter
import re
from typing import List, Dict, Tuple, Callable


class Tokenizer:
    def __init__(self):
        pass

    @staticmethod
    def tokenize(text: str) -> List[str]:
        raise NotImplementedError()

    @staticmethod
    def counter(token_lists: List[str]) -> Counter:
        raise NotImplementedError()

    @classmethod
    def build_vocab(cls, token_lists: List[str], min_freq: int = 1) -> Tuple[Dict[str, int], Dict[int, str]]:
        counter: Counter = cls.counter(token_lists)
        vocab: List[str] = [char for char, freq in counter.items() if freq >= min_freq]
        vocab: List[str] = sorted(vocab)

        word2idx: Dict[str, int] = {char: idx for idx, char in enumerate(vocab)}
        idx2word: Dict[int, str] = {idx: char for char, idx in word2idx.items()}
        return word2idx, idx2word


class TokenizerWordLevel(Tokenizer):
    def __init__(self):
        super(TokenizerWordLevel, self).__init__()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)

    @staticmethod
    def counter(token_lists: List[str]) -> Counter:
        return Counter(token for tokens in token_lists for token in tokens)


class TokenizerCharLevel(Tokenizer):
    def __init__(self):
        super(TokenizerCharLevel, self).__init__()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return list(text)

    @staticmethod
    def counter(token_lists: List[str]) -> Counter:
        return Counter(token_lists)
