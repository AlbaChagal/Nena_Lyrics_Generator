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
    def build_vocab(cls, token_lists: List[str], min_freq: int = 1, is_debug: bool = False) -> Tuple[Dict[str, int], Dict[int, str]]:
        if is_debug:
            print(f"[Tokenizer] Building vocabulary with min_freq={min_freq}")
        counter: Counter = cls.counter(token_lists)
        if is_debug:
            print(f'Counter most common: {counter.most_common(10)}')

        # Always include special tokens
        special_tokens = ["<UNK>", "<BOS>", "<EOS>", "<SEP>", "\n"]
        vocab: List[str] = [char for char, freq in counter.items() if freq >= min_freq]
        for token in special_tokens:
            if token not in vocab:
                vocab.append(token)
        vocab: List[str] = sorted(vocab)

        if is_debug:
            print(f'[Tokenizer] Vocabulary size: {len(vocab)}')
            print(f'[Tokenizer] First 10 words: {vocab[:10]}')
            print(f'[Tokenizer] Last 10 words: {vocab[-10:]}')

        word2idx: Dict[str, int] = {char: idx for idx, char in enumerate(vocab)}
        idx2word: Dict[int, str] = {idx: char for char, idx in word2idx.items()}

        if is_debug:
            print(f'[Tokenizer] word2idx: {len(word2idx)} pairs')
            print(f'[Tokenizer] idx2word: {len(idx2word)} pairs')
            print(f'[Tokenizer] first 10 words in word2idx: {list(word2idx.keys())[:10]}')
        return word2idx, idx2word


class TokenizerWordLevel(Tokenizer):
    def __init__(self):
        super(TokenizerWordLevel, self).__init__()

    @staticmethod
    def tokenize(text: str, is_debug: bool = False) -> List[str]:
        lower_text = text.lower()
        if is_debug:
            print(f"[TokenizerWordLevel] Tokenizing text: {lower_text[:100]}...")
        # Extract special tokens and newlines as-is (case-sensitive)
        pattern = r"<BOS>|<EOS>|<SEP>|<UNK>|\n"  # Match actual newline character
        special_tokens = re.findall(pattern, lower_text)
        # Remove special tokens/newlines for further tokenization
        text_wo_special = re.sub(pattern, "", lower_text)
        # Tokenize the rest (word-level, keep punctuation)
        tokens = re.findall(r"\w+|[^\w\s]", text_wo_special, re.UNICODE)
        # Combine special tokens and regular tokens in order
        return special_tokens + tokens

    @staticmethod
    def counter(token_lists: List[str]) -> Counter:
        return Counter(token_lists)

    
    @classmethod
    def build_vocab(cls, token_lists: List[str], min_freq: int = 1, is_debug: bool = False) -> Tuple[Dict[str, int], Dict[int, str]]:
        if is_debug:
            print(f"[TokenizerWordLevel] Building vocabulary with min_freq={min_freq}")
        counter: Counter = cls.counter(token_lists)
        if is_debug:
            print(f'[TokenizerWordLevel] Counter most common: {counter.most_common(10)}')

        # Always include special tokens, including '\n'
        special_tokens = ["<UNK>", "<BOS>", "<EOS>", "<SEP>", "\n"]
        vocab: List[str] = [word for word, freq in counter.items() if freq >= min_freq]
        for token in special_tokens:
            if token not in vocab:
                vocab.append(token)
        vocab = sorted(vocab)

        if is_debug:
            print(f'[TokenizerWordLevel] Vocabulary size: {len(vocab)}')
            print(f'[TokenizerWordLevel] First 10 words: {vocab[:10]}')
            print(f'[TokenizerWordLevel] Last 10 words: {vocab[-10:]}')

        word2idx: Dict[str, int] = {word: idx for idx, word in enumerate(vocab)}
        idx2word: Dict[int, str] = {idx: word for word, idx in word2idx.items()}

        if is_debug:
            print(f'[TokenizerWordLevel] word2idx: {len(word2idx)} pairs')
            print(f'[TokenizerWordLevel] idx2word: {len(idx2word)} pairs')
            print(f'[TokenizerWordLevel] first 10 words in word2idx: {list(word2idx.keys())[:10]}')
        return word2idx, idx2word


class TokenizerCharLevel(Tokenizer):
    def __init__(self):
        super(TokenizerCharLevel, self).__init__()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return list(text)

    @staticmethod
    def counter(token_lists: List[str]) -> Counter:
        return Counter(token_lists)


class TokenizerTitleToLyrics(TokenizerWordLevel):
    def __init__(self):
        super(TokenizerTitleToLyrics, self).__init__()

    def tokenize(self, text: str) -> List[str]:
        # No longer append <UNK> at the end; just use parent logic
        return super().tokenize(text)
    
    def build_vocab(self, token_lists: List[str], min_freq: int = 1, is_debug: bool = False) -> Tuple[Dict[str, int], Dict[int, str]]:
        word2idx, idx2word = super().build_vocab(token_lists=token_lists, 
                                                 min_freq=min_freq, 
                                                 is_debug=is_debug)
        # Ensure <UNK> and '\n' are always present
        for special in ["<UNK>", "\n"]:
            if special not in word2idx:
                next_idx = max(word2idx.values(), default=-1) + 1
                word2idx[special] = next_idx
                idx2word[next_idx] = special
        return word2idx, idx2word