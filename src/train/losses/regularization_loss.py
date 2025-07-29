import re
from typing import Dict, List

import torch
from wordfreq import word_frequency

class WordsInTextCounter:
    def __init__(self, is_debug: bool = False):
        self.is_debug: bool = is_debug

    @staticmethod
    def is_german_word(word: str) -> bool:
        return word_frequency(word, 'de') > 0.0005 and len(word) >= 2

    @staticmethod
    def get_words_from_logits(logits: torch.Tensor, idx2char: Dict[int, str], is_debug: bool = False) -> List[str]:
        predicted_chars: torch.Tensor = logits.argmax(dim=-1)  # shape: (batch, seq_len)
        generated_text: str = "".join([idx2char[idx] for idx in predicted_chars[0].tolist()])
        words: List[str] = re.findall(r"\w+", generated_text)
        if is_debug:
            print(f'Generated text: {generated_text}...')
            print(f'Extracted words: {words[:10]}')
        return words

    def count_valid_words(self, logits: torch.Tensor, idx2char: Dict[int, str], is_debug: bool = False) -> float:
        predicted_words: List[str] = self.get_words_from_logits(logits=logits, 
                                                                idx2char=idx2char, 
                                                                is_debug=self.is_debug)
        if len(predicted_words) == 0:
            return 0.
        num_invalid = sum(1 for word in predicted_words if not self.is_german_word(word))
        if is_debug:
            print(f'got words from logits, num invalid: {num_invalid} / {len(predicted_words)}')
        return 1. - (num_invalid / len(predicted_words))



