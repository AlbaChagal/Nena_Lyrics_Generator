import re
from typing import Dict, List

import torch
from wordfreq import word_frequency

class WordRegulator:
    def __init__(self):
        pass

    @staticmethod
    def is_german_word(word: str) -> bool:
        return word_frequency(word, 'de') > 0

    @staticmethod
    def get_words_from_logits(logits: torch.Tensor, idx2char: Dict[int, str]) -> List[str]:
        predicted_chars: torch.Tensor = logits.argmax(dim=-1)  # shape: (batch, seq_len)
        generated_text: str = "".join([idx2char[idx] for idx in predicted_chars[0].tolist()])
        words: List[str] = re.findall(r"\w+", generated_text)
        return words

    def forward(self, logits: torch.Tensor, idx2char: Dict[int, str]) -> float:
        predicted_words: List[str] = self.get_words_from_logits(logits, idx2char)
        num_real_words: int = 0
        for word in predicted_words:
            num_real_words += int(self.is_german_word(word))
        if len(predicted_words) == 0:
            return 1.
        return 1. - (num_real_words / len(predicted_words))



