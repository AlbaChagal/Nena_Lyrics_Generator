from typing import Optional, Dict, Tuple, Union, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.train.dataset.title_to_lyrics_dataset import TitleToLyricsDataset
from src.train.dataset.tokenizer import Tokenizer, TokenizerWordLevel, TokenizerCharLevel, TokenizerTitleToLyrics
from src.global_constants import clean_lyrics_dir
from src.train.dataset.char_dataset import CharDataset
from src.train.dataset.word_dataset import WordDataset
from src.train.pretrained_embedder import Embedder
from src.train.training_config import TrainingConfig, DatasetType


class DataManager:
    def __init__(self,
                 training_config: TrainingConfig,
                 random_state: np.random.RandomState,
                 embedding_matrix: Optional[torch.Tensor] = None,
                 is_debug: bool = False):
        self.is_debug: bool = is_debug
        self.training_config: TrainingConfig = training_config
        self.random_state: np.random.RandomState = random_state
        self.dataset: Optional[Union[CharDataset, WordDataset, TitleToLyricsDataset]] = None
        self.tokenizer: Optional[Tokenizer] = None
        self.embedding_matrix: Optional[torch.Tensor] = embedding_matrix
        self.word2idx: Optional[Dict[str, int]] = None
        self.idx2word: Optional[Dict[int, str]] = None

    @staticmethod
    def get_embedding_matrix(training_config: TrainingConfig,
                             word2idx: Optional[Dict[str, int]]) -> Optional[torch.Tensor]:
        embedding_matrix: Optional[torch.Tensor] = None
        if training_config.is_use_pretrained_model:
            assert word2idx is not None
            embedding_matrix = Embedder(training_config).create_embedding_matrix(word2idx)
        return embedding_matrix

    @property
    def vocab(self):
        assert self.dataset is not None, "Dataset is not initialized"
        return self.dataset.vocab

    def load_data(self) -> DataLoader:
        text: str = self.load_vocab_file(training_config=self.training_config, is_debug=self.is_debug)

        self.dataset = \
            self.get_datasets(training_config=self.training_config, 
                              text=text, 
                              random_state=self.random_state, 
                              is_debug=self.is_debug)
        if self.is_debug:
            print(f"[DataManager][load_data] Loaded dataset with vocab size: {len(self.dataset.vocab)}")
            print(f"[DataManager][load_data] Loaded word2idx {list(self.dataset.idx2word.values())[:100]}...")
        ttl_dataloader: DataLoader = DataLoader(self.dataset,
                                                batch_size=1,
                                                shuffle=False)
        return ttl_dataloader

    @staticmethod
    def load_vocab_file(training_config: TrainingConfig = None, is_debug: bool = False) -> str:
        from src.global_constants import char_vocab_file_path, word_vocab_file_path
        vocab_file = char_vocab_file_path
        if training_config is not None and hasattr(training_config, 'dataset_class'):
            if training_config.dataset_class == DatasetType.WordDataset or str(training_config.dataset_class) == "WordDataset":
                vocab_file = word_vocab_file_path
        with open(vocab_file, "r", encoding="utf-8") as f:
            text = f.read()
        if is_debug:
            print(f"[load_vocab_file] Loaded vocab file: {vocab_file} with length {len(text)}")
        return text

    @staticmethod
    def get_datasets(
            training_config: TrainingConfig,
            text: str,
            random_state: np.random.RandomState,
            is_debug: bool = False
    ) -> TitleToLyricsDataset:
        if training_config.dataset_class == DatasetType.CharDataset:
            tokenizer = TokenizerCharLevel()
        elif training_config.dataset_class == DatasetType.WordDataset:
            tokenizer = TokenizerWordLevel()
        else:
            raise ValueError(f'unknown dataset class {training_config.dataset_class}')

        # TitleToLyricsDataset always uses TokenizerTitleToLyrics
        title_to_lyrics_tokenizer: TokenizerTitleToLyrics = TokenizerTitleToLyrics()
        ttl_word2idx, ttl_idx2word = title_to_lyrics_tokenizer.build_vocab(title_to_lyrics_tokenizer.tokenize(text))
        title_to_lyrics_dataset: TitleToLyricsDataset = TitleToLyricsDataset(
            inputs_path=clean_lyrics_dir,
            tokenizer=tokenizer,
            word2idx=ttl_word2idx,
            idx2word=ttl_idx2word,
            is_debug=is_debug
        )
        title_to_lyrics_dataset.load_title_lyrics_pairs()

        return title_to_lyrics_dataset
