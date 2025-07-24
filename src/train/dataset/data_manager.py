from typing import Optional, Dict, Tuple, Union

import torch
from torch.utils.data import DataLoader

from src.train.dataset.tokenizer import Tokenizer, TokenizerWordLevel, TokenizerCharLevel
from src.global_constants import vocab_file_path
from src.train.dataset.char_dataset import CharDataset
from src.train.dataset.word_dataset import WordDataset
from src.train.pretrained_embedder import Embedder
from src.train.training_config import TrainingConfig


class DataManager:
    def __init__(self, training_config: TrainingConfig, embedding_matrix: Optional[torch.Tensor] = None):
        self.training_config: TrainingConfig = training_config
        self.dataset: Optional[Union[CharDataset, WordDataset]] = None
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
        return self.dataset.vocab

    def load_data(self):
        print(f'load_data - load vocab file - start')
        text: str = self.load_vocab_file()
        print(f'load_data - load vocab file - finish')

        self.dataset, self.word2idx, self.idx2word = \
            self.get_dataset(training_config=self.training_config, text=text)
        print(f'load_data - create emdedding matrix - start')
        if self.embedding_matrix is None:
            self.embedding_matrix = self.get_embedding_matrix(training_config=self.training_config,
                                                              word2idx=self.word2idx)
        print(f'load_data - create emdedding matrix - finish')

        print(f'load_data - create DataLoader - start')
        dataloader: DataLoader = DataLoader(self.dataset,
                                            batch_size=self.training_config.batch_size,
                                            shuffle=True)
        print(f'load_data - create DataLoader - finish')
        return dataloader, self.embedding_matrix

    @staticmethod
    def load_vocab_file() -> str:
        with open(vocab_file_path, "r", encoding="utf-8") as f:
            text: str = f.read()
        return text

    @staticmethod
    def get_dataset(
            training_config: TrainingConfig,
            text: str
    ) -> Tuple[Union[CharDataset, WordDataset],
               Optional[Dict[str, int]],
               Optional[Dict[int, str]]]:
        dataset: Union[CharDataset, WordDataset]
        if training_config.dataset_class == CharDataset:
            tokenizer = TokenizerCharLevel()
        elif training_config.dataset_class == WordDataset:
            tokenizer = TokenizerWordLevel()
        else:
            raise ValueError(f'unknown dataset class {training_config.dataset_class}')

        print('get_dataset - tokenize text - start')
        tokens = tokenizer.tokenize(text)
        print('get_dataset - tokenize text - finish')
        print('get_dataset - build vocabulary - start')
        word2idx, idx2word = tokenizer.build_vocab(tokens)
        vocab = list(word2idx.keys())
        print('get_dataset - build vocabulary - finish')
        token_ids = [word2idx[w] for w in tokens if w in word2idx]

        # Dataset
        print('get_dataset - create word dataset - start')
        if training_config.dataset_class == CharDataset:
            dataset: CharDataset = CharDataset(text, block_size=training_config.char_block_size)
        elif training_config.dataset_class == WordDataset:
            dataset: WordDataset = WordDataset(token_ids, block_size=training_config.word_block_size, vocab=vocab)
        else:
            raise ValueError(f'unknown dataset class {training_config.dataset_class}')
        print('get_dataset - create word dataset - finish')
        return dataset, word2idx, idx2word