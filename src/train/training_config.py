import json
from typing import Union

from torch.utils.data import Dataset

from src.train.dataset.char_dataset import CharDataset
from src.train.dataset.word_dataset import WordDataset


class TrainingConfig:
    def __init__(self):

        # Pretrained embedder params
        self.embedder_dim: int = 300

        # Tokenizer params
        self.char_block_size: int = 128
        self.word_block_size: int = 30

        # Data params
        self.dataset_class: Union[type[CharDataset], type[WordDataset]] = CharDataset
        self.num_epochs: int = 600
        self.batch_size: int = 32

        # Checkpoint params
        self.save_checkpoint_freq: int = 1000

        # Loss params
        self.learning_rate = 0.0001
        self.weight_decay = 1e-5

        # Log params
        self.logging_level = 'info'

        # Model structure params
        self.is_use_pretrained_model: bool = False
        self.pretrained_model_name: str = 'fasttext-wiki-news-subwords-300'


    def to_string(self) -> str:
        return str(self.__dict__)

    @staticmethod
    def from_string(string: str) -> 'TrainingConfig':
        config: 'TrainingConfig' = TrainingConfig()
        config.__dict__ = json.loads(string)
        return config

    @classmethod
    def from_json(cls, json_str: str) -> 'TrainingConfig':
        config: 'TrainingConfig' = TrainingConfig.from_string(json_str)
        return config
