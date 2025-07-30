import json
import os
from enum import Enum
from typing import Union

from src.global_constants import config_json_name

class DatasetType(Enum):
    WordDataset = 'WordDataset'
    CharDataset = 'CharDataset'

class TrainingConfig:
    def __init__(self):

        # Pretrained embedder params
        self.loss_weight = 1.0
        self.newline_loss_weight: float = 1.0
        self.embedder_dim: int = 300

        # Tokenizer params
        self.char_block_size: int = 128
        self.word_block_size: int = 30

        # Data params
        self.random_seed: int = 0
        self.dataset_class: Union[DatasetType, str] = DatasetType.WordDataset
        self.num_epochs: int = 600
        self.batch_size: int = 32

        # Checkpoint params
        self.save_checkpoint_freq: int = 1000

        # Loss params
        self.learning_rate: float = 0.0001
        self.weight_decay: float = 1e-5

        # Log params
        self.logging_level: str = 'info'

        # Model structure params
        self.is_use_pretrained_model: bool = False
        self.pretrained_model_name: str = 'fasttext-wiki-news-subwords-300'

    def serialize(self):
        serialized = self.__dict__
        serialized['dataset_class'] = serialized['dataset_class'].value
        return serialized

    def to_string(self) -> str:
        return str(self.serialize())

    @staticmethod
    def from_string(string: str) -> 'TrainingConfig':
        config: 'TrainingConfig' = TrainingConfig()
        config.__dict__ = dict(json.loads(string))
        return config

    @classmethod
    def from_json(cls, json_file_path: str) -> 'TrainingConfig':
        with open(json_file_path, 'r', encoding='utf-8') as f:
            config_json = f.read()
        config: 'TrainingConfig' = TrainingConfig()
        config.__dict__ = json.loads(config_json)
        config.dataset_class = getattr(DatasetType, config.dataset_class)
        return config

if __name__ == '__main__':
    config = TrainingConfig()
    path = '/Users/shaharheyman/PycharmProjects/nena_lyrics_generator/checkpoints/20250724_105120'
    with open(os.path.join(path, config_json_name), 'w', encoding='utf-8') as f:
        json.dump(config.serialize(), f)