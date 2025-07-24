from typing import Dict

import gensim.downloader as api
import torch
import numpy as np
from tqdm import tqdm

from src.train.training_config import TrainingConfig


class Embedder:
    def __init__(self, training_config: TrainingConfig) -> None:
        self.training_config = training_config
        self.model: torch.nn.Module = api.load(self.training_config.pretrained_model_name)
        self.embedding_dim: int = self.training_config.embedder_dim

    def create_embedding_matrix(self, word2idx: Dict[str, int]) -> torch.Tensor:
        embedding_matrix = torch.zeros((len(word2idx), self.embedding_dim))
        for word, idx in tqdm(word2idx.items()):
            if word in self.model:
                embedding_matrix[idx] = torch.tensor(self.model[word])
            else:
                embedding_matrix[idx] = torch.randn(self.embedding_dim) * 0.01  # random for OOV
        return embedding_matrix
