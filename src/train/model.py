from typing import Optional

import torch
from torch import nn

from src.train.pretrained_embedder import Embedder
from src.train.training_config import TrainingConfig


class Model(nn.Module):
    def __init__(self,
                 training_config: TrainingConfig,
                 vocab_size: Optional[int] = None,
                 embedding_matrix: Optional[torch.Tensor] = None,
                 embedding_dim: int = 256,
                 hidden_dim: int = 512,
                 dropout: float = 0.2):

        super(Model, self).__init__()

        self.training_config = training_config
        if self.training_config.is_use_pretrained_model:
            assert embedding_matrix is not None, \
                'embedding_matrix must be provided when loading a pretrained embedder'
            self.embedder = nn.Embedding.from_pretrained(
                embeddings=embedding_matrix,
                freeze=True
            )
            embedding_dim = embedding_matrix.shape[1]  # Ensure consistency
        else:
            assert vocab_size is not None, \
                'vocab_size must be provided when loading an untrained embedder'
            self.embedder = nn.Embedding(num_embeddings=vocab_size,
                                         embedding_dim=embedding_dim)

        self.lstm: nn.LSTM = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=8,
            dropout=dropout,
            batch_first=True
        )
        self.fc: nn.Linear = nn.Linear(in_features=hidden_dim,
                                       out_features=vocab_size)

    def forward(self, x):
        x_embedded = self.embedder(x)
        output, hidden = self.lstm(x_embedded)
        logits = self.fc(output)
        return logits, hidden


