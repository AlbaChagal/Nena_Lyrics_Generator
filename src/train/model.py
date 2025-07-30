from typing import Optional

import torch
from torch import nn

from src.train.pretrained_embedder import Embedder
from src.train.training_config import TrainingConfig

class Model(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 training_config: TrainingConfig,
                 embedding_dim: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 max_seq_len: int = 1024):
        super().__init__()

        self.embedding: nn.Embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = self._generate_positional_encoding(max_seq_len, embedding_dim)

        self.transformer: nn.Transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.output_fc: nn.Linear = nn.Linear(embedding_dim, vocab_size)

    @staticmethod
    def _generate_positional_encoding(max_len: int, model_dim: int) -> torch.Tensor:
        position: torch.Tensor = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) * (-torch.log(torch.tensor(10000.0)) / model_dim))
        pe: torch.Tensor = torch.zeros(max_len, model_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # shape (1, max_len, d_model)

    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        max_length: int = 128,
        start_token_idx: int = 0,
        return_logits: bool = False,
        return_last_logits: bool = False
    ) -> torch.Tensor:
        batch_size = src.size(0)
        device = src.device
        src_emb = self.embedding(src) + self.positional_encoding[:, :src.size(1), :].to(device)

        if return_logits and tgt is not None:
            # Teacher forcing for training: return logits for all steps
            tgt_input = tgt[:, :-1]  # Remove last token (usually <EOS>)
            tgt_emb = self.embedding(tgt_input) + self.positional_encoding[:, :tgt_input.size(1), :].to(device)
            tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            output = self.transformer(src=src_emb, tgt=tgt_emb, tgt_mask=tgt_mask)
            logits = self.output_fc(output)  # (batch_size, tgt_seq_len, vocab_size)
            return logits
        elif return_last_logits:
            # For inference: return logits for the last token only
            seq_len = src.size(1)
            generated = src
            tgt_emb = self.embedding(generated) + self.positional_encoding[:, :generated.size(1), :].to(device)
            tgt_mask = self.transformer.generate_square_subsequent_mask(generated.size(1)).to(device)
            output = self.transformer(src=src_emb, tgt=tgt_emb, tgt_mask=tgt_mask)
            logits = self.output_fc(output)  # (batch_size, seq_len, vocab_size)
            return logits[:, -1, :]  # (batch_size, vocab_size)
        else:
            # Autoregressive generation for inference (greedy)
            generated = torch.full((batch_size, 1), start_token_idx, dtype=torch.long, device=device)
            for _ in range(max_length):
                tgt_emb = self.embedding(generated) + self.positional_encoding[:, :generated.size(1), :].to(device)
                tgt_mask = self.transformer.generate_square_subsequent_mask(generated.size(1)).to(device)
                output = self.transformer(src=src_emb, tgt=tgt_emb, tgt_mask=tgt_mask)
                logits = self.output_fc(output[:, -1:, :])  # last token logits
                next_token = torch.argmax(logits, dim=-1)
                generated = torch.cat([generated, next_token], dim=1)
            return generated[:, 1:]
