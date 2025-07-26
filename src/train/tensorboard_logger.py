import numpy as np
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    def __init__(self, log_dir: str) -> None:
        self.writer: SummaryWriter = SummaryWriter(log_dir)
        self.loss: float = 0.
        self.next_loss: float = 0.
        self.masked_loss: float = 0.
        self.word_regularization_loss: float = 0.
        self.num_updates: int = 0

    def update(self,
               loss: float,
               next_loss: float,
               masked_loss: float,
               word_regularization_loss: float) -> None:
        self.loss += loss
        self.next_loss += next_loss
        self.masked_loss += masked_loss
        self.word_regularization_loss += word_regularization_loss
        self.num_updates += 1

    def reset(self):
        self.loss: float = 0.
        self.next_loss: float = 0.
        self.masked_loss: float = 0.
        self.word_regularization_loss: float = 0.
        self.num_updates: int = 0

    def close(self) -> None:
        self.reset()
        self.writer.close()

    def log(self, step: int) -> None:
        # Losses
        avg_loss: float = self.loss / self.num_updates
        avg_next_loss: float = self.next_loss / self.num_updates
        avg_masked_loss: float = self.masked_loss / self.num_updates
        avg_regularization_loss: float = self.word_regularization_loss / self.num_updates

        # Perplexity
        avg_perplexity: float = np.exp(avg_loss)
        avg_next_perplexity: float = np.exp(avg_next_loss)
        avg_masked_perplexity: float = np.exp(avg_masked_loss)
        avg_regularization_perplexity: float = np.exp(avg_regularization_loss)

        self.writer.add_scalar("losses/1.loss", avg_loss, step)
        self.writer.add_scalar("losses/2.next_loss", avg_next_loss, step)
        self.writer.add_scalar("losses/3.masked_loss", avg_masked_loss, step)
        self.writer.add_scalar("losses/4.regularization_loss", avg_regularization_loss, step)
        self.writer.add_scalar("perplexity/1.perplexity", avg_perplexity, step)
        self.writer.add_scalar("perplexity/2.next_perplexity", avg_next_perplexity, step)
        self.writer.add_scalar("perplexity/3.masked_perplexity", avg_masked_perplexity, step)
        self.writer.add_scalar("perplexity/4.regularization_loss", avg_regularization_perplexity, step)
        self.reset()
