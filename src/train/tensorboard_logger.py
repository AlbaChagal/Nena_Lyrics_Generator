import numpy as np
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    def __init__(self, log_dir: str) -> None:
        self.writer: SummaryWriter = SummaryWriter(log_dir)
        self.loss: float = 0.
        self.next_loss: float = 0.
        self.masked_loss: float = 0.
        self.num_updates: int = 0

    def update(self,
               loss: float,
               next_loss: float,
               masked_loss: float) -> None:
        self.loss += loss
        self.next_loss += next_loss
        self.masked_loss += masked_loss
        self.num_updates += 1

    def reset(self):
        self.loss: float = 0.
        self.next_loss: float = 0.
        self.masked_loss: float = 0.
        self.num_updates: int = 0

    def close(self) -> None:
        self.reset()
        self.writer.close()

    def log(self, step: int) -> None:
        # Losses
        avg_loss: float = self.loss / self.num_updates
        avg_next_loss: float = self.next_loss / self.num_updates
        avg_masked_loss: float = self.masked_loss / self.num_updates

        # Perplexity
        avg_perplexity: float = np.exp(avg_loss)
        avg_next_perplexity: float = np.exp(avg_next_loss)
        avg_masked_perplexity: float = np.exp(avg_masked_loss)

        self.writer.add_scalar("1. loss", avg_loss, step)
        self.writer.add_scalar("2. next_loss", avg_next_loss, step)
        self.writer.add_scalar("3. masked_loss", avg_masked_loss, step)
        self.writer.add_scalar("4. perplexity", avg_perplexity, step)
        self.writer.add_scalar("5. next_perplexity", avg_next_perplexity, step)
        self.writer.add_scalar("6. masked_perplexity", avg_masked_perplexity, step)
        self.reset()
