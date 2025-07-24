import numpy as np
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    def __init__(self, log_dir: str) -> None:
        self.writer: SummaryWriter = SummaryWriter(log_dir)
        self.loss: float = 0.
        self.num_updates: int = 0

    def update(self, loss: float) -> None:
        self.loss += loss
        self.num_updates += 1

    def reset(self):
        self.loss: float = 0.
        self.num_updates: int = 0

    def close(self) -> None:
        self.reset()
        self.writer.close()

    def log(self, step: int) -> None:
        avg_loss: float = self.loss / self.num_updates
        avg_perplexity: float = np.exp(avg_loss)

        self.writer.add_scalar("loss", avg_loss, step)
        self.writer.add_scalar("perplexity", avg_perplexity, step)
        self.reset()
