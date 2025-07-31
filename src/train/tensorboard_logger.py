import numpy as np
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    def __init__(self, log_dir: str) -> None:
        self.log_dir: str = log_dir
        self.writer: SummaryWriter = SummaryWriter(log_dir)
        self.loss: float = 0.
        self.word_percentage_in_output: float = 0.
        self.newline_loss: float = 0.
        self.output_length: float = 0.
        self.newline_tpr: float = 0.
        self.newline_tnr: float = 0.
        self.eos_loss: float = 0.
        self.eos_tpr: float = 0.
        self.eos_tnr: float = 0.
        self.num_updates: int = 0

    def update(self,
               loss: float,
               word_percentage_in_output: float,
               newline_loss: float = 0.0,
               output_length: float = 0.0,
               newline_tpr: float = 0.0,
               newline_tnr: float = 0.0,
               eos_loss: float = 0.0,
               eos_tpr: float = 0.0,
               eos_tnr: float = 0.0) -> None:
        self.loss += loss
        self.word_percentage_in_output += word_percentage_in_output
        self.newline_loss += newline_loss
        self.output_length += output_length
        self.newline_tpr += newline_tpr
        self.newline_tnr += newline_tnr
        self.eos_loss += eos_loss
        self.eos_tpr += eos_tpr
        self.eos_tnr += eos_tnr
        self.num_updates += 1

    def reset(self):
        self.loss: float = 0.
        self.word_percentage_in_output: float = 0.
        self.newline_loss: float = 0.
        self.output_length: float = 0.
        self.newline_tpr: float = 0.
        self.newline_tnr: float = 0.
        self.eos_loss: float = 0.
        self.eos_tpr: float = 0.
        self.eos_tnr: float = 0.
        self.num_updates: int = 0

    def close(self) -> None:
        self.reset()
        self.writer.close()

    def log(self, step: int) -> None:
        # Losses
        avg_loss: float = self.loss / self.num_updates
        avg_word_percentage_in_output: float = self.word_percentage_in_output / self.num_updates
        avg_newline_loss: float = self.newline_loss / self.num_updates
        avg_output_length: float = self.output_length / self.num_updates
        avg_newline_tpr: float = self.newline_tpr / self.num_updates
        avg_newline_tnr: float = self.newline_tnr / self.num_updates
        avg_eos_loss: float = self.eos_loss / self.num_updates
        avg_eos_tpr: float = self.eos_tpr / self.num_updates
        avg_eos_tnr: float = self.eos_tnr / self.num_updates

        # Perplexity
        avg_perplexity: float = np.exp(avg_loss)
        avg_word_percentage_in_output_preplexity: float = np.exp(avg_word_percentage_in_output)
        avg_newline_perplexity: float = np.exp(avg_newline_loss)

        self.writer.add_scalar("losses/1.loss", avg_loss, step)
        self.writer.add_scalar("losses/1.word_percentage_in_output", avg_word_percentage_in_output, step)
        self.writer.add_scalar("losses/2.newline_loss", avg_newline_loss, step)
        self.writer.add_scalar("losses/3.eos_loss", avg_eos_loss, step)
        self.writer.add_scalar("outputs/avg_output_length", avg_output_length, step)
        self.writer.add_scalar("newline/TPR", avg_newline_tpr, step)
        self.writer.add_scalar("newline/TNR", avg_newline_tnr, step)
        self.writer.add_scalar("eos/TPR", avg_eos_tpr, step)
        self.writer.add_scalar("eos/TNR", avg_eos_tnr, step)

        self.writer.add_scalar("perplexity/1.perplexity", avg_perplexity, step)
        self.writer.add_scalar("perplexity/1.word_percentage_in_output_perplexity", avg_word_percentage_in_output_preplexity, step)
        self.writer.add_scalar("perplexity/2.newline_perplexity", avg_newline_perplexity, step)
        self.reset()

