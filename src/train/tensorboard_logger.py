from typing import Union

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.train.losses.losses_datastructs import AllLosses
from src.train.metrics.metrics_datastructs import AllMetrics, ConfusionMatrix, Metrics


class TensorboardLogger:
    def __init__(self, log_dir: str) -> None:
        self.log_dir: str = log_dir
        self.writer: SummaryWriter = SummaryWriter(log_dir)
        self.all_losses: AllLosses = AllLosses()
        self.word_percentage_in_output: float = 0.
        self.newline_loss: float = 0.
        self.output_length: float = 0.
        self.all_metrics: AllMetrics = AllMetrics.init_empty_instance()
        self.num_updates: int = 0

    def update(self,
               all_losses: AllLosses,
               word_percentage_in_output: float,
               all_metrics: AllMetrics,
               newline_loss: float = 0.0,
               output_length: float = 0.0) -> None:
        self.all_losses += all_losses
        self.word_percentage_in_output += word_percentage_in_output
        self.newline_loss += newline_loss
        self.output_length += output_length
        self.all_metrics += all_metrics
        self.num_updates += 1

    def reset(self):
        self.all_losses: AllLosses = AllLosses()
        self.word_percentage_in_output: float = 0.
        self.newline_loss: float = 0.
        self.output_length: float = 0.
        self.all_metrics: AllMetrics = AllMetrics.init_empty_instance()
        self.num_updates = 0

    def close(self) -> None:
        self.reset()
        self.writer.close()

    def log(self, step: int) -> None:
        # Compute averages
        all_avg_losses = self.all_losses / self.num_updates
        all_avg_metrics = self.all_metrics / self.num_updates
        avg_word_percentage_in_output: float = self.word_percentage_in_output / self.num_updates
        avg_output_length: float = self.output_length / self.num_updates

        # Log all losses
        for loss_name in self.all_losses.__slots__:
            self.writer.add_scalar(f"losses/{loss_name}", getattr(all_avg_losses, loss_name), step)

        # Log all metrics
        for single_metric_name in self.all_metrics.__slots__:
            single_metrics: Metrics = getattr(all_avg_metrics, single_metric_name)
            for slot in single_metrics.__slots__:
                attr = getattr(single_metrics, slot)
                if type(attr) == float:
                    self.writer.add_scalar(f"metrics/{single_metric_name}", attr, step)
                elif type(attr) == ConfusionMatrix:
                    for slot in attr.__slots__:
                        self.writer.add_scalar(f"metrics/{single_metric_name}_{slot}",
                                               getattr(attr, slot), step)
                else:
                    raise TypeError(f'all_metrics.{attr} is not a float or '
                                    f'ConfusionMatrix, got: {type(attr)}')

        # Log output stats
        self.writer.add_scalar("outputs/avg_output_length", avg_output_length, step)
        self.writer.add_scalar("outputs/avg_word_percentage_in_output", avg_word_percentage_in_output, step)

        # Log perplexity for each loss (if meaningful)
        for loss_name in self.all_losses.__slots__:
            avg_loss_val = getattr(all_avg_losses, loss_name)
            if avg_loss_val > 0:
                self.writer.add_scalar(f"perplexity/{loss_name}", float(np.exp(avg_loss_val)), step)

        self.reset()

