from typing import Dict

import torch

from src.global_constants import get_all_special_tokens, eos_token, new_line_token, separator_token, bos_token, \
    unknown_token
from src.train.metrics.confusion_matrix_calculators import MaskedConfusionMatrixCalculator
from src.train.metrics.metrics_datastructs import ConfusionMatrix, Metrics, AllMetrics


class AllMetricsCalculator(object):
    def __init__(self, word2idx: Dict[str, int], device: torch.device):
        self.word2idx: Dict[str, int] = word2idx
        self.device: torch.device = device
        self.general_confusion_matrix: MaskedConfusionMatrixCalculator = \
            MaskedConfusionMatrixCalculator(
                word2idx=self.word2idx,
                desired_tokens=torch.tensor([self.word2idx[token] for token in get_all_special_tokens()],
                                            device=self.device),
                is_mask_opposite=True
        )
        self.eos_confusion_matrix: MaskedConfusionMatrixCalculator = \
            MaskedConfusionMatrixCalculator(
                word2idx=self.word2idx,
                desired_tokens=self.word2idx[eos_token],
                is_mask_opposite=False
            )
        self.new_line_confusion_matrix: MaskedConfusionMatrixCalculator = \
            MaskedConfusionMatrixCalculator(
                word2idx=self.word2idx,
                desired_tokens=self.word2idx[new_line_token],
                is_mask_opposite=False
            )
        self.separator_confusion_matrix: MaskedConfusionMatrixCalculator = \
            MaskedConfusionMatrixCalculator(
                word2idx=self.word2idx,
                desired_tokens=self.word2idx[separator_token],
                is_mask_opposite=False
            )
        self.bos_confusion_matrix: MaskedConfusionMatrixCalculator = \
            MaskedConfusionMatrixCalculator(
                word2idx=self.word2idx,
                desired_tokens=self.word2idx[bos_token],
                is_mask_opposite=False
            )
        self.unknown_confusion_matrix: MaskedConfusionMatrixCalculator = \
            MaskedConfusionMatrixCalculator(
                word2idx=self.word2idx,
                desired_tokens=self.word2idx[unknown_token],
                is_mask_opposite=False
            )

    def calculate(self, logits: torch.Tensor, labels: torch.Tensor) -> AllMetrics:

        general_confusion_matrix: ConfusionMatrix = \
            self.general_confusion_matrix.calculate_matrix(logits, labels)
        eos_confusion_matrix: ConfusionMatrix = \
            self.eos_confusion_matrix.calculate_matrix(logits, labels)
        new_line_confusion_matrix: ConfusionMatrix = \
            self.new_line_confusion_matrix.calculate_matrix(logits, labels)
        separator_confusion_matrix: ConfusionMatrix = \
            self.separator_confusion_matrix.calculate_matrix(logits, labels)
        bos_confusion_matrix: ConfusionMatrix = \
            self.bos_confusion_matrix.calculate_matrix(logits, labels)
        unknown_confusion_matrix: ConfusionMatrix = \
            self.unknown_confusion_matrix.calculate_matrix(logits, labels)

        general_token_metrics: Metrics = \
            self.general_confusion_matrix.calculate_metrics(general_confusion_matrix)
        eos_token_metrics: Metrics = \
            self.eos_confusion_matrix.calculate_metrics(eos_confusion_matrix)
        new_line_token_metrics: Metrics = \
            self.new_line_confusion_matrix.calculate_metrics(new_line_confusion_matrix)
        separator_token_metrics: Metrics = \
            self.separator_confusion_matrix.calculate_metrics(separator_confusion_matrix)
        bos_token_metrics: Metrics = \
            self.bos_confusion_matrix.calculate_metrics(bos_confusion_matrix)
        unknown_token_metrics: Metrics = \
            self.unknown_confusion_matrix.calculate_metrics(unknown_confusion_matrix)

        return AllMetrics(
            general_token_metrics=general_token_metrics,
            eos_token_metrics=eos_token_metrics,
            new_line_token_metrics=new_line_token_metrics,
            separator_token_metrics=separator_token_metrics,
            bos_token_metrics=bos_token_metrics,
            unknown_token_metrics=unknown_token_metrics
        )
