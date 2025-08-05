from typing import Dict, Union, List

import torch

from src.train.metrics.metrics_datastructs import ConfusionMatrix, Metrics


class MaskedConfusionMatrixCalculator(object):
    def __init__(self,
                 word2idx: Dict[str, int],
                 desired_tokens: Union[torch.Tensor, int],
                 is_mask_opposite: bool = False):

        self.word2idx: Dict[str, int] = word2idx
        self.desired_tokens: Union[torch.Tensor, int] = desired_tokens
        self.is_mask_opposite: bool = is_mask_opposite

    @staticmethod
    def get_tensor_mask(
            src: torch.Tensor,
            desired_tokens: Union[torch.Tensor, int],
    ) -> torch.Tensor:

        is_desired_tokens: torch.Tensor
        if isinstance(desired_tokens, int):
            is_desired_tokens = src == desired_tokens
        elif isinstance(desired_tokens, torch.Tensor):
            is_desired_tokens: torch.Tensor = torch.isin(src, desired_tokens)
        else:
            raise TypeError(f'desired_tokens should be a tensor, '
                            f'string or a list of strings, got type: {type(desired_tokens)}')

        return is_desired_tokens

    def calculate_matrix(self,
                         logits: torch.Tensor,
                         target: torch.Tensor) -> ConfusionMatrix:

        masked_preds: torch.Tensor = \
            self.get_tensor_mask(logits.argmax(dim=-1), self.desired_tokens)
        masked_labels: torch.Tensor = \
            self.get_tensor_mask(target, self.desired_tokens)

        is_pred_positive: torch.Tensor
        is_label_positive: torch.Tensor
        if self.is_mask_opposite:
            is_pred_positive = ~masked_preds
            is_label_positive = ~masked_labels
        else:
            is_pred_positive = masked_preds
            is_label_positive = masked_labels

        gt_positive_inds: torch.Tensor = torch.where(is_label_positive)[1]
        gt_negative_inds: torch.Tensor = torch.where(~is_label_positive)[1]
        positive_pred_inds: torch.Tensor = torch.where(is_pred_positive)[1]
        negative_pred_inds: torch.Tensor = torch.where(~is_pred_positive)[1]

        tp: int = torch.count_nonzero(torch.isin(positive_pred_inds, gt_positive_inds)).detach().cpu().item()
        tn: int = torch.count_nonzero(torch.isin(negative_pred_inds, gt_negative_inds)).detach().cpu().item()
        fp: int = torch.count_nonzero(torch.isin(positive_pred_inds, gt_negative_inds)).detach().cpu().item()
        fn: int = torch.count_nonzero(torch.isin(negative_pred_inds, gt_positive_inds)).detach().cpu().item()

        return ConfusionMatrix(tp=tp, tn=tn, fp=fp, fn=fn)

    @staticmethod
    def calculate_metrics(confusion_matrix: ConfusionMatrix) -> Metrics:

        if confusion_matrix.tp + confusion_matrix.fn != 0:
            tpr: float = confusion_matrix.tp / float(confusion_matrix.tp + confusion_matrix.fn)
        else:
            tpr: float = 0
        if confusion_matrix.tp + confusion_matrix.fp != 0:
            tnr: float = confusion_matrix.tn / float(confusion_matrix.tp + confusion_matrix.fp)
        else:
            tnr: float = 0

        acc: float = (confusion_matrix.tp + confusion_matrix.tn) / \
                     (confusion_matrix.tp + confusion_matrix.fp + confusion_matrix.fn + confusion_matrix.tn)

        return Metrics(tpr=tpr, tnr=tnr, acc=acc, confusion_matrix=confusion_matrix)

