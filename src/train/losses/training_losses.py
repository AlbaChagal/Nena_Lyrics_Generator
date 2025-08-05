from typing import Callable, Dict, List, Tuple

import torch

from src.global_constants import new_line_token, eos_token, unknown_token, separator_token, get_all_special_tokens, \
    bos_token
from src.train.losses.losses_datastructs import AllLosses
from src.train.training_config import TrainingConfig


class LossFunction(torch.nn.Module):
    def __init__(self, training_config: TrainingConfig, device: torch.device):
        super(LossFunction, self).__init__()
        self.device: torch.device = device
        self.general_character_loss: GeneralCharacterLoss = \
            GeneralCharacterLoss(self.device,
                                 weight=training_config.general_character_loss_weight)
        self.eos_loss = \
            EOSLoss(self.device,
                    weight=training_config.eos_loss_weight)
        self.unknown_loss: UnknownLoss = \
            UnknownLoss(self.device,
                        weight=training_config.unknown_loss_weight)
        self.new_line_loss: NewLineLoss = \
            NewLineLoss(self.device,
                        weight=training_config.newline_loss_weight)
        self.separator_loss: SeparatorLoss = \
            SeparatorLoss(self.device,
                          weight=training_config.separator_loss_weight)
        self.bos_loss: BOSLoss = \
            BOSLoss(self.device,
                    weight=training_config.bos_loss_weight)

    def forward(self,
                logits: torch.Tensor,
                labels: torch.Tensor,
                word2idx: Dict[str, int]) -> Tuple[torch.Tensor, AllLosses]:
        general_character_loss: torch.Tensor = self.general_character_loss(logits=logits,
                                                                           labels=labels,
                                                                           word2idx=word2idx)
        eos_loss: torch.Tensor = self.eos_loss(logits=logits, labels=labels, word2idx=word2idx)
        unknown_loss: torch.Tensor = self.unknown_loss(logits=logits, labels=labels, word2idx=word2idx)
        new_line_loss: torch.Tensor = self.new_line_loss(logits=logits, labels=labels, word2idx=word2idx)
        separator_loss: torch.Tensor = self.separator_loss(logits=logits, labels=labels, word2idx=word2idx)
        bos_loss: torch.Tensor = self.bos_loss(logits=logits, labels=labels, word2idx=word2idx)
        loss: torch.Tensor = \
            general_character_loss + eos_loss + unknown_loss + new_line_loss + separator_loss + bos_loss
        all_losses: AllLosses = AllLosses(
            total_loss=loss.detach().cpu().item(),
            general_token_loss=general_character_loss.detach().cpu().item(),
            eos_token_loss=eos_loss.detach().cpu().item(),
            unknown_token_loss=unknown_loss.detach().cpu().item(),
            new_line_token_loss=new_line_loss.detach().cpu().item(),
            separator_token_loss=separator_loss.detach().cpu().item(),
            bos_token_loss=bos_loss.detach().cpu().item(),
        )
        return loss, all_losses


class GeneralCharacterLoss(torch.nn.Module):
    def __init__(self, device: torch.device, weight: float):
        super(GeneralCharacterLoss, self).__init__()
        self.device: torch.device = device
        self.weight: float = weight
        self.loss_fn: Callable = torch.nn.functional.cross_entropy
        self.special_characters: List[str] = get_all_special_tokens()

    def forward(self,
                logits: torch.Tensor,
                labels: torch.Tensor,
                word2idx: Dict[str, int]) -> torch.Tensor:

        special_character_tokens: List[int] = \
            [word2idx[token] for token in self.special_characters]
        special_character_tokens_tensor: torch.Tensor = \
            torch.tensor(special_character_tokens,
                         dtype=torch.long,
                         device=self.device)

        mask: torch.Tensor = ~torch.isin(labels, special_character_tokens_tensor)
        if mask.sum() == 0:
            return torch.tensor(0., device=self.device)
        logits_flat: torch.Tensor = logits[mask]  # (num_eos, vocab_size)
        targets_flat: torch.Tensor = labels[mask]  # (num_eos,)
        loss: torch.Tensor = self.loss_fn(logits_flat, targets_flat)
        return self.weight * loss

class SpecialCharacterLossBase(torch.nn.Module):
    def __init__(self, device: torch.device, weight: float):
        super(SpecialCharacterLossBase, self).__init__()
        self.loss_fn: Callable = torch.nn.functional.cross_entropy
        self.special_character: str = ''
        self.device: torch.device = device
        self.weight: float = weight

    def forward(self,
                logits: torch.Tensor,
                labels: torch.Tensor,
                word2idx: Dict[str, int]) -> torch.Tensor:
        assert self.special_character in word2idx, 'special_character must be in word2idx'
        assert self.special_character, 'special_character must be defined'
        eos_idx: int = word2idx[self.special_character]
        mask: torch.Tensor = (labels == eos_idx)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        logits_flat: torch.Tensor = logits[mask]  # (num_eos, vocab_size)
        targets_flat: torch.Tensor = labels[mask]  # (num_eos,)
        loss: torch.Tensor = self.loss_fn(logits_flat, targets_flat)
        return self.weight * loss

class EOSLoss(SpecialCharacterLossBase):
    def __init__(self, device: torch.device, weight: float):
        super(EOSLoss, self).__init__(device, weight)
        self.special_character: str = eos_token

class NewLineLoss(SpecialCharacterLossBase):
    def __init__(self, device: torch.device, weight: float):
        super(NewLineLoss, self).__init__(device, weight)
        self.special_character: str = new_line_token

class UnknownLoss(SpecialCharacterLossBase):
    def __init__(self, device: torch.device, weight: float):
        super(UnknownLoss, self).__init__(device, weight)
        self.special_character: str = unknown_token

class SeparatorLoss(SpecialCharacterLossBase):
    def __init__(self, device: torch.device, weight: float):
        super(SeparatorLoss, self).__init__(device, weight)
        self.special_character: str = separator_token

class BOSLoss(SpecialCharacterLossBase):
    def __init__(self, device: torch.device, weight: float):
        super(BOSLoss, self).__init__(device, weight)
        self.special_character: str = bos_token
