import torch


class NextCharacterLoss(torch.nn.Module):
    def __init__(self):
        super(NextCharacterLoss, self).__init__()
        self.loss_func = torch.nn.functional.cross_entropy

    def forward(self, logits: torch.Tensor, y: torch.Tensor, vocab_size: int) -> torch.Tensor:
        return self.loss_func(logits.view(-1, vocab_size), y.view(-1))


class MaskedCharacterLoss(torch.nn.Module):
    def __init__(self):
        super(MaskedCharacterLoss, self).__init__()
        self.loss_func = torch.nn.functional.cross_entropy

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                logits: torch.Tensor,
                mask_token_idx: int) -> torch.Tensor:
        mask_targets = (x == mask_token_idx)
        if mask_targets.any():
            logits_masked = logits[mask_targets]
            targets_masked = y[mask_targets]
            loss_masked = self.loss_func(logits_masked, targets_masked)
        else:
            loss_masked = torch.tensor(0.0, device=logits.device)

        return loss_masked


class TitleToLyricsLoss(NextCharacterLoss):
    def __init__(self):
        super(TitleToLyricsLoss, self).__init__()
