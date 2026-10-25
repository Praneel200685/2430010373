"""Class-weighted focal loss (Eq. 7)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, y):
        logpt = F.log_softmax(logits, dim=1).gather(1, y[:, None]).squeeze(1)
        loss = -((1.0 - logpt.exp()) ** self.gamma) * logpt
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device)[y]
        return loss.mean()
