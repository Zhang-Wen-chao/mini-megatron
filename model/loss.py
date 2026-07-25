import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels, loss_mask):
        vocab_size = logits.size(-1)
        logits = logits.view(-1, vocab_size)
        labels = labels.view(-1)
        loss_mask = loss_mask.view(-1)

        loss = nn.functional.cross_entropy(logits, labels, reduction="none")
        loss = torch.sum(loss * loss_mask) / loss_mask.sum()
        return loss
