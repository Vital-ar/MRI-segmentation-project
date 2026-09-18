import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0, dice_weight=1.0, bce_weight=1.0):
        super().__init__()

        self.smooth = smooth
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

        self.bce = nn.BCEWithLogitsLoss()



    def forward(self, logits, targets):

        bce = self.bce(logits, targets.float())

        probs = torch.sigmoid(logits)

        probs = probs.view(-1)
        targets = targets.float().view(-1)

        intersection = (probs * targets).sum()

        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth)

        dice_loss = 1.0 - dice

        loss = (self.bce_weight * bce + self.dice_weight * dice_loss)

        return loss