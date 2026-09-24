import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self, pos_weight, num_classes = 3, smooth=1.0, dice_weight=1.0, bce_weight=1.0):
        super().__init__()

        self.smooth = smooth
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

        pos_weight = torch.tensor(pos_weight)
        if len(pos_weight.shape) == 1:
            pos_weight = pos_weight.view(num_classes, 1, 1)

        self.bce = nn.BCEWithLogitsLoss(pos_weight = pos_weight)



    def forward(self, logits, targets):

        bce = self.bce(logits, targets.float())

        probs = torch.sigmoid(logits)

        probs = probs.view(probs.size(0), -1)
        targets = targets.float().view(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        dice_loss = 1.0 - dice.mean()

        loss = (self.bce_weight * bce + self.dice_weight * dice_loss)

        return loss, dice_loss, bce