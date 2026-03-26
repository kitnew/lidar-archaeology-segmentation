import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight, reduction="none", bce_weight=0.5, dice_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        pos_weight_tensor = torch.tensor(pos_weight)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor, reduction=reduction)
        self.dice = DiceLoss()
        self.bce_w = bce_weight
        self.dice_w = dice_weight

    def forward(self, logits, targets):
        # Move pos_weight to correct device dynamically
        if self.bce.pos_weight is not None and self.bce.pos_weight.device != logits.device:
            self.bce.pos_weight = self.bce.pos_weight.to(logits.device)
            
        bce = self.bce(logits, targets)
        dice = self.dice(logits, targets)
        return self.bce_w * bce + self.dice_w * dice
