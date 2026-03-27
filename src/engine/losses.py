import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets, valid=None):
        probs = torch.sigmoid(logits)
        if valid is not None:
            probs = probs * valid
            targets = targets * valid
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

    def forward(self, logits, targets, valid=None):
        # Move pos_weight to correct device dynamically
        if self.bce.pos_weight is not None and self.bce.pos_weight.device != logits.device:
            self.bce.pos_weight = self.bce.pos_weight.to(logits.device)
            
        bce = self.bce(logits, targets)
        dice = self.dice(logits, targets, valid=valid)
        return self.bce_w * bce + self.dice_w * dice

class TanimotoLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(TanimotoLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets, valid=None):
        """
        Tanimoto Loss with complement for multiclass imbalanced problems.
        Based on:
        - T(p, l) = (sum w_J * intersection_J) / (sum w_J * (p^2 + l^2 - intersection)_J)
        - w_J = V_J^-2 where V_J is the volume of class J
        - Final loss = 1 - (T(p, l) + T(1-p, 1-l)) / 2
        """
        # Ensure targets is the same shape as logits
        if targets.shape != logits.shape:
            targets = targets.view(logits.shape)

        probs = torch.sigmoid(logits)
        
        # Flatten spatial dimensions: (B, C, H, W) -> (B, C, N)
        b, c = probs.shape[0], probs.shape[1]
        p = probs.view(b, c, -1)
        l = targets.view(b, c, -1)
        
        v = valid.view(b, 1, -1) if valid is not None else None
        
        def calculate_weighted_tanimoto(p_in, l_in):
            # p_in, l_in: (B, C, N)
            if v is not None:
                p_in = p_in * v
                l_in = l_in * v
            
            # Volume per class J: V_J = sum over pixels i
            v_j = torch.sum(l_in, dim=-1)  # (B, C)
            
            # Weights w_J = V_J^-2 (Inverse "volume" weighting scheme)
            w_j = 1.0 / (torch.square(v_j) + self.smooth)
            
            # Intersection sum_i p_iJ * l_iJ
            intersection = torch.sum(p_in * l_in, dim=-1)  # (B, C)
            
            # sum_i p_iJ^2 and sum_i l_iJ^2
            p2 = torch.sum(torch.square(p_in), dim=-1)  # (B, C)
            l2 = torch.sum(torch.square(l_in), dim=-1)  # (B, C)
            
            # Equation (5) components
            # Numerator: \sum_J w_J \sum_i p_iJ l_iJ
            numerator = torch.sum(w_j * intersection, dim=-1)  # (B,)
            
            # Denominator: \sum_J w_J \sum_i (p_iJ^2 + l_iJ^2 - p_iJ l_iJ)
            denominator = torch.sum(w_j * (p2 + l2 - intersection), dim=-1)  # (B,)
            
            return (numerator + self.smooth) / (denominator + self.smooth)

        # T(p, l)
        t_score = calculate_weighted_tanimoto(p, l)
        
        # T(1-p, 1-l) - Tanimoto with complement
        t_complement_score = calculate_weighted_tanimoto(1.0 - p, 1.0 - l)
        
        # Average score
        tanimoto_score = (t_score + t_complement_score) / 2.0
        
        return torch.mean(1.0 - tanimoto_score)

from typing import Optional

class BinaryWeightedTanimotoWithComplementLoss(nn.Module):
    """
    Binary Tanimoto loss with complement and fixed class weights.

    Expected:
        logits:  [B, 1, H, W]
        targets: [B, 1, H, W] or [B, H, W], values in {0, 1}

    score = w_pos * T(p, l) + w_neg * T(1-p, 1-l)
    loss  = 1 - score
    """
    def __init__(self, w_pos: float, w_neg: float, smooth: float = 1e-6):
        super().__init__()
        self.w_pos = float(w_pos)
        self.w_neg = float(w_neg)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError(f"logits must have shape [B, 1, H, W], got {tuple(logits.shape)}")

        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        elif targets.ndim != 4:
            raise ValueError(f"targets must have shape [B, H, W] or [B, 1, H, W], got {tuple(targets.shape)}")

        if targets.shape != logits.shape:
            raise ValueError(
                f"targets shape {tuple(targets.shape)} must match logits shape {tuple(logits.shape)}"
            )

        targets = targets.to(dtype=logits.dtype)
        probs = torch.sigmoid(logits)

        p = probs.reshape(probs.shape[0], -1)
        l = targets.reshape(targets.shape[0], -1)
        v = valid.reshape(valid.shape[0], -1) if valid is not None else None

        def tanimoto(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            if v is not None:
                a = a * v
                b = b * v
            intersection = (a * b).sum(dim=1)
            a2 = (a * a).sum(dim=1)
            b2 = (b * b).sum(dim=1)
            return (intersection + self.smooth) / (a2 + b2 - intersection + self.smooth)

        t_pos = tanimoto(p, l)
        t_neg = tanimoto(1.0 - p, 1.0 - l)

        score = self.w_pos * t_pos + self.w_neg * t_neg
        loss = 1.0 - score
        return loss.mean()