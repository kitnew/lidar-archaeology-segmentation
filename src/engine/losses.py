import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def _prepare_binary_tensors(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
):
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

    if valid is not None:
        if valid.ndim == 3:
            valid = valid.unsqueeze(1)
        elif valid.ndim != 4:
            raise ValueError(f"valid must have shape [B, H, W] or [B, 1, H, W], got {tuple(valid.shape)}")

        if valid.shape != logits.shape:
            raise ValueError(
                f"valid shape {tuple(valid.shape)} must match logits shape {tuple(logits.shape)}"
            )

        valid = valid.to(dtype=logits.dtype)

    return logits, targets, valid


def _masked_mean(loss_map: torch.Tensor, valid: Optional[torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
    if valid is None:
        return loss_map.mean()

    denom = valid.sum().clamp_min(eps)
    return (loss_map * valid).sum() / denom


class BCEDiceLoss(nn.Module):
    """
    Binary BCE + Dice loss with internal valid masking.

    Interface:
        forward(logits, targets, valid=None) -> scalar
    """
    def __init__(
        self,
        pos_weight: Optional[float] = None,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-6,
    ):
        super().__init__()

        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)

        if pos_weight is None:
            self.register_buffer("pos_weight", None)
        else:
            self.register_buffer("pos_weight", torch.tensor(float(pos_weight), dtype=torch.float32))

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits, targets, valid = _prepare_binary_tensors(logits, targets, valid)

        # BCE part
        bce_map = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,  # pyright: ignore[reportArgumentType]
            reduction="none",
        )
        bce = _masked_mean(bce_map, valid)

        # Dice part
        probs = torch.sigmoid(logits)

        if valid is not None:
            probs = probs * valid
            targets = targets * valid

        probs = probs.reshape(probs.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)

        intersection = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)
        dice_score = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        dice = 1.0 - dice_score.mean()

        return self.bce_weight * bce + self.dice_weight * dice


class BinaryWeightedTanimotoWithComplementLoss(nn.Module):
    """
    Binary weighted Tanimoto with complement and internal valid masking.

    Interface:
        forward(logits, targets, valid=None) -> scalar
    """
    def __init__(
        self,
        w_pos: float,
        w_neg: float,
        smooth: float = 1e-6,
    ):
        super().__init__()

        w_pos = float(w_pos)
        w_neg = float(w_neg)

        s = w_pos + w_neg
        if s <= 0:
            raise ValueError("w_pos + w_neg must be > 0")

        self.w_pos = w_pos / s
        self.w_neg = w_neg / s
        self.smooth = float(smooth)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits, targets, valid = _prepare_binary_tensors(logits, targets, valid)

        probs = torch.sigmoid(logits)

        p = probs.reshape(probs.shape[0], -1)
        t = targets.reshape(targets.shape[0], -1)

        if valid is not None:
            v = valid.reshape(valid.shape[0], -1)
        else:
            v = None

        def tanimoto(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            if v is not None:
                a = a * v
                b = b * v

            intersection = (a * b).sum(dim=1)
            a2 = (a * a).sum(dim=1)
            b2 = (b * b).sum(dim=1)

            return (intersection + self.smooth) / (a2 + b2 - intersection + self.smooth)

        t_pos = tanimoto(p, t)
        t_neg = tanimoto(1.0 - p, 1.0 - t)

        score = self.w_pos * t_pos + self.w_neg * t_neg
        return (1.0 - score).mean()