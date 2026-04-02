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

class BinaryTanimotoWithComplementLoss(nn.Module):
    """
    Canonical binary Tanimoto loss with complement.

    Based on the binary formulation:
        score = (T(p, t) + T(1-p, 1-t)) / 2
        loss  = 1 - score

    Interface:
        forward(logits, targets, valid=None) -> scalar
    """
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
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

        score = 0.5 * (t_pos + t_neg)
        return (1.0 - score).mean()

class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss with internal valid masking.

    Based on the binary focal loss form:
        FL(pt) = -(1 - pt)^gamma * log(pt)

    Interface:
        forward(logits, targets, valid=None) -> scalar
    """
    def __init__(
        self,
        gamma: float = 2.0,
    ):
        super().__init__()

        gamma = float(gamma)
        if gamma < 0:
            raise ValueError("gamma must be >= 0")

        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits, targets, valid = _prepare_binary_tensors(logits, targets, valid)

        # Stable binary cross-entropy from logits:
        # CE = -log(pt)
        ce_map = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        # Since CE = -log(pt), then pt = exp(-CE)
        pt = torch.exp(-ce_map)

        focal_map = ((1.0 - pt) ** self.gamma) * ce_map

        return _masked_mean(focal_map, valid)

class FocalDiceLoss(nn.Module):
    """
    Binary Focal + Dice loss with internal valid masking.

    Interface:
        forward(logits, targets, valid=None) -> scalar
    """
    def __init__(
        self,
        gamma: float = 2.0,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-6,
    ):
        super().__init__()

        gamma = float(gamma)
        if gamma < 0:
            raise ValueError("gamma must be >= 0")

        self.gamma = gamma
        self.focal_weight = float(focal_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits, targets, valid = _prepare_binary_tensors(logits, targets, valid)

        # Focal part
        probs = torch.sigmoid(logits)
        bce_map = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        pt = targets * probs + (1.0 - targets) * (1.0 - probs)
        focal_map = ((1.0 - pt) ** self.gamma) * bce_map
        focal = _masked_mean(focal_map, valid)

        # Dice part
        if valid is not None:
            probs_dice = probs * valid
            targets_dice = targets * valid
        else:
            probs_dice = probs
            targets_dice = targets

        probs_dice = probs_dice.reshape(probs_dice.shape[0], -1)
        targets_dice = targets_dice.reshape(targets_dice.shape[0], -1)

        intersection = (probs_dice * targets_dice).sum(dim=1)
        denom = probs_dice.sum(dim=1) + targets_dice.sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        dice = 1.0 - dice_score.mean()

        return self.focal_weight * focal + self.dice_weight * dice

class BinaryDECBLoss(nn.Module):
    """
    Binary adaptation of DECB for one-logit segmentation.

    Interface:
        forward(logits, targets, valid=None) -> scalar

    Modes:
        - mode='bce'   : DECB-weighted BCEWithLogits
        - mode='focal' : DECB-weighted binary focal loss

    Important:
        This is a binary adaptation for a one-logit setup.
        The original paper defines DECB in the multiclass softmax setting. :contentReference[oaicite:0]{index=0}
    """
    def __init__(
        self,
        mode: str = "bce",
        gamma: float = 3.0,
        smooth: float = 1e-8,
        normalize_weights: bool = False,
    ):
        super().__init__()

        if mode not in {"bce", "focal"}:
            raise ValueError(f"mode must be 'bce' or 'focal', got {mode}")

        self.mode = mode
        self.gamma = float(gamma)
        self.smooth = float(smooth)
        self.normalize_weights = bool(normalize_weights)

    def _effective_subspace_size(self, n: torch.Tensor) -> torch.Tensor:
        """
        Effective sample subspace size based on Eq. (11) and Eq. (15).

        beta(n) = (1 / 1001) ** (1 / n)
        E(n)    = (1000 / 1001) / (1 - beta(n))
        """
        n = n.clamp_min(1.0)
        beta = torch.pow(
            torch.tensor(1.0 / 1001.0, device=n.device, dtype=n.dtype),
            1.0 / n,
        )
        return (1000.0 / 1001.0) / (1.0 - beta + self.smooth)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits, targets, valid = _prepare_binary_tensors(logits, targets, valid)

        if valid is None:
            valid_eff = torch.ones_like(targets)
        else:
            valid_eff = valid

        valid_bin = (valid_eff > 0).to(dtype=logits.dtype)

        pos_mask = targets * valid_bin
        neg_mask = (1.0 - targets) * valid_bin

        n_pos = pos_mask.sum()
        n_neg = neg_mask.sum()
        n_batch = (n_pos + n_neg).clamp_min(1.0)

        e_nbatch = self._effective_subspace_size(n_batch.unsqueeze(0)).squeeze(0)
        e_pos = self._effective_subspace_size(n_pos.unsqueeze(0)).squeeze(0)
        e_neg = self._effective_subspace_size(n_neg.unsqueeze(0)).squeeze(0)

        # DECB-inspired weights:
        # minority class -> use effective count
        # majority class -> use raw count
        if n_pos < e_nbatch:
            w_pos = 1.0 - e_pos / n_batch
        else:
            w_pos = 1.0 - n_pos / n_batch

        if n_neg < e_nbatch:
            w_neg = 1.0 - e_neg / n_batch
        else:
            w_neg = 1.0 - n_neg / n_batch

        weights = targets * w_pos + (1.0 - targets) * w_neg

        if self.normalize_weights:
            mean_w = (weights * valid_eff).sum() / valid_eff.sum().clamp_min(self.smooth)
            weights = weights / mean_w.clamp_min(self.smooth)

        if self.mode == "bce":
            loss_map = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                reduction="none",
            )
        else:
            probs = torch.sigmoid(logits)
            pt = targets * probs + (1.0 - targets) * (1.0 - probs)
            pt = pt.clamp(self.smooth, 1.0 - self.smooth)
            loss_map = -((1.0 - pt) ** self.gamma) * torch.log(pt)

        loss_map = loss_map * weights
        return _masked_mean(loss_map, valid_eff, eps=self.smooth)

