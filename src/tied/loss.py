"""TEED losses with built-in tolerance (cats_loss uses a neighbourhood
radius around each edge pixel, so off-by-one predictions are not
punished — exactly the "mozliwosc tolerancji" from TEED).

Ported from TEED's loss2.py.

``bdcn_loss2`` is the per-pixel weighted BCE used on the 3 multi-scale
heads. ``cats_loss`` is the tracing loss used on the final fused output;
its ``bdr_factor`` controls the tolerance band around target edges.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bdcn_loss2(logits: torch.Tensor, targets: torch.Tensor,
               l_weight: float = 1.1) -> torch.Tensor:
    """Class-balanced BCE. Targets are soft in [0, 1]."""
    mask = targets.float().clone()
    num_pos = (mask > 0.0).float().sum()
    num_neg = (mask <= 0.0).float().sum()
    total = num_pos + num_neg + 1e-12
    mask[targets > 0.0] = (1.0 * num_neg / total).item()
    mask[targets <= 0.0] = (1.1 * num_pos / total).item()
    prob = torch.sigmoid(logits)
    cost = F.binary_cross_entropy(prob, targets.float(), weight=mask, reduction="none")
    return l_weight * cost.mean((1, 2, 3)).sum()


def _bdrloss(pred: torch.Tensor, label: torch.Tensor, radius: int) -> torch.Tensor:
    filt = torch.ones(1, 1, 2 * radius + 1, 2 * radius + 1,
                      device=pred.device, dtype=pred.dtype)
    bdr_pred = pred * label
    pred_bdr_sum = label * F.conv2d(bdr_pred, filt, padding=radius)
    texture_mask = F.conv2d(label, filt, padding=radius)
    mask = (texture_mask != 0).float()
    mask[label == 1] = 0
    pred_texture_sum = F.conv2d(pred * (1 - label) * mask, filt, padding=radius)
    softmax_map = torch.clamp(
        pred_bdr_sum / (pred_texture_sum + pred_bdr_sum + 1e-10),
        1e-10, 1 - 1e-10)
    cost = -label * torch.log(softmax_map)
    cost[label == 0] = 0
    return cost.mean((1, 2, 3)).sum()


def _textureloss(pred: torch.Tensor, label: torch.Tensor,
                 mask_radius: int) -> torch.Tensor:
    filt1 = torch.ones(1, 1, 3, 3, device=pred.device, dtype=pred.dtype)
    filt2 = torch.ones(1, 1, 2 * mask_radius + 1, 2 * mask_radius + 1,
                       device=pred.device, dtype=pred.dtype)
    pred_sums = F.conv2d(pred, filt1, padding=1)
    label_sums = F.conv2d(label, filt2, padding=mask_radius)
    mask = 1 - (label_sums > 0).float()
    loss = -torch.log(torch.clamp(1 - pred_sums / 9, 1e-10, 1 - 1e-10))
    loss[mask == 0] = 0
    return loss.mean((1, 2, 3)).sum()


def cats_loss(logits: torch.Tensor, targets: torch.Tensor,
              l_weight=(0.01, 3.0), radius: int = 4) -> torch.Tensor:
    """Tracing loss with tolerance: pixels within ``radius`` of a true
    edge are treated as the boundary band; off-by-one predictions are
    forgiven by the bdr term. ``l_weight = (tex_factor, bdr_factor)``."""
    tex_factor, bdr_factor = l_weight
    targets = targets.float()
    with torch.no_grad():
        mask = targets.clone()
        num_pos = (mask == 1).float().sum()
        num_neg = (mask == 0).float().sum()
        beta = num_neg / (num_pos + num_neg + 1e-12)
        mask[targets == 1] = beta
        mask[targets == 0] = 1.1 * (1 - beta)
        mask[(targets != 0) & (targets != 1)] = 0
    prob = torch.sigmoid(logits)
    cost = F.binary_cross_entropy(prob, targets, weight=mask, reduction="none")
    cost = cost.mean((1, 2, 3)).sum()
    label_w = (targets != 0).float()
    tex = _textureloss(prob, label_w, mask_radius=radius)
    bdr = _bdrloss(prob, label_w, radius=radius)
    return cost + bdr_factor * bdr + tex_factor * tex


def _pos_weight(targets: torch.Tensor) -> torch.Tensor:
    """Class-balance weight = sum(1-t) / sum(t) over the whole batch.
    Pushes the soft losses out of the trivial p ~ mean(t) minimum that
    dominates when the background is overwhelming (typical: ~98% of
    pixels are 0). Clamped at >=1 so it never down-weights positives."""
    t = targets.float()
    pos = t.sum().clamp_min(1.0)
    neg = (1.0 - t).sum().clamp_min(1.0)
    return (neg / pos).clamp_min(1.0)


def soft_bce_loss(logits: torch.Tensor, targets: torch.Tensor,
                  radius: int = 0) -> torch.Tensor:
    """Class-balanced BCE between sigmoid(logits) and float targets.

    Tonal: the per-pixel optimum is ``p == t``, but the class weight
    keeps faint edges from being drowned by the dark background.

    ``radius > 0`` enables a spatial tolerance band: the target is
    max-pooled by a (2r+1) kernel before BCE, so a prediction within
    ``r`` pixels of a true edge is no longer punished as a false
    positive. Max-pool preserves intensity, so the tonal property is
    kept — a faint gray line dilates to a faint gray band of the same
    intensity. Trade-off: predicted lines also become up to ``r``
    pixels thicker because the model is rewarded for matching the band.
    """
    t = targets.float()
    if radius > 0:
        k = 2 * radius + 1
        t = F.max_pool2d(t, kernel_size=k, stride=1, padding=radius)
    return F.binary_cross_entropy_with_logits(
        logits, t, pos_weight=_pos_weight(t), reduction="mean")


def soft_jaccard_loss(logits: torch.Tensor, targets: torch.Tensor,
                      smooth: float = 1.0) -> torch.Tensor:
    """Differentiable soft IoU distance: ``1 - sum(p*t) / sum(p + t - p*t)``.

    Works directly on soft targets (no binarisation), so faint edges in
    a gray outline contribute proportionally to their intensity. The
    ``smooth`` term keeps the gradient finite when both p and t are 0.
    Reduced as the mean over the batch.
    """
    prob = torch.sigmoid(logits)
    p = prob.flatten(1)
    t = targets.flatten(1)
    inter = (p * t).sum(dim=1)
    union = (p + t - p * t).sum(dim=1)
    iou = (inter + smooth) / (union + smooth)
    return (1.0 - iou).mean()


LOSS_KINDS = ("teed", "soft_jaccard", "soft_bce")


def resolve_loss(kind: str, outline_mode: str) -> str:
    """Auto-routing: mono -> teed, gray -> soft_jaccard. Any explicit
    kind passes through unchanged."""
    if kind != "auto":
        return kind
    return "teed" if outline_mode == "mono" else "soft_jaccard"


def compute_loss(kind: str, preds, target, radius: int = 4):
    """Apply the named loss to the model's outputs. Teed uses all 4
    heads; the soft losses use only the fused output."""
    if kind == "teed":
        return teed_total_loss(preds, target, radius=radius)
    if kind == "soft_jaccard":
        return soft_jaccard_loss(preds[-1], target)
    if kind == "soft_bce":
        return soft_bce_loss(preds[-1], target, radius=radius)
    raise ValueError(f"unknown loss kind: {kind!r}")


@torch.no_grad()
def hard_pixel_counts(logits: torch.Tensor, targets: torch.Tensor,
                      threshold: float = 0.5) -> dict:
    """Binarise sigmoid(logits) and targets at ``threshold``, then count
    wrong pixels and union pixels — same shape as MCED's hard_pixel_counts.

    For outline="mono" targets are already in {0., 1.} so the threshold
    is irrelevant on that side. For outline="gray" we threshold both
    sides at 0.5 which gives a coarse but consistent IoU-style signal.
    """
    pred_b = (torch.sigmoid(logits) >= threshold)
    targ_b = (targets >= threshold)
    wrong = (pred_b != targ_b).sum().item()
    union = (pred_b | targ_b).sum().item()
    total = int(targ_b.numel())
    return {"wrong_px": int(wrong), "union_px": int(union), "total_px": total}


SCALE_WEIGHTS = (1.1, 0.7, 1.1, 1.3)
CATS_WEIGHTS = (0.01, 3.0)


def teed_total_loss(preds: list[torch.Tensor], target: torch.Tensor,
                    scale_weights=None,
                    cats_weights=CATS_WEIGHTS,
                    radius: int = 4) -> torch.Tensor:
    """Combined TEED loss: bdcn_loss2 on every head plus cats_loss on
    the fused output.

    ``scale_weights`` length must match ``len(preds)``. Default uses
    TEED's original (1.1, 0.7, 1.1, 1.3) for 4-head models (TED,
    TEDup); for any other count (e.g. TEDdeep's 5 heads) falls back
    to uniform 1.0.
    """
    if scale_weights is None:
        if len(preds) == 4:
            scale_weights = SCALE_WEIGHTS              # (1.1, 0.7, 1.1, 1.3)
        elif len(preds) == 5:
            # Prepend a weight for the new shallowest head, keep TEED's
            # pattern for the rest — fused head must stay at 1.3 so the
            # output we actually use at inference gets prioritised.
            scale_weights = (1.0,) + SCALE_WEIGHTS     # (1.0, 1.1, 0.7, 1.1, 1.3)
        else:
            scale_weights = tuple(1.0 for _ in preds)
    assert len(preds) == len(scale_weights), \
        f"preds={len(preds)} scale_weights={len(scale_weights)}"
    loss1 = sum(bdcn_loss2(p, target, w) for p, w in zip(preds, scale_weights))
    loss2 = cats_loss(preds[-1], target, cats_weights, radius=radius)
    return loss1 + loss2
