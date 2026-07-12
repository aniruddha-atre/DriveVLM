"""
Evaluation metric for Talk2Car grounding.

One referred object per command, so accuracy@0.5 (Talk2Car's "AP50") is just the fraction of
predictions that overlap the ground-truth box by more than 0.5 IoU. Boxes are [x1, y1, x2, y2]
in pixels.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

Box = tuple[float, float, float, float]

# Matches one signed integer or decimal. Model box output is parsed by taking the first four.
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_box(text: str) -> Box | None:
    """Pull the box out of the VLM's text, e.g. "[975, 463, 1141, 575]" -> (x1, y1, x2, y2).

    Uses the first four numbers, or None if there aren't four. Shared by the Qwen eval/train/demo
    paths, so it's unit-tested.
    """
    nums = _NUM_RE.findall(text)
    if len(nums) < 4:
        return None
    x1, y1, x2, y2 = (float(n) for n in nums[:4])
    return (x1, y1, x2, y2)


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def accuracy_at_50(preds: Sequence[Box | None], gts: Sequence[Box]) -> float:
    """Fraction of samples with IoU(pred, gt) > 0.5. A missing prediction (None) counts as 0."""
    if len(preds) != len(gts):
        raise ValueError(f"preds/gts length mismatch: {len(preds)} vs {len(gts)}")
    if not gts:
        return 0.0
    hits = sum(1 for p, g in zip(preds, gts, strict=True) if p is not None and iou(p, g) > 0.5)
    return hits / len(gts)
