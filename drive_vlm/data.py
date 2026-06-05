"""Talk2Car data loading.

One sample = (image path, command text, ground-truth box, object label). Boxes are normalized
to [x1, y1, x2, y2] in pixels here, so the rest of the codebase never deals with raw formats.

IMPORTANT: Talk2Car's `2d_box` in the JSON is [x, y, w, h] (top-left + width/height), NOT
[x1, y1, x2, y2]. We convert on load. Images are 1600x900 (nuScenes front camera).

Expected layout under `data_dir`:
    {train,val,test}_commands.json   # top-level dict with a "commands" list
    images/img_{split}_{idx}.jpg
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Sample:
    image_path: Path
    command: str
    box: Box  # [x1, y1, x2, y2] in pixels
    label: str  # nuScenes class, e.g. "vehicle.car"


def xywh_to_xyxy(b: list[float]) -> Box:
    x, y, w, h = b
    return (x, y, x + w, y + h)


def load_split(data_dir: Path, split: str) -> list[Sample]:
    """Load Talk2Car samples for a split ('train' | 'val' | 'test')."""
    data_dir = Path(data_dir)
    records = json.loads((data_dir / f"{split}_commands.json").read_text())["commands"]
    images = data_dir / "images"
    return [
        Sample(
            image_path=images / r["t2c_img"],
            command=r["command"],
            box=xywh_to_xyxy(r["2d_box"]),
            label=r["obj_name"],
        )
        for r in records
    ]
