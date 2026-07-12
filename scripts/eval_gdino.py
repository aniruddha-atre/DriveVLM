"""
Evaluate a (fine-tuned) Grounding DINO checkpoint on a Talk2Car split.

One object per command, so we always take the top-scoring box (threshold=0). Reports
accuracy@50. Defaults to the fine-tuned checkpoint; pass the HF id for zero-shot.

    python scripts/eval_gdino.py --checkpoint checkpoints/gdino-t2c --split test
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import mlflow
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, GroundingDinoForObjectDetection

from drive_vlm.data import load_split
from drive_vlm.eval import Box, accuracy_at_50


def fmt(command: str) -> str:
    return command.lower().strip().rstrip(".") + " ."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/gdino-t2c")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = load_split(Path(args.data_dir), args.split)
    if args.limit:
        samples = samples[: args.limit]

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    model = GroundingDinoForObjectDetection.from_pretrained(args.checkpoint)
    cast(torch.nn.Module, model).to(device)
    model.eval()

    preds: list[Box | None] = []
    with torch.no_grad():
        for s in tqdm(samples, desc="gdino"):
            img = Image.open(s.image_path).convert("RGB")
            inp = processor(images=img, text=fmt(s.command), return_tensors="pt").to(device)
            out = model(**inp)
            res = processor.post_process_grounded_object_detection(
                out,
                inp["input_ids"],
                threshold=0.0,
                text_threshold=0.0,
                target_sizes=[img.size[::-1]],
            )[0]
            if len(res["scores"]) == 0:
                preds.append(None)
            else:
                best = int(res["scores"].argmax())
                preds.append(tuple(res["boxes"][best].tolist()))
    gts = [s.box for s in samples]

    metrics = {
        "accuracy_at_50": accuracy_at_50(preds, gts),
        "n": len(samples),
        "n_no_prediction": sum(p is None for p in preds),
    }
    mlflow.set_experiment("drive-vlm")
    with mlflow.start_run(run_name=f"gdino-{args.split}-eval"):
        mlflow.set_tag("stage", "eval")
        mlflow.log_params({"checkpoint": args.checkpoint, "split": args.split, "n": len(samples)})
        mlflow.log_metrics(metrics)
    print(f"=== Grounding DINO ({args.checkpoint}) {args.split} ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    main()
