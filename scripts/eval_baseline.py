"""Zero-shot Grounding DINO baseline on Talk2Car, logged to MLflow.

    uv run python scripts/eval_baseline.py --split val --limit 20

Runs the model on `--limit` samples (omit for the full split) and logs params + accuracy@0.5
to MLflow. On CPU, keep --limit small; run the full split on a GPU/HPC.

MLflow tracking dir defaults to ./mlruns (override with MLFLOW_TRACKING_URI). View with:
    uv run mlflow ui
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow

from drive_vlm.data import load_split
from drive_vlm.infer import evaluate
from drive_vlm.model import DEFAULT_CHECKPOINT, GroundingDINO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None, help="max samples (default: all)")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--box-threshold", type=float, default=0.3)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    args = ap.parse_args()

    samples = load_split(Path(args.data_dir), args.split)
    if args.limit:
        samples = samples[: args.limit]

    print(f"Loaded {len(samples)} {args.split} samples. Loading {args.checkpoint}...")
    model = GroundingDINO(checkpoint=args.checkpoint)
    print(f"Running zero-shot on device={model.device}...")

    mlflow.set_experiment("drive-vlm")
    with mlflow.start_run(run_name=f"zeroshot-gdino-{args.split}"):
        mlflow.set_tag("stage", "zero-shot")
        mlflow.log_params(
            {
                "model": args.checkpoint,
                "split": args.split,
                "n_samples": len(samples),
                "box_threshold": args.box_threshold,
                "device": model.device,
                "command_input": "full_command",
            }
        )
        metrics = evaluate(model, samples, box_threshold=args.box_threshold)
        mlflow.log_metrics(metrics)

    print("\n=== zero-shot Grounding DINO ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print("\nLogged to MLflow. View with:  uv run mlflow ui")


if __name__ == "__main__":
    main()
