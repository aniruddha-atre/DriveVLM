"""
Inference benchmark for Grounding DINO: latency / throughput / VRAM / accuracy across precisions.

For each (precision, batch size) it reports p50/p90 latency, throughput, and peak VRAM; per
precision it reports accuracy@0.5 on a small slice, so a memory saving only counts if quality
holds. Model-agnostic: a Backend needs `.load()` and `.infer(images, commands) -> list[Box|None]`.

    uv run python scripts/benchmark.py --precisions fp16 bf16 --batch-sizes 1 4 8 --acc-samples 50
"""

from __future__ import annotations

import argparse
import contextlib
import statistics
import time
from pathlib import Path
from typing import cast

import torch
from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection

from drive_vlm.data import load_split
from drive_vlm.eval import Box, accuracy_at_50

# fp16/bf16 use autocast, NOT model.half(): Grounding DINO's deformable attention calls
# grid_sample, which errors on half inputs — autocast keeps that op in fp32 automatically.
PRECISIONS = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


def fmt(command: str) -> str:
    return command.lower().strip().rstrip(".") + " ."


class GDinoBackend:
    """Fine-tuned Grounding DINO at a given precision. Always returns the top-scoring box."""

    def __init__(self, checkpoint: str, precision: str, device: torch.device):
        self.checkpoint, self.device = checkpoint, device
        self.amp_dtype = PRECISIONS[precision]  # None for fp32
        self.name = f"gdino-{precision}" + ("" if self.amp_dtype is None else " (amp)")

    def load(self) -> None:
        self.processor = AutoProcessor.from_pretrained(self.checkpoint)
        model = GroundingDinoForObjectDetection.from_pretrained(self.checkpoint)
        cast(torch.nn.Module, model).to(self.device)
        self.model = model.eval()

    def infer(self, images: list[Image.Image], commands: list[str]) -> list[Box | None]:
        inp = self.processor(
            images=images, text=[fmt(c) for c in commands], return_tensors="pt", padding=True
        ).to(self.device)
        amp = (
            torch.autocast("cuda", dtype=self.amp_dtype)
            if self.amp_dtype is not None
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), amp:
            out = self.model(**inp)
        results = self.processor.post_process_grounded_object_detection(
            out,
            inp["input_ids"],
            threshold=0.0,
            text_threshold=0.0,
            target_sizes=[im.size[::-1] for im in images],
        )
        boxes: list[Box | None] = []
        for r in results:
            if len(r["scores"]) == 0:
                boxes.append(None)
            else:
                boxes.append(tuple(r["boxes"][int(r["scores"].argmax())].tolist()))
        return boxes


def measure_latency(backend, images, commands, batch_sizes, iters, warmup) -> list[dict]:
    rows = []
    for bs in batch_sizes:
        imgs = [images[i % len(images)] for i in range(bs)]
        cmds = [commands[i % len(commands)] for i in range(bs)]
        try:
            for _ in range(warmup):
                backend.infer(imgs, cmds)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            lat = []
            for _ in range(iters):
                t0 = time.perf_counter()
                backend.infer(imgs, cmds)
                torch.cuda.synchronize()
                lat.append(time.perf_counter() - t0)
            p50 = statistics.median(lat)
            p90 = sorted(lat)[int(0.9 * (len(lat) - 1))]
            peak = torch.cuda.max_memory_allocated() / 1e9
            rows.append(
                {
                    "batch": bs,
                    "p50_ms": p50 * 1e3,
                    "p90_ms": p90 * 1e3,
                    "img_s": bs / p50,
                    "vram_gb": peak,
                }
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            rows.append(
                {"batch": bs, "p50_ms": None, "p90_ms": None, "img_s": None, "vram_gb": None}
            )
    return rows


@torch.inference_mode()
def measure_accuracy(backend, samples) -> float:
    preds, gts = [], []
    for s in samples:
        img = Image.open(s.image_path).convert("RGB")
        preds.append(backend.infer([img], [s.command])[0])
        gts.append(s.box)
    return accuracy_at_50(preds, gts)


def to_markdown(results: dict[str, dict]) -> str:
    lines = [
        "| Config | Batch | p50 (ms) | p90 (ms) | Throughput (img/s) | Peak VRAM (GB) | acc@0.5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, info in results.items():
        acc = f"{info['acc']:.3f}" if info["acc"] is not None else "—"
        for i, r in enumerate(info["rows"]):
            if r["p50_ms"] is None:
                cells = ["OOM", "OOM", "OOM", "OOM"]
            else:
                cells = [
                    f"{r['p50_ms']:.1f}",
                    f"{r['p90_ms']:.1f}",
                    f"{r['img_s']:.1f}",
                    f"{r['vram_gb']:.2f}",
                ]
            lines.append(
                f"| {name if i == 0 else ''} | {r['batch']} | {cells[0]} | {cells[1]} | "
                f"{cells[2]} | {cells[3]} | {acc if i == 0 else ''} |"
            )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/gdino-t2c")
    ap.add_argument(
        "--precisions", nargs="+", default=["fp32", "fp16", "bf16"], choices=list(PRECISIONS)
    )
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--acc-samples", type=int, default=50, help="val slice for accuracy (0=skip)")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="assets/benchmark.md")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("benchmark needs a CUDA GPU.")
    device = torch.device("cuda")
    gpu = torch.cuda.get_device_name(0)

    val = load_split(Path(args.data_dir), "val")
    lat_imgs = [Image.open(val[i].image_path).convert("RGB") for i in range(8)]
    lat_cmds = [val[i].command for i in range(8)]
    acc_samples = val[: args.acc_samples] if args.acc_samples else []

    results: dict[str, dict] = {}
    for prec in args.precisions:
        backend = GDinoBackend(args.checkpoint, prec, device)
        backend.load()
        rows = measure_latency(
            backend, lat_imgs, lat_cmds, args.batch_sizes, args.iters, args.warmup
        )
        acc = measure_accuracy(backend, acc_samples) if acc_samples else None
        results[backend.name] = {"rows": rows, "acc": acc}
        print(f"done: {backend.name}", flush=True)
        del backend
        torch.cuda.empty_cache()

    header = (
        f"# Inference benchmark — Grounding DINO (Talk2Car)\n\n"
        f"GPU: {gpu} · {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB · "
        f"iters={args.iters}, acc slice n={args.acc_samples}\n\n"
    )
    table = to_markdown(results)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + table + "\n")
    print("\n" + header + table)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
