"""
Fine-tune Grounding DINO on Talk2Car referring-expression grounding.

Each command describes ONE object, so for HF's GroundingDINO loss every image is a single
"class" (index 0, whose text positive-map the model builds from the command) with one box.

    # smoke test (tiny, validates forward+backward on GPU in seconds):
    python scripts/finetune_gdino.py --limit 8 --max-steps 2 --epochs 1 --allow-cpu
    # proper run (train backbone at 0.1x lr, warmup+cosine, full val each epoch):
    python scripts/finetune_gdino.py --epochs 8 --batch-size 4 --lr 1e-4

Defaults: backbone trainable at lr*0.1, warmup 500 + cosine decay, full val eval, best-epoch
checkpointing. Logged to MLflow: train_loss + lr per step, accuracy_at_50 on val per epoch.
"""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
from typing import cast

import mlflow
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoProcessor,
    GroundingDinoForObjectDetection,
    get_cosine_schedule_with_warmup,
)

from drive_vlm.data import Sample, load_split
from drive_vlm.eval import accuracy_at_50

CHECKPOINT = "IDEA-Research/grounding-dino-tiny"


class SampleDataset(Dataset):
    """Map-style dataset over a list of Samples (so DataLoader is correctly typed)."""

    def __init__(self, items: list[Sample]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Sample:
        return self.items[i]


def format_text(command: str) -> str:
    """GroundingDINO wants lowercase text delimited by a period (defines one class)."""
    return command.lower().strip().rstrip(".") + " ."


def box_xyxy_to_norm_cxcywh(box, w: int, h: int) -> list[float]:
    x1, y1, x2, y2 = box
    return [((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h, (x2 - x1) / w, (y2 - y1) / h]


def make_collate(processor):
    def collate(batch: list[Sample]):
        images = [Image.open(s.image_path).convert("RGB") for s in batch]
        texts = [format_text(s.command) for s in batch]
        inputs = processor(images=images, text=texts, return_tensors="pt", padding=True)
        labels = []
        for s, img in zip(batch, images, strict=True):
            w, h = img.size
            labels.append(
                {
                    "class_labels": torch.tensor([0], dtype=torch.long),
                    "boxes": torch.tensor(
                        [box_xyxy_to_norm_cxcywh(s.box, w, h)], dtype=torch.float
                    ),
                }
            )
        return inputs, labels

    return collate


@torch.no_grad()
def eval_accuracy(
    model: GroundingDinoForObjectDetection, processor, samples: list[Sample], device: torch.device
) -> dict:
    """REC has exactly one object per command -> ALWAYS take the top-scoring box (no threshold)."""
    model.eval()
    preds, gts = [], []
    for s in samples:
        image = Image.open(s.image_path).convert("RGB")
        inputs = processor(images=image, text=format_text(s.command), return_tensors="pt").to(
            device
        )
        out = model(**inputs)
        # threshold=0 -> keep all queries, then pick the single highest-scoring box.
        res = processor.post_process_grounded_object_detection(
            out,
            inputs["input_ids"],
            threshold=0.0,
            text_threshold=0.0,
            target_sizes=[image.size[::-1]],
        )[0]
        if len(res["scores"]) == 0:
            preds.append(None)
        else:
            best = int(res["scores"].argmax())
            preds.append(tuple(res["boxes"][best].tolist()))
        gts.append(s.box)
    return {
        "accuracy_at_50": accuracy_at_50(preds, gts),
        "n": len(samples),
        "n_no_prediction": sum(p is None for p in preds),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--checkpoint", default=CHECKPOINT, help="HF Grounding DINO checkpoint")
    ap.add_argument(
        "--epochs", type=int, default=10, help="max epochs (early stopping may cut short)"
    )
    ap.add_argument(
        "--patience", type=int, default=3, help="early stop after N epochs w/o val gain"
    )
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4, help="lr for transformer/heads")
    ap.add_argument("--backbone-lr-mult", type=float, default=0.1, help="backbone lr = lr * this")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None, help="cap train samples (smoke tests)")
    ap.add_argument(
        "--val-limit", type=int, default=None, help="cap val samples (default: full val)"
    )
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (smoke tests)")
    ap.add_argument("--freeze-backbone", action="store_true", help="freeze vision+text backbones")
    ap.add_argument("--out", default="checkpoints/gdino-t2c")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--allow-cpu", action="store_true", help="local logic tests only")
    args = ap.parse_args()

    use_cuda = torch.cuda.is_available()
    if not use_cuda and not args.allow_cpu:
        raise SystemExit("No GPU detected — refusing to train on CPU. (Use --allow-cpu for tests.)")
    device = torch.device("cuda" if use_cuda else "cpu")

    train = load_split(Path(args.data_dir), "train")
    if args.limit:
        train = train[: args.limit]
    val = load_split(Path(args.data_dir), "val")[: args.val_limit]

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    model = GroundingDinoForObjectDetection.from_pretrained(args.checkpoint)
    assert isinstance(model, GroundingDinoForObjectDetection)  # narrow from_pretrained's union type
    # Move to device in place; the nn.Module view avoids a broken HF .to() type stub.
    cast(torch.nn.Module, model).to(device)

    if args.freeze_backbone:
        for n, p in model.named_parameters():
            if "backbone" in n:
                p.requires_grad_(False)

    # Two param groups: the pretrained backbone trains at a lower LR than the heads/transformer.
    backbone = [p for n, p in model.named_parameters() if p.requires_grad and "backbone" in n]
    heads = [p for n, p in model.named_parameters() if p.requires_grad and "backbone" not in n]
    groups = [{"params": heads, "lr": args.lr}]
    if backbone:
        groups.append({"params": backbone, "lr": args.lr * args.backbone_lr_mult})
    trainable = heads + backbone
    n_train = sum(p.numel() for p in trainable)

    loader = DataLoader(
        dataset=SampleDataset(train),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate(processor),
        num_workers=args.num_workers,
    )
    optim = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    total_steps = len(loader) * args.epochs
    sched = get_cosine_schedule_with_warmup(optim, args.warmup_steps, total_steps)

    mlflow.set_experiment("drive-vlm")
    with mlflow.start_run(run_name="finetune-gdino"):
        mlflow.set_tag("stage", "finetune")
        mlflow.log_params(
            {
                "model": args.checkpoint,
                "epochs": args.epochs,
                "patience": args.patience,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "backbone_lr_mult": args.backbone_lr_mult,
                "warmup_steps": args.warmup_steps,
                "weight_decay": args.weight_decay,
                "freeze_backbone": args.freeze_backbone,
                "n_train": len(train),
                "n_val": len(val),
                "trainable_params": n_train,
            }
        )
        print(f"train={len(train)} val={len(val)} trainable_params={n_train:,} device={device}")

        # Zero-shot baseline with the SAME (top-1) metric, for an apples-to-apples comparison.
        base = eval_accuracy(model, processor, val, device)
        print(f"[baseline/zero-shot] val {base}", flush=True)
        mlflow.log_metric("accuracy_at_50", base["accuracy_at_50"], step=0)
        mlflow.log_metric("baseline_accuracy_at_50", base["accuracy_at_50"])

        # Save processor up front; checkpoint the BEST epoch as we go (c23g_low is preemptible,
        # so a long run can be killed mid-way — never rely on a single save at the end).
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        processor.save_pretrained(out_dir)
        best_acc = base["accuracy_at_50"]
        best_epoch = -1  # -1 = baseline/zero-shot is still the best
        epochs_no_improve = 0

        step = 0
        for epoch in range(args.epochs):
            model.train()
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = [{k: v.to(device) for k, v in t.items()} for t in labels]
                amp = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if device.type == "cuda"
                    else contextlib.nullcontext()
                )
                with amp:
                    out = model(**inputs, labels=labels)
                loss = out.loss
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 0.1)
                optim.step()
                sched.step()
                if step % args.log_every == 0:
                    lr_now = float(sched.get_last_lr()[0])
                    print(
                        f"epoch {epoch} step {step} loss {loss.item():.4f} lr {lr_now:.2e}",
                        flush=True,
                    )
                    mlflow.log_metric("train_loss", loss.item(), step=step)
                    mlflow.log_metric("lr", lr_now, step=step)
                step += 1
                if args.max_steps and step >= args.max_steps:
                    break
            if args.max_steps and step >= args.max_steps:
                print("max-steps reached (smoke test) — stopping.")
                break

            metrics = eval_accuracy(model, processor, val, device)
            print(f"[epoch {epoch}] val {metrics}", flush=True)
            mlflow.log_metric("accuracy_at_50", metrics["accuracy_at_50"], step=epoch + 1)

            if metrics["accuracy_at_50"] > best_acc:
                best_acc = metrics["accuracy_at_50"]
                best_epoch = epoch
                epochs_no_improve = 0
                model.save_pretrained(out_dir)
                print(f"  ^ new best {best_acc:.4f} -> checkpoint saved to {out_dir}", flush=True)
            else:
                epochs_no_improve += 1
                print(f"  no improvement ({epochs_no_improve}/{args.patience})", flush=True)
                if epochs_no_improve >= args.patience:
                    print(f"early stopping: no val gain for {args.patience} epochs", flush=True)
                    break

        mlflow.log_metric("best_accuracy_at_50", best_acc)
        mlflow.log_metric("best_epoch", best_epoch)
        print(
            f"done. best accuracy_at_50={best_acc:.4f} @ epoch {best_epoch} "
            f"(baseline was {base['accuracy_at_50']:.4f})"
        )


if __name__ == "__main__":
    main()
