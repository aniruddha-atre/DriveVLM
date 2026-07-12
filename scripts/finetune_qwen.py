"""
LoRA fine-tune Qwen2.5-VL on Talk2Car referring-expression grounding (box-as-text SFT).

Qwen2.5-VL writes the box out as text. We fine-tune it with LoRA (PEFT) to output the right
`[x1, y1, x2, y2]` for the referred object, training the loss only on the box tokens (the prompt
is masked out). Qwen keeps coordinates close to the original 1600x900 image, so targets use raw
pixel coords — the same space the eval parser reads.

Container deps: peft, qwen-vl-utils, accelerate. VALIDATE first: `--limit 8 --max-steps 2`.

    python scripts/finetune_qwen.py --epochs 2 --lr 1e-4 --grad-accum 8
"""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
from typing import cast

import mlflow
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)

from drive_vlm.data import Sample, load_split
from drive_vlm.eval import Box, accuracy_at_50, parse_box

CKPT = "Qwen/Qwen2.5-VL-3B-Instruct"


def grounding_prompt(command: str) -> str:
    return (
        "This is a front-camera image from a car. The following driving command refers to "
        f'exactly one object in the scene: "{command}". '
        "Output only the bounding box of that referred object as [x1, y1, x2, y2] in pixel "
        "coordinates of this image."
    )


def target_str(box: Box) -> str:
    x1, y1, x2, y2 = (round(v) for v in box)
    return f"[{x1}, {y1}, {x2}, {y2}]"


class SampleDataset(Dataset):
    def __init__(self, items: list[Sample]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Sample:
        return self.items[i]


def build_collate(processor):
    def collate(batch: list[Sample]):
        # batch_size=1 keeps label-masking simple and correct (no padding edge cases).
        s = batch[0]
        img = Image.open(s.image_path).convert("RGB")
        user = {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": grounding_prompt(s.command)},
            ],
        }
        asst = {"role": "assistant", "content": [{"type": "text", "text": target_str(s.box)}]}
        full = processor.apply_chat_template(
            [user, asst], tokenize=False, add_generation_prompt=False
        )
        prompt = processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[full], images=[img], return_tensors="pt")
        prompt_len = processor(text=[prompt], images=[img], return_tensors="pt")["input_ids"].shape[
            1
        ]
        labels = inputs["input_ids"].clone()
        labels[:, :prompt_len] = -100  # loss only on the assistant's box tokens
        inputs["labels"] = labels
        return inputs

    return collate


@torch.no_grad()
def eval_accuracy(model, processor, samples: list[Sample], device: torch.device) -> dict:
    model.eval()
    preds: list[Box | None] = []
    for s in samples:
        img = Image.open(s.image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": grounding_prompt(s.command)},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt").to(device)
        gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        out = processor.batch_decode(
            gen[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]
        preds.append(parse_box(out))
    gts = [s.box for s in samples]
    return {
        "accuracy_at_50": accuracy_at_50(preds, gts),
        "n": len(samples),
        "n_no_prediction": sum(p is None for p in preds),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--val-limit", type=int, default=300, help="val subset for per-epoch eval")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out", default="checkpoints/qwen25vl-t2c-lora")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--allow-cpu", action="store_true")
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
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16
    )
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    cast(torch.nn.Module, model).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)

    loader = DataLoader(
        SampleDataset(train),
        batch_size=1,
        shuffle=True,
        collate_fn=build_collate(processor),
        num_workers=args.num_workers,
    )
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    total_updates = (len(loader) * args.epochs) // args.grad_accum
    sched = get_cosine_schedule_with_warmup(optim, args.warmup_steps, max(total_updates, 1))

    mlflow.set_experiment("drive-vlm")
    with mlflow.start_run(run_name="finetune-qwen25vl-lora"):
        mlflow.set_tag("stage", "finetune")
        mlflow.log_params(
            {
                "model": args.checkpoint,
                "method": "lora",
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "epochs": args.epochs,
                "lr": args.lr,
                "grad_accum": args.grad_accum,
                "n_train": len(train),
                "trainable_params": n_train,
            }
        )
        print(f"train={len(train)} val={len(val)} LoRA trainable={n_train:,} device={device}")

        base = eval_accuracy(model, processor, val, device)
        print(f"[baseline/zero-shot] val {base}", flush=True)
        mlflow.log_metric("accuracy_at_50", base["accuracy_at_50"], step=0)
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        processor.save_pretrained(str(out_dir))
        best_acc, best_epoch, no_improve = base["accuracy_at_50"], -1, 0

        step = 0
        optim.zero_grad()
        for epoch in range(args.epochs):
            model.train()
            for inputs in loader:
                inputs = inputs.to(device)
                with (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if device.type == "cuda"
                    else contextlib.nullcontext()
                ):
                    loss = model(**inputs).loss / args.grad_accum
                loss.backward()
                if (step + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optim.step()
                    sched.step()
                    optim.zero_grad()
                if step % args.log_every == 0:
                    print(
                        f"epoch {epoch} step {step} loss {loss.item() * args.grad_accum:.4f}",
                        flush=True,
                    )
                    mlflow.log_metric("train_loss", loss.item() * args.grad_accum, step=step)
                step += 1
                if args.max_steps and step >= args.max_steps:
                    break
            if args.max_steps and step >= args.max_steps:
                print("max-steps reached (smoke) — stopping.")
                break

            metrics = eval_accuracy(model, processor, val, device)
            print(f"[epoch {epoch}] val {metrics}", flush=True)
            mlflow.log_metric("accuracy_at_50", metrics["accuracy_at_50"], step=epoch + 1)
            if metrics["accuracy_at_50"] > best_acc:
                best_acc, best_epoch, no_improve = metrics["accuracy_at_50"], epoch, 0
                model.save_pretrained(str(out_dir))  # saves LoRA adapter only
                print(f"  ^ new best {best_acc:.4f} -> adapter saved to {out_dir}", flush=True)
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"early stopping: no val gain for {args.patience} epochs", flush=True)
                    break

        mlflow.log_metric("best_accuracy_at_50", best_acc)
        base_acc = base["accuracy_at_50"]
        print(f"done. best={best_acc:.4f} @ epoch {best_epoch} (baseline {base_acc:.4f})")


if __name__ == "__main__":
    main()
