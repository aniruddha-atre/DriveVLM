"""
Zero-shot Qwen2.5-VL referring-expression grounding on Talk2Car -> accuracy@50.

Qwen2.5-VL is an autoregressive VLM: it emits the box as TEXT. We prompt for the box and parse
the first [x1,y1,x2,y2]. REC = one object, so we take that single box.

Container deps needed (add to .venvc): `qwen-vl-utils`, `accelerate`.
CAVEAT — VALIDATE FIRST on a small --limit: Qwen2.5-VL returns coordinates in the *processed*
(smart-resized) image space, so they may need scaling back to the original 1600x900. Check a few
predictions visually before trusting the number; add a scale factor here if boxes are off.

    python scripts/eval_qwen.py --split val --limit 30
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import mlflow
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from drive_vlm.data import load_split
from drive_vlm.eval import Box, accuracy_at_50, parse_box

CKPT = "Qwen/Qwen2.5-VL-3B-Instruct"


def predict(model, processor, image: Image.Image, command: str, device) -> Box | None:
    prompt = (
        "This is a front-camera image from a car. The following driving command refers to "
        f'exactly one object in the scene: "{command}". '
        "Output only the bounding box of that referred object as [x1, y1, x2, y2] in pixel "
        "coordinates of this image."
    )
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
    gen = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    out = processor.batch_decode(gen[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[
        0
    ]
    return parse_box(out)  # xyxy pixels — MAY need scaling (see caveat above)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--adapter", default=None, help="path to a LoRA adapter (fine-tuned eval)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = load_split(Path(args.data_dir), args.split)
    if args.limit:
        samples = samples[: args.limit]

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    cast(torch.nn.Module, model).to(device)  # in-place; nn.Module view avoids broken HF .to() stub
    model.eval()

    preds: list[Box | None] = []
    with torch.no_grad():
        for s in tqdm(samples, desc="qwen2.5-vl"):
            image = Image.open(s.image_path).convert("RGB")
            preds.append(predict(model, processor, image, s.command, device))
    gts = [s.box for s in samples]

    metrics = {
        "accuracy_at_50": accuracy_at_50(preds, gts),
        "n": len(samples),
        "n_no_prediction": sum(p is None for p in preds),
    }
    mode = "finetuned" if args.adapter else "zeroshot"
    mlflow.set_experiment("drive-vlm")
    with mlflow.start_run(run_name=f"{mode}-qwen25vl-{args.split}"):
        mlflow.set_tag("stage", mode)
        mlflow.log_params(
            {
                "model": args.checkpoint,
                "split": args.split,
                "n": len(samples),
                "adapter": args.adapter,
            }
        )
        mlflow.log_metrics(metrics)
    print(f"=== {mode} Qwen2.5-VL ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    main()
