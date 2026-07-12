"""
Serving benchmark for the LoRA-fine-tuned Qwen2.5-VL across precision/quantization tiers.

Metrics are generation-shaped: TTFT (prefill), end-to-end latency, decode tokens/s, peak VRAM, and
accuracy@0.5 on a val slice (a memory saving only counts if quality holds). The same LoRA adapter
sits on the base in every tier.

Tiers: bf16 (native precision, the serving baseline), fp16, and 4-bit bitsandbytes NF4. Qwen is
bf16-trained, so fp16 is included to show it is not a free swap: its narrower range can overflow
transformer activations.

    python scripts/benchmark_qwen.py --tiers bf16 fp16 nf4 --n 60 --out assets/benchmark_qwen.md
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import cast

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from drive_vlm.data import load_split
from drive_vlm.eval import accuracy_at_50, parse_box

BASE = "Qwen/Qwen2.5-VL-3B-Instruct"


def grounding_prompt(command: str) -> str:
    return (
        "This is a front-camera image from a car. The following driving command refers to "
        f'exactly one object in the scene: "{command}". '
        "Output only the bounding box of that referred object as [x1, y1, x2, y2] in pixel "
        "coordinates of this image."
    )


class QwenBackend:
    def __init__(self, base: str, adapter: str | None, tier: str, device: torch.device):
        self.base, self.adapter, self.tier, self.device = base, adapter, tier, device
        self.name = f"qwen-{tier}" + ("+lora" if adapter else "")

    def load(self) -> None:
        from peft import PeftModel

        self.processor = AutoProcessor.from_pretrained(self.base)
        if self.tier == "nf4":
            qcfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.base, quantization_config=qcfg, torch_dtype=torch.bfloat16, device_map={"": 0}
            )
        else:  # bf16 (native) or fp16
            dtype = torch.float16 if self.tier == "fp16" else torch.bfloat16
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.base, torch_dtype=dtype
            )
            cast(torch.nn.Module, model).to(self.device)
        if self.adapter:
            model = PeftModel.from_pretrained(model, self.adapter)
        self.model = model.eval()

    def _inputs(self, img: Image.Image, command: str):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": grounding_prompt(command)},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.processor(text=[text], images=[img], return_tensors="pt").to(self.device)

    @torch.inference_mode()
    def generate(self, img: Image.Image, command: str, max_new_tokens: int) -> tuple[str, int]:
        inp = self._inputs(img, command)
        gen = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        n_new = gen.shape[1] - inp["input_ids"].shape[1]
        text = self.processor.batch_decode(
            gen[:, inp["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]
        return text, int(n_new)


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def benchmark(backend: QwenBackend, samples, max_new_tokens: int, warmup: int) -> dict:
    img0 = Image.open(samples[0].image_path).convert("RGB")
    for _ in range(warmup):
        backend.generate(img0, samples[0].command, max_new_tokens)
    torch.cuda.reset_peak_memory_stats()

    ttfts, e2es, decode_rates, preds, gts = [], [], [], [], []
    for s in samples:
        img = Image.open(s.image_path).convert("RGB")
        (_, _), t_ttft = timed(lambda im=img, c=s.command: backend.generate(im, c, 1))
        (text, n_new), t_e2e = timed(
            lambda im=img, c=s.command: backend.generate(im, c, max_new_tokens)
        )
        ttfts.append(t_ttft)
        e2es.append(t_e2e)
        if n_new > 1 and t_e2e > t_ttft:
            decode_rates.append((n_new - 1) / (t_e2e - t_ttft))
        preds.append(parse_box(text))
        gts.append(s.box)

    def p(xs, q):
        return sorted(xs)[int(q * (len(xs) - 1))]

    return {
        "ttft_ms": statistics.median(ttfts) * 1e3,
        "e2e_ms": statistics.median(e2es) * 1e3,
        "e2e_p90_ms": p(e2es, 0.9) * 1e3,
        "decode_tok_s": statistics.mean(decode_rates) if decode_rates else 0.0,
        "req_s": 1.0 / statistics.median(e2es),
        "vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "acc": accuracy_at_50(preds, gts),
    }


def to_markdown(results: dict[str, dict], gpu: str, n: int) -> str:
    head = (
        f"# Serving benchmark — Qwen2.5-VL-3B + LoRA (Talk2Car)\n\n"
        f"GPU: {gpu} · batch 1 · n={n} · greedy\n\n"
        "| Tier | TTFT p50 (ms) | e2e p50 (ms) | e2e p90 (ms) | decode (tok/s) | "
        "req/s | Peak VRAM (GB) | acc@0.5 |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = []
    for name, r in results.items():
        rows.append(
            f"| {name} | {r['ttft_ms']:.0f} | {r['e2e_ms']:.0f} | {r['e2e_p90_ms']:.0f} | "
            f"{r['decode_tok_s']:.1f} | {r['req_s']:.2f} | {r['vram_gb']:.2f} | {r['acc']:.3f} |"
        )
    return head + "\n".join(rows) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--adapter", default="checkpoints/qwen25vl-t2c-lora-v2")
    ap.add_argument(
        "--tiers", nargs="+", default=["bf16", "fp16", "nf4"], choices=["bf16", "fp16", "nf4"]
    )
    ap.add_argument("--n", type=int, default=60, help="val samples for latency + accuracy")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="assets/benchmark_qwen.md")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("Qwen benchmark needs a CUDA GPU (won't fit a small card).")
    device = torch.device("cuda")
    gpu = torch.cuda.get_device_name(0)
    samples = load_split(Path(args.data_dir), "val")[: args.n]

    results: dict[str, dict] = {}
    for tier in args.tiers:
        backend = QwenBackend(args.base, args.adapter, tier, device)
        backend.load()
        results[backend.name] = benchmark(backend, samples, args.max_new_tokens, args.warmup)
        print(f"done: {backend.name} -> {results[backend.name]}", flush=True)
        del backend
        torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = to_markdown(results, gpu, args.n)
    out.write_text(table)
    print("\n" + table + f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
