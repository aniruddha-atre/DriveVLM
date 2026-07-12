# Serving benchmark — Qwen2.5-VL-3B + LoRA (Talk2Car)

GPU: NVIDIA H100 · batch 1 · n=60 · greedy

| Tier | TTFT p50 (ms) | e2e p50 (ms) | e2e p90 (ms) | decode (tok/s) | req/s | Peak VRAM (GB) | acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen-bf16+lora | 216 | 759 | 795 | 38.4 | 1.32 | 8.11 | 0.767 |
| qwen-fp16+lora | 214 | 797 | 1030 | 38.1 | 1.26 | 8.11 | 0.433 |
| qwen-nf4+lora | 239 | 1151 | 1216 | 22.7 | 0.87 | 2.89 | 0.783 |
