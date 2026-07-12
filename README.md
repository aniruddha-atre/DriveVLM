# Drive-VLM

**Language-guided object grounding for autonomous driving.** Given a front-camera image and a
natural-language command, predict the bounding box of the referred object — end-to-end, from
versioned data to two fine-tuned models to a served demo, on the
[Talk2Car](https://talk2car.github.io/) benchmark.

![Two architectures](assets/architecture.png)

## The task

Talk2Car is built on nuScenes: ~12k commands over ~9.2k front-camera driving images, each command
referring to **one** object with a 2D box. Given the image + a command (e.g. *"stop alongside that
truck"*), output a single `[x1, y1, x2, y2]`.

**Metric:** accuracy @ IoU > 0.5 (Talk2Car's "AP50" — one object per command).

## Results

Two models from opposite paradigms, both fine-tuned into the Talk2Car SOTA band (~0.70–0.76):

| Model | Approach | val acc@0.5 | test acc@0.5 |
|---|---|---|---|
| **Grounding DINO-base** | fine-tuned (frozen backbone) | **0.757** | **0.755** |
| **Qwen2.5-VL-3B** | **LoRA fine-tune** (box-as-text) | **0.752** | **0.745** |
| Grounding DINO-base | zero-shot | 0.514 | — |
| Qwen2.5-VL-3B | zero-shot | 0.498 | — |

Negligible val→test drop (0.002 / 0.007) — the models generalize, they don't overfit.

![Grounding DINO vs Qwen2.5-VL](assets/comparison_grid.png)

## Approach

- **Grounding DINO** — a discriminative detector. Fine-tuned the decoder + detection heads with a
  **frozen backbone**; the box comes out as regressed coordinates.
- **Qwen2.5-VL-3B** — an autoregressive VLM. Fine-tuned with **LoRA/PEFT** on the attention layers
  (~15M trainable params), *box-as-text* SFT with answer-only loss masking; the box is **generated
  as text** (`[x1, y1, x2, y2]`) and parsed.
- A single **format-agnostic AP50** harness scores both identically (always top-1 box, since there
  is exactly one referred object).

### What didn't work

- **Unfreezing the Grounding DINO backbone** caused catastrophic forgetting — accuracy collapsed to
  0.37 and clawed back to only 0.58. Keeping the backbone frozen reached 0.757 within an epoch.
- **LoRA on a 3.5k subset (r=16)** plateaued at 0.707. Closing the gap to 0.752 took the full 8.3k
  training set *and* more adapter capacity (r=32) — the box-as-text VLM needed both.
- **Naive zero-shot prompting** left accuracy on the table; a driving-scene grounding prompt recovered
  +16 pts on Qwen, though zero-shot still trailed fine-tuning by ~25 pts.

## MLOps

![Pipeline](assets/mlops_pipeline.png)

- **DVC** versions the dataset — pointer files committed to git, image bytes in a remote.
- **MLflow** logs every run (params, metrics, checkpoints); `mlflow_hpc.db` holds all experiments.
- **Apptainer + SLURM** — training ran in an Apptainer container on RWTH HPC (H100) via SLURM.
- **Gradio** serves the fine-tuned checkpoint as an interactive demo.

## Deployment

### Serving API (FastAPI + Docker)

The fine-tuned Grounding DINO is served behind a REST API — `POST /predict` (multipart image +
command → box as JSON), a `GET /health` probe, typed request/response, model loaded once at
startup. Containerized CPU-only so it runs anywhere.

```bash
# run the API locally
uv run --extra serving python scripts/serve.py            # → http://localhost:8000  (docs at /docs)
curl -s -F file=@car.jpg -F command="the white truck" localhost:8000/predict
# {"box":[281.1,484.4,454.2,555.1],"confidence":0.51,"latency_ms":779,"model":"...","device":"cuda"}

# or run it as a container (weights baked in, self-contained)
docker build -t drive-vlm-api .
docker run --rm -p 8000:8000 drive-vlm-api
```

### Serving benchmark

A model-agnostic harness (`scripts/benchmark*.py`) measures latency / throughput / VRAM **and**
accuracy@0.5 on an H100, so a memory saving only counts if quality survives it. The deliverable is
the table and the finding — not a demo.

**Qwen2.5-VL-3B + LoRA — precision tiers (H100, batch 1, n=60):**

| Tier | e2e p50 | decode (tok/s) | Peak VRAM | acc@0.5 |
|---|---:|---:|---:|---:|
| bf16 (native) | 759 ms | 38.4 | 8.11 GB | 0.767 |
| fp16 | 797 ms | 38.1 | 8.11 GB | **0.433** |
| 4-bit NF4 | 1151 ms | 22.7 | **2.89 GB** | 0.783 |

**Findings:**
- **4-bit NF4** shrinks the VLM **2.8× in VRAM (8.1 → 2.9 GB) at no accuracy cost**, but runs
  **~50% slower** — at batch 1 the dequantization overhead outweighs the memory-bandwidth saving.
  A *fit-on-smaller-hardware* win, not a latency win.
- **fp16 is strictly worse than bf16** here: same speed and VRAM, but accuracy collapses
  (0.767 → 0.433). Qwen is bf16-trained, and fp16's narrower range overflows its activations — so
  bf16, not fp16, is the correct serving baseline.

**Grounding DINO — precision × batch:** latency is precision-invariant to within ~10% (mixed
precision via autocast, since naive `.half()` crashes in the deformable-attention `grid_sample`),
accuracy holds across fp32/fp16/bf16, and throughput is compute-bound — batching scales latency
roughly linearly rather than raising images/sec. Full tables in `assets/benchmark_*.md`.

## Quickstart

```bash
uv sync
uv run pytest                                    # AP50 / IoU metric tests
uv run --extra serving python scripts/demo.py    # Gradio demo (needs checkpoints/gdino-t2c)
```

Evaluate / fine-tune (GPU):

```bash
uv run python scripts/eval_gdino.py --checkpoint checkpoints/gdino-t2c --split test
uv run python scripts/finetune_gdino.py --freeze-backbone --epochs 6      # Grounding DINO
uv run python scripts/finetune_qwen.py --lora-r 32 --epochs 3             # Qwen2.5-VL LoRA
```

## Repository

```text
drive_vlm/
  data.py            Talk2Car → list[Sample(image, command, box)]
  eval.py            IoU + accuracy@0.5 (AP50) metric
scripts/
  finetune_gdino.py  Grounding DINO fine-tuning (frozen backbone, early stopping)
  finetune_qwen.py   Qwen2.5-VL LoRA fine-tuning (box-as-text SFT)
  eval_gdino.py      eval a GDINO checkpoint on any split
  eval_qwen.py       eval Qwen (zero-shot or with a LoRA adapter)
  benchmark.py       GDINO latency/throughput/VRAM across precisions
  benchmark_qwen.py  Qwen serving benchmark (bf16 vs 4-bit NF4)
  serve.py           FastAPI REST API (POST /predict, GET /health)
  demo.py            Gradio app on the fine-tuned checkpoint
Dockerfile           containerizes the serving API (CPU, weights baked in)
tests/               pytest for the metric & box parser
data/                Talk2Car (DVC-tracked, git-ignored)
```

## Stack

Python 3.12 · uv · PyTorch · HF Transformers · PEFT · bitsandbytes · DVC · MLflow · Apptainer ·
SLURM · Gradio

## Dataset & citation

This project uses the [Talk2Car](https://talk2car.github.io/) dataset (built on
[nuScenes](https://www.nuscenes.org/)).

> Deruyttere, T., Vandenhende, S., Grujicic, D., Van Gool, L., & Moens, M.-F. (2019).
> *Talk2Car: Taking Control of Your Self-Driving Car.* In Proceedings of the 2019 Conference on
> Empirical Methods in Natural Language Processing and the 9th International Joint Conference on
> Natural Language Processing (EMNLP-IJCNLP), 2088–2098.

```bibtex
@inproceedings{deruyttere2019talk2car,
  title     = {Talk2Car: Taking Control of Your Self-Driving Car},
  author    = {Deruyttere, Thierry and Vandenhende, Simon and Grujicic, Dusan
               and Van Gool, Luc and Moens, Marie-Francine},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language
               Processing and the 9th International Joint Conference on Natural Language
               Processing (EMNLP-IJCNLP)},
  pages     = {2088--2098},
  year      = {2019}
}
```
