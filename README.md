# Drive-VLM

Language-guided object grounding for autonomous driving — an **end-to-end MLOps project** on
the [Talk2Car](https://talk2car.github.io/) dataset.

> Given a front-camera image and a natural-language command 
(*"Park behind the white truck."*), predict the bounding box of the referred object.

> Create a full production lifecycle - data versioning, config-driven training, experiment tracking, a model registry, a served API, a demo UI, and monitoring.

## Model

V1-  **Grounding DINO** (discriminative grounding detector).

## Metric

**AP50** (IoU > 0.5 against ground-truth box) — the official Talk2Car benchmark.
