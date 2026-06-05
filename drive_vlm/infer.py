"""Run a model over Talk2Car samples and score it.

Keeps the loop tiny and model-agnostic: anything with a `.predict(image, command) -> box`
works here, so the same code scores zero-shot and (later) fine-tuned models.
"""

from __future__ import annotations

from PIL import Image
from tqdm import tqdm

from drive_vlm.data import Sample
from drive_vlm.eval import Box, accuracy_at_50


def predict_over(model, samples: list[Sample], box_threshold: float = 0.3) -> list[Box | None]:
    """Run the model on each sample's (image, command), returning predicted boxes."""
    preds: list[Box | None] = []
    for s in tqdm(samples, desc="predict"):
        image = Image.open(s.image_path).convert("RGB")
        preds.append(model.predict(image, s.command, box_threshold=box_threshold))
    return preds


def evaluate(model, samples: list[Sample], box_threshold: float = 0.3) -> dict:
    """Predict over samples and return {accuracy@0.5, n, n_missed}."""
    preds = predict_over(model, samples, box_threshold=box_threshold)
    gts = [s.box for s in samples]
    return {
        "accuracy_at_50": accuracy_at_50(preds, gts),
        "n": len(samples),
        "n_no_prediction": sum(p is None for p in preds),
    }
