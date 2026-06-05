"""Grounding DINO wrapper: image + text command -> best bounding box.

Thin layer over the HF `transformers` Grounding DINO model so the rest of the code never
touches model internals. Zero-shot for now; fine-tuning comes later.
"""

from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

Box = tuple[float, float, float, float]

DEFAULT_CHECKPOINT = "IDEA-Research/grounding-dino-tiny"


class GroundingDINO:
    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image: Image.Image, command: str, box_threshold: float = 0.3) -> Box | None:
        """Return the highest-scoring box for the command, or None if nothing clears threshold."""
        inputs = self.processor(images=image, text=command, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=box_threshold,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]],  # (height, width)
        )[0]

        if len(results["scores"]) == 0:
            return None
        best = int(results["scores"].argmax())
        x1, y1, x2, y2 = results["boxes"][best].tolist()
        return (x1, y1, x2, y2)
