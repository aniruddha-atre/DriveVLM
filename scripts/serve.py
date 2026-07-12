"""FastAPI service for Talk2Car referring-expression grounding.

POST an image + command to /predict, get the predicted box back as JSON. The model is loaded once
at startup and reused across requests.

    uv run --extra serving python scripts/serve.py
    curl -s -F file=@car.jpg -F command="the white truck" localhost:8000/predict
"""

from __future__ import annotations

import argparse
import io
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, cast

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from transformers import AutoProcessor, GroundingDinoForObjectDetection

CHECKPOINT = "checkpoints/gdino-t2c"


@dataclass
class Runtime:
    # HF processor/model classes are loosely typed, so Any here avoids a cast at every call site.
    processor: Any
    model: Any
    device: torch.device
    checkpoint: str


RT: Runtime | None = None  # populated on startup, cleared on shutdown


class Prediction(BaseModel):
    box: list[float] | None  # [x1, y1, x2, y2] in pixels; None if nothing matched
    confidence: float | None
    latency_ms: float
    model: str
    device: str


def load_runtime(checkpoint: str) -> Runtime:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(checkpoint)
    model = GroundingDinoForObjectDetection.from_pretrained(checkpoint)
    cast(torch.nn.Module, model).to(device)  # cast works around a wrong .to() stub on HF models
    model.eval()
    return Runtime(processor, model, device, checkpoint)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global RT
    RT = load_runtime(app.state.checkpoint)
    _predict(Image.new("RGB", (640, 384)), "a car")  # warm up so the first real request isn't cold
    yield
    RT = None


def _predict(image: Image.Image, command: str) -> tuple[list[float] | None, float | None]:
    assert RT is not None
    text = command.lower().strip().rstrip(".") + " ."
    inp = RT.processor(images=image, text=text, return_tensors="pt").to(RT.device)
    with torch.inference_mode():
        out = RT.model(**inp)
    res = RT.processor.post_process_grounded_object_detection(
        out, inp["input_ids"], threshold=0.0, text_threshold=0.0, target_sizes=[image.size[::-1]]
    )[0]
    if len(res["scores"]) == 0:
        return None, None
    best = int(res["scores"].argmax())  # one referred object per command, so take the top box
    return [round(v, 1) for v in res["boxes"][best].tolist()], float(res["scores"][best])


app = FastAPI(title="Drive-VLM grounding API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Readiness probe."""
    return {
        "status": "ok" if RT is not None else "loading",
        "model": RT.checkpoint if RT is not None else None,
        "device": str(RT.device) if RT is not None else None,
    }


@app.post("/predict", response_model=Prediction)
async def predict(
    file: UploadFile = File(...),  # noqa: B008 — File/Form in defaults is how FastAPI declares params
    command: str = Form(...),  # noqa: B008
) -> Prediction:
    """Ground the referred object in an uploaded image."""
    if RT is None:
        raise HTTPException(status_code=503, detail="model still loading")
    if not command.strip():
        raise HTTPException(status_code=422, detail="command must not be empty")
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"could not read image: {e}") from e

    start = time.perf_counter()
    box, confidence = _predict(image, command)
    return Prediction(
        box=box,
        confidence=confidence,
        latency_ms=round((time.perf_counter() - start) * 1e3, 1),
        model=RT.checkpoint,
        device=str(RT.device),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    app.state.checkpoint = args.checkpoint
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
