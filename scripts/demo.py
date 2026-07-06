"""
Gradio demo — Talk2Car referring-expression grounding.

Pick a model, give it an image (or a video) plus a natural-language command; the model draws a
box around the referred object.

- **Grounding DINO** (fine-tuned) runs locally from `checkpoints/gdino-t2c`.
- **Qwen2.5-VL LoRA** (fine-tuned) runs locally from `checkpoints/qwen25vl-t2c-lora-v2` on top of
  `Qwen/Qwen2.5-VL-3B-Instruct`.
- **Video** is processed *frame by frame* — these are single-image models, so there is no temporal
  tracking; each sampled frame is grounded independently.
  
    uv run --extra serving python scripts/demo.py
    uv run --extra serving python scripts/demo.py --share
    uv run --extra serving python scripts/demo.py --qwen-adapter checkpoints/qwen25vl-t2c-lora-v2
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, GroundingDinoForObjectDetection

Box = tuple[float, float, float, float]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# On CPU, use every core — this is the single biggest latency win for GDINO.
if DEVICE.type == "cpu":
    torch.set_num_threads(os.cpu_count() or 1)

GDINO_NAME = "Grounding DINO (fine-tuned)"
QWEN_NAME = "Qwen2.5-VL LoRA (fine-tuned)"
BOX_RE = re.compile(r"(-?\d+\.?\d*)\D+(-?\d+\.?\d*)\D+(-?\d+\.?\d*)\D+(-?\d+\.?\d*)")


def grounding_prompt(command: str) -> str:
    return (
        "This is a front-camera image from a car. The following driving command refers to "
        f'exactly one object in the scene: "{command}". '
        "Output only the bounding box of that referred object as [x1, y1, x2, y2] in pixel "
        "coordinates of this image."
    )


class GDINOModel:
    """Fine-tuned Grounding DINO — one referred object, so always take the top-1 box."""

    def __init__(self, checkpoint: str):
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        model = GroundingDinoForObjectDetection.from_pretrained(checkpoint)
        cast(torch.nn.Module, model).to(DEVICE)
        self.model = model.eval()

    def predict(self, img: Image.Image, command: str) -> tuple[Box | None, float | None]:
        text = command.lower().strip().rstrip(".") + " ."
        inp = self.processor(images=img, text=text, return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            out = self.model(**inp)
        res = self.processor.post_process_grounded_object_detection(
            out, inp["input_ids"], threshold=0.0, text_threshold=0.0, target_sizes=[img.size[::-1]]
        )[0]
        if len(res["scores"]) == 0:
            return None, None
        best = int(res["scores"].argmax())
        return tuple(res["boxes"][best].tolist()), float(res["scores"][best])


class QwenModel:
    """Qwen2.5-VL + LoRA adapter — emits the box as text, which we parse."""

    def __init__(self, base: str, adapter: str | None):
        from peft import PeftModel  # noqa: PLC0415 — optional dep, import only when selected
        from transformers import Qwen2_5_VLForConditionalGeneration  # noqa: PLC0415

        dtype = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(base)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base, torch_dtype=dtype)
        if adapter:
            if not Path(adapter).exists():
                raise FileNotFoundError(f"adapter not found: {adapter}")
            model = PeftModel.from_pretrained(model, adapter)
        cast(torch.nn.Module, model).to(DEVICE)
        # Union of base model / PeftModel with HF's loose generate stub — type as Any.
        self.model: Any = model.eval()

    def predict(self, img: Image.Image, command: str) -> tuple[Box | None, float | None]:
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
        inp = self.processor(text=[text], images=[img], return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            gen = self.model.generate(**inp, max_new_tokens=64, do_sample=False)
        out = self.processor.batch_decode(
            gen[:, inp["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]
        m = BOX_RE.search(out)
        if not m:
            return None, None
        return (float(m[1]), float(m[2]), float(m[3]), float(m[4])), None  # no calibrated score


def draw(img: Image.Image, box: Box | None) -> Image.Image:
    if box is None:
        return img
    out = img.copy()
    ImageDraw.Draw(out).rectangle([round(v) for v in box], outline=(255, 60, 60), width=5)
    return out


def build_app(args: argparse.Namespace) -> gr.Blocks:
    loaders = {
        GDINO_NAME: lambda: GDINOModel(args.gdino_checkpoint),
        QWEN_NAME: lambda: QwenModel(args.qwen_base, args.qwen_adapter),
    }
    cache: dict[str, object] = {}

    def get_model(name: str):
        if name not in cache:
            cache[name] = loaders[name]()
        return cache[name]

    def ground_image(name: str, image: Image.Image | None, command: str):
        if image is None or not command.strip():
            return None, "Upload an image and enter a command."
        try:
            model = get_model(name)
        except Exception as e:  # missing adapter / deps / weights → say so, don't crash
            return None, f"⚠️ {name} is unavailable here: {e}"
        img = image.convert("RGB")
        box, score = model.predict(img, command)  # type: ignore[attr-defined]
        if box is None:
            return img, "No object found for that command."
        rounded = [round(v) for v in box]
        tail = f"   ·   confidence: {score:.2f}" if score is not None else ""
        return draw(img, box), f"box (x1,y1,x2,y2): {rounded}{tail}"

    def ground_video(name: str, path: str | None, command: str, fps: float, progress=gr.Progress()):  # noqa: B008 — gradio injects progress via this default
        if not path or not command.strip():
            raise gr.Error("Upload a video and enter a command.")
        try:
            import imageio.v2 as imageio  # noqa: PLC0415 — optional dep for the video tab
        except ImportError as e:
            raise gr.Error(f"video support needs imageio[ffmpeg]: {e}") from e
        try:
            model = get_model(name)
        except Exception as e:
            raise gr.Error(f"{name} is unavailable here: {e}") from e

        reader = imageio.get_reader(path)
        src_fps = float(reader.get_meta_data().get("fps", 30.0))
        stride = max(1, round(src_fps / fps))
        fd, out_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        writer = imageio.get_writer(out_path, fps=fps, macro_block_size=None)
        hits = kept = 0
        frames = cast(Iterator[np.ndarray], reader)  # imageio Reader stub mistypes __iter__
        for i, frame in enumerate(frames):
            if i % stride:
                continue
            if kept >= args.max_frames:
                break
            img = Image.fromarray(frame).convert("RGB")
            box, _ = model.predict(img, command)  # type: ignore[attr-defined]
            hits += box is not None
            writer.append_data(np.asarray(draw(img, box)))
            kept += 1
            progress(min(kept / args.max_frames, 0.99), desc=f"grounded {kept} frames")
        reader.close()
        writer.close()
        return out_path, f"processed {kept} frames @ {fps:g} fps · object found in {hits}/{kept}"

    # Warm up the default (local) model so the first real request is fast, not cold.
    try:
        warm = get_model(GDINO_NAME)
        warm.predict(Image.new("RGB", (640, 384)), "a car")  # type: ignore[attr-defined]
    except Exception:
        pass

    examples = None
    img_dir = Path("data/images")
    if img_dir.exists():
        picks = [("img_val_0.jpg", "the silver car"), ("img_val_6.jpg", "the grey vehicle")]
        examples = [[str(img_dir / n), c] for n, c in picks if (img_dir / n).exists()] or None

    with gr.Blocks(title="Drive-VLM · Talk2Car referring grounding") as app:
        gr.Markdown(
            "# Drive-VLM · Talk2Car referring grounding\n"
            "Describe **one** object in a driving scene; the model localizes it.\n"
            "Grounding DINO 0.757 · Qwen2.5-VL LoRA 0.752 accuracy@0.5 (val)."
        )
        model_sel = gr.Radio(
            [GDINO_NAME, QWEN_NAME], value=GDINO_NAME, label="Model"
        )
        with gr.Tab("Image"):
            with gr.Row():
                img_in = gr.Image(type="pil", label="Front-camera image")
                img_out = gr.Image(type="pil", label="Referred object")
            cmd = gr.Textbox(label="Command", placeholder="e.g. park behind the white truck")
            img_txt = gr.Textbox(label="Prediction", interactive=False)
            img_btn = gr.Button("Ground", variant="primary")
            if examples:
                gr.Examples(examples, inputs=[img_in, cmd])
            img_btn.click(ground_image, [model_sel, img_in, cmd], [img_out, img_txt])
            cmd.submit(ground_image, [model_sel, img_in, cmd], [img_out, img_txt])
        with gr.Tab("Video (frame-by-frame)"):
            gr.Markdown(
                "Single-image models applied per frame — **no temporal tracking**. "
                f"Capped at {args.max_frames} sampled frames; on CPU this can take a while."
            )
            with gr.Row():
                vid_in = gr.Video(label="Driving clip")
                vid_out = gr.Video(label="Grounded clip")
            vcmd = gr.Textbox(label="Command", placeholder="e.g. the pedestrian on the right")
            vfps = gr.Slider(1, 10, value=3, step=1, label="Sample rate (frames/sec)")
            vid_txt = gr.Textbox(label="Status", interactive=False)
            vid_btn = gr.Button("Ground video", variant="primary")
            vid_btn.click(ground_video, [model_sel, vid_in, vcmd, vfps], [vid_out, vid_txt])

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gdino-checkpoint", default="checkpoints/gdino-t2c")
    ap.add_argument("--qwen-base", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--qwen-adapter", default="checkpoints/qwen25vl-t2c-lora-v2")
    ap.add_argument("--max-frames", type=int, default=90, help="cap on frames processed per video")
    ap.add_argument("--share", action="store_true", help="create a public gradio link")
    args = ap.parse_args()

    build_app(args).queue().launch(share=args.share)


if __name__ == "__main__":
    main()
