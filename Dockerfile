# Serving image for the Grounding DINO grounding API (scripts/serve.py).
# CPU-only on purpose: the image runs anywhere, no GPU required. The fine-tuned checkpoint is
# baked in so the container is self-contained — `docker run` and it serves.
#
#   docker build -t drive-vlm-api .
#   docker run --rm -p 8000:8000 drive-vlm-api
#   curl -s -F file=@car.jpg -F command="the white truck" localhost:8000/predict

FROM python:3.12-slim

WORKDIR /app

# Dependencies first so this layer is cached across code changes. CPU build of torch keeps the
# image portable and much smaller than the CUDA build.
RUN pip install --no-cache-dir "torch>=2.7" --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir \
      "transformers>=5" pillow "fastapi>=0.110" "uvicorn[standard]>=0.29" python-multipart

# App code + the fine-tuned weights.
COPY scripts/serve.py ./scripts/serve.py
COPY checkpoints/gdino-t2c ./checkpoints/gdino-t2c

# Weights are baked in; never reach out to the Hub at runtime.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
EXPOSE 8000

# Container-level health check hitting our own /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8000"]
