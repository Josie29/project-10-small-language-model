# CPU-only image for the base-vs-tuned demo. Hugging Face gates Gradio Spaces behind a
# paid plan, so the demo is containerised and hosted elsewhere; `app.py` is unchanged and
# still runs as a Space if that ever becomes free again.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# The default torch wheel on PyPI bundles CUDA and is several gigabytes. This host has no
# GPU, so pull from the CPU index and keep the image small enough to deploy quickly.
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Bake both checkpoints into the image rather than downloading them on first request. A
# recorded demo cannot afford a two-minute cold start on the grader's live prompt.
ARG BASE_MODEL=Qwen/Qwen3-0.6B
ARG TUNED_MODEL=machalek29/qwen3-0.6b-state-lifetime-tutor-n500
ENV BASE_MODEL=${BASE_MODEL} TUNED_MODEL=${TUNED_MODEL}
RUN python -c "\
from huggingface_hub import snapshot_download; \
import os; \
[snapshot_download(r, allow_patterns=['*.json','*.safetensors','*.jinja','*.txt']) \
 for r in (os.environ['BASE_MODEL'], os.environ['TUNED_MODEL'])]"

COPY app.py .
COPY assets ./assets

EXPOSE 7860
CMD ["python", "app.py"]
