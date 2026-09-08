FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime@sha256:1e26efd426b0fecbfe7cf3d3ae5003fada6ac5a76eddc1e042857f5d049605ee

LABEL org.opencontainers.image.title="SAM_CLIP"
LABEL org.opencontainers.image.description="Reproducible NVIDIA GPU environment for SAM_CLIP training, inference, and evaluation"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    SAM_CLIP_HOME=/opt/sam_clip

WORKDIR ${SAM_CLIP_HOME}

COPY requirements.txt constraints.txt ./
RUN python -m pip install --upgrade "pip==23.3.2" \
    && python -m pip install --constraint constraints.txt --requirement requirements.txt

COPY app/ ./
COPY scripts/gpu_check.py ./tools/gpu_check.py
COPY docker/entrypoint.sh /usr/local/bin/sam-clip

RUN chmod 0755 /usr/local/bin/sam-clip \
    && mkdir -p /workspace/data /workspace/weights /workspace/checkpoints /workspace/outputs \
    && python -m compileall -q ${SAM_CLIP_HOME}

WORKDIR /workspace
ENTRYPOINT ["sam-clip"]
CMD ["help"]
