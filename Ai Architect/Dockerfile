# CUDA-ready Dockerfile for Evo-Hateful-Memes
# Requires Docker + NVIDIA Container Toolkit
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

ENV TOKENIZERS_PARALLELISM=false

# Example usage:
# docker build -t evohm:latest .
# docker run --gpus all -it --rm -v $PWD:/workspace evohm:latest \
#   torchrun --nproc_per_node=2 scripts/train.py --config configs/ddp_2xa100.yaml
