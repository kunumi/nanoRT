# syntax=docker/dockerfile:1
#
# Reproducible H200 environment. It warms the two slow parts of the bootstrap
# so a run skips the large dependency downloads:
#   1. pixi package cache  (torch 2.6.0 + CUDA 12.4 wheels, rust, maturin, uv, ...)
#   2. cargo crate cache   (rustler dependencies)
# At run time `run.sh` does a fast `pixi install` (re-links the environment from
# the warm cache), builds the pre-fetched Rust sampler, installs the frozen HF
# artifacts, and starts training.
#
# The pixi env is self-contained (python 3.12, torch==2.6.0+cu124, rust, maturin,
# uv), so the base only supplies CUDA 12.4 + cuDNN system libs matching the
# cluster driver line. No base torch (unlike pytorch/pytorch) -> smaller image.
# If the cluster's NVIDIA driver line changes, bump this tag to match.
#
# Build & push (from repo root; requires x86_64 or buildx --platform):
#   IMAGE=<registry>/<ns>/nanort:cu124-torch2.6 ./docker_build.sh
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIXI_CACHE_DIR=/opt/pixi-cache \
    CARGO_HOME=/opt/cargo \
    PATH=/root/.pixi/bin:${PATH}

# System dependencies for pixi and the Rust extension.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates build-essential pkg-config git \
    && rm -rf /var/lib/apt/lists/*

# pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash

WORKDIR /app

# --- Warm the pixi + cargo caches ------------------------------------------
# Copy only manifests + locks so this layer stays cached unless deps change.
# A stub rt/__init__.py lets the editable self-dependency
# (`rt = { path = ".", editable = true }`) resolve during the build. The heavy
# downloads land in PIXI_CACHE_DIR / CARGO_HOME.
COPY pyproject.toml pixi.lock ./
COPY rustler/Cargo.toml rustler/Cargo.lock ./rustler/
# Stubs so the manifests parse during the build: rt/__init__.py for the editable
# self-dep, and rustler/src/lib.rs because Cargo.toml declares [lib] rustler ->
# src/lib.rs (cargo fetch parses targets even though it only downloads deps).
# Mount the repository at run time to supply the training sources.
RUN mkdir -p rt && : > rt/__init__.py \
 && mkdir -p rustler/src && : > rustler/src/lib.rs \
 && pixi install --locked \
 && (cd rustler && pixi run --manifest-path /app/pyproject.toml cargo fetch --locked)

# --- Build-time sanity check (fail fast on a broken image) ------------------
# Verify the env resolves and the GPU/build stack imports before the image
# ships. (No GPU at build time, so we only import + read versions, never
# torch.cuda.is_available().)
RUN pixi run --manifest-path /app/pyproject.toml python -c \
      "import torch, sklearn, sentence_transformers, einops, polars; \
print('torch', torch.__version__, '| cuda build', torch.version.cuda)" \
 && pixi run --manifest-path /app/pyproject.toml maturin --version \
 && pixi run --manifest-path /app/pyproject.toml cargo --version

# Drop the path-bound env; keep only the warm caches. The run recreates .pixi
# from PIXI_CACHE_DIR (no download) in its own workdir, with the editable
# install pointing at that workdir instead of /app.
RUN rm -rf /app/.pixi /app/rt
