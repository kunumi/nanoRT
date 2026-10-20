#!/usr/bin/env bash
# Build the reproducible linux/amd64 nanoRT image.
set -euo pipefail

IMAGE="${IMAGE:-modded-nanort:cu124-torch2.6}"
PUSH="${PUSH:-0}"

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: no reachable Docker daemon." >&2
  exit 1
fi

arch="$(uname -m)"
args=(--platform linux/amd64 -t "${IMAGE}")
if [[ "${PUSH}" == "1" ]]; then
  args+=(--push)
else
  args+=(--load)
fi

if docker buildx version >/dev/null 2>&1; then
  docker buildx build "${args[@]}" .
elif [[ "${arch}" == "x86_64" || "${arch}" == "amd64" ]]; then
  docker build -t "${IMAGE}" .
  if [[ "${PUSH}" == "1" ]]; then
    docker push "${IMAGE}"
  fi
else
  echo "ERROR: buildx is required to target linux/amd64 from ${arch}." >&2
  exit 1
fi

echo "Built: ${IMAGE}"
