#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
base_image=${QWEN38_BASE_IMAGE:-qwen38-flash-next-sm120:2026-08-31}
patch_sha=$(sha256sum "$repo_root/patches/0003-nvidia-mixed-host-offload.patch" | awk '{print $1}')
docker build --platform linux/amd64 --network=none --pull=false \
 --file "$repo_root/docker/Dockerfile.nvidia" \
 --build-arg "QWEN38_BASE_IMAGE=$base_image" --build-arg "NVIDIA_PATCH_SHA256=$patch_sha" \
 --tag qwen38-flash-next-sm120:nvidia-200k "$repo_root"
