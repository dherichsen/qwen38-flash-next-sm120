#!/usr/bin/env bash
# Build the pinned runtime image after composing and verifying its source tree.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
# shellcheck disable=SC1091
. "$repo_root/config/pins.env"

archive="$repo_root/build/sglang-source.tar.gz"
if [[ ! -f "$archive" ]]; then
  "$script_dir/compose_sglang_source.sh" "$archive"
fi

archive_sha=$(sha256sum "$archive" | awk '{print $1}')
image_tag=${QWEN38_BUILD_TAG:-qwen38-flash-next-sm120:2026-08-31}

DOCKER_BUILDKIT=1 docker build \
  --platform linux/amd64 \
  --network=none \
  --pull=false \
  --file "$repo_root/docker/Dockerfile" \
  --build-arg "SGLANG_BASE_IMAGE=$SGLANG_BASE_IMAGE" \
  --build-arg "SOURCE_ARCHIVE_SHA256=$archive_sha" \
  --tag "$image_tag" \
  "$repo_root"

docker image inspect \
  --format 'image_id={{.Id}} source_tree={{index .Config.Labels "ai.qwen38.sglang.source-tree"}}' \
  "$image_tag"
