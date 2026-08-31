#!/usr/bin/env bash
# Compose the exact tested SGLang tree from public source and vendored patches.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)

# shellcheck disable=SC1091
. "$repo_root/config/pins.env"

output=${1:-"$repo_root/build/sglang-source.tar.gz"}
output_dir=$(dirname -- "$output")
mkdir -p -- "$output_dir"

for command_name in git gzip sha256sum tar; do
  command -v "$command_name" >/dev/null || {
    echo "missing required command: $command_name" >&2
    exit 2
  }
done

patch_1="$repo_root/patches/0001-kv-budget-gc-before-profile.patch"
patch_2="$repo_root/patches/0002-qsa-fp8-kv-cache.patch"
echo "$PATCH_1_SHA256  $patch_1" | sha256sum --check --strict
echo "$PATCH_2_SHA256  $patch_2" | sha256sum --check --strict

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/qwen38-source.XXXXXX")
cleanup() {
  if [[ -n "${work_dir:-}" && -d "$work_dir" ]]; then
    find "$work_dir" -depth -delete
  fi
}
trap cleanup EXIT

source_dir="$work_dir/sglang"
git init -q "$source_dir"
git -C "$source_dir" remote add origin "$SGLANG_REPOSITORY"
git -C "$source_dir" fetch --depth=1 --filter=blob:none origin "$SGLANG_BASE_COMMIT"
git -C "$source_dir" checkout -q --detach FETCH_HEAD

actual_base=$(git -C "$source_dir" rev-parse 'HEAD^{tree}')
if [[ "$actual_base" != "$SGLANG_BASE_TREE" ]]; then
  echo "base tree mismatch: expected=$SGLANG_BASE_TREE actual=$actual_base" >&2
  exit 3
fi

git -C "$source_dir" config user.name "Qwen38 source composer"
git -C "$source_dir" config user.email "noreply@example.invalid"
git -C "$source_dir" am --quiet "$patch_1" "$patch_2"

actual_tree=$(git -C "$source_dir" rev-parse 'HEAD^{tree}')
if [[ "$actual_tree" != "$SGLANG_EXPECTED_TREE" ]]; then
  echo "composed tree mismatch: expected=$SGLANG_EXPECTED_TREE actual=$actual_tree" >&2
  exit 4
fi
if [[ -n "$(git -C "$source_dir" status --porcelain)" ]]; then
  echo "composed source is unexpectedly dirty" >&2
  exit 5
fi

archive_tmp="$work_dir/sglang-source.tar.gz"
tar \
  --sort=name \
  --mtime='@0' \
  --owner=0 \
  --group=0 \
  --numeric-owner \
  --transform='s#^#sglang/#' \
  -C "$source_dir" \
  -cf - python test | gzip -n >"$archive_tmp"

mv -- "$archive_tmp" "$output"
actual_archive_sha=$(sha256sum "$output" | awk '{print $1}')
if [[ "$actual_archive_sha" != "$SGLANG_SOURCE_ARCHIVE_SHA256" ]]; then
  echo "source archive mismatch: expected=$SGLANG_SOURCE_ARCHIVE_SHA256 " \
       "actual=$actual_archive_sha" >&2
  exit 6
fi
sha256sum "$output" >"$output.sha256"
printf 'source_tree=%s\narchive=%s\n' "$actual_tree" "$output"
cat "$output.sha256"
