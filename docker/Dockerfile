# syntax=docker/dockerfile:1.7

ARG SGLANG_BASE_IMAGE=lmsysorg/sglang@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1
FROM ${SGLANG_BASE_IMAGE}

ARG SOURCE_ARCHIVE=build/sglang-source.tar.gz
ARG SOURCE_ARCHIVE_SHA256
ARG SOURCE_TREE=7848e2094b30690a40a47a56f1e8f4e68a9929d3

WORKDIR /sgl-workspace/sglang

# The base image uses an editable install rooted here. Replace only the pinned
# Python and test source while preserving generated version and egg metadata.
RUN --mount=type=bind,source=${SOURCE_ARCHIVE},target=/tmp/sglang-source.tar.gz,readonly \
    set -eux; \
    test -n "${SOURCE_ARCHIVE_SHA256}"; \
    echo "${SOURCE_ARCHIVE_SHA256}  /tmp/sglang-source.tar.gz" | sha256sum --check --strict; \
    mkdir -p /tmp/sglang-preserve /tmp/sglang-unpack; \
    cp python/sglang/_version.py /tmp/sglang-preserve/_version.py; \
    cp -a python/sglang.egg-info /tmp/sglang-preserve/sglang.egg-info; \
    tar -xzf /tmp/sglang-source.tar.gz -C /tmp/sglang-unpack; \
    test -d /tmp/sglang-unpack/sglang/python/sglang; \
    test -d /tmp/sglang-unpack/sglang/test; \
    rm -rf /sgl-workspace/sglang/python/sglang \
           /sgl-workspace/sglang/python/sglang.egg-info \
           /sgl-workspace/sglang/test; \
    rm -f /sgl-workspace/sglang/python/pyproject.toml \
          /sgl-workspace/sglang/python/pyproject_cpu.toml \
          /sgl-workspace/sglang/python/pyproject_xpu.toml \
          /sgl-workspace/sglang/python/pyproject_other.toml; \
    cp -a /tmp/sglang-unpack/sglang/python/. /sgl-workspace/sglang/python/; \
    cp -a /tmp/sglang-unpack/sglang/test /sgl-workspace/sglang/test; \
    cp /tmp/sglang-preserve/_version.py /sgl-workspace/sglang/python/sglang/_version.py; \
    cp -a /tmp/sglang-preserve/sglang.egg-info /sgl-workspace/sglang/python/sglang.egg-info; \
    find /sgl-workspace/sglang/python -name __pycache__ -type d -prune -exec rm -rf '{}' +; \
    python3 -c "import sglang, sglang.srt.models.qwen4_exp; print(sglang.__file__)"; \
    rm -rf /tmp/sglang-preserve /tmp/sglang-unpack

LABEL org.opencontainers.image.source="https://github.com/dherichsen/qwen38-flash-next-sm120" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.description="Pinned SGLang runtime for Qwen3.8 Flash-Next on SM120" \
      ai.qwen38.sglang.base-commit="99c9362e6685db579c469f6e0e566b08827b3477" \
      ai.qwen38.sglang.source-tree="${SOURCE_TREE}" \
      ai.qwen38.sglang.patch-gc="a1ee15f2a7a0ad6d804d9fc48c1560d7e50f0bdc" \
      ai.qwen38.sglang.patch-fp8-kv="4318970cabd6ff1109eae352c5a2d9f6bbffff19"
