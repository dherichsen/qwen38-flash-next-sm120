#!/usr/bin/env python3
"""Fail-closed host, image, model, GPU, cache, and port checks."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import ConfigError, EXPECTED_SOURCE_TREE, RuntimeConfig


class PreflightError(RuntimeError):
    """A required launch condition is not satisfied."""


def run(argv: list[str], timeout: int = 60) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv, text=True, capture_output=True, check=False, timeout=timeout
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PreflightError(f"cannot execute {argv[0]}: {exc}") from exc


def check_model(config: RuntimeConfig) -> dict[str, Any]:
    path = config.model_path
    if not path.is_dir():
        raise PreflightError(f"model directory is missing: {path}")
    required = [path / "config.json"]
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise PreflightError(f"model metadata is missing: {missing}")
    indexes = sorted(path.glob("*.safetensors.index.json"))
    shards = sorted(path.glob("*.safetensors"))
    if not indexes and not shards:
        raise PreflightError("model directory has no safetensors index or shards")
    return {
        "path": str(path),
        "config": str(required[0]),
        "index_files": len(indexes),
        "root_shards": len(shards),
    }


def check_image(config: RuntimeConfig) -> dict[str, Any]:
    process = run(["docker", "image", "inspect", config.image])
    if process.returncode != 0:
        raise PreflightError(f"Docker image is unavailable: {config.image}")
    try:
        image = json.loads(process.stdout)[0]
    except (json.JSONDecodeError, IndexError, TypeError) as exc:
        raise PreflightError("Docker returned malformed image metadata") from exc
    image_id = image.get("Id")
    if config.image_id and image_id != config.image_id:
        raise PreflightError(
            f"image ID mismatch: expected={config.image_id} actual={image_id}"
        )
    labels = (image.get("Config") or {}).get("Labels") or {}
    source_tree = labels.get("ai.qwen38.sglang.source-tree")
    if source_tree != EXPECTED_SOURCE_TREE:
        raise PreflightError(
            f"source-tree label mismatch: expected={EXPECTED_SOURCE_TREE} "
            f"actual={source_tree}"
        )
    return {"reference": config.image, "id": image_id, "source_tree": source_tree}


def check_gpu(config: RuntimeConfig) -> dict[str, Any]:
    fields = "uuid,name,memory.total,compute_cap"
    process = run(
        [
            "nvidia-smi",
            "-i",
            config.gpu_device,
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ]
    )
    if process.returncode != 0:
        raise PreflightError(f"GPU query failed: {process.stderr.strip()[-500:]}")
    parts = [part.strip() for part in process.stdout.strip().split(",")]
    if len(parts) != 4:
        raise PreflightError(f"unexpected GPU query result: {process.stdout!r}")
    uuid, name, memory_text, compute_cap = parts
    try:
        memory_mib = int(memory_text)
    except ValueError as exc:
        raise PreflightError(f"invalid GPU memory value: {memory_text!r}") from exc
    if config.require_sm120 and compute_cap != "12.0":
        raise PreflightError(f"expected SM120 compute capability 12.0, got {compute_cap}")
    if memory_mib < config.min_gpu_memory_mib:
        raise PreflightError(
            f"GPU memory {memory_mib} MiB is below required "
            f"{config.min_gpu_memory_mib} MiB"
        )
    apps = run(
        [
            "nvidia-smi",
            "-i",
            config.gpu_device,
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if apps.returncode != 0:
        raise PreflightError("cannot enumerate GPU compute processes")
    active_apps = [line.strip() for line in apps.stdout.splitlines() if line.strip()]
    if config.require_idle_gpu and active_apps:
        raise PreflightError(f"selected GPU has active compute processes: {active_apps}")
    return {
        "uuid": uuid,
        "name": name,
        "memory_mib": memory_mib,
        "compute_capability": compute_cap,
        "active_compute_processes": active_apps,
    }


def check_cache(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise PreflightError(
            f"kernel-cache directory is missing: {path}; create it before launch"
        )
    if not os.access(path, os.R_OK | os.W_OK | os.X_OK):
        raise PreflightError(f"kernel-cache directory is not writable: {path}")
    return {"path": str(path), "read_write": True}


def check_port(config: RuntimeConfig) -> dict[str, Any]:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((config.bind_host, config.host_port))
    except OSError as exc:
        raise PreflightError(
            f"cannot bind {config.bind_host}:{config.host_port}: {exc}"
        ) from exc
    finally:
        probe.close()
    return {"host": config.bind_host, "port": config.host_port, "available": True}


def main() -> int:
    receipt: dict[str, Any] = {"schema": "qwen38-sm120-preflight-v1"}
    try:
        config = RuntimeConfig.from_env()
        receipt.update(
            status="pass",
            config=config.public_dict(),
            model=check_model(config),
            image=check_image(config),
            gpu=check_gpu(config),
            cache=check_cache(config.kernel_cache),
            port=check_port(config),
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    except (ConfigError, PreflightError) as exc:
        receipt.update(status="fail", failure=str(exc))
        print(json.dumps(receipt, indent=2, sort_keys=True), file=sys.stderr)
        return 42


if __name__ == "__main__":
    raise SystemExit(main())
