#!/usr/bin/env python3
"""Shared, dependency-free runtime configuration and Docker argv builder."""

from __future__ import annotations

import ipaddress
import json
import os
import re
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping


EXPECTED_SOURCE_TREE = "7848e2094b30690a40a47a56f1e8f4e68a9929d3"
CONTAINER_MODEL_PATH = "/model"
CONTAINER_CACHE_PATH = "/root/.cache/flashinfer"


class ConfigError(ValueError):
    """Raised when a runtime setting is missing or unsafe."""


def _text(env: Mapping[str, str], name: str, default: str = "") -> str:
    value = env.get(name, default).strip()
    if "\x00" in value or "\n" in value or "\r" in value:
        raise ConfigError(f"{name} contains a control character")
    return value


def _integer(
    env: Mapping[str, str], name: str, default: int, minimum: int, maximum: int
) -> int:
    raw = _text(env, name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ConfigError(f"{name} must be between {minimum} and {maximum}")
    return value


def _floating(
    env: Mapping[str, str], name: str, default: float, minimum: float, maximum: float
) -> float:
    raw = _text(env, name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be numeric") from exc
    if not minimum <= value <= maximum:
        raise ConfigError(f"{name} must be between {minimum} and {maximum}")
    return value


def env_flag(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = _text(env, name, "1" if default else "0").lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be a boolean")


@dataclass(frozen=True)
class RuntimeConfig:
    profile: str
    model_path: Path
    image: str
    image_id: str
    container_name: str
    served_model_name: str
    bind_host: str
    host_port: int
    container_port: int
    gpu_device: str
    cpu_set: str
    kernel_cache: Path
    context_length: int
    max_total_tokens: int
    mem_fraction_static: float
    require_sm120: bool
    require_idle_gpu: bool
    min_gpu_memory_mib: int

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "RuntimeConfig":
        values = os.environ if env is None else env
        model_raw = _text(values, "QWEN38_MODEL_PATH")
        if not model_raw:
            raise ConfigError("QWEN38_MODEL_PATH is required")

        config = cls(
            profile=_text(values, "QWEN38_PROFILE", "legacy-131k"),
            model_path=Path(model_raw),
            image=_text(
                values,
                "QWEN38_IMAGE",
                "qwen38-flash-next-sm120:2026-08-31",
            ),
            image_id=_text(values, "QWEN38_IMAGE_ID"),
            container_name=_text(
                values, "QWEN38_CONTAINER_NAME", "qwen38-flash-next"
            ),
            served_model_name=_text(
                values,
                "QWEN38_SERVED_MODEL_NAME",
                "qwen3.8-flash-next-sglang",
            ),
            bind_host=_text(values, "QWEN38_BIND_HOST", "127.0.0.1"),
            host_port=_integer(values, "QWEN38_HOST_PORT", 11438, 1, 65535),
            container_port=_integer(
                values, "QWEN38_CONTAINER_PORT", 11437, 1, 65535
            ),
            gpu_device=_text(values, "QWEN38_GPU_DEVICE", "0"),
            cpu_set=_text(values, "QWEN38_CPU_SET", "0-31"),
            kernel_cache=Path(
                _text(
                    values,
                    "QWEN38_KERNEL_CACHE",
                    "/var/lib/qwen38-flash-next/kernel-cache",
                )
            ),
            context_length=_integer(
                values, "QWEN38_CONTEXT_LENGTH", 131072, 1024, 1048576
            ),
            max_total_tokens=_integer(
                values, "QWEN38_MAX_TOTAL_TOKENS", 131072, 1024, 1048576
            ),
            mem_fraction_static=_floating(
                values, "QWEN38_MEM_FRACTION_STATIC", 0.96, 0.1, 1.0
            ),
            require_sm120=env_flag(values, "QWEN38_REQUIRE_SM120", True),
            require_idle_gpu=env_flag(values, "QWEN38_REQUIRE_IDLE_GPU", True),
            min_gpu_memory_mib=_integer(
                values, "QWEN38_MIN_GPU_MEMORY_MIB", 90000, 0, 1048576
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.profile not in {"legacy-131k", "nvidia-200k"}:
            raise ConfigError("unknown QWEN38_PROFILE")
        if not self.model_path.is_absolute():
            raise ConfigError("QWEN38_MODEL_PATH must be absolute")
        if not self.kernel_cache.is_absolute():
            raise ConfigError("QWEN38_KERNEL_CACHE must be absolute")
        for name, path in (
            ("QWEN38_MODEL_PATH", self.model_path),
            ("QWEN38_KERNEL_CACHE", self.kernel_cache),
        ):
            if "," in str(path):
                raise ConfigError(f"{name} cannot contain a comma")
        if not self.image or any(character.isspace() for character in self.image):
            raise ConfigError("QWEN38_IMAGE must be one non-empty image reference")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.container_name):
            raise ConfigError("QWEN38_CONTAINER_NAME has unsafe characters")
        if not self.served_model_name or any(
            character.isspace() for character in self.served_model_name
        ):
            raise ConfigError("QWEN38_SERVED_MODEL_NAME must not contain whitespace")
        try:
            address = ipaddress.ip_address(self.bind_host)
        except ValueError as exc:
            raise ConfigError("QWEN38_BIND_HOST must be an IP address") from exc
        if address.version != 4:
            raise ConfigError("the public template currently supports IPv4 binding")
        if not re.fullmatch(r"(?:[0-9]+|GPU-[A-Fa-f0-9-]+)", self.gpu_device):
            raise ConfigError("QWEN38_GPU_DEVICE must select exactly one GPU")
        if not re.fullmatch(r"[0-9,-]+", self.cpu_set):
            raise ConfigError("QWEN38_CPU_SET must be a Docker CPU-set expression")
        if self.max_total_tokens > self.context_length:
            raise ConfigError(
                "QWEN38_MAX_TOTAL_TOKENS cannot exceed QWEN38_CONTEXT_LENGTH"
            )

    @property
    def base_url(self) -> str:
        return f"http://{self.bind_host}:{self.host_port}"

    def public_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["model_path"] = str(self.model_path)
        value["kernel_cache"] = str(self.kernel_cache)
        value["base_url"] = self.base_url
        return value

    def docker_argv(self) -> list[str]:
        server_defaults = json.dumps(
            {
                "enable_thinking": True,
                "preserve_thinking": True,
                "reasoning_effort": "medium",
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        return [
            "docker",
            "run",
            "--rm",
            "--name",
            self.container_name,
            "--restart",
            "no",
            "--pull",
            "never",
            "--cpuset-cpus",
            self.cpu_set,
            "--gpus",
            f"device={self.gpu_device}",
            "--ipc",
            "host",
            "--ulimit",
            "memlock=-1:-1",
            "--ulimit",
            "stack=67108864:67108864",
            "--cap-drop",
            "ALL",
            "--cap-add",
            "IPC_LOCK",
            "--security-opt",
            "no-new-privileges",
            "--publish",
            f"{self.bind_host}:{self.host_port}:{self.container_port}",
            "--env",
            "HF_HUB_OFFLINE=1",
            "--env",
            "TRANSFORMERS_OFFLINE=1",
            "--env",
            "HF_HOME=/root/.cache/huggingface",
            "--env",
            "TOKENIZERS_PARALLELISM=false",
            "--env",
            "PYTHONUNBUFFERED=1",
            "--mount",
            f"type=bind,src={self.model_path},dst={CONTAINER_MODEL_PATH},readonly",
            "--mount",
            f"type=bind,src={self.kernel_cache},dst={CONTAINER_CACHE_PATH}",
            "--log-driver",
            "local",
            "--log-opt",
            "max-size=200m",
            "--log-opt",
            "max-file=5",
            self.image,
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            CONTAINER_MODEL_PATH,
            "--served-model-name",
            self.served_model_name,
            "--tp",
            "1",
            "--trust-remote-code",
            "--quantization",
            "modelopt_mixed" if self.profile == "nvidia-200k" else "modelopt_fp4",
            "--fp4-gemm-backend",
            "flashinfer_cutlass",
            "--context-length",
            str(self.context_length),
            "--max-total-tokens",
            str(self.max_total_tokens),
            "--mem-fraction-static",
            str(self.mem_fraction_static),
            "--page-size",
            "64",
            "--chunked-prefill-size",
            "4096",
            "--ple-offload-embedding",
            "--linear-attn-prefill-backend",
            "triton",
            "--linear-attn-decode-backend",
            "flashinfer",
            "--linear-attn-verify-backend",
            "triton",
            "--mamba-ssm-dtype",
            "bfloat16",
            "--mamba-radix-cache-strategy",
            "extra_buffer",
            "--mamba-track-interval",
            "64",
            "--max-mamba-cache-size",
            "16",
            "--max-running-requests",
            "1",
            "--cuda-graph-max-bs-decode",
            "1",
            "--reasoning-parser",
            "auto",
            "--tool-call-parser",
            "auto",
            "--default-chat-template-kwargs",
            server_defaults,
            "--sampling-defaults",
            "model",
            "--speculative-algorithm",
            "NEXTN",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--host",
            "0.0.0.0",
            "--port",
            str(self.container_port),
        ] + (["--moe-runner-backend", "flashinfer_cutlass", "--offload-embedding-to-host"] if self.profile == "nvidia-200k" else [])


def shell_join(argv: list[str]) -> str:
    return shlex.join(argv)
