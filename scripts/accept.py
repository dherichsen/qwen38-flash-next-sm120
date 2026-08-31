#!/usr/bin/env python3
"""Run reproducible public acceptance gates and write a redacted receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from api_checks import (
    CheckError,
    check_exact_prefill,
    choose_probe_token,
    public_ready_checks,
)


FATAL_LOG_MARKERS = (
    "mlirerror",
    "cuda error",
    "out of memory",
    "traceback (most recent call last)",
    "segmentation fault",
    "illegal memory",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("QWEN38_BASE_URL", "http://127.0.0.1:11438"),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "QWEN38_SERVED_MODEL_NAME", "qwen3.8-flash-next-sglang"
        ),
    )
    parser.add_argument(
        "--container-name",
        default=os.environ.get("QWEN38_CONTAINER_NAME", "qwen38-flash-next"),
    )
    parser.add_argument("--health-timeout", type=int, default=1800)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--skip-long-context", action="store_true")
    parser.add_argument("--skip-log-check", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def check_container_logs(name: str) -> dict[str, Any]:
    try:
        process = subprocess.run(
            ["docker", "logs", name],
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CheckError(f"cannot read container logs: {exc}") from exc
    output = process.stdout + process.stderr
    if process.returncode != 0:
        raise CheckError(f"docker logs failed for {name!r}")
    lowered = output.lower()
    hits = [marker for marker in FATAL_LOG_MARKERS if marker in lowered]
    if hits:
        raise CheckError(f"fatal markers in container logs: {hits}")
    return {
        "status": "pass",
        "bytes": len(output.encode("utf-8")),
        "sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "fatal_marker_count": 0,
    }


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("QWEN38_API_KEY", "")
    receipt: dict[str, Any] = {
        "schema": "qwen38-sm120-public-acceptance-v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "private_grader_used": False,
    }
    try:
        checks = public_ready_checks(
            args.base_url,
            args.model,
            health_timeout=args.health_timeout,
            request_timeout=args.request_timeout,
            api_key=api_key,
        )
        if args.skip_long_context:
            checks["long_context"] = {"status": "skipped_by_operator"}
        else:
            token_id = choose_probe_token(
                args.base_url,
                args.model,
                timeout=args.request_timeout,
                api_key=api_key,
            )
            checks["long_context"] = [
                check_exact_prefill(
                    args.base_url,
                    args.model,
                    count,
                    timeout=args.request_timeout,
                    api_key=api_key,
                    token_id=token_id,
                )
                for count in (65536, 120000)
            ]
        if args.skip_log_check:
            checks["container_logs"] = {"status": "skipped_by_operator"}
        else:
            checks["container_logs"] = check_container_logs(args.container_name)
        receipt.update(
            status="pass",
            checks=checks,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        if args.output:
            atomic_write(args.output, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    except CheckError as exc:
        receipt.update(
            status="fail",
            failure=str(exc),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        if args.output:
            atomic_write(args.output, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True), file=sys.stderr)
        return 42


if __name__ == "__main__":
    raise SystemExit(main())
