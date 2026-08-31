#!/usr/bin/env python3
"""Wait for the server and run public model, reasoning, and tool gates."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

from api_checks import CheckError, public_ready_checks


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
        "--health-timeout",
        type=int,
        default=int(os.environ.get("QWEN38_READY_TIMEOUT", "1800")),
    )
    parser.add_argument("--request-timeout", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt: dict[str, object] = {
        "schema": "qwen38-sm120-ready-v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
    }
    try:
        receipt["checks"] = public_ready_checks(
            args.base_url,
            args.model,
            health_timeout=args.health_timeout,
            request_timeout=args.request_timeout,
            api_key=os.environ.get("QWEN38_API_KEY", ""),
        )
        receipt["status"] = "pass"
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    except CheckError as exc:
        receipt.update(
            status="fail",
            failure=str(exc),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        print(json.dumps(receipt, indent=2, sort_keys=True), file=sys.stderr)
        return 42


if __name__ == "__main__":
    raise SystemExit(main())
