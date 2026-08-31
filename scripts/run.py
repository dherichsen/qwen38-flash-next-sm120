#!/usr/bin/env python3
"""Print or explicitly execute the configured Docker launch."""

from __future__ import annotations

import argparse
import json
import os
import sys

from common import ConfigError, RuntimeConfig, shell_join


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--print-shell", action="store_true")
    output.add_argument("--print-config", action="store_true")
    output.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = RuntimeConfig.from_env()
        argv = config.docker_argv()
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    if args.print_shell:
        print(shell_join(argv))
    elif args.print_config:
        print(json.dumps(config.public_dict(), indent=2, sort_keys=True))
    elif args.execute:
        os.execvp(argv[0], argv)
    else:
        print(json.dumps(argv, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
