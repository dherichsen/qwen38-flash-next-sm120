from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from common import ConfigError, RuntimeConfig


class RuntimeConfigTests(unittest.TestCase):
    def base_env(self) -> dict[str, str]:
        return {
            "QWEN38_MODEL_PATH": "/srv/models/compatible-checkpoint",
            "QWEN38_KERNEL_CACHE": "/var/lib/qwen38-flash-next/kernel-cache",
        }

    def test_defaults_encode_accepted_single_gpu_profile(self) -> None:
        config = RuntimeConfig.from_env(self.base_env())
        argv = config.docker_argv()
        self.assertEqual(config.context_length, 131072)
        self.assertEqual(config.max_total_tokens, 131072)
        self.assertIn("device=0", argv)
        self.assertIn("fp8_e4m3", argv)
        self.assertIn("NEXTN", argv)
        self.assertIn("modelopt_fp4", argv)
        self.assertIn("127.0.0.1:11438:11437", argv)
        self.assertNotIn("--privileged", argv)
        self.assertIn("no-new-privileges", argv)

    def test_execution_is_one_gpu_only(self) -> None:
        env = self.base_env() | {"QWEN38_GPU_DEVICE": "0,1"}
        with self.assertRaisesRegex(ConfigError, "exactly one GPU"):
            RuntimeConfig.from_env(env)

    def test_paths_must_be_absolute_and_mount_safe(self) -> None:
        with self.assertRaisesRegex(ConfigError, "must be absolute"):
            RuntimeConfig.from_env(
                self.base_env() | {"QWEN38_MODEL_PATH": "relative/model"}
            )
        with self.assertRaisesRegex(ConfigError, "cannot contain a comma"):
            RuntimeConfig.from_env(
                self.base_env() | {"QWEN38_MODEL_PATH": "/srv/models/a,b"}
            )

    def test_token_pool_cannot_exceed_context(self) -> None:
        env = self.base_env() | {
            "QWEN38_CONTEXT_LENGTH": "65536",
            "QWEN38_MAX_TOTAL_TOKENS": "131072",
        }
        with self.assertRaisesRegex(ConfigError, "cannot exceed"):
            RuntimeConfig.from_env(env)

    def test_model_path_is_required(self) -> None:
        with self.assertRaisesRegex(ConfigError, "required"):
            RuntimeConfig.from_env({})


if __name__ == "__main__":
    unittest.main()
