from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import api_checks


class EmptyHealthResponse:
    status = 200

    def __enter__(self) -> "EmptyHealthResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return b""


class ApiCheckTests(unittest.TestCase):
    @mock.patch("urllib.request.urlopen", return_value=EmptyHealthResponse())
    def test_empty_200_health_body_is_success(self, _urlopen: mock.Mock) -> None:
        self.assertEqual(
            api_checks.http_status("http://127.0.0.1:11438", "/health_generate"),
            200,
        )

    def server_info(self) -> dict[str, object]:
        return {
            "context_length": 131072,
            "max_total_num_tokens": 131072,
            "kv_cache_dtype": "fp8_e4m3",
            "quantization": "modelopt_fp4",
            "speculative_algorithm": "EAGLE",
            "speculative_num_steps": 3,
            "speculative_eagle_topk": 1,
            "speculative_num_draft_tokens": 4,
            "mem_fraction_static": 0.96,
            "page_size": 64,
            "chunked_prefill_size": 4096,
            "max_running_requests": 1,
            "served_model_name": "qwen3.8-flash-next-sglang",
            "default_chat_template_kwargs": {
                "enable_thinking": True,
                "preserve_thinking": True,
                "reasoning_effort": "medium",
            },
            "internal_states": [
                {
                    "avg_spec_accept_length": 3.4,
                    "memory_usage": {"token_capacity": 131072},
                }
            ],
        }

    @mock.patch("api_checks.http_json")
    def test_server_info_requires_runtime_and_active_nextn(
        self, http_json: mock.Mock
    ) -> None:
        http_json.return_value = self.server_info()
        result = api_checks.check_server_info(
            "http://127.0.0.1:11438",
            "qwen3.8-flash-next-sglang",
            timeout=30,
        )
        self.assertEqual(result["token_capacities"], [131072])
        self.assertEqual(result["active_spec_accept_lengths"], [3.4])

        broken = self.server_info()
        broken["internal_states"] = [
            {
                "avg_spec_accept_length": 1.0,
                "memory_usage": {"token_capacity": 131072},
            }
        ]
        http_json.return_value = broken
        with self.assertRaisesRegex(api_checks.CheckError, "configured"):
            api_checks.check_server_info(
                "http://127.0.0.1:11438",
                "qwen3.8-flash-next-sglang",
                timeout=30,
            )

    @mock.patch("api_checks.http_json")
    def test_exact_prefill_requires_exact_usage(self, http_json: mock.Mock) -> None:
        http_json.return_value = {
            "choices": [{"finish_reason": "length", "text": "ok"}],
            "usage": {"prompt_tokens": 65536, "completion_tokens": 32},
        }
        result = api_checks.check_exact_prefill(
            "http://127.0.0.1:11438",
            "qwen3.8-flash-next-sglang",
            65536,
            timeout=30,
            token_id=42,
        )
        self.assertEqual(result["prompt_tokens"], 65536)
        payload = http_json.call_args.kwargs["payload"]
        self.assertEqual(len(payload["prompt"]), 65536)


if __name__ == "__main__":
    unittest.main()
