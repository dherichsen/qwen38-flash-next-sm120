#!/usr/bin/env python3
"""Public, dependency-free SGLang API checks used by ready and acceptance."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from typing import Any


class CheckError(RuntimeError):
    """An HTTP or response-contract check failed."""


def _url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def http_json(
    base_url: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: int = 60,
    api_key: str = "",
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(_url(base_url, path), data=body)
    if body is not None:
        request.add_header("Content-Type", "application/json")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            if response.status != 200:
                raise CheckError(f"HTTP {response.status} from {path}")
    except urllib.error.HTTPError as exc:
        detail = exc.read()[-1000:].decode("utf-8", errors="replace")
        raise CheckError(f"HTTP {exc.code} from {path}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise CheckError(f"request failed for {path}: {exc}") from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CheckError(f"non-JSON response from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CheckError(f"JSON response from {path} is not an object")
    return value


def http_status(
    base_url: str, path: str, *, timeout: int = 60, api_key: str = ""
) -> int:
    request = urllib.request.Request(_url(base_url, path))
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        raise CheckError(f"HTTP {exc.code} from {path}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise CheckError(f"request failed for {path}: {exc}") from exc


def wait_for_health(
    base_url: str, *, timeout: int, api_key: str = "", interval: float = 2.0
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    attempts = 0
    last_error = "not attempted"
    while time.monotonic() < deadline:
        attempts += 1
        try:
            status = http_status(
                base_url, "/health_generate", timeout=30, api_key=api_key
            )
            if status != 200:
                raise CheckError(f"HTTP {status} from /health_generate")
            return {"status": "pass", "attempts": attempts}
        except CheckError as exc:
            last_error = str(exc)
            time.sleep(interval)
    raise CheckError(f"health timeout after {timeout}s: {last_error}")


def check_models(
    base_url: str, model: str, *, timeout: int, api_key: str = ""
) -> dict[str, Any]:
    response = http_json(
        base_url, "/v1/models", timeout=min(timeout, 120), api_key=api_key
    )
    data = response.get("data")
    if not isinstance(data, list):
        raise CheckError("/v1/models has no data list")
    model_ids = [item.get("id") for item in data if isinstance(item, dict)]
    if model not in model_ids:
        raise CheckError(f"served model {model!r} not found in {model_ids!r}")
    return {"status": "pass", "served_model": model, "model_count": len(model_ids)}


def check_server_info(
    base_url: str, model: str, *, timeout: int, api_key: str = ""
) -> dict[str, Any]:
    response = http_json(
        base_url, "/get_server_info", timeout=min(timeout, 120), api_key=api_key
    )
    profile = os.environ.get("QWEN38_PROFILE", "legacy-131k")
    if profile not in {"legacy-131k", "nvidia-200k"}:
        raise CheckError("unknown QWEN38_PROFILE")
    nvidia = profile == "nvidia-200k"
    context = int(os.environ.get("QWEN38_CONTEXT_LENGTH", "200000" if nvidia else "131072"))
    capacity = int(os.environ.get("QWEN38_MAX_TOTAL_TOKENS", str(context)))
    expected: dict[str, Any] = {
        "context_length": context,
        "max_total_num_tokens": capacity,
        "kv_cache_dtype": "fp8_e4m3",
        "quantization": "modelopt_mixed" if nvidia else "modelopt_fp4",
        "speculative_algorithm": "EAGLE",
        "speculative_num_steps": 3,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 4,
        "mem_fraction_static": 0.96,
        "page_size": 64,
        "chunked_prefill_size": 4096,
        "max_running_requests": 1,
        "served_model_name": model,
    }
    if nvidia:
        expected.update(offload_embedding_to_host=True, moe_runner_backend="flashinfer_cutlass")
    mismatches: dict[str, dict[str, Any]] = {}
    for field, wanted in expected.items():
        actual = response.get(field)
        if isinstance(wanted, float) and isinstance(actual, (int, float)):
            equal = abs(float(actual) - wanted) < 1e-9
        else:
            equal = actual == wanted
        if not equal:
            mismatches[field] = {"expected": wanted, "actual": actual}
    template_kwargs = response.get("default_chat_template_kwargs")
    expected_kwargs = {
        "enable_thinking": True,
        "preserve_thinking": True,
        "reasoning_effort": "medium",
    }
    if not isinstance(template_kwargs, dict):
        mismatches["default_chat_template_kwargs"] = {
            "expected": expected_kwargs,
            "actual": template_kwargs,
        }
    else:
        for key, wanted in expected_kwargs.items():
            if template_kwargs.get(key) != wanted:
                mismatches[f"default_chat_template_kwargs.{key}"] = {
                    "expected": wanted,
                    "actual": template_kwargs.get(key),
                }
    if mismatches:
        raise CheckError(f"server runtime contract mismatch: {mismatches}")
    states = response.get("internal_states")
    if not isinstance(states, list) or not states:
        raise CheckError("server runtime contract has no internal scheduler states")
    accept_lengths = [
        state.get("avg_spec_accept_length")
        for state in states
        if isinstance(state, dict)
        and isinstance(state.get("avg_spec_accept_length"), (int, float))
    ]
    active_accept_lengths = [
        float(value)
        for value in accept_lengths
        if math.isfinite(float(value)) and float(value) > 1.0
    ]
    if not active_accept_lengths:
        raise CheckError(
            f"NEXTN is configured but no active acceptance length exceeds 1: "
            f"{accept_lengths!r}"
        )
    token_capacities = [
        (state.get("memory_usage") or {}).get("token_capacity")
        for state in states
        if isinstance(state, dict)
    ]
    if not token_capacities or any(value != capacity for value in token_capacities):
        raise CheckError(
            f"internal token-capacity mismatch: expected {capacity}, got {token_capacities!r}"
        )
    return {
        "status": "pass",
        "context_length": response["context_length"],
        "max_total_num_tokens": response["max_total_num_tokens"],
        "kv_cache_dtype": response["kv_cache_dtype"],
        "quantization": response["quantization"],
        "speculative_algorithm": response["speculative_algorithm"],
        "active_spec_accept_lengths": active_accept_lengths,
        "token_capacities": token_capacities,
        "served_model_name": response["served_model_name"],
    }


def _choice_message(response: dict[str, Any], label: str) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise CheckError(f"{label} response has no choices")
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    if not isinstance(message, dict):
        raise CheckError(f"{label} response has no assistant message")
    return message


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def check_reasoning(
    base_url: str, model: str, *, timeout: int, api_key: str = ""
) -> dict[str, Any]:
    response = http_json(
        base_url,
        "/v1/chat/completions",
        payload={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Think through 17 multiplied by 19, then return the integer.",
                }
            ],
            "max_tokens": 512,
            "temperature": 0,
            "reasoning_effort": "medium",
            "chat_template_kwargs": {
                "enable_thinking": True,
                "preserve_thinking": True,
            },
        },
        timeout=timeout,
        api_key=api_key,
    )
    message = _choice_message(response, "reasoning")
    reasoning = message.get("reasoning_content")
    content = message.get("content")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise CheckError("reasoning_content was not preserved")
    if not isinstance(content, str) or "323" not in content:
        raise CheckError("reasoning probe did not return the expected result")
    return {
        "status": "pass",
        "reasoning_chars": len(reasoning),
        "reasoning_sha256": _digest_text(reasoning),
        "content_chars": len(content),
        "content_sha256": _digest_text(content),
    }


def check_tool_round_trip(
    base_url: str, model: str, *, timeout: int, api_key: str = ""
) -> dict[str, Any]:
    user_message = {
        "role": "user",
        "content": "Call multiply with a=7 and b=8. Use the tool before answering.",
    }
    tool = {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two integers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    }
    first = http_json(
        base_url,
        "/v1/chat/completions",
        payload={
            "model": model,
            "messages": [user_message],
            "tools": [tool],
            "tool_choice": "required",
            "max_tokens": 512,
            "temperature": 0,
            "reasoning_effort": "medium",
        },
        timeout=timeout,
        api_key=api_key,
    )
    assistant = _choice_message(first, "tool-first")
    calls = assistant.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise CheckError("tool-first response did not contain exactly one tool call")
    call = calls[0]
    function = call.get("function") if isinstance(call, dict) else None
    if not isinstance(function, dict) or function.get("name") != "multiply":
        raise CheckError("tool-first response called the wrong function")
    arguments_raw = function.get("arguments")
    try:
        arguments = json.loads(arguments_raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise CheckError("tool-call arguments are not valid JSON") from exc
    if arguments != {"a": 7, "b": 8}:
        raise CheckError(f"unexpected tool-call arguments: {arguments!r}")
    call_id = call.get("id")
    if not isinstance(call_id, str) or not call_id:
        raise CheckError("tool call has no ID")

    second = http_json(
        base_url,
        "/v1/chat/completions",
        payload={
            "model": model,
            "messages": [
                user_message,
                assistant,
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": "multiply",
                    "content": json.dumps({"result": 56}),
                },
            ],
            "tools": [tool],
            "tool_choice": "none",
            "max_tokens": 256,
            "temperature": 0,
            "reasoning_effort": "medium",
        },
        timeout=timeout,
        api_key=api_key,
    )
    final_message = _choice_message(second, "tool-final")
    content = final_message.get("content")
    if not isinstance(content, str) or "56" not in content:
        raise CheckError("tool-final response did not use the supplied result")
    return {
        "status": "pass",
        "function": "multiply",
        "arguments": arguments,
        "final_content_chars": len(content),
        "final_content_sha256": _digest_text(content),
    }


def choose_probe_token(
    base_url: str, model: str, *, timeout: int, api_key: str = ""
) -> int:
    response = http_json(
        base_url,
        "/v1/tokenize",
        payload={
            "model": model,
            "prompt": " stability",
            "add_special_tokens": False,
        },
        timeout=min(timeout, 120),
        api_key=api_key,
    )
    tokens = response.get("tokens")
    if not isinstance(tokens, list) or not tokens or not all(
        isinstance(item, int) and item >= 0 for item in tokens
    ):
        raise CheckError("/v1/tokenize did not return usable token IDs")
    return tokens[-1]


def check_exact_prefill(
    base_url: str,
    model: str,
    prompt_tokens: int,
    *,
    timeout: int,
    api_key: str = "",
    token_id: int | None = None,
) -> dict[str, Any]:
    selected = (
        choose_probe_token(base_url, model, timeout=timeout, api_key=api_key)
        if token_id is None
        else token_id
    )
    response = http_json(
        base_url,
        "/v1/completions",
        payload={
            "model": model,
            "prompt": [selected] * prompt_tokens,
            "max_tokens": 32,
            "temperature": 0,
            "stream": False,
        },
        timeout=timeout,
        api_key=api_key,
    )
    usage = response.get("usage")
    choices = response.get("choices")
    if not isinstance(usage, dict):
        raise CheckError(f"{prompt_tokens}-token probe has no usage object")
    actual = usage.get("prompt_tokens")
    if actual != prompt_tokens:
        raise CheckError(
            f"prompt-token mismatch: expected={prompt_tokens} actual={actual}"
        )
    if not isinstance(choices, list) or not choices:
        raise CheckError(f"{prompt_tokens}-token probe has no completion")
    first = choices[0] if isinstance(choices[0], dict) else {}
    finish_reason = first.get("finish_reason")
    if finish_reason not in {"length", "stop"}:
        raise CheckError(f"unexpected finish reason: {finish_reason!r}")
    return {
        "status": "pass",
        "prompt_tokens": actual,
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": finish_reason,
        "probe_token_id": selected,
    }


def public_ready_checks(
    base_url: str,
    model: str,
    *,
    health_timeout: int,
    request_timeout: int,
    api_key: str = "",
) -> dict[str, Any]:
    return {
        "health": wait_for_health(
            base_url, timeout=health_timeout, api_key=api_key
        ),
        "models": check_models(
            base_url, model, timeout=request_timeout, api_key=api_key
        ),
        "reasoning": check_reasoning(
            base_url, model, timeout=request_timeout, api_key=api_key
        ),
        "tools": check_tool_round_trip(
            base_url, model, timeout=request_timeout, api_key=api_key
        ),
        "server_info": check_server_info(
            base_url, model, timeout=request_timeout, api_key=api_key
        ),
    }
