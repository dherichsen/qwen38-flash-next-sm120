# Caveats and non-claims

## Upstream status

The tested stack was based on SGLang's Qwen3.8 branch at commit `99c9362e` plus
two patches. On 2026-08-31, Qwen3.8 support PR #36497 and FP8-QSA-KV PR #36644
were still open. A future merge, rebase, or release is a new candidate and must
be retested rather than assumed equivalent.

## FP8 KV scales

The server warned that FP8 KV scales were not calibrated and used `1.0`. The
local quality and soak gates passed with that behavior, but this repository
does not claim the warning is harmless for all prompts or checkpoints.

## SM120 fallback

On exact SM120 with the tested stack, sparse decode selected the FlashInfer
TRT-LLM backend. There was no supported runtime switch or exception-triggered
fallback to the FlashAttention branch. A backend-specific fault can therefore
fail a request or service and requires service-level rollback.

## Reproduction boundary

The exact tested NVFP4 checkpoint was separately supplied and described itself
as a private candidate. A public user may have to quantize a compatible source
checkpoint. Different weights, scales, drivers, CUDA versions, FlashInfer
artifacts, or GPUs can change correctness, memory capacity, and performance.

The published benchmark is one-host evidence using a private coding harness.
It is not an independent benchmark, statistical population study, or world
record claim.

## GPU ownership is external

The test host used a private GPU-lease and restoration controller that is not
included here. This service consumes almost the entire 96 GiB device when
warmed. Before starting a trainer or another GPU0 workload, stop and drain this
service; after releasing the device, restore it only through health-gated
orchestration. The generic systemd unit supervises one service but does not
coordinate GPU ownership across services.
