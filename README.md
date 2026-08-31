# Qwen3.8 Flash-Next on one SM120 GPU

This repository captures a reproducible SGLang source composition and a
single-GPU deployment profile for Qwen3.8 Flash-Next on an NVIDIA RTX PRO
6000 Blackwell (SM120, 96 GiB). It preserves reasoning, tool calling, 131,072
tokens of context, FP8 E4M3 KV cache, and built-in NEXTN speculative decoding.

The important result is local and deliberately narrow. The like-for-like active
worker timestamps for warm tasks 2–9 were **22.34 s** for this profile versus
**28.70 s** for the contemporaneous `worker-36` control (22.2% lower). Both
profiles passed 9/9 workers with hidden score 100. Coarser campaign-level
measurements also favored Flash-Next (32.111 versus 40.784 seconds per clean),
but those two campaign controllers polled at different intervals and are only
corroborating evidence. This is not a world record or an independently
reproduced benchmark; see
[docs/results-2026-08-31.md](docs/results-2026-08-31.md).

## What is included

- Two attributed patches that reproduce the exact tested SGLang source tree.
- A deterministic source composer that refuses a tree-hash mismatch.
- A Dockerfile based on an immutable official SGLang image digest.
- A parameterized, least-privilege Docker launcher and systemd template.
- Public readiness and acceptance checks for models, reasoning, tools, and
  exact 65,536- and 120,000-token prefill probes.

No model weights, private prompts, hidden graders, production logs,
credentials, or machine-specific paths are included.

## Compatibility boundary

The tested checkpoint was a separately supplied NVFP4 candidate whose card
described it as a private candidate. This repository does **not** imply that
those exact weights are publicly available. Bring or quantize a compatible
checkpoint, verify its provenance, and accept its license before use. The
source model uses the Qwen Community License 1.0, not Apache-2.0; commercial
Model-as-a-Service or AI Work Assistant use may require a separate Qwen
license. See [docs/licensing.md](docs/licensing.md).

## Pinned source

The composer starts from SGLang commit `99c9362e6685db579c469f6e0e566b08827b3477`
and applies two rebased, author-preserving patches. It accepts the result only
when Git reports tree:

```text
7848e2094b30690a40a47a56f1e8f4e68a9929d3
```

The first patch is the merged KV-budget GC fix from SGLang PR #36583. The
second is the FP8 QSA KV-cache work from open PR #36644, rebased onto the tested
SM120 branch. The tested Qwen3.8 support itself was still an open upstream PR
(#36497) on 2026-08-31. Read [patches/PROVENANCE.md](patches/PROVENANCE.md)
before rebasing anything.

## Build

Requirements: Linux, Git, GNU `tar`, `gzip`, Docker with BuildKit, NVIDIA
Container Toolkit, and an SM120 GPU.

This is a heavy deployment. The tested checkpoint occupied about 135.25 GB on
disk, warmed GPU use was 92,430 MiB (about 90.3 GiB), CPU PLE offload adds
substantial host memory pressure, and cold boot plus kernel JIT took roughly
6–8 minutes. Plan for fast local storage and ample host RAM; a 192 GiB-class
machine is a sensible starting point for reproducing the tested shape.

```bash
./scripts/compose_sglang_source.sh
./scripts/build_image.sh
```

The first command downloads public SGLang source only, applies the vendored
patches, verifies the final tree, and creates an ignored deterministic archive
under `build/`. The second command verifies that archive and builds
`qwen38-flash-next-sm120:2026-08-31`.

## Prepare a compatible model

The image build does not fetch or create weights. Start with the official
[Qwen3.8 Flash-Next model](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
review its license, and use
[NVIDIA TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
if you intend to produce ModelOpt-compatible NVFP4 weights. The exact candidate
quantization pipeline used in the local result is outside this repository and
is not claimed reproducible here. Treat a newly quantized checkpoint as a new
candidate: verify tensor structure, scale finiteness, unchanged tensors,
quality, long context, and soak behavior before production use.

## Configure and inspect

```bash
cp config/server.env.example .env
# Edit .env: QWEN38_MODEL_PATH is required.
set -a
. ./.env
set +a

python3 scripts/preflight.py
python3 scripts/run.py --print-shell
```

`run.py` prints the audited Docker argv by default. It launches only with an
explicit `--execute`:

```bash
python3 scripts/run.py --execute
```

The default endpoint is `http://127.0.0.1:11438/v1`. The example remains
loopback-only.

## Public verification

After the server is ready:

```bash
python3 scripts/ready.py
python3 scripts/accept.py --output evidence/acceptance.json
```

`ready.py` checks health, model identity, preserved reasoning, and a complete
tool round-trip. `accept.py` adds exact token-ID prefills at 65,536 and 120,000
tokens and checks container logs when `QWEN38_CONTAINER_NAME` is available.
Neither script invokes a private grader or mutates routing.

The 120K probe is intentionally expensive. Use `--skip-long-context` for a
quick functional acceptance run.

## systemd

`systemd/qwen38-flash-next.service.in` is a template, not an install script.
Replace `@INSTALL_DIR@` with the absolute checkout path, copy the edited unit
to `/etc/systemd/system`, and place the reviewed environment file at
`/etc/qwen38-flash-next.env`. Inspect both files before enabling the unit.

## Known caveats

- The tested SGLang support stack included unreleased/open pull requests.
- FP8 KV scales emitted warnings and defaulted to `1.0`; that caveat is not
  hidden or claimed fixed here.
- The exact SM120 sparse-decode route had no supported in-process
  FlashAttention fallback. Failure recovery is service-level rollback.
- This is a single-GPU profile. Do not infer that tensor parallelism across
  mismatched, non-P2P GPUs will improve it.
- The private GPU lease/resource scheduler used on the test host is not part of
  this repository. The persistent service occupies about 92,430 MiB and must be
  drained before another job uses the same GPU. Supply your own exclusive lease
  and health-gated restoration; the systemd template alone is not collision-safe
  with trainers or other GPU services.
- Runtime behavior depends on the exact checkpoint, driver, CUDA stack, and
  container architecture selected by the pinned multi-platform digest.

See [docs/caveats.md](docs/caveats.md) for the full boundary.

## License

Repository-authored code is Apache-2.0. Vendored patches remain attributed to
their SGLang authors under Apache-2.0. Model weights are not included and are
subject to separate terms. See [LICENSE](LICENSE), [NOTICE](NOTICE), and
[docs/licensing.md](docs/licensing.md).
