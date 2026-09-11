# Qwen3.8 Flash-Next: the current 200K setup

A technical field guide for one RTX PRO 6000 Blackwell 96 GB. Recorded September 11, 2026. [Rendered guide](https://dherichsen.github.io/qwen38-flash-next-sm120/).

## Is the original repository still used?

Yes, as the source foundation. Production runs a separately built image based on the same SGLang tree `7848e2094b30690a40a47a56f1e8f4e68a9929d3`, plus the NVIDIA compatibility overlay now included here. It does not execute this Git checkout directly. The original August 31 configuration remains available; merely raising its context limit does not reproduce the new deployment.

| Component | August 31 profile | Current September 7 deployment |
|---|---|---|
| Checkpoint | Earlier RadixArk candidate | NVIDIA Qwen3.8-Flash-Next-NVFP4 |
| Context / shared token pool | 131,072 / 131,072 | 200,000 / 200,000 |
| Quantization loader | `modelopt_fp4` | `modelopt_mixed` |
| Offload | PLE embeddings | PLE + vocabulary embeddings to host |
| Runtime | Pinned SGLang + GC / FP8 KV patches | Same foundation + NVIDIA compatibility patches |
| Unchanged | Single GPU, FP8 E4M3 KV, NEXTN | Single GPU, FP8 E4M3 KV, NEXTN |

The [NVIDIA checkpoint](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) is public. This guide pins revision `fc694b54fb0174e0913e6adf86691ef85a4ead47`. Obtain the weights from the publisher; no weights are redistributed here. Review the model card’s linked NVIDIA and Qwen terms.

## Hardware and operating envelope

- Linux x86-64; NVIDIA RTX PRO 6000 **Blackwell** with 96 GB. An older RTX 6000 is a different architecture. This full profile does not fit the 32 GB RTX 5090; the second GPU in our machine is reserved for separate research.
- Our host has 192 GB-class RAM (186 GiB reported by Linux). Offloaded embeddings and checkpoint loading require substantial host memory. This is an observed working configuration, not a measured minimum.
- Fast local storage: roughly 123.6 GiB for checkpoint shards, plus the model cache, approximately 47 GB container image and build workspace. Allow ample additional space for downloads and intermediate layers.
- Observed driver: 595.84. The container toolchain is digest-pinned in `config/pins.env`; host driver/container compatibility must be checked on your machine.
- One generation at a time; CPUs 0–31 on our host. Change CPU affinity to valid CPUs on yours. Cold startup took approximately nine minutes in the deployment record.
- About 87 GiB GPU memory was in use during the September 11 spot check; this is not a peak bound or a guarantee for every request.
- Default client profile: medium preserved thinking, temperature 0, up to 32,768 output tokens per request. Input, history, reasoning and output share the 200,000-token context. The output allowance is a client setting, not a server-wide cap.

## 1. Prepare the host and download the checkpoint

Install Docker, NVIDIA Container Toolkit, Git, Python 3 and `uv`. Confirm Docker GPU access before proceeding. Use the [NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for your distribution.

```bash
nvidia-smi --query-gpu=uuid,name,memory.total,compute_cap --format=csv
lscpu

git clone https://github.com/dherichsen/qwen38-flash-next-sm120.git
cd qwen38-flash-next-sm120

uv tool install huggingface_hub
mkdir -p "$HOME/models/qwen38-flash-next-nvfp4"
hf download nvidia/Qwen3.8-Flash-Next-NVFP4 \
  --revision fc694b54fb0174e0913e6adf86691ef85a4ead47 \
  --local-dir "$HOME/models/qwen38-flash-next-nvfp4"
```

Allow Hugging Face’s default Xet transfer support; disabling it broke the large-file download in our preparation. If authentication is required, use `hf auth login` locally. Never put tokens in this repository. The download is large; do not point the runtime at a partially downloaded directory.

## 2. Build the original foundation and current overlay

```bash
./scripts/compose_sglang_source.sh
./scripts/build_image.sh
./scripts/build_nvidia_image.sh
python3 -m unittest discover -s tests -v
```

The first stage reconstructs the pinned public SGLang tree. The second image adds `patches/0003-nvidia-mixed-host-offload.patch`, checking every changed file before and after application. It includes mixed ModelOpt loading, SM120 FP4/FP8 backend compatibility, routing fixes and host embedding offload. See [patch provenance](../patches/PROVENANCE.md) and [source hashes](../config/nvidia-source-manifest.json).

Image builds use CPU and disk; they do not load the model. Image IDs will differ from our original production image because the layer history differs. A local image ID is not a downloadable registry reference.

## 3. Configure one GPU and inspect the launch

```bash
cp config/nvidia-200k.env.example .env
```

Edit `.env`: set `QWEN38_MODEL_PATH` to the absolute downloaded directory, replace `QWEN38_GPU_DEVICE` with the **UUID of your idle 96 GB Blackwell GPU**, set `QWEN38_CPU_SET` to valid CPUs, and choose a writable absolute `QWEN38_KERNEL_CACHE` directory. Create the cache directory first. Keep the API bound to `127.0.0.1`.

```bash
set -a
. ./.env
set +a
mkdir -p "$QWEN38_KERNEL_CACHE"
python3 scripts/preflight.py
python3 scripts/run.py --print-shell
python3 scripts/run.py --execute
```

The last command runs in the foreground. Preflight rejects a busy GPU, insufficient VRAM, an incompatible image, missing model files or an occupied port. Coordinate exclusive GPU ownership before launch; do not stop unrelated services to make preflight pass. The public kit does not include our private GPU lease and power-management services.

The trusted-model code flag is part of the pinned runtime. Review the publisher and pinned code before launching. Nothing here exposes your inference endpoint to the internet.

## 4. Verify readiness in a second terminal

From the same checkout, load the same `.env`, then run:

```bash
set -a
. ./.env
set +a
python3 scripts/ready.py
python3 scripts/accept.py --output evidence/acceptance.json
```

Readiness checks identity, runtime configuration, reasoning and native tool roundtrips. Acceptance adds exact 65,536- and 120,000-token prefill probes and log checks; these consume GPU time. `--skip-long-context` gives a shorter functional pass, which does not validate long context. A configured 200K window is not proof of high-quality reasoning at 200K. Failed gates need investigation, not threshold changes.

## 5. Connect a client

Base URL: `http://127.0.0.1:11438/v1`. Model: `qwen3.8-flash-next-sglang`. API style: Chat Completions, with streaming and native tool calls. Our coding client uses the name `flash-medium`; that client profile is not bundled or required for API use.

```bash
curl --fail-with-body http://127.0.0.1:11438/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-flash-next-sglang",
    "messages": [{"role": "user", "content": "Write a Python function that deduplicates a list while preserving order."}],
    "temperature": 0,
    "max_tokens": 32768,
    "chat_template_kwargs": {
      "enable_thinking": true,
      "preserve_thinking": true,
      "reasoning_effort": "medium"
    }
  }'
```

For another machine, prefer an SSH tunnel to the loopback API (`ssh -L 11438:127.0.0.1:11438 user@your-server`). For unattended service use, adapt the existing [systemd template](../systemd/qwen38-flash-next.service.in) after a successful foreground acceptance run. Keep exclusive GPU ownership and readiness checks. The public documentation page has no access to your inference server.

## Evidence and limits

The September 7 production record reports startup, 65K prefill, reasoning, tool-roundtrip and resource checks passed. One small Hermes coding integration passed its visible and independent checks. This does not establish broad coding superiority or quality throughout the full context window. The older nine-task speed comparison belongs to the August 31 profile and must not be presented as a benchmark of this configuration.

For this publication, we built the public NVIDIA overlay, verified its changed runtime files against the running service, matched all 3,717 Python/CUDA source files with no differences, and passed 11 configuration/API tests. [Build verification](current-build-verification.json) records the exact comparison and any differences. We did **not** load a second copy of the rebuilt image on the occupied GPU or repeat a 200K quality benchmark. Your friend should run acceptance on their own machine.

If startup is still loading, allow time for checkpoint I/O and kernel compilation. If out of memory, check exclusive GPU ownership, host RAM and the selected UUID; a reduced context profile is a separate configuration to validate. If readiness reports a loader/offload mismatch, verify that both the NVIDIA overlay image and `config/nvidia-200k.env.example` are selected.
