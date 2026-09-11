# Licensing boundary

The deployment code and documentation in this repository are licensed under
Apache-2.0. SGLang is also Apache-2.0; the two vendored patches retain their
authors and upstream pull-request provenance in `NOTICE` and
`patches/PROVENANCE.md`.

Model weights are not included. As checked on 2026-08-31, both the Qwen3.8
Flash-Next source card and the tested derivative card declared `license: other`
and referred users to the **Qwen Community License 1.0**. That license includes
conditions beyond Apache-2.0. In particular, its text says commercial use in a
Model-as-a-Service or AI Work Assistant business requires a separate license,
subject to the license's definitions and internal-use exception.

Review the current source-model files directly before downloading, quantizing,
redistributing, or serving any checkpoint:

- https://huggingface.co/Qwen/Qwen3.8-Flash-Next
- https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE

Nothing in this repository grants rights to a checkpoint, Qwen trademarks, or
third-party datasets. This summary is informational, not legal advice.

## Current NVIDIA profile (September 11, 2026)

The new guide uses the public `nvidia/Qwen3.8-Flash-Next-NVFP4` checkpoint. Its current model card names the NVIDIA Open Model License as governing terms and links the Qwen Community License as additional information. Read those linked terms and the pinned model files directly; the older private-candidate description above concerns the historical profile. No model weights are distributed here.

https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4
