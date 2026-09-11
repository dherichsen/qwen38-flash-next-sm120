# Patch provenance

The patch series applies directly to SGLang commit
`99c9362e6685db579c469f6e0e566b08827b3477`, in the listed order.

| File | Original author | Upstream source | Rebased commit | SHA-256 |
|---|---|---|---|---|
| `0001-kv-budget-gc-before-profile.patch` | Alison Shao | [PR #36583](https://github.com/sgl-project/sglang/pull/36583), commit `6d399a46b4e8e1eaac67fdf1d683c6b05267bb6e` | `a1ee15f2a7a0ad6d804d9fc48c1560d7e50f0bdc` | `a4a8765ccbb5e318816c2350adbe9b339aee5f35f3a022ba8d0ba4fb2b17476b` |
| `0002-qsa-fp8-kv-cache.patch` | LING ZHI | [PR #36644](https://github.com/sgl-project/sglang/pull/36644), commit `d2c8b42cb1091db9de760663fbc56aef05b22a57` | `4318970cabd6ff1109eae352c5a2d9f6bbffff19` | `479141f9b3a1d248dcde09b869ca81421f3cd1612f1adf3302f5a89327182f16` |

The base already contains the exact-SM120 sparse-decode routing merged into
the Qwen3.8 branch through PR #36806. The FP8 patch was rebased onto that base,
so its patch ID differs from the original PR commit while retaining both sets
of changes.

Validated tree sequence:

```text
base                  83bb87574bd055206919798faf2d1c9997b87ddf
after 0001            43100cba34280ccd9f0f52e46d3c8bbcc36210b7
after 0002 (accepted) 7848e2094b30690a40a47a56f1e8f4e68a9929d3
```

The files were produced by Git 2.43.0 with:

```bash
git format-patch --stdout --full-index --binary -1 <rebased-commit>
```

Do not replace the pins with moving PR heads. Any update is a new candidate and
must receive new tree hashes and acceptance evidence.

## September 7 NVIDIA compatibility overlay (published September 11)

`0003-nvidia-mixed-host-offload.patch` is the local deployment delta on top of the original composed tree. It was reconstructed from the exact 16 runtime files and seven test files copied into the production NVIDIA compatibility image, not from a broader experimental branch. Existing source copyright and license headers remain intact; original SGLang code remains Apache-2.0. This local integration is not claimed to be a merged upstream patch. The delta covers mixed-precision ModelOpt loading, host embedding gather/offload, SM120 backend selection, routing order and related compatibility tests. Unrelated DFlash experiments are excluded.

`config/nvidia-source-manifest.json` records original and deployed SHA256 per file. The new-file entries have a null base hash. The builder refuses a base or result mismatch. One upstream source whitespace-only line is retained for byte-for-byte runtime matching. `docs/current-build-verification.json` records comparison with the running deployment. The original base-tree label identifies ancestry, not the entire patched source tree.
