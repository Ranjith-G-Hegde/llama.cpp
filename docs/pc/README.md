# llama.cpp for a 12 GB RTX 40 + GTX 16xx + 96 GB DDR5 PC

This fork tunes llama.cpp for one class of consumer PC and shares what was measured on it:

| Part | This PC | Similar setups that should benefit |
|---|---|---|
| Main GPU | RTX 4070 SUPER 12 GB, PCIe 4.0 x16 | any 12-16 GB RTX 30/40/50 card |
| Second GPU | GTX 1660 SUPER 6 GB in a chipset PCIe 3.0 x4 slot | a GTX 16xx / older card without tensor cores in a narrow slot |
| CPU | Ryzen 7 7800X3D (8 cores, Zen 4, AVX-512) | 8-16 core Zen 4 / Zen 5 |
| RAM | 96 GB DDR5 (4 DIMMs, 3200 MT/s) | 64-128 GB DDR5 |
| OS | Windows 11, MSVC 2022, CUDA 12.6 | Linux works too; the Windows notes are below |

Everything here is **full quality**: no expert skipping, no reduced expert counts, no routing changes. Every speed-up
either produces the same tokens or only changes rounding the way any CPU/GPU split does.

## Branches

| Branch | Base | What it is for |
|---|---|---|
| `pc-main` (default) | upstream master 2026-11-15 | MoE models with experts in RAM (Laguna S 2.1, Nemotron 3 Super, Qwen3.8 Flash Next), plus the second-GPU patch. The base for further work |
| `beellama-moe` | [BeeLlama](https://github.com/Anbeeld/beellama.cpp) preview-v0.5.0 + upstream MoE-cache commits | the build the MoE numbers below were validated on |
| `dual-gpu-1660` | BeeLlama | dense Qwen3.8-27B with FFN layers spilled to the GTX 16xx (the second-GPU patch below) |

`pc-main` is upstream master plus these open upstream PRs: #32065 (few-row MMQ for MoE), #31802 (fused q8_0 KV dequant
in flash attention), #30685 + #31970 (async host input copies), #30493, #32177, #30853 (AVX-512 iq3_xxs dot), #31031
(trimmed draft vocab), #31333, #32159, #31345, #31095, #30797, #31850. The same Laguna config ran within noise of
`beellama-moe` (fresh 18.3 vs 18.2 tok/s, 8K prompt 720 vs 772).

## The second-GPU patch (GTX 16xx helper)

Three opt-in environment variables, all off by default, for a second GPU with no tensor cores in a narrow slot:

| Variable | What it does |
|---|---|
| `GGML_SCHED_GPU_WEIGHT_OFFLOAD=<min batch>` | a large-batch `MUL_MAT` whose weights live on the second GPU runs on the first GPU; decode and speculative verify stay on the second GPU. 32 worked best here |
| `GGML_CUDA_PEER_WEIGHT_MIRROR=1` | keeps a pinned-RAM copy of those weights, so the copy crosses the main GPU's x16 link instead of the helper's x4 link |
| `GGML_CUDA_TU116_CUBLAS=1` | on cc 7.5 cards without tensor cores, batches of 64+ use dequantize + cuBLAS instead of MMA MMQ. Pair with `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` |

Measured on Qwen3.8-27B IQ3_S with 30-44 FFN layers on the 1660 at 128K-256K context: prompts 230-270 -> 640-790
tok/s, decode unchanged (about 24-33 tok/s with DFlash + ngram drafting). The mirror costs about 5.5-6 GiB of pinned RAM.

## MoE models with experts in RAM (validated twice)

Common arguments:

```
-cmoe -ngl 99 --device CUDA0 -t 8 -c 65536 -fa on -ctk q8_0 -ctv q8_0 -lm none --jinja -np 1
-b 2048 -ub 2048 --spec-type ngram-mod --spec-ngram-mod-n-match 24
```

| Model | `--moe-cache-mib` / `--moe-cache-cpu-share` | fresh decode | rewrite | prompt 2.5K / 8K / 32K | vs a plain `-cmoe` start |
|---|---|---|---|---|---|
| Laguna S 2.1 UD-Q4_K_S | 4000 / 0.8 | 18.2 | 36-38 | 547 / 772 / 776 | decode +37%, rewrite 3.1x, prompts 2.3x |
| Qwen3.8 Flash Next IQ3_S | 5000 / 0.5 | 31.8 | 62-70 | 402 / 544 / 661 | rewrite 2.1x, prompts 1.3-1.6x |
| Nemotron 3 Super Q4_K | 2000 / 0.8 | 15.0 | 29-32 | 410 / 584 / 660 | decode +36%, rewrite 3.0x, prompts 1.9-2.4x |

- **Qwen chat:** add the separate MTP head on the second GPU (`-md mtp-Qwen3.8-Flash-Next-Q4_0.gguf -devd CUDA1
  --spec-type draft-mtp --spec-draft-n-max 5 -ubd 2048`): fresh decode 31.8 -> 38.9 (+22%), prompts about 25-40%
  slower. Keep ngram-only for long-prompt and agent work.
- Caches of 6000 MiB or more do not start on a 12 GB card; Nemotron with `-ub 2048` needs 2000 or less.
- `ngram-mod` costs 6-13% on fresh text and doubles copy-heavy work (edits, rewrites).

### Tried and not worth it on this PC

| Try | Result |
|---|---|
| `GGML_CUDA_BLOCKING_SYNC=1` (#26487) | Qwen decode 31.8 -> 28.7 |
| #32076 cache without syncs | asserts with any `--moe-cache-cpu-share` > 0; at share 0 Laguna is 37% slower |
| #30594 hybrid profile cache | 57 conflicts; on a 12 GB card perplexity +5% (KL 0.158) |
| Nemotron 3 Nano 4B as draft | decode 15.0 -> 7.9 (each verify reads its own experts from RAM) |
| poolside Laguna DFlash draft | barely drafts; 9.9 -> 9.7 / 6.8 |
| ik_llama.cpp (+ #2850 for Nemotron) with `classlru -muge -sas -ub 4096` | 2-4x slower than the table above |

## Why Strata is still faster on Qwen, and what is missing here

Strata (Qwen Flash Next, full quality) reaches 59-61 tok/s fresh decode and 1748 tok/s at 32K on the same PC. The
gap is how experts are placed and streamed, not the GPU kernels:

| Strata technique | Here |
|---|---|
| VRAM expert cache that adapts while you chat, all experts in pinned RAM | yes |
| start-up placement from a saved usage profile | not yet |
| CPU vs PCIe split of misses tuned from the measured CPU speed | a fixed share, tuned by hand |
| CPU computes misses while the GPU runs cached experts | yes, with a sync per layer |
| tuned AVX-512 CPU rows (fused gate+up for 2-4 tokens) | generic ggml kernels |
| next layer's experts streamed during the current layer (prompts) | not yet |
| CPU computes the rarely-routed experts of a prompt | not yet |
| expert-cache slots lent to the prompt buffers | not yet |
| prompt lookup only where its measured cost pays | ngram always on |

These are the next steps for `pc-main`. The
[GenerelSchwerz/llama.cpp](https://github.com/GenerelSchwerz/llama.cpp) `moe-cache` fork already implements most of
them (profile-aware allocation, automatic miss split, layer-ahead prompt staging, workspace lending, Strata-derived CPU
kernels) and is the reference for porting.

## Building on Windows

```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=75;89" ^
  -DGGML_SCHED_MAX_COPIES=1 -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_NATIVE=OFF -DGGML_AVX512=ON -DGGML_AVX512_VBMI=ON ^
  -DGGML_AVX512_VNNI=ON -DGGML_AVX512_BF16=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_UI=OFF
cmake --build build --config Release -j 16 --target llama-server llama-bench
```

- `75;89` builds for both the GTX 16xx and the RTX 40 card. A build of every quant type for both fits under the 2 GB
  DLL limit here; ik_llama.cpp's does not.
- `GGML_AVX512_BF16=OFF`: MSVC cannot compile upstream's BF16 casts in `sgemm.cpp`. Quantized models do not use it.
- `LLAMA_BUILD_UI=OFF` fetches the prebuilt web UI instead of running `npm` (which writes its cache to the system drive).
- CUDA 12.6 with VS 2022 17.12 (MSVC 19.42) is the tested pair.
