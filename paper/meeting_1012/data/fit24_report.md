| config (14B alone, 2-step, fused + partial commit, 161 frames) | weights (+ CLIP image encoder) ↓ | KV window (18 latent frames) ↓ | resident after prepare ↓ | peak, generation ↓ | peak, streamed decode ↓ | peak reserved ↓ | fits 24 GB (≤ 23 GiB reserved) | fits 32 GB | steady fps ↑ |
|---|---|---|---|---|---|---|---|---|---|
| 704x384, FP8 weights, bf16 KV (the realtime mode as run) | 17.6 GiB | 14.2 GiB | 32.7 GiB | 33.5 GiB | 37.6 GiB | 38.9 GiB | no | no | 16.0 |
| 704x384, FP8 weights, FP8 KV | 17.6 GiB | 7.1 GiB | 25.6 GiB | 26.8 GiB | 30.5 GiB | 31.8 GiB | no | no | 15.0 |
| 704x384, INT4 weights, bf16 KV | 8.9 GiB | 14.2 GiB | 24.1 GiB | 25.6 GiB | 28.9 GiB | 30.7 GiB | no | yes | 10.7 |
| 704x384, INT4 weights, FP8 KV | 8.9 GiB | 7.1 GiB | 17.0 GiB | 18.5 GiB | 21.8 GiB | 23.3 GiB | no | yes | 10.3 |
| 704x384, AWQ-INT4 weights, FP8 KV | 9.1 GiB | 7.1 GiB | 17.0 GiB | 18.6 GiB | 21.8 GiB | 23.2 GiB | no | yes | 10.2 |
| 832x480, FP8 weights, bf16 KV (the quality mode as run) | 17.6 GiB | 20.7 GiB | 39.3 GiB | 40.4 GiB | 46.4 GiB | 48.4 GiB | no | no | 10.0 |
| 832x480, FP8 weights, FP8 KV | 17.6 GiB | 10.3 GiB | 29.0 GiB | 30.6 GiB | 36.0 GiB | 38.7 GiB | no | no | 9.5 |
| 832x480, INT4 weights, FP8 KV | 8.9 GiB | 10.3 GiB | 20.3 GiB | 21.9 GiB | 27.4 GiB | 29.7 GiB | no | yes | 6.9 |
| 832x480, AWQ-INT4 weights, FP8 KV | 9.1 GiB | 10.3 GiB | 20.4 GiB | 22.0 GiB | 27.4 GiB | 29.5 GiB | no | yes | 6.8 |
| 704x384, GPTQ-4 weights (Triton dequant), FP8 KV | 9.1 GiB | 7.1 GiB | 17.0 GiB | 18.1 GiB | 21.8 GiB | 23.2 GiB | no | yes | 10.3 |
| 704x384, FP8 weights, FP8 KV, window 12 | 17.6 GiB | 4.7 GiB | 23.3 GiB | 24.2 GiB | 28.1 GiB | 29.4 GiB | no | yes | 17.5 |
| 832x480, FP8 weights, FP8 KV, window 12 | 17.6 GiB | 6.9 GiB | 25.5 GiB | 26.8 GiB | 32.6 GiB | 34.7 GiB | no | no | 11.2 |
| 704x384, GPTQ-4 weights, FP8 KV, window 12 | 9.1 GiB | 4.7 GiB | 14.7 GiB | 15.6 GiB | 19.5 GiB | 20.9 GiB | yes | yes | 11.3 |

Allocator-cap runs (device emulation: the CUDA caching allocator may not grow past the cap):
| run (tiled* = spatial 2x2 tiled chunk decode instead of per-frame streamed decode) | cap | result | peak reserved ↓ | steady fps ↑ | decode s/chunk ↓ |
|---|---|---|---|---|---|
| cap24_a704_int4_bf16 | 24 GB | OOM at prepare | – | – | – |
| cap24_a704_int4_fp8kv | 24 GB | ran to the end | 23.3 GiB | 10.3 | 0.43 |
| cap24_a832_int4_fp8kv | 24 GB | OOM at decode | – | 6.8 | – |
| cap32_a704_fp8_fp8kv | 32 GB | ran to the end | 31.9 GiB | 15.1 | 0.43 |
| cap32_a704_int4_bf16 | 32 GB | ran to the end | 30.7 GiB | 10.7 | 0.43 |
| cap32_a832_int4_fp8kv | 32 GB | ran to the end | 29.7 GiB | 6.9 | 0.63 |
| tiled24_a704_int4_fp8kv | 24 GB | ran to the end | 19.4 GiB | 10.3 | 0.49 |
| tiled24_a704_int4t_fp8kv | 24 GB | ran to the end | 19.4 GiB | 10.3 | 0.49 |
| tiled24_a704w12_gptq4_fp8kv | 24 GB | ran to the end | 17.0 GiB | 11.4 | 0.49 |
| tiled24_a832_int4_fp8kv | 24 GB | ran to the end | 23.9 GiB | 6.9 | 0.69 |
| tiled32_a704_fp8_fp8kv | 32 GB | ran to the end | 27.9 GiB | 15.1 | 0.49 |
| tiled32_a832_fp8_fp8kv | 32 GB | ran to the end | 32.0 GiB | 9.5 | 0.90 |
| tiled32_a832w12_fp8_fp8kv | 32 GB | ran to the end | 28.8 GiB | 11.3 | 0.69 |

Quality check, seed 1234, 8 bench-43 prompts (easy_lake + 7 physics), paired with the FP8 weights / bf16 KV run at the same resolution:
| config | clips | VLM quality ↑ | VLM drift ↓ | Δ quality vs reference (paired) ↑ | Δ drift vs reference (paired) ↓ | quality: better / equal / worse |
|---|---|---|---|---|---|---|
| 704x384, FP8 weights, bf16 KV (the realtime mode as run) | 8 | 6.75 ± 0.25 | 63.8 ± 7.4 | (reference) | (reference) | – |
| 704x384, FP8 weights, FP8 KV | 8 | 6.62 ± 0.26 | 58.1 ± 9.4 | -0.12 ± 0.23 | -5.6 ± 7.2 | 1 / 5 / 2 |
| 704x384, INT4 weights, bf16 KV | 8 | 5.88 ± 0.35 | 70.6 ± 7.4 | -0.88 ± 0.35 | +6.9 ± 5.9 | 0 / 3 / 5 |
| 704x384, INT4 weights, FP8 KV | 8 | 5.75 ± 0.31 | 62.5 ± 8.2 | -1.00 ± 0.27 | -1.2 ± 1.2 | 0 / 2 / 6 |
| 832x480, FP8 weights, bf16 KV (the quality mode as run) | 8 | 7.50 ± 0.27 | 58.1 ± 9.4 | (reference) | (reference) | – |
| 832x480, FP8 weights, FP8 KV | 8 | 6.88 ± 0.23 | 51.2 ± 9.1 | -0.62 ± 0.32 | -6.9 ± 10.3 | 1 / 2 / 5 |
| 832x480, INT4 weights, FP8 KV | 8 | 6.50 ± 0.27 | 76.2 ± 1.2 | -1.00 ± 0.27 | +18.1 ± 8.9 | 0 / 2 / 6 |
| 832x480, AWQ-INT4 weights, FP8 KV | 8 | 6.38 ± 0.18 | 69.4 ± 5.6 | -1.12 ± 0.23 | +11.2 ± 8.6 | 0 / 1 / 7 |
| 704x384, GPTQ-4 weights (Triton dequant), FP8 KV | 43 | 6.65 ± 0.10 | 58.3 ± 3.7 | -1.12 ± 0.23 | +8.8 ± 7.0 | 0 / 1 / 7 |
| 704x384, FP8 weights, FP8 KV, window 12 | 43 | 6.81 ± 0.13 | 52.6 ± 4.2 | -0.62 ± 0.26 | +11.2 ± 7.8 | 1 / 1 / 6 |
| 832x480, FP8 weights, FP8 KV, window 12 | 43 | 7.26 ± 0.12 | 52.9 ± 4.1 | -0.38 ± 0.26 | +6.9 ± 5.9 | 1 / 3 / 4 |
| 704x384, GPTQ-4 weights, FP8 KV, window 12 | 43 | 6.74 ± 0.10 | 55.9 ± 3.7 | -0.88 ± 0.23 | +10.0 ± 9.1 | 0 / 2 / 6 |

43-prompt quality check (seed 1234), paired with the bench-43 clip of the same mode without quantization (FP8 weights, bf16 KV):
| config | pairs | VLM quality ↑ | VLM drift ↓ | reference quality ↑ | reference drift ↓ | Δ quality (paired) ↑ | Δ drift (paired) ↓ | quality: better / equal / worse |
|---|---|---|---|---|---|---|---|---|
| 704x384, FP8 weights, FP8 KV | 43 | 6.77 ± 0.09 | 57.6 ± 4.0 | 6.88 | 57.3 | -0.12 ± 0.11 | +0.2 ± 3.5 | 8 / 22 / 13 |
| 704x384, INT4 weights, FP8 KV | 43 | 6.19 ± 0.14 | 60.6 ± 3.5 | 6.88 | 57.3 | -0.70 ± 0.14 | +3.3 ± 3.5 | 4 / 13 / 26 |
| 704x384, AWQ-INT4 weights, FP8 KV | 43 | 6.28 ± 0.15 | 60.9 ± 3.8 | 6.88 | 57.3 | -0.60 ± 0.13 | +3.6 ± 4.0 | 2 / 19 / 22 |
| 832x480, FP8 weights, FP8 KV | 43 | 7.12 ± 0.10 | 49.4 ± 4.2 | 7.30 | 50.6 | -0.19 ± 0.08 | -1.2 ± 2.9 | 2 / 32 / 9 |
| 704x384, GPTQ-4 weights, FP8 KV | 43 | 6.65 ± 0.10 | 58.3 ± 3.7 | 6.88 | 57.3 | -0.23 ± 0.09 | +0.9 ± 3.2 | 4 / 25 / 14 |
| 704x384, FP8 weights, FP8 KV, KV window 12 | 43 | 6.81 ± 0.13 | 52.6 ± 4.2 | 6.88 | 57.3 | -0.07 ± 0.12 | -4.8 ± 4.3 | 8 / 26 / 9 |
| 832x480, FP8 weights, FP8 KV, KV window 12 | 43 | 7.26 ± 0.12 | 52.9 ± 4.1 | 7.30 | 50.6 | -0.05 ± 0.10 | +2.3 ± 2.5 | 6 / 28 / 9 |
| 704x384, FP8 weights, FP8 KV, seed 4321 | 43 | 6.77 ± 0.13 | 57.7 ± 4.1 | 6.91 | 56.5 | -0.14 ± 0.09 | +1.2 ± 3.5 | 5 / 27 / 11 |
| 832x480, GPTQ-4 weights, FP8 KV | 43 | 6.72 ± 0.13 | 52.2 ± 4.0 | 7.30 | 50.6 | -0.58 ± 0.12 | +1.6 ± 4.2 | 3 / 16 / 24 |
| 704x384, GPTQ-4 weights, FP8 KV, KV window 12 (24 GB candidate) | 43 | 6.74 ± 0.10 | 55.9 ± 3.7 | 6.88 | 57.3 | -0.14 ± 0.09 | -1.4 ± 3.8 | 3 / 30 / 10 |
