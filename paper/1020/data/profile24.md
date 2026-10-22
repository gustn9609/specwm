# Device-tier profile on an idle GPU (10/20, GPU 5 held exclusively, 6 clips of 10 s per cell, medians over clips)

steady fps (gen) ↑ = 12 frames / median per-chunk DiT wall (chunks ≥ 3, CUDA-synced, decode excluded); chunk_s ↓; TTFF ↓ = prepare + first decoded frame; steady fps (e2e) ↑ includes the side-stream VAE decode; peak = torch reserved GiB ↓ (the capped arms decode in bf16 2x2 tiles and have no timed side-stream decode -> TTFF / e2e n/a).

| res | device tier | n | steady fps gen ↑ | chunk_s ↓ | TTFF s ↓ | steady fps e2e ↑ | decode_s ↓ | peak GiB ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 832x480 | original (FP8 w, bf16 KV, window 18) | 6 | <u>9.60</u> | <u>1.249</u> | 7.57 | 6.30 | 9.01 | 72.5 |
| 832x480 | 32 GB class (FP8 w, FP8 KV, window 12) | 6 | **10.82** | **1.109** | n/a | n/a | n/a | <u>27.9</u> |
| 832x480 | 24 GB class (GPTQ-4 w, FP8 KV, window 12) | 6 | 7.42 | 1.616 | n/a | n/a | n/a | **19.7** |
| 640x352 | original (FP8 w, bf16 KV, window 18) | 6 | <u>19.35</u> | <u>0.620</u> | 4.50 | 12.14 | 5.06 | 49.1 |
| 640x352 | 32 GB class (FP8 w, FP8 KV, window 12) | 6 | **21.05** | **0.570** | n/a | n/a | n/a | <u>24.0</u> |
| 640x352 | 24 GB class (GPTQ-4 w, FP8 KV, window 12) | 6 | 12.50 | 0.960 | n/a | n/a | n/a | **15.8** |
