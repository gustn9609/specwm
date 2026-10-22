# 05 Time breakdown of the default action-conditioned SDVG (LingBot-World v2 1.3B → 14B, ImageReward τ −0.7)

jar/castle × seeds 1234/4321/9999, 13 chunks (3 latent frames = 12 video frames; chunk 0 forced to the target), original code, one RTX PRO 6000.
Per-component times from specwm_out/loop1005/*/loop.json; verify split into VAE decode (0.44 s) and ImageReward (0.31 s) per chunk from the 10/1 measurement.

| Step | What runs | Per chunk ↓ (s) | Per clip ↓ (s) | Share |
|---|---|---|---|---|
| prepare | T5 (CPU) + VAE encode of the input image | – | 11.3 | 11 % |
| D | 1.3B draft, 4 steps | 0.71 | 8.5 | 9 % |
| VAE → x^d | decode the draft chunk (fp32) | 0.44 | 5.3 | 5 % |
| V | ImageReward, 12 frames | 0.32 | 3.8 | 4 % |
| T (reject) | 14B regenerate, 4 steps | 3.94 | 47.3 | 47 % |
| KV commit | 14B 0.98 + 1.3B 0.17 | 1.15 | 15.0 | 15 % |
| VAE → x^f | decode the output (fp32) | 0.66 | 8.6 | 9 % |
| total | 1/12 accepted | | 99.9 | |

Per chunk: accepted 2.62 s, rejected 6.56 s, target-only 4.91 s → SDVG wins only above 42 % acceptance (actual: 8 % at τ −0.7, 32 % at τ −2.2).
![breakdown](breakdown.png)
