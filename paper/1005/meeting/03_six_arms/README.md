# 03 Six arms on the action-conditioned world model (jar/castle × seeds 1234/4321/9999)

Times per clip, model loading excluded. AR arms: prepare (11.3 s) + latent generation + final VAE decode. Bidirectional: WanI2V.generate (encoding + 70 UniPC steps, guide 5, shift 10 + decode).

| Arm | e2e mean ↓ (s) | min–max (s) | accepted chunks (of 13) |
|---|---|---|---|
| Draft only (1.3B) | 31.1 | 31.0–31.4 | 13.0 |
| SDVG τ −0.7 (default) | 99.9 | 97.8–105.6 | 1.0 |
| SDVG τ −1.5 | 100.0 | 97.8–106.2 | 1.0 |
| SDVG τ −2.2 | 87.9 | 74.6–100.1 | 3.8 |
| Target only (14B AR) | 83.3 | 82.3–84.3 | 0.0 |
| Bidirectional 14B | 2778.0 | 2729.2–2827.4 | – |

![e2e](e2e_six_arms.png)

Videos: `videos/<scene>__seed<k>__six_arms.mp4`, top row: draft | SDVG −0.7 | SDVG −1.5; bottom row: SDVG −2.2 | target | Bidirectional.
