> **INVALID (10/5):** latent-critic arms here accepted the drift critic on logit >= tau (inverted; the most drifted chunks were accepted). See step3_*_fixed.

# sdvg_steps_compare

Times on dedicated GPUs (10/2 clean re-run), model loading excluded; 832x480 unless noted, 16 fps.

| scene | seed | variant | total time ↓ (s) | accepted |
|---|---|---|---|---|
| jar | 1234 | 1: Self-Forcing 1.3B -> Krea 14B, ImageReward tau -0.7 (T2V) | 28.9 | 12/13 |
| jar | 1234 | 2: LingBot 1.3B -> 14B, ImageReward tau -0.7 (action) | 94.3 | 1/13 |
| jar | 1234 | 3: LingBot 1.3B -> 14B, latent drift critic tau 0.05 (action) | 58.9 | 6/13 |
| jar | 4321 | 1: Self-Forcing 1.3B -> Krea 14B, ImageReward tau -0.7 (T2V) | 29.0 | 12/13 |
| jar | 4321 | 2: LingBot 1.3B -> 14B, ImageReward tau -0.7 (action) | 88.6 | 1/13 |
| jar | 4321 | 3: LingBot 1.3B -> 14B, latent drift critic tau 0.05 (action) | 62.9 | 5/13 |
| jar | 9999 | 1: Self-Forcing 1.3B -> Krea 14B, ImageReward tau -0.7 (T2V) | 29.5 | 12/13 |
| jar | 9999 | 2: LingBot 1.3B -> 14B, ImageReward tau -0.7 (action) | 88.7 | 1/13 |
| jar | 9999 | 3: LingBot 1.3B -> 14B, latent drift critic tau 0.05 (action) | 54.7 | 7/13 |
| castle | 1234 | 1: Self-Forcing 1.3B -> Krea 14B, ImageReward tau -0.7 (T2V) | 30.3 | 12/13 |
| castle | 1234 | 2: LingBot 1.3B -> 14B, ImageReward tau -0.7 (action) | 86.5 | 1/13 |
| castle | 1234 | 3: LingBot 1.3B -> 14B, latent drift critic tau 0.05 (action) | 65.3 | 4/13 |
| castle | 4321 | 1: Self-Forcing 1.3B -> Krea 14B, ImageReward tau -0.7 (T2V) | 30.1 | 12/13 |
| castle | 4321 | 2: LingBot 1.3B -> 14B, ImageReward tau -0.7 (action) | 86.5 | 1/13 |
| castle | 4321 | 3: LingBot 1.3B -> 14B, latent drift critic tau 0.05 (action) | 53.1 | 7/13 |
| castle | 9999 | 1: Self-Forcing 1.3B -> Krea 14B, ImageReward tau -0.7 (T2V) | 29.3 | 12/13 |
| castle | 9999 | 2: LingBot 1.3B -> 14B, ImageReward tau -0.7 (action) | 86.7 | 1/13 |
| castle | 9999 | 3: LingBot 1.3B -> 14B, latent drift critic tau 0.05 (action) | 73.6 | 2/13 |
