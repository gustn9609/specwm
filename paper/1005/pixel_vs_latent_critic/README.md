# pixel_vs_latent_critic

Same loop (spec_ar_loop, original-speed 14B, 1 GPU), 832x480. Pixel = SDVG: VAE-decode the draft chunk + ImageReward worst frame >= tau. Latent = drift critic on the latent, accept logit <= tau (fixed 10/5).

| scene | seed | arm | latent gen ↓ (s) | + decode (s) | total ↓ (s) | accepted (of 12) | draft / verify / target / commit (s) | plan |
|---|---|---|---|---|---|---|---|---|
| jar | 1234 | drafter only | 11.5 | 8.6 | 20.1 | 12 | 9.2 / 0.0 / 0.0 / 2.2 | DDDDDDDDDDDDD |
| jar | 1234 | pixel: ImageReward tau -0.7 | 85.7 | 8.6 | 94.3 | 1 | 8.6 / 13.9 / 48.0 / 15.2 | TDTTTTTTTTTTT |
| jar | 1234 | latent critic tau 0.05 (fixed) | 31.4 | 8.7 | 40.0 | 11 | 8.6 / 0.1 / 7.4 / 15.3 | TDDDDDDDDDDDT |
| jar | 1234 | target only | 64.4 | 8.6 | 73.0 | 0 | 0.0 / 0.0 / 51.3 / 12.8 | TTTTTTTTTTTTT |
| jar | 4321 | drafter only | 11.3 | 8.6 | 19.9 | 12 | 9.0 / 0.0 / 0.0 / 2.3 | DDDDDDDDDDDDD |
| jar | 4321 | pixel: ImageReward tau -0.7 | 80.0 | 8.6 | 88.6 | 1 | 8.5 / 8.2 / 48.0 / 15.2 | TDTTTTTTTTTTT |
| jar | 4321 | latent critic tau 0.05 (fixed) | 27.1 | 8.7 | 35.8 | 12 | 8.6 / 0.1 / 3.1 / 15.3 | TDDDDDDDDDDDD |
| jar | 4321 | target only | 64.4 | 8.6 | 73.0 | 0 | 0.0 / 0.0 / 51.3 / 12.8 | TTTTTTTTTTTTT |
| jar | 9999 | drafter only | 11.4 | 8.6 | 20.0 | 12 | 9.0 / 0.0 / 0.0 / 2.3 | DDDDDDDDDDDDD |
| jar | 9999 | pixel: ImageReward tau -0.7 | 80.1 | 8.6 | 88.7 | 1 | 8.6 / 8.2 / 48.0 / 15.2 | TDTTTTTTTTTTT |
| jar | 9999 | latent critic tau 0.05 (fixed) | 35.5 | 8.7 | 44.2 | 10 | 8.6 / 0.1 / 11.6 / 15.3 | TDDDDDDDDDDTT |
| jar | 9999 | target only | 64.4 | 8.6 | 73.0 | 0 | 0.0 / 0.0 / 51.4 / 12.8 | TTTTTTTTTTTTT |
| castle | 1234 | drafter only | 11.2 | 8.5 | 19.7 | 12 | 8.9 / 0.0 / 0.0 / 2.2 | DDDDDDDDDDDDD |
| castle | 1234 | pixel: ImageReward tau -0.7 | 78.0 | 8.5 | 86.5 | 1 | 8.4 / 8.1 / 46.6 / 14.8 | TDTTTTTTTTTTT |
| castle | 1234 | latent critic tau 0.05 (fixed) | 42.9 | 8.5 | 51.4 | 8 | 8.4 / 0.1 / 19.5 / 14.8 | TDDDDDDDTTDTT |
| castle | 1234 | target only | 62.5 | 8.5 | 71.0 | 0 | 0.0 / 0.0 / 49.8 / 12.4 | TTTTTTTTTTTTT |
| castle | 4321 | drafter only | 11.2 | 8.5 | 19.7 | 12 | 8.9 / 0.0 / 0.0 / 2.2 | DDDDDDDDDDDDD |
| castle | 4321 | pixel: ImageReward tau -0.7 | 78.0 | 8.5 | 86.5 | 1 | 8.4 / 8.1 / 46.6 / 14.8 | TDTTTTTTTTTTT |
| castle | 4321 | latent critic tau 0.05 (fixed) | 34.5 | 8.5 | 43.0 | 10 | 8.4 / 0.1 / 11.2 / 14.8 | TDDDDDDDDDDTT |
| castle | 4321 | target only | 62.5 | 8.5 | 71.0 | 0 | 0.0 / 0.0 / 49.8 / 12.4 | TTTTTTTTTTTTT |
| castle | 9999 | drafter only | 11.2 | 8.5 | 19.7 | 12 | 9.0 / 0.0 / 0.0 / 2.2 | DDDDDDDDDDDDD |
| castle | 9999 | pixel: ImageReward tau -0.7 | 78.2 | 8.5 | 86.7 | 1 | 8.5 / 8.1 / 46.7 / 14.8 | TDTTTTTTTTTTT |
| castle | 9999 | latent critic tau 0.05 (fixed) | 30.5 | 8.5 | 39.0 | 11 | 8.4 / 0.1 / 7.1 / 14.8 | TDDDDDDDDDDDT |
| castle | 9999 | target only | 62.5 | 8.5 | 71.0 | 0 | 0.0 / 0.0 / 49.8 / 12.4 | TTTTTTTTTTTTT |
