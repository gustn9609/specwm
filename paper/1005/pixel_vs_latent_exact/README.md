# pixel (ImageReward) vs latent drift critic, per-clip EXACT acceptance match

For every clip, both verifiers were run on dense tau grids (pixel 16 taus, latent 13 taus, fixed accept direction for latent); a pair is listed when both
accepted exactly k of the 12 draft chunks in that clip. If several taus gave the same k, times/distances are averaged (video = first tau).
e2e = prepare 11.3 s + latent generation + final VAE decode (original-speed loop, 1 GPU, 832x480). dist = mean |pixel diff| to the same-seed target-only video
(coarse: AR rollouts are chaotic). Panels: draft | pixel | latent | target.

| scene | seed | k accepted (both) | pixel τ | latent τ | pixel e2e ↓ (s) | latent e2e ↓ (s) | Δ (s) | pixel dist ↓ | latent dist ↓ | pixel plan | latent plan |
|---|---|---|---|---|---|---|---|---|---|---|---|
| jar | 1234 | 1 | -1.5,-0.7 | -4.0 | 105.9 | 92.0 | -13.9 | 67.1 | 62.1 | TDTTTTTTTTTTT | TTDTTTTTTTTTT |
| jar | 1234 | 4 | -2.25 | -1.0 | 88.3 | 80.9 | -7.4 | 74.4 | 67.9 | TDDTTTTTDTTDT | TDDDDTTTTTTTT |
| jar | 1234 | 7 | -2.265,-2.26,-2.255 | -2.5 | 75.6 | 67.7 | -7.9 | 64.2 | 57.1 | TDDTTTDDDTDTD | TDDTDTDDDDTTT |
| jar | 1234 | 8 | -2.27 | -3.87,-3.5 | 71.4 | 63.3 | -8.1 | 62.6 | 56.3 | TDDTTTDDDDTDD | TDTDDDDDDTDTT |
| jar | 4321 | 8 | -2.21 | -3.87,-3.5 | 71.9 | 63.3 | -8.6 | 64.6 | 58.9 | TDDDTDTDTDDDT | TDTDDDDDDTDTT |
| jar | 4321 | 11 | -2.25,-2.245,-2.265,-2.26,-2.255,-2.27 | -3.0,-2.5 | 59.4 | 51.3 | -8.1 | 58.0 | 57.1 | TDDDDDDDDDDTD | TDDDDDDDDDDDT |
| jar | 4321 | 12 | -2.275 | 3.0,-1.0,1.0,0.05,-2.0 | 55.2 | 47.0 | -8.1 | 57.1 | 57.1 | TDDDDDDDDDDDD | TDDDDDDDDDDDD |
| jar | 9999 | 10 | -2.275 | 0.05,1.0,3.0 | 63.6 | 55.5 | -8.2 | 58.5 | 59.7 | TDDDDDTTDDDDD | TDDDDDDDDDDTT |
| castle | 1234 | 4 | -2.1 | -5.25,-5.5 | 92.2 | 78.0 | -14.2 | 73.1 | 56.9 | TDDDTTTDTTTTT | TTDDDDTTTTTTT |
| castle | 4321 | 5 | -2.2,-2.18 | -3.0,-4.0,-3.87,-4.5,-4.25,-3.5 | 82.2 | 74.8 | -7.4 | 66.1 | 65.2 | TDDTDTTDDTTTT | TDDDDDTTTTTTT |
| castle | 4321 | 7 | -2.15 | -4.75,-2.5,-5.0 | 74.1 | 66.4 | -7.7 | 65.7 | 69.9 | TDDTTTDDTDTDD | TDDDTDTDDDTTT |
| castle | 4321 | 9 | -2.23 | -2.0 | 66.4 | 58.4 | -8.0 | 71.0 | 80.1 | TDDDDDDTTDTDD | TDDDDDDDDDTTT |
| castle | 4321 | 10 | -2.25,-2.245,-2.24 | 3.0,-1.0,1.0,0.05 | 62.4 | 54.3 | -8.1 | 71.1 | 81.8 | TDDDDDDTTDDDD | TDDDDDDDDDDTT |
| castle | 9999 | 9 | -2.24 | -4.25,-4.5 | 65.5 | 58.4 | -7.1 | 53.1 | 56.2 | TDTDTDDDDDTDD | TDDDDDDDDDTTT |
| castle | 9999 | 10 | -2.25,-2.255,-2.245 | -3.5,-4.75,-4.0,-3.87,-5.0 | 61.9 | 54.3 | -7.6 | 56.0 | 56.3 | TDDTDDDDDDTDD | TDDDDDDDDDDTT |
| castle | 9999 | 11 | -2.26 | -1.0,0.05,-2.0,-2.5,-3.0,1.0,3.0 | 57.8 | 50.2 | -7.6 | 56.3 | 57.6 | TDDTDDDDDDDDD | TDDDDDDDDDDDT |
