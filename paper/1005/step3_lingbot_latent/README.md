> **INVALID (10/5):** latent-critic arms here accepted the drift critic on logit >= tau (inverted; the most drifted chunks were accepted). See step3_*_fixed.

# step3_lingbot_latent

Times on dedicated GPUs (10/2 clean re-run), model loading excluded; 832x480 unless noted, 16 fps.

| scene | seed | arm | latent gen ↓ (s) | + final decode (s) | total ↓ (s) | chunks accepted | draft / verify / target / KV commit (s) |
|---|---|---|---|---|---|---|---|
| jar | 1234 | drafter only | 11.5 | 8.6 | 20.1 | 13/13 | 9.2 / 0.0 / 0.0 / 2.2 |
| jar | 1234 | latent critic tau 0.05 | 50.3 | 8.7 | 58.9 | 6/13 | 8.6 / 0.1 / 26.3 / 15.3 |
| jar | 1234 | latent critic tau -3.87 | 46.6 | 8.6 | 55.2 | 7/13 | 8.6 / 0.1 / 22.6 / 15.3 |
| jar | 1234 | target only | 64.4 | 8.6 | 73.0 | 0/13 | 0.0 / 0.0 / 51.3 / 12.8 |
| jar | 4321 | drafter only | 11.3 | 8.6 | 19.9 | 13/13 | 9.0 / 0.0 / 0.0 / 2.3 |
| jar | 4321 | latent critic tau 0.05 | 54.3 | 8.6 | 62.9 | 5/13 | 8.6 / 0.1 / 30.2 / 15.3 |
| jar | 4321 | latent critic tau -3.87 | 58.4 | 8.6 | 67.0 | 4/13 | 8.5 / 0.1 / 34.5 / 15.3 |
| jar | 4321 | target only | 64.4 | 8.6 | 73.0 | 0/13 | 0.0 / 0.0 / 51.3 / 12.8 |
| jar | 9999 | drafter only | 11.4 | 8.6 | 20.0 | 13/13 | 9.0 / 0.0 / 0.0 / 2.3 |
| jar | 9999 | latent critic tau 0.05 | 46.1 | 8.6 | 54.7 | 7/13 | 8.6 / 0.1 / 22.1 / 15.3 |
| jar | 9999 | latent critic tau -3.87 | 43.3 | 8.7 | 52.0 | 8/13 | 8.6 / 0.1 / 19.3 / 15.3 |
| jar | 9999 | target only | 64.4 | 8.6 | 73.0 | 0/13 | 0.0 / 0.0 / 51.4 / 12.8 |
| castle | 1234 | drafter only | 11.2 | 8.5 | 19.7 | 13/13 | 8.9 / 0.0 / 0.0 / 2.2 |
| castle | 1234 | latent critic tau 0.05 | 56.8 | 8.5 | 65.3 | 4/13 | 8.4 / 0.1 / 33.4 / 14.9 |
| castle | 1234 | latent critic tau -3.87 | 40.6 | 8.6 | 49.2 | 8/13 | 8.5 / 0.1 / 17.1 / 14.9 |
| castle | 1234 | target only | 62.5 | 8.5 | 71.0 | 0/13 | 0.0 / 0.0 / 49.8 / 12.4 |
| castle | 4321 | drafter only | 11.2 | 8.5 | 19.7 | 13/13 | 8.9 / 0.0 / 0.0 / 2.2 |
| castle | 4321 | latent critic tau 0.05 | 44.6 | 8.5 | 53.1 | 7/13 | 8.4 / 0.1 / 21.2 / 14.9 |
| castle | 4321 | latent critic tau -3.87 | 40.8 | 8.5 | 49.3 | 8/13 | 8.5 / 0.1 / 17.3 / 14.9 |
| castle | 4321 | target only | 62.5 | 8.5 | 71.0 | 0/13 | 0.0 / 0.0 / 49.8 / 12.4 |
| castle | 9999 | drafter only | 11.2 | 8.5 | 19.7 | 13/13 | 9.0 / 0.0 / 0.0 / 2.2 |
| castle | 9999 | latent critic tau 0.05 | 65.1 | 8.5 | 73.6 | 2/13 | 8.4 / 0.1 / 41.7 / 14.9 |
| castle | 9999 | latent critic tau -3.87 | 61.1 | 8.5 | 69.6 | 3/13 | 8.4 / 0.1 / 37.6 / 14.9 |
| castle | 9999 | target only | 62.5 | 8.5 | 71.0 | 0/13 | 0.0 / 0.0 / 49.8 / 12.4 |
