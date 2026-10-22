# step3_lingbot_latent_fixed

**Fixed 10/5:** the drift critic (higher = more drift) is accepted on logit <= tau. The old step3 folders accepted logit >= tau (inverted).

Plan: T = chunk made by the 14B, D = draft chunk accepted (chunk 0 always T).

| scene | seed | arm | latent gen ↓ (s) | + decode (s) | total ↓ (s) | accepted | plan |
|---|---|---|---|---|---|---|---|
| jar | 1234 | drafter only | 11.5 | 8.6 | 20.1 | 13/13 | DDDDDDDDDDDDD |
| jar | 1234 | latent critic tau 0.05 | 31.4 | 8.7 | 40.0 | 11/13 | TDDDDDDDDDDDT |
| jar | 1234 | latent critic tau -3.87 | 43.3 | 8.7 | 52.0 | 8/13 | TDTDDDDDDTDTT |
| jar | 1234 | target only | 64.4 | 8.6 | 73.0 | 0/13 | TTTTTTTTTTTTT |
| jar | 4321 | drafter only | 11.3 | 8.6 | 19.9 | 13/13 | DDDDDDDDDDDDD |
| jar | 4321 | latent critic tau 0.05 | 27.1 | 8.7 | 35.8 | 12/13 | TDDDDDDDDDDDD |
| jar | 4321 | latent critic tau -3.87 | 43.3 | 8.7 | 52.0 | 8/13 | TDTDDDDDDTDTT |
| jar | 4321 | target only | 64.4 | 8.6 | 73.0 | 0/13 | TTTTTTTTTTTTT |
| jar | 9999 | drafter only | 11.4 | 8.6 | 20.0 | 13/13 | DDDDDDDDDDDDD |
| jar | 9999 | latent critic tau 0.05 | 35.5 | 8.7 | 44.2 | 10/13 | TDDDDDDDDDDTT |
| jar | 9999 | latent critic tau -3.87 | 46.6 | 8.7 | 55.3 | 7/13 | TTTDDDDDTDDTT |
| jar | 9999 | target only | 64.4 | 8.6 | 73.0 | 0/13 | TTTTTTTTTTTTT |
| castle | 1234 | drafter only | 11.2 | 8.5 | 19.7 | 13/13 | DDDDDDDDDDDDD |
| castle | 1234 | latent critic tau 0.05 | 42.9 | 8.5 | 51.4 | 8/13 | TDDDDDDDTTDTT |
| castle | 1234 | latent critic tau -3.87 | 55.0 | 8.5 | 63.5 | 5/13 | TDDDDDTTTTTTT |
| castle | 1234 | target only | 62.5 | 8.5 | 71.0 | 0/13 | TTTTTTTTTTTTT |
| castle | 4321 | drafter only | 11.2 | 8.5 | 19.7 | 13/13 | DDDDDDDDDDDDD |
| castle | 4321 | latent critic tau 0.05 | 34.5 | 8.5 | 43.0 | 10/13 | TDDDDDDDDDDTT |
| castle | 4321 | latent critic tau -3.87 | 55.0 | 8.5 | 63.5 | 5/13 | TDDDDTDTTTTTT |
| castle | 4321 | target only | 62.5 | 8.5 | 71.0 | 0/13 | TTTTTTTTTTTTT |
| castle | 9999 | drafter only | 11.2 | 8.5 | 19.7 | 13/13 | DDDDDDDDDDDDD |
| castle | 9999 | latent critic tau 0.05 | 30.5 | 8.5 | 39.0 | 11/13 | TDDDDDDDDDDDT |
| castle | 9999 | latent critic tau -3.87 | 34.5 | 8.5 | 43.0 | 10/13 | TDDDDDDDDDDTT |
| castle | 9999 | target only | 62.5 | 8.5 | 71.0 | 0/13 | TTTTTTTTTTTTT |
