
**Table A - following after the re-anchor (chunks 7-12), 14B 2-step device <- 14B 4-step cloud, k = 6. follow ratio ↑ = mean of d_G/(d_T+d_G), same-seed latent distances (1 = on the cloud's same-seed path, 0 = on the device's own path); envelope distance ↓ = |arm seed-mean path - cloud seed-mean path| / cloud seed spread (PCA; < 1 = inside the cloud's seed spread); d_T ↓ = same-seed latent distance to the cloud clip (1 = two independent cloud seeds). Mean over seeds (follow: ± s.e.). Bold best / underline 2nd among the five re-anchor arms; alone = alpha 0, every / self@3 = ceiling / floor for reference.**

| scene | n | metric | alpha 0 (alone) | alpha 0.25 | alpha 0.5 | alpha 0.75 | alpha 1 (once@6) | progressive 0.25/0.5/1 | every | self@3 |
|---|---|---|---|---|---|---|---|---|---|---|
| sandcastle | 10 | follow ↑ | 0.00 | 0.36 ± 0.01 | 0.49 ± 0.01 | <u>0.61</u> ± 0.01 | **0.66** ± 0.01 | 0.59 ± 0.01 | 0.79 | 0.36 |
| sandcastle | 10 | envelope ↓ | 0.30 | 0.28 | 0.14 | 0.10 | **0.05** | <u>0.07</u> | 0.01 | 0.29 |
| sandcastle | 10 | d_T ↓ | 0.96 | 0.89 | 0.73 | 0.56 | **0.49** | <u>0.51</u> | 0.25 | 0.96 |
| stonehenge | 10 | follow ↑ | 0.00 | 0.34 ± 0.01 | 0.49 ± 0.01 | <u>0.61</u> ± 0.01 | **0.64** ± 0.01 | 0.56 ± 0.01 | 0.78 | 0.36 |
| stonehenge | 10 | envelope ↓ | 0.23 | 0.18 | 0.26 | 0.19 | **0.09** | <u>0.15</u> | 0.06 | 0.24 |
| stonehenge | 10 | d_T ↓ | 0.96 | 0.90 | 0.75 | 0.57 | **0.53** | <u>0.57</u> | 0.27 | 0.96 |
| jar | 10 | follow ↑ | 0.00 | 0.36 ± 0.01 | 0.48 ± 0.01 | <u>0.60</u> ± 0.01 | **0.66** ± 0.01 | 0.59 ± 0.01 | 0.80 | 0.37 |
| jar | 10 | envelope ↓ | 0.59 | 0.52 | 0.36 | 0.14 | **0.05** | <u>0.13</u> | 0.03 | 0.58 |
| jar | 10 | d_T ↓ | 0.98 | 0.88 | 0.78 | 0.59 | **0.50** | <u>0.54</u> | 0.23 | 0.99 |
| mean of 3 scenes | | follow ↑ | 0.00 | 0.35 | 0.48 | <u>0.60</u> | **0.65** | 0.58 | 0.79 | 0.36 |
| mean of 3 scenes | | envelope ↓ | 0.37 | 0.32 | 0.25 | 0.15 | **0.06** | <u>0.12</u> | 0.03 | 0.37 |

**Table A2 - same-seed distance to the cloud clip d_T ↓ in the first three chunks after the re-anchor (7 / 8 / 9; 1 = independent cloud seeds), mean over seeds:**

| scene | alpha 0.25 | alpha 0.5 | alpha 0.75 | alpha 1 (once@6) | progressive 0.25/0.5/1 |
|---|---|---|---|---|---|
| sandcastle | 0.79 / 0.86 / 0.89 | 0.58 / 0.67 / 0.73 | 0.34 / 0.45 / 0.55 | 0.25 / 0.38 / 0.49 | 0.79 / 0.52 / 0.29 |
| stonehenge | 0.80 / 0.85 / 0.87 | 0.59 / 0.65 / 0.70 | 0.32 / 0.42 / 0.55 | 0.23 / 0.35 / 0.51 | 0.80 / 0.49 / 0.36 |
| jar | 0.83 / 0.89 / 0.90 | 0.60 / 0.69 / 0.77 | 0.36 / 0.48 / 0.59 | 0.25 / 0.40 / 0.49 | 0.83 / 0.53 / 0.31 |

**Table B - re-anchor boundary jump ↓ at chunk boundary 6|7 (device view = the device's own chunks 0-6 followed by the arm's chunks 7-12, decoded jointly; frame-to-frame pixel L1 on 8-bit, pixel frames 80|81). ordinary = mean of the device-alone and the cloud clip at the same boundary; ratio = splice / ordinary (1 = an ordinary chunk boundary; the 1.3B hard re-anchor was 2.55x). n = seeds. Bold best / underline 2nd.**

| scene | n | ordinary L1 (alone / cloud) | alpha 0.25: L1, ratio | alpha 0.5: L1, ratio | alpha 0.75: L1, ratio | alpha 1 (once@6): L1, ratio | progressive 0.25/0.5/1: L1, ratio | seeds ratio > 1.5 |
|---|---|---|---|---|---|---|---|---|
| sandcastle | 10 | 20.8 (21.7 / 19.9) | 21.7 ± 1.4, **1.06** | 26.3 ± 2.1, 1.28 | 33.8 ± 3.1, 1.64 | 42.8 ± 4.2, 2.08 | 21.7 ± 1.4, **1.06** | 0 / 2 / 6 / 8 / 0 |
| stonehenge | 10 | 25.5 (25.0 / 26.0) | 25.4 ± 1.1, **1.00** | 27.4 ± 1.1, 1.08 | 31.5 ± 1.4, 1.24 | 36.2 ± 1.7, 1.43 | 25.4 ± 1.1, **1.00** | 0 / 0 / 1 / 1 / 0 |
| jar | 10 | 21.4 (20.9 / 22.0) | 21.1 ± 1.2, **0.99** | 24.0 ± 1.5, 1.14 | 29.1 ± 2.0, 1.38 | 35.0 ± 2.5, 1.67 | 21.1 ± 1.2, **0.99** | 0 / 1 / 4 / 4 / 0 |
| pooled | 30 | 22.6 (22.5 / 22.6) | 22.7 ± 0.8, **1.02** | 25.9 ± 0.9, 1.17 | 31.5 ± 1.3, 1.42 | 38.0 ± 1.8, 1.73 | 22.7 ± 0.8, **1.02** | 0 / 3 / 11 / 13 / 0 |

Progressive arm, later re-blend boundaries of the device view (L1, ratio vs ordinary): sandcastle: 7|8 26.4 (1.30x) 8|9 32.1 (1.18x); stonehenge: 7|8 27.1 (1.07x) 8|9 30.1 (1.01x); jar: 7|8 23.3 (1.07x) 8|9 27.9 (1.10x); pooled: 7|8 25.6 (1.15x) 8|9 30.0 (1.10x)
Own boundary of the un-spliced clips (blended context as stored, then the continuation; L1 at 6|7, pooled): alpha 0.25 20.7, alpha 0.5 19.4, alpha 0.75 20.2, alpha 1 (once@6) 23.0, progressive 0.25/0.5/1 20.7

**Table C - judge (Qwen3-VL-32B, 8 frames per clip) on the DEVICE-VIEW clips (own chunks 0-6 + re-anchored continuation): quality 1-10 ↑, drift 0-100 ↓; mean ± s.e. over seeds; T = the cloud clip itself. Bold best / underline 2nd among the re-anchor arms.**

| quality ↑ | T (cloud) | alpha 0 (alone) | alpha 0.25 | alpha 0.5 | alpha 0.75 | alpha 1 (once@6) | progressive 0.25/0.5/1 |
|---|---|---|---|---|---|---|---|
| sandcastle | 6.90 ± 0.10 | 7.00 ± 0.00 | **7.00** ± 0.00 (n=10) | **7.00** ± 0.00 (n=10) | **7.00** ± 0.00 (n=10) | **7.00** ± 0.00 (n=10) | 6.80 ± 0.13 (n=10) |
| stonehenge | 7.60 ± 0.16 | 7.10 ± 0.10 | 6.90 ± 0.10 (n=10) | 6.70 ± 0.15 (n=10) | <u>7.10</u> ± 0.10 (n=10) | **7.40** ± 0.16 (n=10) | 7.00 ± 0.15 (n=10) |
| jar | 8.30 ± 0.21 | 8.10 ± 0.23 | 7.80 ± 0.13 (n=10) | 7.50 ± 0.22 (n=10) | 7.70 ± 0.21 (n=10) | **8.00** ± 0.21 (n=10) | <u>7.90</u> ± 0.23 (n=10) |
| pooled | 7.62 ± 0.10 | 7.40 ± 0.12 | 7.23 ± 0.09 (n=30) | 7.07 ± 0.11 (n=30) | <u>7.27</u> ± 0.10 (n=30) | **7.47** ± 0.11 (n=30) | 7.23 ± 0.13 (n=30) |

| drift ↓ | T (cloud) | alpha 0 (alone) | alpha 0.25 | alpha 0.5 | alpha 0.75 | alpha 1 (once@6) | progressive 0.25/0.5/1 |
|---|---|---|---|---|---|---|---|
| sandcastle | 76.00 ± 1.00 | 70.50 ± 4.50 | **71.50** ± 4.72 (n=10) | 75.00 ± 0.00 (n=10) | **71.50** ± 4.72 (n=10) | 77.00 ± 1.33 (n=10) | 76.00 ± 1.00 (n=10) |
| stonehenge | 9.50 ± 7.54 | 12.00 ± 3.27 | 14.00 ± 3.06 (n=10) | 40.00 ± 9.72 (n=10) | **10.00** ± 3.33 (n=10) | <u>12.00</u> ± 3.27 (n=10) | 17.00 ± 3.00 (n=10) |
| jar | 30.00 ± 8.60 | 36.50 ± 8.40 | 36.50 ± 8.40 (n=10) | 49.50 ± 9.26 (n=10) | 41.50 ± 8.98 (n=10) | **32.00** ± 7.23 (n=10) | **32.00** ± 7.23 (n=10) |
| pooled | 46.00 ± 4.45 | 39.67 ± 5.51 | <u>40.67</u> ± 5.47 (n=30) | 54.83 ± 5.12 (n=30) | 41.00 ± 5.79 (n=30) | **40.33** ± 5.67 (n=30) | 41.67 ± 5.30 (n=30) |

Paired vs alpha 0 (alone), pooled over the 3 scenes, mean ± s.e. (better / worse / tie in the metric's own direction): alpha 0.25: quality -0.17 ± 0.08 (1/6/23), drift +1.00 ± 2.74 (1/3/26); alpha 0.5: quality -0.33 ± 0.10 (0/9/21), drift +15.17 ± 5.80 (2/10/18); alpha 0.75: quality -0.13 ± 0.09 (1/4/25), drift +1.33 ± 4.20 (3/4/23); alpha 1 (once@6): quality +0.07 ± 0.10 (5/3/22), drift +0.67 ± 3.62 (2/5/23); progressive 0.25/0.5/1: quality -0.17 ± 0.12 (2/5/23), drift +2.00 ± 3.73 (2/7/21)
