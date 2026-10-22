
**14b 2step <- 14B 4-step: follow ratio (higher = follows the injected 14B context; 1 = on T's same-seed path, 0 = back on its own path), mean over chunks after the injection of d_G/(d_T+d_G) with same-seed latent distances; bold best, underline 2nd; self@3 = re-commit of G's OWN context (numerical floor). s.e. over seeds in the data json.**

| scene (k*_pca / k*_lat) | n seeds | once@1 ↑ | once@3 ↑ | once@6 ↑ | once@9 ↑ | every ↑ | self@3 ↑ |
|---|---|---|---|---|---|---|---|
| sand_filling_up_jar (None / None) | 10 | 0.55 | 0.60 | 0.66 | <u>0.74</u> | **0.76** | 0.33 |
| sandcastle_wave_impact (None / None) | 10 | 0.54 | 0.59 | 0.66 | <u>0.73</u> | **0.74** | 0.31 |

**Envelope distance ↓ (f20 PCA space): |arm seed-mean path - T seed-mean path| / T seed spread, mean over chunks after the injection; in parentheses G-alone over the SAME chunks; < 1 = inside the 14B envelope. Bold = best arm per scene.**

| scene | once@1 ↓ (alone) | once@3 ↓ (alone) | once@6 ↓ (alone) | once@9 ↓ (alone) | every ↓ (alone) | self@3 ↓ (alone) |
|---|---|---|---|---|---|---|
| sand_filling_up_jar | 0.34 (0.51) | 0.25 (0.53) | 0.05 (0.59) | **0.01** (0.60) | <u>0.05</u> (0.49) | 0.53 (0.53) |
| sandcastle_wave_impact | 0.28 (0.27) | 0.18 (0.25) | 0.05 (0.30) | **0.03** (0.48) | <u>0.03</u> (0.30) | 0.25 (0.25) |

**Distance to G-alone's own mean path ↑ (same units; large = the arm left the generator's own path):**

| scene | once@1 | once@3 | once@6 | once@9 | every | self@3 |
|---|---|---|---|---|---|---|
| sand_filling_up_jar | 0.22 | 0.33 | 0.54 | 0.60 | 0.45 | 0.06 |
| sandcastle_wave_impact | 0.20 | 0.16 | 0.26 | 0.47 | 0.29 | 0.03 |

**First 4 chunks after the injection (same-seed latent distances in units of the T seed-to-seed distance at that chunk; mean over seeds):**

| scene | arm | d_T k+1 | d_T k+2 | d_T k+3 | d_T k+4 | d_G k+1 | d_G k+2 | d_G k+3 | d_G k+4 | follow (first 4) |
|---|---|---|---|---|---|---|---|---|---|---|
| sand_filling_up_jar | once@1 | 0.43 | 0.61 | 0.70 | 0.72 | 0.94 | 0.90 | 0.87 | 0.88 | 0.59 |
| sand_filling_up_jar | once@3 | 0.37 | 0.50 | 0.58 | 0.65 | 0.92 | 0.92 | 0.91 | 0.93 | 0.64 |
| sand_filling_up_jar | once@6 | 0.25 | 0.40 | 0.49 | 0.61 | 1.00 | 1.01 | 0.95 | 0.90 | 0.69 |
| sand_filling_up_jar | once@9 | 0.27 | 0.34 | 0.37 | - | 0.91 | 0.92 | 0.95 | - | 0.74 |
| sand_filling_up_jar | every | 0.55 | 0.43 | 0.38 | 0.37 | 0.85 | 0.94 | 0.93 | 0.92 | 0.68 |
| sand_filling_up_jar | self@3 | 0.95 | 0.97 | 0.98 | 1.00 | 0.14 | 0.33 | 0.46 | 0.52 | 0.26 |
| sandcastle_wave_impact | once@1 | 0.44 | 0.59 | 0.67 | 0.72 | 0.85 | 0.85 | 0.80 | 0.81 | 0.58 |
| sandcastle_wave_impact | once@3 | 0.37 | 0.52 | 0.59 | 0.61 | 0.83 | 0.83 | 0.83 | 0.86 | 0.62 |
| sandcastle_wave_impact | once@6 | 0.25 | 0.38 | 0.49 | 0.58 | 0.93 | 0.96 | 0.95 | 0.90 | 0.69 |
| sandcastle_wave_impact | once@9 | 0.27 | 0.35 | 0.40 | - | 0.94 | 0.93 | 0.93 | - | 0.73 |
| sandcastle_wave_impact | every | 0.51 | 0.44 | 0.39 | 0.37 | 0.76 | 0.85 | 0.87 | 0.83 | 0.66 |
| sandcastle_wave_impact | self@3 | 0.87 | 0.89 | 0.91 | 0.95 | 0.13 | 0.25 | 0.34 | 0.41 | 0.23 |

## Same-model control, complete (10/20 23:50 CDT after the reboot): 14B 2-step <- 14B 4-step, 5 scenes x 10 seeds (specwm/inject_probe.py --model 14b --steps 2; figure figs/f21_inject_clean_14b_2step.png)

**14b 2step <- 14B 4-step: follow ratio (higher = follows the injected 14B context; 1 = on T's same-seed path, 0 = back on its own path), mean over chunks after the injection of d_G/(d_T+d_G) with same-seed latent distances; bold best, underline 2nd; self@3 = re-commit of G's OWN context (numerical floor). s.e. over seeds in the data json.**

| scene (k*_pca / k*_lat) | n seeds | once@1 ↑ | once@3 ↑ | once@6 ↑ | once@9 ↑ | every ↑ | self@3 ↑ |
|---|---|---|---|---|---|---|---|
| sand_filling_up_jar (None / None) | 10 | 0.55 | 0.60 | 0.66 | <u>0.74</u> | **0.76** | 0.33 |
| sandcastle_wave_impact (None / None) | 10 | 0.54 | 0.59 | 0.66 | <u>0.73</u> | **0.74** | 0.31 |
| easy_lake (0 / None) | 10 | 0.48 | 0.60 | 0.66 | <u>0.70</u> | **0.73** | 0.27 |
| easy_stonehenge_sunset (3 / None) | 10 | 0.53 | 0.58 | 0.64 | <u>0.71</u> | **0.73** | 0.31 |
| pouring_water_into_cup (None / None) | 10 | 0.52 | 0.57 | 0.61 | <u>0.67</u> | **0.74** | 0.37 |

**Envelope distance ↓ (f20 PCA space): |arm seed-mean path - T seed-mean path| / T seed spread, mean over chunks after the injection; in parentheses G-alone over the SAME chunks; < 1 = inside the 14B envelope. Bold = best arm per scene.**

| scene | once@1 ↓ (alone) | once@3 ↓ (alone) | once@6 ↓ (alone) | once@9 ↓ (alone) | every ↓ (alone) | self@3 ↓ (alone) |
|---|---|---|---|---|---|---|
| sand_filling_up_jar | 0.34 (0.51) | 0.25 (0.53) | 0.05 (0.59) | **0.01** (0.60) | <u>0.05</u> (0.49) | 0.53 (0.53) |
| sandcastle_wave_impact | 0.28 (0.27) | 0.18 (0.25) | 0.05 (0.30) | **0.03** (0.47) | <u>0.03</u> (0.30) | 0.25 (0.25) |
| easy_lake | 0.15 (0.20) | 0.07 (0.22) | 0.07 (0.25) | <u>0.04</u> (0.20) | **0.04** (0.21) | 0.22 (0.22) |
| easy_stonehenge_sunset | 0.38 (0.55) | 0.22 (0.47) | <u>0.09</u> (0.23) | **0.08** (0.29) | 0.14 (0.53) | 0.48 (0.47) |
| pouring_water_into_cup | 0.20 (0.28) | 0.25 (0.30) | 0.19 (0.38) | **0.05** (0.42) | <u>0.07</u> (0.30) | 0.29 (0.30) |

**Distance to G-alone's own mean path ↑ (same units; large = the arm left the generator's own path):**

| scene | once@1 | once@3 | once@6 | once@9 | every | self@3 |
|---|---|---|---|---|---|---|
| sand_filling_up_jar | 0.22 | 0.34 | 0.55 | 0.60 | 0.46 | 0.06 |
| sandcastle_wave_impact | 0.21 | 0.16 | 0.25 | 0.47 | 0.29 | 0.03 |
| easy_lake | 0.18 | 0.15 | 0.20 | 0.17 | 0.19 | 0.03 |
| easy_stonehenge_sunset | 0.28 | 0.39 | 0.20 | 0.22 | 0.39 | 0.04 |
| pouring_water_into_cup | 0.20 | 0.21 | 0.24 | 0.40 | 0.26 | 0.09 |

**First 4 chunks after the injection (same-seed latent distances in units of the T seed-to-seed distance at that chunk; mean over seeds):**

| scene | arm | d_T k+1 | d_T k+2 | d_T k+3 | d_T k+4 | d_G k+1 | d_G k+2 | d_G k+3 | d_G k+4 | follow (first 4) |
|---|---|---|---|---|---|---|---|---|---|---|
| sand_filling_up_jar | once@1 | 0.43 | 0.61 | 0.70 | 0.72 | 0.94 | 0.90 | 0.87 | 0.88 | 0.59 |
| sand_filling_up_jar | once@3 | 0.37 | 0.50 | 0.58 | 0.65 | 0.92 | 0.92 | 0.91 | 0.93 | 0.64 |
| sand_filling_up_jar | once@6 | 0.25 | 0.40 | 0.49 | 0.61 | 1.00 | 1.01 | 0.95 | 0.90 | 0.69 |
| sand_filling_up_jar | once@9 | 0.27 | 0.34 | 0.37 | - | 0.91 | 0.92 | 0.95 | - | 0.74 |
| sand_filling_up_jar | every | 0.55 | 0.43 | 0.38 | 0.37 | 0.85 | 0.94 | 0.93 | 0.92 | 0.68 |
| sand_filling_up_jar | self@3 | 0.95 | 0.97 | 0.98 | 1.00 | 0.14 | 0.33 | 0.46 | 0.52 | 0.26 |
| sandcastle_wave_impact | once@1 | 0.44 | 0.59 | 0.67 | 0.72 | 0.85 | 0.85 | 0.80 | 0.81 | 0.58 |
| sandcastle_wave_impact | once@3 | 0.37 | 0.52 | 0.59 | 0.61 | 0.83 | 0.83 | 0.83 | 0.86 | 0.62 |
| sandcastle_wave_impact | once@6 | 0.25 | 0.38 | 0.49 | 0.58 | 0.93 | 0.96 | 0.95 | 0.90 | 0.69 |
| sandcastle_wave_impact | once@9 | 0.27 | 0.35 | 0.40 | - | 0.94 | 0.93 | 0.93 | - | 0.73 |
| sandcastle_wave_impact | every | 0.51 | 0.44 | 0.39 | 0.37 | 0.76 | 0.85 | 0.87 | 0.83 | 0.66 |
| sandcastle_wave_impact | self@3 | 0.87 | 0.89 | 0.91 | 0.95 | 0.13 | 0.25 | 0.34 | 0.41 | 0.23 |
| easy_lake | once@1 | 0.44 | 0.61 | 0.75 | 0.80 | 0.66 | 0.63 | 0.65 | 0.66 | 0.51 |
| easy_lake | once@3 | 0.32 | 0.43 | 0.51 | 0.59 | 0.78 | 0.80 | 0.83 | 0.87 | 0.64 |
| easy_lake | once@6 | 0.21 | 0.30 | 0.40 | 0.55 | 0.89 | 0.91 | 0.85 | 0.81 | 0.71 |
| easy_lake | once@9 | 0.31 | 0.36 | 0.37 | - | 0.84 | 0.83 | 0.85 | - | 0.70 |
| easy_lake | every | 0.48 | 0.44 | 0.37 | 0.32 | 0.63 | 0.66 | 0.71 | 0.78 | 0.63 |
| easy_lake | self@3 | 0.82 | 0.84 | 0.86 | 0.91 | 0.09 | 0.20 | 0.30 | 0.36 | 0.21 |
| easy_stonehenge_sunset | once@1 | 0.46 | 0.72 | 0.76 | 0.77 | 0.80 | 0.82 | 0.78 | 0.80 | 0.54 |
| easy_stonehenge_sunset | once@3 | 0.42 | 0.55 | 0.63 | 0.69 | 0.88 | 0.88 | 0.93 | 0.94 | 0.62 |
| easy_stonehenge_sunset | once@6 | 0.23 | 0.35 | 0.51 | 0.68 | 0.93 | 0.92 | 0.90 | 0.92 | 0.68 |
| easy_stonehenge_sunset | once@9 | 0.35 | 0.41 | 0.42 | - | 0.95 | 0.94 | 0.95 | - | 0.71 |
| easy_stonehenge_sunset | every | 0.48 | 0.46 | 0.48 | 0.42 | 0.69 | 0.80 | 0.87 | 0.88 | 0.63 |
| easy_stonehenge_sunset | self@3 | 0.93 | 0.93 | 0.95 | 0.94 | 0.12 | 0.23 | 0.35 | 0.40 | 0.22 |
| pouring_water_into_cup | once@1 | 0.44 | 0.68 | 0.84 | 0.86 | 0.83 | 0.89 | 0.90 | 0.92 | 0.56 |
| pouring_water_into_cup | once@3 | 0.35 | 0.53 | 0.62 | 0.68 | 0.90 | 0.92 | 0.94 | 0.92 | 0.63 |
| pouring_water_into_cup | once@6 | 0.22 | 0.43 | 0.65 | 0.77 | 0.93 | 0.94 | 0.94 | 0.90 | 0.66 |
| pouring_water_into_cup | once@9 | 0.34 | 0.49 | 0.59 | - | 0.93 | 0.94 | 0.94 | - | 0.67 |
| pouring_water_into_cup | every | 0.51 | 0.44 | 0.37 | 0.35 | 0.80 | 0.83 | 0.90 | 0.90 | 0.67 |
| pouring_water_into_cup | self@3 | 0.91 | 0.94 | 0.97 | 0.95 | 0.19 | 0.40 | 0.55 | 0.64 | 0.31 |
