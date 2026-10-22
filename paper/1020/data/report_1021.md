# Frozen baseline table (generated 2026-10-21 17:14 CDT by baselines/report.py)

runs: `/home/hk28965/specwm/baselines/runs` (201 arms, 8699 clips); metrics: `/home/hk28965/specwm/baselines/metrics, /home/hk28965/specwm/baselines/metrics_g7, /home/hk28965/specwm/baselines/metrics_g0, /home/hk28965/specwm/baselines/metrics_g1, /home/hk28965/specwm/baselines/metrics_g2, /home/hk28965/specwm/baselines/metrics_g3` (judge 8531, VBench 8531, RPE 8531, reaction 1133, luminance 7119 clips with values).

Columns: GPUs = GPUs the arm runs on (timing.json `gpus`, default 1); fps gen = steady DiT-only fps, fps e2e = steady fps incl. decode, gen s = DiT-only generation time of the 10-s clip, TTFF = input to first decoded frame (s), VRAM = peak allocated GiB (timing.json, PROTOCOL section 4; '(dev)' = device GPU only for 2-GPU hybrid arms); judge Q (1-10) / drift (0-100) = Qwen3-VL clip judge on 8 evenly spaced frames; VBench Q = 6-dim quality score (0-100, clips resampled to 16 fps), imaging = MUSIQ (0-100), MUSIQ drift = std of the 2 s segment means; RPE Rot = mean relative rotation error (deg per 1/8 s step) of DA3-recovered poses vs the commanded program (Sim(3)-aligned per fifth); react = P5 reaction time from the model's effective input switch until the optical-flow sign follows (s / frames at native fps, mean over valid switches); black = number of the row's clips that go black (last-second mean luminance of the 16x9 gray downscale < 10 or < 30 % of the first second; metrics/luminance.json). set = <scene set><suffix> of the clip names (suffix _nocap = empty-prompt arm, _long = 60-s P6 clips: each ranked and paired only within its own suffix). n = clips in the row; '(partial)' = fewer than the frozen task list tasks/<set>.json expects (generation or scoring still running). Arrows give the direction; per scene set, **bold** = best arm, <u>underline</u> = second among the main rows; '(k)' = metric present on k of n clips. Means over clips. Extra rows (split by scene type, P2p+P4p only, excl. black) are shown in italics and are never ranked.

## 1. Per model arm x scene set (one table per set, footnotes below each)

### set `abot` (n expected 60 per arm = 5 programs P1-P5)

| model_arm | set | n | res | GPUs | fps gen ↑ | fps e2e ↑ | gen s ↓ | TTFF s ↓ | VRAM GiB ↓ | judge Q ↑ | judge drift ↓ | VBench Q ↑ | imaging ↑ | MUSIQ drift ↓ | RPE Rot ° ↓ | react s ↓ | react fr ↓ | black |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| abot | abot | 60 | 1280x704 | 1 | **15.4** | **12.6** | **7.9** | **0.95** | <u>32.1</u> | 8.03 | 26.5 | 80.50 | 71.5 | 3.08 | 0.543 | 0.46 (11) | 5.5 (11) | 0 |
| *abot (_1st scenes only, n=35)* | abot | 35 | 1280x704 | 1 | 15.5 | 12.7 | 7.9 | 0.94 | 32.1 | 7.94 | 27.0 | 78.65 | 70.1 | 3.78 | 0.538 | 0.32 (6) | 3.9 (6) | 0 |
| *abot (_3rd scenes, n=25)* | abot | 25 | 1280x704 | 1 | 15.4 | 12.4 | 7.9 | 0.96 | 32.1 | 8.16 | 25.8 | 83.11 | 73.6 | 2.11 | 0.550 | 0.62 (5) | 7.5 (5) | 0 |
| abot_t5offload | abot | 15 (partial) | 1280x704 | 1 | **15.4** | <u>12.5</u> | <u>8.0</u> | <u>0.97</u> | **22.5** | 8.07 | 22.7 | 80.23 | 72.5 | 3.17 | 0.579 | 0.49 (3) | 5.9 (3) | 0 |
| *abot_t5offload (_1st scenes only, n=5)* | abot | 5 (partial) | 1280x704 | 1 | 15.3 | 12.4 | 8.0 | 1.01 | 22.5 | 7.40 | 16.0 | 75.42 | 67.8 | 6.29 | 0.623 | 0.36 (1) | 4.3 (1) | 0 |
| *abot_t5offload (_3rd scenes, n=10)* | abot | 10 (partial) | 1280x704 | 1 | 15.4 | 12.6 | 7.9 | 0.95 | 22.5 | 8.40 | 26.0 | 82.64 | 74.8 | 1.61 | 0.557 | 0.56 (2) | 6.7 (2) | 0 |
| ours_adapt25_abot | abot | 60 | 1280x704 | 1 | 3.6 | 2.5 | 47.0 | 17.37 | 85.1 | 8.75 | 13.0 | 82.96 | 74.9 | 1.57 | 0.729 | 0.34 (11) | 5.4 (11) | 0 |
| *ours_adapt25_abot (_1st scenes only, n=35)* | abot | 35 | 1280x704 | 1 | 3.6 | 2.5 | 47.1 | 17.31 | 85.1 | 8.77 | 12.0 | 81.71 | 74.2 | 1.79 | 0.732 | 0.15 (6) | 2.4 (6) | 0 |
| *ours_adapt25_abot (_3rd scenes, n=25)* | abot | 25 | 1280x704 | 1 | 3.6 | 2.5 | 46.8 | 17.45 | 85.1 | 8.72 | 14.4 | 84.71 | 75.8 | 1.26 | 0.726 | 0.56 (5) | 9.0 (5) | 0 |
| *ours_adapt25_abot (P2p+P4p only, n=24)* | abot | 24 | 1280x704 | 1 | 3.7 | 2.5 | 47.4 | 17.20 | 85.1 | 8.62 | 24.8 | 82.38 | 74.7 | 1.93 | 1.070 | - | - | 0 |
| ours_adapt25c1hyb_cloud_abot | abot | 60 | 1280x704 | 2 | 2.3 | - | 81.2 | - | 84.3 (dev) | 8.85 | 12.2 | 83.56 | <u>75.0</u> | 1.61 | 0.564 | **0.09 (12)** | <u>1.5 (12)</u> | 0 |
| *ours_adapt25c1hyb_cloud_abot (_1st scenes only, n=35)* | abot | 35 | 1280x704 | 2 | 2.3 | - | 83.0 | - | 84.3 (dev) | 8.91 | 8.6 | 82.59 | 74.6 | 1.77 | 0.550 | 0.09 (7) | 1.5 (7) | 0 |
| *ours_adapt25c1hyb_cloud_abot (_3rd scenes, n=25)* | abot | 25 | 1280x704 | 2 | 2.4 | - | 78.7 | - | 84.3 (dev) | 8.76 | 17.4 | 84.93 | 75.6 | 1.38 | 0.583 | 0.09 (5) | 1.5 (5) | 0 |
| *ours_adapt25c1hyb_cloud_abot (P2p+P4p only, n=24)* | abot | 24 | 1280x704 | 2 | 2.3 | - | 83.3 | - | 84.3 (dev) | 8.75 | 16.2 | 82.92 | 75.0 | 1.63 | 0.803 | - | - | 0 |
| ours_adapt25c1hyb_dev_abot | abot | 60 | 1280x704 | 2 | 2.3 | - | 79.3 | - | 84.6 (dev) | 8.80 | 12.8 | 83.62 | <u>75.0</u> | 1.57 | 0.563 | <u>0.10 (11)</u> | 1.6 (11) | 0 |
| *ours_adapt25c1hyb_dev_abot (_1st scenes only, n=35)* | abot | 35 | 1280x704 | 2 | 2.3 | - | 81.0 | - | 84.6 (dev) | 8.86 | 9.3 | 82.63 | 74.5 | 1.75 | 0.549 | 0.11 (6) | 1.8 (6) | 0 |
| *ours_adapt25c1hyb_dev_abot (_3rd scenes, n=25)* | abot | 25 | 1280x704 | 2 | 2.4 | - | 76.8 | - | 84.6 (dev) | 8.72 | 17.8 | 85.02 | 75.7 | 1.31 | 0.582 | 0.09 (5) | 1.4 (5) | 0 |
| *ours_adapt25c1hyb_dev_abot (P2p+P4p only, n=24)* | abot | 24 | 1280x704 | 2 | 2.3 | - | 81.4 | - | 84.6 (dev) | 8.71 | 20.0 | 82.95 | 75.0 | 1.63 | 0.804 | - | - | 0 |
| ours_rt2_abot | abot | 60 | 1280x704 | 1 | 3.7 | 2.5 | 43.4 | 16.50 | 84.8 | 8.78 | <u>10.9</u> | 82.98 | 74.3 | 1.58 | 0.752 | 0.27 (12) | 4.4 (12) | 0 |
| *ours_rt2_abot (_1st scenes only, n=35)* | abot | 35 | 1280x704 | 1 | 3.7 | 2.5 | 43.5 | 16.44 | 84.8 | 8.77 | 9.6 | 82.24 | 73.5 | 1.86 | 0.732 | 0.08 (7) | 1.2 (7) | 0 |
| *ours_rt2_abot (_3rd scenes, n=25)* | abot | 25 | 1280x704 | 1 | 3.7 | 2.5 | 43.3 | 16.58 | 84.8 | 8.80 | 12.8 | 84.01 | 75.5 | 1.20 | 0.781 | 0.55 (5) | 8.8 (5) | 0 |
| *ours_rt2_abot (P2p+P4p only, n=24)* | abot | 24 | 1280x704 | 1 | 3.7 | 2.5 | 43.5 | 16.38 | 84.8 | 8.79 | 17.7 | 81.24 | 74.7 | 1.65 | 1.073 | - | - | 0 |
| ours_rt2c1hyb_cloud_abot | abot | 60 | 1280x704 | 2 | 2.6 | - | 60.6 | - | 84.3 (dev) | 8.87 | 12.8 | 83.78 | 74.6 | 1.46 | 0.586 | **0.09 (12)** | **1.4 (12)** | - |
| *ours_rt2c1hyb_cloud_abot (_1st scenes only, n=35)* | abot | 35 | 1280x704 | 2 | 2.6 | - | 60.6 | - | 84.3 (dev) | 8.89 | 9.0 | 82.75 | 73.8 | 1.65 | 0.581 | 0.09 (7) | 1.4 (7) | - |
| *ours_rt2c1hyb_cloud_abot (_3rd scenes, n=25)* | abot | 25 | 1280x704 | 2 | 2.6 | - | 60.7 | - | 84.3 (dev) | 8.84 | 18.0 | 85.22 | 75.9 | 1.20 | 0.593 | 0.09 (5) | 1.4 (5) | - |
| *ours_rt2c1hyb_cloud_abot (P2p+P4p only, n=24)* | abot | 24 | 1280x704 | 2 | 2.6 | - | 60.6 | - | 84.3 (dev) | 8.75 | 16.5 | 82.87 | 74.9 | 1.64 | 0.810 | - | - | - |
| ours_rt2c1hyb_dev_abot | abot | 60 | 1280x704 | 2 | 2.6 | - | 58.7 | - | 84.3 (dev) | <u>8.88</u> | 13.2 | <u>83.81</u> | 74.5 | 1.48 | 0.582 | **0.09 (12)** | **1.4 (12)** | - |
| *ours_rt2c1hyb_dev_abot (_1st scenes only, n=35)* | abot | 35 | 1280x704 | 2 | 2.6 | - | 58.6 | - | 84.3 (dev) | 8.91 | 9.3 | 82.78 | 73.6 | 1.63 | 0.577 | 0.09 (7) | 1.4 (7) | - |
| *ours_rt2c1hyb_dev_abot (_3rd scenes, n=25)* | abot | 25 | 1280x704 | 2 | 2.6 | - | 58.8 | - | 84.3 (dev) | 8.84 | 18.6 | 85.26 | 75.8 | 1.27 | 0.590 | 0.09 (5) | 1.4 (5) | - |
| *ours_rt2c1hyb_dev_abot (P2p+P4p only, n=24)* | abot | 24 | 1280x704 | 2 | 2.6 | - | 58.7 | - | 84.3 (dev) | 8.71 | 18.1 | 82.91 | 74.7 | 1.65 | 0.808 | - | - | - |
| ours_sf24_abot | abot | 60 | 1280x704 | 1 | 2.6 | 2.4 | 62.5 | 23.54 | - | 8.85 | 13.9 | 83.58 | 74.5 | <u>1.45</u> | 0.533 | **0.09 (12)** | <u>1.5 (12)</u> | 0 |
| *ours_sf24_abot (_1st scenes only, n=35)* | abot | 35 | 1280x704 | 1 | 2.7 | 2.4 | 61.7 | 23.36 | - | 8.91 | 12.4 | 82.69 | 74.0 | 1.65 | 0.530 | 0.11 (7) | 1.8 (7) | 0 |
| *ours_sf24_abot (_3rd scenes, n=25)* | abot | 25 | 1280x704 | 1 | 2.5 | 2.3 | 63.8 | 23.79 | - | 8.76 | 16.0 | 84.83 | 75.3 | 1.16 | 0.538 | 0.07 (5) | 1.1 (5) | 0 |
| *ours_sf24_abot (P2p+P4p only, n=24)* | abot | 24 | 1280x704 | 1 | 2.5 | 2.3 | 64.6 | 23.54 | - | 8.75 | 20.8 | 83.06 | 74.6 | 1.70 | 0.765 | - | - | 0 |
| ours_sf24_abot_dit | abot | 2 (partial) | 1280x704 | 1 | 3.4 | - | 48.9 | - | - | **9.00** | **10.0** | **84.56** | **76.3** | **0.71** | **0.334** | - | - | - |
| *ours_sf24_abot_dit (_3rd scenes, n=2)* | abot | 2 (partial) | 1280x704 | 1 | 3.4 | - | 48.9 | - | - | 9.00 | 10.0 | 84.56 | 76.3 | 0.71 | 0.334 | - | - | - |
| ours_sf24c1hyb_cloud_abot | abot | 60 | 1280x704 | 2 | 2.4 | - | 67.9 | - | 84.3 (dev) | 8.82 | 13.4 | 83.53 | 74.8 | 1.49 | 0.532 | **0.09 (12)** | <u>1.5 (12)</u> | 0 |
| *ours_sf24c1hyb_cloud_abot (_1st scenes only, n=35)* | abot | 35 | 1280x704 | 2 | 2.4 | - | 66.8 | - | 84.3 (dev) | 8.89 | 11.7 | 82.50 | 74.3 | 1.65 | 0.518 | 0.10 (7) | 1.6 (7) | 0 |
| *ours_sf24c1hyb_cloud_abot (_3rd scenes, n=25)* | abot | 25 | 1280x704 | 2 | 2.3 | - | 69.3 | - | 84.3 (dev) | 8.72 | 15.8 | 84.97 | 75.7 | 1.27 | 0.550 | 0.09 (5) | 1.4 (5) | 0 |
| *ours_sf24c1hyb_cloud_abot (P2p+P4p only, n=24)* | abot | 24 | 1280x704 | 2 | 2.3 | - | 69.9 | - | 84.3 (dev) | 8.71 | 21.5 | 82.97 | 75.0 | 1.83 | 0.754 | - | - | 0 |
| ours_sf24c1hyb_dev_abot | abot | 60 | 1280x704 | 2 | 2.4 | - | 65.9 | - | 84.4 (dev) | 8.78 | 13.2 | 83.54 | 74.8 | 1.46 | <u>0.527</u> | 0.18 (12) | 2.8 (12) | 0 |
| *ours_sf24c1hyb_dev_abot (_1st scenes only, n=35)* | abot | 35 | 1280x704 | 2 | 2.4 | - | 64.9 | - | 84.4 (dev) | 8.91 | 12.4 | 82.56 | 74.2 | 1.60 | 0.514 | 0.26 (7) | 4.1 (7) | 0 |
| *ours_sf24c1hyb_dev_abot (_3rd scenes, n=25)* | abot | 25 | 1280x704 | 2 | 2.3 | - | 67.4 | - | 84.4 (dev) | 8.60 | 14.4 | 84.92 | 75.6 | 1.26 | 0.545 | 0.07 (5) | 1.1 (5) | 0 |
| *ours_sf24c1hyb_dev_abot (P2p+P4p only, n=24)* | abot | 24 | 1280x704 | 2 | 2.3 | - | 68.0 | - | 84.4 (dev) | 8.75 | 19.6 | 83.03 | 74.9 | 1.90 | 0.753 | - | - | 0 |
| ours_student1p3b_abot | abot | 60 | 1280x704 | 1 | <u>4.5</u> | 3.7 | 36.8 | 21.03 | - | 8.53 | 24.2 | 82.07 | 69.5 | 3.60 | 0.671 | 0.11 (11) | 1.8 (11) | 3 |
| *ours_student1p3b_abot (_1st scenes only, n=35)* | abot | 35 | 1280x704 | 1 | 4.5 | 3.7 | 36.8 | 21.00 | - | 8.46 | 28.4 | 81.27 | 68.8 | 4.06 | 0.656 | 0.12 (6) | 1.9 (6) | 3 |
| *ours_student1p3b_abot (_3rd scenes, n=25)* | abot | 25 | 1280x704 | 1 | 4.5 | 3.6 | 36.8 | 21.07 | - | 8.64 | 18.2 | 83.19 | 70.3 | 2.95 | 0.691 | 0.10 (5) | 1.7 (5) | 0 |
| *ours_student1p3b_abot (P2p+P4p only, n=24)* | abot | 24 | 1280x704 | 1 | 4.5 | 3.7 | 36.3 | 20.58 | - | 8.46 | 20.4 | 81.75 | 70.5 | 3.58 | 1.129 | - | - | 2 |
| *ours_student1p3b_abot excl. black (n=57)* | abot | 57 | 1280x704 | 1 | 4.5 | 3.6 | 37.0 | 21.10 | - | 8.58 | 23.8 | 82.33 | 69.9 | 3.25 | 0.633 | 0.11 (11) | 1.8 (11) | 0 |
| ours_teacher_abot | abot | 60 | 1280x704 | 1 | 2.0 | 1.6 | 81.5 | 18.53 | 84.8 | 8.83 | 13.7 | 82.86 | 74.6 | 1.73 | 0.727 | 0.12 (11) | 2.0 (11) | 0 |
| *ours_teacher_abot (_1st scenes only, n=35)* | abot | 35 | 1280x704 | 1 | 2.0 | 1.6 | 79.2 | 18.16 | 84.8 | 8.86 | 11.0 | 81.77 | 73.7 | 1.97 | 0.699 | 0.14 (7) | 2.2 (7) | 0 |
| *ours_teacher_abot (_3rd scenes, n=25)* | abot | 25 | 1280x704 | 1 | 1.9 | 1.5 | 84.7 | 19.05 | 84.8 | 8.80 | 17.4 | 84.37 | 75.8 | 1.40 | 0.767 | 0.10 (4) | 1.6 (4) | 0 |
| *ours_teacher_abot (P2p+P4p only, n=24)* | abot | 24 | 1280x704 | 1 | 2.0 | 1.6 | 79.9 | 18.80 | 84.8 | 8.58 | 18.1 | 82.30 | 74.7 | 2.00 | 1.040 | - | - | 0 |

- Set means use programs P1-P5 only for EVERY arm (ABot has no key magnitude, so its P2p = P2 and P4p = P4 videos are identical); ours' P2p/P4p clips are listed in separate '(P2p+P4p only)' rows and are not ranked.
- 5 of the 12 ABot scenes are third-person (`_3rd`: aurora_ice, case4_desert_caravan, desert_camel, desert_valley, ruins_explorer), where WASD moves the character and not the camera: RPE Rot / reaction on them is not a camera comparison; see the '_1st scenes only' (7 scenes) and '_3rd scenes' (5 scenes) split rows (not ranked).
- `abot` (24 clips): P2p/P4p clips exist but are byte-identical to the P2/P4 clips (no key magnitude) and are not counted.
- `ours_adapt25c1hyb_cloud_abot`, `ours_adapt25c1hyb_dev_abot`, `ours_rt2c1hyb_cloud_abot`, `ours_rt2c1hyb_dev_abot`, `ours_sf24c1hyb_cloud_abot`, `ours_sf24c1hyb_dev_abot`: 2 GPUs (device + cloud): VRAM = device GPU peak only (the cloud GPU's peak is timing.json `cloud_peak_alloc_GiB`); TTFF / fps e2e are not measured in the two-process mode.
- `ours_sf24_abot` (60 of 60 clips), `ours_sf24_abot_dit` (2 of 2 clips), `ours_student1p3b_abot` (60 of 60 clips): fps gen / gen s are DiT-only, corrected for the side-stream decode overlap (audit 10/19).
- `ours_student1p3b_abot` (3 of 60): clips go black (see the 'excl. black' rows).
- '(partial)' rows: n below the 60 clips of tasks/abot.json (generation or scoring still running).

### set `abot_long` (n expected 4 per arm = 4 scenes x P6)

| model_arm | set | n | res | GPUs | fps gen ↑ | fps e2e ↑ | gen s ↓ | TTFF s ↓ | VRAM GiB ↓ | judge Q ↑ | judge drift ↓ | VBench Q ↑ | imaging ↑ | MUSIQ drift ↓ | RPE Rot ° ↓ | react s ↓ | react fr ↓ | black |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| abot | abot_long | 4 | 1280x704 | 1 | **15.3** | **12.8** | **47.1** | **0.98** | **32.1** | 7.00 | 83.8 | 73.95 | 69.2 | 7.10 | 0.603 | - | - | 0 (1) |
| ours_adapt25_abot_long | abot_long | 4 | 1280x704 | 1 | 3.6 | <u>2.5</u> | 272.2 | 83.72 | 71.6 | 8.00 | 55.0 | 76.02 | 75.4 | 3.48 | 0.464 | - | - | - |
| ours_adapt25c1hyb_cloud_abot_long | abot_long | 4 | 1280x704 | 2 | 2.6 | - | 377.0 | - | 67.5 (dev) | <u>8.25</u> | 61.2 | 78.45 | 76.2 | 2.74 | 0.286 | - | - | - |
| ours_adapt25c1hyb_dev_abot_long | abot_long | 4 | 1280x704 | 2 | 2.6 | - | 375.1 | - | 67.7 (dev) | <u>8.25</u> | **38.8** | 78.59 | 76.2 | 2.76 | 0.276 | - | - | - |
| ours_rt2_abot_long | abot_long | 4 | 1280x704 | 1 | <u>3.7</u> | <u>2.5</u> | <u>262.4</u> | 83.78 | 71.4 | 8.00 | 50.0 | 78.11 | 76.4 | 3.01 | 0.518 | - | - | - |
| ours_rt2c1hyb_cloud_abot_long | abot_long | 4 | 1280x704 | 2 | 2.6 | - | 370.2 | - | 67.5 (dev) | 8.00 | <u>47.5</u> | 78.71 | <u>76.9</u> | <u>2.58</u> | 0.276 | - | - | - |
| ours_rt2c1hyb_dev_abot_long | abot_long | 4 | 1280x704 | 2 | 2.6 | - | 368.3 | - | 67.5 (dev) | 7.75 | 50.0 | 78.83 | 76.7 | **2.55** | <u>0.275</u> | - | - | - |
| ours_sf24_abot_long | abot_long | 4 | 1280x704 | 1 | 2.5 | 2.3 | 397.2 | 86.33 | - | **8.50** | 52.5 | 78.67 | 75.6 | 3.28 | 0.348 | - | - | - |
| ours_sf24c1hyb_cloud_abot_long | abot_long | 4 | 1280x704 | 2 | 2.4 | - | 420.5 | - | <u>63.9 (dev)</u> | <u>8.25</u> | 52.5 | <u>80.27</u> | 76.5 | 2.97 | 0.278 | - | - | - |
| ours_sf24c1hyb_dev_abot_long | abot_long | 4 | 1280x704 | 2 | 2.4 | - | 418.6 | - | <u>63.9 (dev)</u> | **8.50** | 52.5 | **80.52** | 76.3 | 3.04 | **0.270** | - | - | - |
| ours_teacher_abot_long | abot_long | 4 | 1280x704 | 1 | 2.0 | 1.6 | 484.5 | <u>83.06</u> | 71.2 | 7.50 | 63.8 | 77.71 | **77.0** | <u>2.58</u> | 0.505 | - | - | 0 (1) |

- `ours_adapt25c1hyb_cloud_abot_long`, `ours_adapt25c1hyb_dev_abot_long`, `ours_rt2c1hyb_cloud_abot_long`, `ours_rt2c1hyb_dev_abot_long`, `ours_sf24c1hyb_cloud_abot_long`, `ours_sf24c1hyb_dev_abot_long`: 2 GPUs (device + cloud): VRAM = device GPU peak only (the cloud GPU's peak is timing.json `cloud_peak_alloc_GiB`); TTFF / fps e2e are not measured in the two-process mode.
- `ours_sf24_abot_long` (4 of 4 clips): fps gen / gen s are DiT-only, corrected for the side-stream decode overlap (audit 10/19).

### set `astra` (n expected 21 per arm)

| model_arm | set | n | res | GPUs | fps gen ↑ | fps e2e ↑ | gen s ↓ | TTFF s ↓ | VRAM GiB ↓ | judge Q ↑ | judge drift ↓ | VBench Q ↑ | imaging ↑ | MUSIQ drift ↓ | RPE Rot ° ↓ | react s ↓ | react fr ↓ | black |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| astra | astra | 21 | 832x480 | 1 | 1.3 | 1.1 | 173.0 | 25.80 | **14.9** | 6.52 | 48.8 | 77.27 | 70.8 | 2.81 | 0.992 | - | - | 0 |
| ours_adapt25_astra | astra | 21 | 832x480 | 1 | 9.5 | 6.3 | 17.5 | 8.76 | 62.8 | 7.90 | 10.5 | 78.87 | 73.0 | 2.71 | 0.522 | 0.14 (3) | 2.2 (3) | 0 |
| ours_adapt25c1hyb_cloud_astra | astra | 21 | 832x480 | 2 | 6.8 | - | 23.8 | - | 62.1 (dev) | <u>8.19</u> | <u>8.6</u> | 79.22 | 73.3 | 2.38 | 0.413 | 0.15 (3) | 2.3 (3) | 0 |
| ours_adapt25c1hyb_dev_astra | astra | 21 | 832x480 | 2 | 6.8 | - | 23.0 | - | 62.3 (dev) | <u>8.19</u> | 11.4 | 79.31 | 73.1 | 2.40 | <u>0.412</u> | 0.20 (3) | 3.2 (3) | 0 |
| ours_rs576_astra | astra | 21 | 832x480 | 1 | **20.5** | 9.7 | **7.7** | **4.44** | 39.3 | 3.67 | 15.7 | 73.70 | 59.4 | 3.17 | 0.452 | 0.16 (3) | 2.6 (3) | - |
| ours_rs576r_astra | astra | 21 | 832x480 | 1 | 12.6 | 7.5 | 12.4 | 11.14 | 48.1 | 8.00 | 14.0 | 77.71 | 74.3 | 2.03 | 0.470 | 0.19 (3) | 3.1 (3) | - |
| ours_rt2_astra | astra | 21 | 832x480 | 1 | 9.6 | 6.3 | 16.1 | 8.29 | 62.6 | 7.76 | 12.4 | 78.46 | 72.9 | 2.34 | 0.516 | 0.16 (3) | 2.6 (3) | 0 |
| ours_rt2c1hyb32_cloud_astra | astra | 21 | 832x480 | 2 | 7.7 | - | 20.7 | - | 62.1 (dev) | 8.14 | 15.5 | 79.06 | 73.1 | 2.24 | 0.434 | 0.24 (3) | 3.8 (3) | - |
| ours_rt2c1hyb32_dev_astra | astra | 21 | 832x480 | 2 | 7.7 | - | 19.9 | - | <u>27.4 (dev)</u> | 7.81 | 15.5 | 79.12 | 72.8 | 2.22 | 0.432 | 0.23 (3) | 3.7 (3) | - |
| ours_rt2c1hyb_cloud_astra | astra | 21 | 832x480 | 2 | 6.8 | - | 22.4 | - | 62.1 (dev) | 8.10 | 11.4 | 78.57 | 73.3 | 2.36 | 0.453 | 0.16 (3) | 2.6 (3) | - |
| ours_rt2c1hyb_dev_astra | astra | 21 | 832x480 | 2 | 6.8 | - | 21.7 | - | 62.1 (dev) | 7.90 | 11.4 | 78.75 | 73.1 | 2.39 | 0.450 | 0.18 (3) | 2.9 (3) | - |
| ours_sf1p3b_cal_astra | astra | 21 | 832x480 | 1 | 16.9 | <u>12.6</u> | 9.7 | 6.77 | 28.7 | 7.48 | 23.1 | 81.12 | <u>75.0</u> | <u>1.87</u> | 0.905 | **0.11 (3)** | **1.8 (3)** | 0 |
| ours_sf1p3b_pub_astra | astra | 21 | 832x480 | 1 | <u>18.6</u> | **13.4** | <u>8.9</u> | <u>6.66</u> | 28.7 | 7.29 | 25.7 | 80.45 | 74.1 | 2.13 | 0.895 | <u>0.13 (3)</u> | <u>2.1 (3)</u> | 0 |
| ours_sf24_astra | astra | 21 | 832x480 | 1 | 7.2 | 6.3 | 22.4 | 7.86 | - | 8.10 | 11.4 | 79.03 | 72.4 | 2.25 | 0.445 | 0.19 (3) | 3.1 (3) | 0 |
| ours_sf24_astra_dit | astra | 2 (partial) | 832x480 | 1 | 10.1 | - | 16.7 | - | - | **9.00** | 10.0 | **84.06** | **80.0** | **0.47** | **0.158** | - | - | - |
| ours_sf24c1hyb_cloud_astra | astra | 21 | 832x480 | 2 | 6.7 | - | 23.4 | - | 62.1 (dev) | 8.05 | 9.5 | 78.65 | 72.3 | 2.27 | 0.438 | 0.20 (3) | 3.2 (3) | - |
| ours_sf24c1hyb_dev_astra | astra | 21 | 832x480 | 2 | 6.8 | - | 22.6 | - | 62.2 (dev) | 8.05 | **7.6** | 78.88 | 72.0 | 2.23 | 0.439 | 0.19 (3) | 3.0 (3) | - |
| ours_student1p3b_astra | astra | 21 | 832x480 | 1 | 11.2 | 9.0 | 14.2 | 6.87 | - | 8.10 | 11.2 | <u>83.16</u> | 74.0 | 2.98 | 0.783 | 0.14 (3) | 2.2 (3) | 0 |
| ours_teacher_astra | astra | 21 | 832x480 | 1 | 5.3 | 4.1 | 30.0 | 9.06 | 62.6 | 7.86 | 11.0 | 78.69 | 74.0 | 2.36 | 0.563 | 0.15 (3) | 2.4 (3) | 0 |

- Astra's own-scene set = 3 scenes x 7 programs = 21 clips per arm (n expected 21).
- `astra` (21 of 21 clips): timing.json flags a shared-GPU measurement (speed columns are a lower bound there).
- `ours_adapt25c1hyb_cloud_astra`, `ours_adapt25c1hyb_dev_astra`, `ours_rt2c1hyb32_cloud_astra`, `ours_rt2c1hyb32_dev_astra`, `ours_rt2c1hyb_cloud_astra`, `ours_rt2c1hyb_dev_astra`, `ours_sf24c1hyb_cloud_astra`, `ours_sf24c1hyb_dev_astra`: 2 GPUs (device + cloud): VRAM = device GPU peak only (the cloud GPU's peak is timing.json `cloud_peak_alloc_GiB`); TTFF / fps e2e are not measured in the two-process mode.
- `ours_sf1p3b_cal_astra` (21 of 21 clips), `ours_sf1p3b_pub_astra` (21 of 21 clips), `ours_sf24_astra` (21 of 21 clips), `ours_sf24_astra_dit` (2 of 2 clips), `ours_student1p3b_astra` (21 of 21 clips): fps gen / gen s are DiT-only, corrected for the side-stream decode overlap (audit 10/19).
- '(partial)' rows: n below the 21 clips of tasks/astra.json (generation or scoring still running).

### set `mg2` (n expected 84 per arm)

| model_arm | set | n | res | GPUs | fps gen ↑ | fps e2e ↑ | gen s ↓ | TTFF s ↓ | VRAM GiB ↓ | judge Q ↑ | judge drift ↓ | VBench Q ↑ | imaging ↑ | MUSIQ drift ↓ | RPE Rot ° ↓ | react s ↓ | react fr ↓ | black |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mg2 | mg2 | 84 | 640x352 | 1 | **34.3** | **21.1** | **3.8** | **2.11** | **11.5** | 8.00 | 27.0 | <u>78.88</u> | 68.7 | 3.28 | 0.391 | 0.26 (12) | 3.1 (12) | 0 |
| ours_adapt25_mg2 | mg2 | 84 | 640x352 | 1 | 19.2 | 12.1 | 8.8 | 4.76 | 43.9 | 7.68 | 23.4 | 77.37 | 68.1 | 4.36 | 1.114 | 0.19 (12) | 3.1 (12) | 0 |
| ours_adapt25c1hyb_cloud_mg2 | mg2 | 84 | 640x352 | 2 | 12.4 | - | 14.1 | - | 43.3 (dev) | 7.88 | 15.1 | 78.29 | 68.9 | 3.56 | 0.717 | 0.11 (12) | 1.8 (12) | 1 |
| *ours_adapt25c1hyb_cloud_mg2 excl. black (n=83)* | mg2 | 83 | 640x352 | 2 | 12.4 | - | 14.1 | - | 43.3 (dev) | 7.89 | 14.3 | 78.30 | 69.1 | 3.56 | 0.724 | 0.11 (12) | 1.8 (12) | 0 |
| ours_adapt25c1hyb_dev_mg2 | mg2 | 84 | 640x352 | 2 | 12.4 | - | 13.7 | - | 43.4 (dev) | 7.98 | 15.1 | 78.40 | 68.7 | 3.59 | 0.709 | 0.12 (12) | 1.9 (12) | 1 |
| *ours_adapt25c1hyb_dev_mg2 excl. black (n=83)* | mg2 | 83 | 640x352 | 2 | 12.4 | - | 13.7 | - | 43.4 (dev) | 7.98 | 14.3 | 78.41 | 68.9 | 3.58 | 0.716 | 0.12 (12) | 1.9 (12) | 0 |
| ours_mx24_mg2_idle5 | mg2 | 6 (partial) | 640x352 | 1 | 12.5 | - | 13.2 | - | <u>15.3</u> | 7.67 | <u>3.3</u> | **80.51** | 69.9 | 2.74 | 0.779 | - | - | - |
| ours_mx32_mg2_idle5 | mg2 | 6 (partial) | 640x352 | 1 | 21.0 | - | 7.7 | - | 23.6 | 7.50 | **0.0** | 78.54 | 68.3 | 2.71 | 0.763 | - | - | - |
| ours_rt2_mg2 | mg2 | 84 | 640x352 | 1 | 19.3 | 12.1 | 8.2 | 4.56 | 43.7 | 7.52 | 21.1 | 77.20 | 67.4 | 4.15 | 1.130 | 0.12 (12) | 2.0 (12) | 0 |
| ours_rt2_mg2_idle5 | mg2 | 6 (partial) | 640x352 | 1 | 19.4 | 12.2 | 8.1 | 4.48 | 43.7 | 8.00 | <u>3.3</u> | 77.51 | 70.1 | 2.79 | 0.756 | - | - | - |
| ours_rt2c1hyb24_cloud_mg2 | mg2 | 1 (partial) | 640x352 | 2 | 12.3 | - | 18.1 | - | 43.2 (dev) | 7.00 | **0.0** | 75.69 | <u>72.8</u> | <u>1.54</u> | <u>0.034</u> | - | - | - |
| ours_rt2c1hyb24_dev_mg2 | mg2 | 1 (partial) | 640x352 | 2 | 12.6 | - | 17.7 | - | <u>15.3 (dev)</u> | 7.00 | **0.0** | 75.95 | **73.2** | **1.23** | **0.033** | - | - | - |
| ours_rt2c1hyb32_cloud_mg2 | mg2 | 84 | 640x352 | 2 | 15.0 | - | 10.8 | - | 43.3 (dev) | 7.74 | 14.8 | 77.98 | 67.8 | 3.71 | 0.697 | 0.14 (12) | 2.2 (12) | - |
| ours_rt2c1hyb32_dev_mg2 | mg2 | 84 | 640x352 | 2 | 15.0 | - | 10.4 | - | 23.6 (dev) | 7.81 | 14.9 | 78.03 | 66.8 | 3.76 | 0.700 | 0.12 (12) | 2.0 (12) | - |
| ours_rt2c1hyb_cloud_mg2 | mg2 | 84 | 640x352 | 2 | 13.8 | - | 11.3 | - | 43.3 (dev) | 7.79 | 18.6 | 78.01 | 68.4 | 3.60 | 0.738 | <u>0.10 (12)</u> | 1.6 (12) | 0 |
| ours_rt2c1hyb_dev_mg2 | mg2 | 84 | 640x352 | 2 | 13.8 | - | 10.9 | - | 43.3 (dev) | 7.83 | 17.6 | 78.10 | 68.1 | 3.67 | 0.723 | 0.11 (12) | 1.7 (12) | 0 |
| ours_sf24_mg2 | mg2 | 84 | 640x352 | 1 | 13.6 | 11.8 | 12.2 | 4.69 | - | 7.98 | 15.2 | 78.31 | 67.8 | 3.40 | 0.627 | **0.09 (12)** | **1.4 (12)** | 0 |
| ours_sf24_mg2_dit | mg2 | 2 (partial) | 640x352 | 1 | 20.7 | - | 9.0 | - | - | 6.50 | 42.5 | 73.66 | 70.5 | 2.44 | 0.329 | - | - | - |
| ours_sf24c1hyb_cloud_mg2 | mg2 | 84 | 640x352 | 2 | 12.9 | - | 12.6 | - | 43.3 (dev) | <u>8.12</u> | 13.9 | 77.98 | 67.6 | 3.28 | 0.648 | <u>0.10 (12)</u> | 1.6 (12) | 0 |
| ours_sf24c1hyb_dev_mg2 | mg2 | 84 | 640x352 | 2 | 13.0 | - | 12.2 | - | 43.3 (dev) | **8.15** | 12.6 | 77.94 | 66.9 | 3.45 | 0.641 | **0.09 (12)** | <u>1.5 (12)</u> | 0 |
| ours_student1p3b_mg2 | mg2 | 84 | 640x352 | 1 | <u>21.8</u> | <u>17.3</u> | <u>7.4</u> | <u>4.11</u> | - | 7.23 | 25.5 | 77.86 | 64.8 | 3.65 | 1.052 | **0.09 (12)** | **1.4 (12)** | 1 |
| *ours_student1p3b_mg2 excl. black (n=83)* | mg2 | 83 | 640x352 | 1 | 21.8 | 17.3 | 7.4 | 4.11 | - | 7.24 | 25.5 | 77.84 | 64.8 | 3.64 | 1.063 | 0.09 (12) | 1.4 (12) | 0 |
| ours_teacher_mg2 | mg2 | 84 | 640x352 | 1 | 10.6 | 8.0 | 14.9 | 4.92 | 43.7 | 7.98 | 19.5 | 77.40 | 69.6 | 3.90 | 1.142 | 0.15 (12) | 2.4 (12) | 0 |

- `ours_adapt25c1hyb_cloud_mg2`, `ours_adapt25c1hyb_dev_mg2`, `ours_rt2c1hyb24_cloud_mg2`, `ours_rt2c1hyb24_dev_mg2`, `ours_rt2c1hyb32_cloud_mg2`, `ours_rt2c1hyb32_dev_mg2`, `ours_rt2c1hyb_cloud_mg2`, `ours_rt2c1hyb_dev_mg2`, `ours_sf24c1hyb_cloud_mg2`, `ours_sf24c1hyb_dev_mg2`: 2 GPUs (device + cloud): VRAM = device GPU peak only (the cloud GPU's peak is timing.json `cloud_peak_alloc_GiB`); TTFF / fps e2e are not measured in the two-process mode.
- `ours_adapt25c1hyb_cloud_mg2` (1 of 84), `ours_adapt25c1hyb_dev_mg2` (1 of 84), `ours_student1p3b_mg2` (1 of 84): clips go black (see the 'excl. black' rows).
- `ours_sf24_mg2` (84 of 84 clips), `ours_sf24_mg2_dit` (2 of 2 clips), `ours_student1p3b_mg2` (84 of 84 clips): fps gen / gen s are DiT-only, corrected for the side-stream decode overlap (audit 10/19).
- '(partial)' rows: n below the 84 clips of tasks/mg2.json (generation or scoring still running).

### set `mg2_long` (n expected 4 per arm = 4 scenes x P6)

| model_arm | set | n | res | GPUs | fps gen ↑ | fps e2e ↑ | gen s ↓ | TTFF s ↓ | VRAM GiB ↓ | judge Q ↑ | judge drift ↓ | VBench Q ↑ | imaging ↑ | MUSIQ drift ↓ | RPE Rot ° ↓ | react s ↓ | react fr ↓ | black |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mg2 | mg2_long | 4 | 640x352 | 1 | **34.6** | **21.2** | **21.1** | **9.51** | **16.2** | 5.50 | 72.5 | 70.22 | 62.3 | 9.51 | 0.914 | - | - | 1 (1) |
| *mg2 excl. black (n=3)* | mg2_long | 3 | 640x352 | 1 | 34.6 | 21.2 | 21.2 | 9.57 | 16.2 | 6.33 | 63.3 | 73.32 | 71.0 | 7.04 | 0.225 | - | - | - |
| ours_adapt25_mg2_long | mg2_long | 4 | 640x352 | 1 | 19.2 | 12.1 | 51.0 | 22.06 | 36.2 | 6.75 | 81.2 | 72.88 | 67.0 | 7.62 | 0.632 | - | - | - |
| ours_adapt25c1hyb_cloud_mg2_long | mg2_long | 4 | 640x352 | 2 | 13.6 | - | 71.0 | - | 34.7 (dev) | 6.50 | 67.5 | 76.03 | 69.3 | 5.96 | 0.209 | - | - | - |
| ours_adapt25c1hyb_dev_mg2_long | mg2_long | 4 | 640x352 | 2 | 13.6 | - | 70.7 | - | 34.9 (dev) | 6.50 | 81.2 | 76.46 | 69.0 | 6.15 | <u>0.205</u> | - | - | - |
| ours_mx24_mg2_long | mg2_long | 4 | 640x352 | 1 | 12.4 | - | 77.9 | - | <u>17.3</u> | 6.75 | 70.0 | 72.30 | 66.4 | 8.15 | 0.921 | - | - | - |
| ours_mx32_mg2_long | mg2_long | 4 | 640x352 | 1 | 20.9 | - | 46.3 | - | 25.5 | <u>7.00</u> | 83.8 | 72.27 | 65.5 | 8.86 | 1.187 | - | - | - |
| ours_mxK8_mg2_long | mg2_long | 4 | 640x352 | 1 | 18.2 | 11.7 | 53.0 | 21.44 | 29.9 | 5.50 | 90.0 | 70.74 | 66.2 | 8.09 | 0.937 | - | - | - |
| ours_mxK8c1hyb_cloud_mg2_long | mg2_long | 4 | 640x352 | 2 | 12.9 | - | 74.3 | - | 34.7 (dev) | 6.50 | 57.5 | 75.49 | 69.0 | 5.61 | 0.209 | - | - | - |
| ours_mxK8c1hyb_dev_mg2_long | mg2_long | 4 | 640x352 | 2 | 12.9 | - | 73.9 | - | 26.2 (dev) | 6.50 | 57.5 | 75.40 | 68.4 | 5.85 | 0.209 | - | - | - |
| ours_mxQ4_mg2_long | mg2_long | 4 | 640x352 | 1 | 11.9 | 8.7 | 81.6 | 22.71 | 27.7 | 6.75 | 77.5 | 72.20 | 67.7 | 8.68 | 0.676 | - | - | - |
| ours_mxQ4c1hyb_cloud_mg2_long | mg2_long | 4 | 640x352 | 2 | 6.9 | - | 114.4 | - | 34.7 (dev) | <u>7.00</u> | 67.5 | 76.08 | **70.9** | 5.49 | 0.232 | - | - | - |
| ours_mxQ4c1hyb_dev_mg2_long | mg2_long | 4 | 640x352 | 2 | 7.1 | - | 114.1 | - | 23.8 (dev) | <u>7.00</u> | 67.5 | 76.39 | <u>70.7</u> | 5.52 | 0.233 | - | - | - |
| ours_mxW12_mg2_long | mg2_long | 4 | 640x352 | 1 | 21.8 | 13.0 | 44.6 | 21.75 | 32.0 | 6.75 | 81.2 | 71.71 | 61.9 | 10.23 | 0.909 | - | - | - |
| ours_mxW12c1hyb_cloud_mg2_long | mg2_long | 4 | 640x352 | 2 | 15.6 | - | 62.1 | - | 34.7 (dev) | 6.75 | 56.2 | 74.47 | 68.2 | **4.98** | 0.276 | - | - | - |
| ours_mxW12c1hyb_dev_mg2_long | mg2_long | 4 | 640x352 | 2 | 15.6 | - | 61.7 | - | 30.7 (dev) | <u>7.00</u> | 58.8 | 74.50 | 67.4 | 5.28 | 0.270 | - | - | - |
| ours_mxW9_mg2_long | mg2_long | 4 | 640x352 | 1 | 23.4 | <u>13.6</u> | 41.6 | 21.38 | 30.0 | 6.75 | 75.0 | 72.18 | 60.0 | 5.55 | 3.796 | - | - | - |
| ours_mxW9c1hyb_cloud_mg2_long | mg2_long | 4 | 640x352 | 2 | <u>23.5</u> | - | 41.6 | - | 34.7 (dev) | 6.75 | 75.0 | 72.85 | 63.2 | <u>5.20</u> | 3.914 | - | - | - |
| ours_mxW9c1hyb_dev_mg2_long | mg2_long | 4 | 640x352 | 2 | <u>23.5</u> | - | <u>41.3</u> | - | 28.7 (dev) | 6.75 | 75.0 | 72.18 | 60.0 | 5.55 | 3.796 | - | - | - |
| ours_rt2_mg2_long | mg2_long | 4 | 640x352 | 1 | 19.3 | 12.1 | 49.9 | 21.34 | 36.1 | 6.50 | 81.2 | 72.42 | 64.6 | 9.66 | 0.711 | - | - | - |
| ours_rt2c1hyb24_cloud_mg2_long | mg2_long | 4 | 640x352 | 2 | 5.9 | - | 146.4 | - | 34.7 (dev) | <u>7.00</u> | 66.2 | 76.37 | 70.6 | 5.93 | 0.328 | - | - | - |
| ours_rt2c1hyb24_dev_mg2_long | mg2_long | 4 | 640x352 | 2 | 6.0 | - | 146.0 | - | <u>17.3 (dev)</u> | <u>7.00</u> | 63.8 | <u>76.48</u> | 69.4 | 6.07 | 0.350 | - | - | - |
| ours_rt2c1hyb32_cloud_mg2_long | mg2_long | 4 | 640x352 | 2 | 14.9 | - | 64.9 | - | 34.7 (dev) | <u>7.00</u> | 55.0 | 76.32 | 68.9 | 5.77 | 0.239 | - | - | - |
| ours_rt2c1hyb32_dev_mg2_long | mg2_long | 4 | 640x352 | 2 | 14.9 | - | 64.5 | - | 25.5 (dev) | 6.25 | 58.8 | 76.31 | 67.9 | 5.99 | 0.231 | - | - | - |
| ours_rt2c1hyb_cloud_mg2_long | mg2_long | 4 | 640x352 | 2 | 13.7 | - | 70.1 | - | 34.7 (dev) | <u>7.00</u> | 75.0 | 75.13 | 69.9 | 5.95 | 0.233 | - | - | - |
| ours_rt2c1hyb_dev_mg2_long | mg2_long | 4 | 640x352 | 2 | 13.7 | - | 69.7 | - | 34.7 (dev) | 6.25 | 80.0 | 75.24 | 69.5 | 6.18 | 0.230 | - | - | - |
| ours_sf24_mg2_long | mg2_long | 4 | 640x352 | 1 | 13.8 | 12.0 | 68.6 | <u>19.80</u> | - | **7.25** | <u>53.8</u> | **76.95** | 68.9 | 5.41 | **0.199** | - | - | - |
| ours_sf24c1hyb_cloud_mg2_long | mg2_long | 4 | 640x352 | 2 | 13.3 | - | 71.8 | - | 34.7 (dev) | **7.25** | **52.5** | 76.02 | 68.3 | 6.42 | 0.267 | - | - | - |
| ours_sf24c1hyb_dev_mg2_long | mg2_long | 4 | 640x352 | 2 | 13.3 | - | 71.5 | - | 34.8 (dev) | <u>7.00</u> | 56.2 | 76.32 | 68.0 | 6.69 | 0.266 | - | - | - |
| ours_teacher_mg2_long | mg2_long | 4 | 640x352 | 1 | 10.5 | 8.0 | 91.3 | 21.89 | 36.1 | 6.50 | 82.5 | 73.27 | 68.9 | 6.25 | 0.580 | - | - | - |

- `mg2` (1 of 4): clips go black (see the 'excl. black' rows).
- `ours_adapt25c1hyb_cloud_mg2_long`, `ours_adapt25c1hyb_dev_mg2_long`, `ours_mxK8c1hyb_cloud_mg2_long`, `ours_mxK8c1hyb_dev_mg2_long`, `ours_mxQ4c1hyb_cloud_mg2_long`, `ours_mxQ4c1hyb_dev_mg2_long`, `ours_mxW12c1hyb_cloud_mg2_long`, `ours_mxW12c1hyb_dev_mg2_long`, `ours_mxW9c1hyb_cloud_mg2_long`, `ours_mxW9c1hyb_dev_mg2_long`, `ours_rt2c1hyb24_cloud_mg2_long`, `ours_rt2c1hyb24_dev_mg2_long`, `ours_rt2c1hyb32_cloud_mg2_long`, `ours_rt2c1hyb32_dev_mg2_long`, `ours_rt2c1hyb_cloud_mg2_long`, `ours_rt2c1hyb_dev_mg2_long`, `ours_sf24c1hyb_cloud_mg2_long`, `ours_sf24c1hyb_dev_mg2_long`: 2 GPUs (device + cloud): VRAM = device GPU peak only (the cloud GPU's peak is timing.json `cloud_peak_alloc_GiB`); TTFF / fps e2e are not measured in the two-process mode.
- `ours_sf24_mg2_long` (4 of 4 clips): fps gen / gen s are DiT-only, corrected for the side-stream decode overlap (audit 10/19).

### set `neutral_832` (n expected 84 per arm)

| model_arm | set | n | res | GPUs | fps gen ↑ | fps e2e ↑ | gen s ↓ | TTFF s ↓ | VRAM GiB ↓ | judge Q ↑ | judge drift ↓ | VBench Q ↑ | imaging ↑ | MUSIQ drift ↓ | RPE Rot ° ↓ | react s ↓ | react fr ↓ | black |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| astra | neutral_832 | 84 | 832x480 | 1 | 1.3 | 1.1 | 173.0 | 25.80 | **14.9** | 4.95 | 77.2 | 70.18 | 55.3 | 6.16 | 1.658 | 0.75 (5) | 15.0 (5) | 6 |
| *astra excl. black (n=78)* | neutral_832 | 78 | 832x480 | 1 | 1.3 | 1.1 | 173.0 | 25.80 | 14.9 | 4.88 | 76.8 | 70.61 | 56.4 | 5.59 | 1.626 | 0.75 (5) | 15.0 (5) | 0 |
| ours_adapt25_neutral_832 | neutral_832 | 84 | 832x480 | 1 | 9.4 | 6.2 | 17.8 | 8.83 | 62.8 | 7.54 | 40.9 | 74.12 | 68.5 | 3.91 | 0.681 | 0.27 (12) | 4.3 (12) | 1 |
| *ours_adapt25_neutral_832 excl. black (n=83)* | neutral_832 | 83 | 832x480 | 1 | 9.4 | 6.2 | 17.8 | 8.75 | 62.8 | 7.54 | 40.4 | 74.18 | 68.8 | 3.76 | 0.688 | 0.27 (12) | 4.3 (12) | 0 |
| ours_adapt25c1hyb_cloud_neutral_832 | neutral_832 | 84 | 832x480 | 2 | 6.7 | - | 24.0 | - | 62.1 (dev) | 7.70 | 39.9 | 74.21 | 68.7 | 3.92 | 0.690 | 0.18 (12) | 2.8 (12) | 3 |
| *ours_adapt25c1hyb_cloud_neutral_832 excl. black (n=81)* | neutral_832 | 81 | 832x480 | 2 | 6.7 | - | 24.0 | - | 62.1 (dev) | 7.73 | 38.6 | 74.45 | 69.2 | 3.59 | 0.676 | 0.18 (12) | 2.8 (12) | 0 |
| ours_adapt25c1hyb_dev_neutral_832 | neutral_832 | 84 | 832x480 | 2 | 6.7 | - | 23.3 | - | 62.3 (dev) | 7.65 | 38.6 | 74.22 | 68.2 | 4.07 | 0.749 | 0.24 (12) | 3.9 (12) | 3 |
| *ours_adapt25c1hyb_dev_neutral_832 excl. black (n=81)* | neutral_832 | 81 | 832x480 | 2 | 6.7 | - | 23.3 | - | 62.3 (dev) | 7.68 | 37.3 | 74.47 | 68.8 | 3.74 | 0.740 | 0.24 (12) | 3.9 (12) | 0 |
| ours_mx24_neutral_832_idle5 | neutral_832 | 6 (partial) | 832x480 | 1 | 7.4 | - | 22.0 | - | <u>19.1</u> | 7.17 | 65.8 | 71.94 | 66.2 | 4.90 | 0.811 | - | - | - |
| ours_mx32_neutral_832_idle5 | neutral_832 | 6 (partial) | 832x480 | 1 | 10.8 | - | 14.9 | - | 27.4 | 7.00 | 64.2 | 70.75 | 63.5 | 6.73 | 0.814 | - | - | - |
| ours_rs576_neutral_832 | neutral_832 | 84 | 832x480 | 1 | **20.5** | 9.7 | **7.7** | **4.62** | 39.3 | 5.42 | 45.0 | 70.77 | 51.7 | 5.68 | 0.756 | **0.09 (11)** | **1.4 (11)** | - |
| ours_rs576r_neutral_832 | neutral_832 | 84 | 832x480 | 1 | 12.6 | 7.5 | 12.5 | 11.57 | 48.1 | 7.54 | 36.4 | 74.24 | 67.1 | 4.70 | 0.756 | **0.09 (12)** | **1.4 (12)** | - |
| ours_rt2_neutral_832 | neutral_832 | 84 | 832x480 | 1 | 9.6 | 6.3 | 16.1 | 8.55 | 62.6 | 7.60 | 39.2 | 74.14 | 68.0 | 4.10 | 0.814 | 0.16 (11) | 2.6 (11) | 1 |
| *ours_rt2_neutral_832 excl. black (n=83)* | neutral_832 | 83 | 832x480 | 1 | 9.6 | 6.3 | 16.1 | 8.47 | 62.6 | 7.60 | 38.6 | 74.20 | 68.3 | 3.98 | 0.822 | 0.16 (11) | 2.6 (11) | 0 |
| ours_rt2_neutral_832_idle5 | neutral_832 | 6 (partial) | 832x480 | 1 | 9.6 | 6.3 | 16.1 | 9.49 | 62.6 | 7.00 | 76.7 | 70.61 | 65.6 | 5.17 | 0.694 | - | - | - |
| ours_rt2c1hyb24_cloud_neutral_832 | neutral_832 | 1 (partial) | 832x480 | 2 | 5.3 | - | 29.9 | - | 62.1 (dev) | 7.00 | 85.0 | 71.82 | **71.9** | 6.27 | <u>0.115</u> | - | - | - |
| ours_rt2c1hyb24_dev_neutral_832 | neutral_832 | 1 (partial) | 832x480 | 2 | 5.4 | - | 29.2 | - | <u>19.1 (dev)</u> | 7.00 | 85.0 | 71.70 | <u>71.3</u> | 6.45 | **0.107** | - | - | - |
| ours_rt2c1hyb32_cloud_neutral_832 | neutral_832 | 84 | 832x480 | 2 | 5.2 | - | 51.6 | - | 62.1 (dev) | **7.82** | **34.5** | 74.71 | 68.8 | <u>3.73</u> | 0.700 | 0.21 (12) | 3.3 (12) | - |
| ours_rt2c1hyb32_dev_neutral_832 | neutral_832 | 84 | 832x480 | 2 | 5.2 | - | 50.9 | - | 27.4 (dev) | 7.69 | 36.0 | 74.63 | 67.8 | 4.01 | 0.754 | 0.20 (12) | 3.2 (12) | - |
| ours_rt2c1hyb_cloud_neutral_832 | neutral_832 | 84 | 832x480 | 2 | 6.8 | - | 22.5 | - | 62.1 (dev) | 7.68 | 37.6 | 74.38 | 68.2 | 3.77 | 0.835 | 0.13 (11) | 2.0 (11) | 3 |
| *ours_rt2c1hyb_cloud_neutral_832 excl. black (n=81)* | neutral_832 | 81 | 832x480 | 2 | 6.8 | - | 22.5 | - | 62.1 (dev) | 7.72 | 36.0 | 74.61 | 68.9 | 3.53 | 0.799 | 0.13 (11) | 2.0 (11) | 0 |
| ours_rt2c1hyb_dev_neutral_832 | neutral_832 | 84 | 832x480 | 2 | 6.8 | - | 21.8 | - | 62.1 (dev) | 7.68 | 37.9 | 74.40 | 67.8 | 3.85 | 0.804 | 0.13 (12) | 2.2 (12) | 3 |
| *ours_rt2c1hyb_dev_neutral_832 excl. black (n=81)* | neutral_832 | 81 | 832x480 | 2 | 6.8 | - | 21.8 | - | 62.1 (dev) | 7.72 | 36.3 | 74.64 | 68.5 | 3.63 | 0.773 | 0.13 (12) | 2.2 (12) | 0 |
| ours_sf1p3b_cal_neutral_832 | neutral_832 | 84 | 832x480 | 1 | 14.6 | <u>11.2</u> | 11.2 | <u>6.78</u> | 28.7 | 6.61 | 46.5 | 74.00 | 63.8 | 3.91 | 1.375 | 0.13 (12) | 2.1 (12) | 1 |
| *ours_sf1p3b_cal_neutral_832 excl. black (n=83)* | neutral_832 | 83 | 832x480 | 1 | 14.7 | 11.2 | 11.2 | 6.78 | 28.7 | 6.59 | 47.1 | 73.91 | 63.7 | 3.92 | 1.390 | 0.13 (12) | 2.1 (12) | 0 |
| ours_sf1p3b_pub_neutral_832 | neutral_832 | 84 | 832x480 | 1 | <u>16.2</u> | **12.2** | <u>10.2</u> | <u>6.78</u> | 28.7 | 5.64 | 55.2 | 71.47 | 59.7 | 4.61 | 1.409 | 0.13 (12) | 2.0 (12) | 0 |
| ours_sf24_neutral_832 | neutral_832 | 84 | 832x480 | 1 | 5.1 | 4.6 | 33.1 | 9.44 | - | <u>7.79</u> | 35.3 | 74.69 | 69.6 | 3.74 | 0.711 | 0.19 (12) | 3.0 (12) | 1 |
| *ours_sf24_neutral_832 excl. black (n=83)* | neutral_832 | 83 | 832x480 | 1 | 5.1 | 4.6 | 33.2 | 9.45 | - | 7.80 | 34.7 | 74.74 | 69.8 | 3.62 | 0.719 | 0.19 (12) | 3.0 (12) | 0 |
| ours_sf24_neutral_832_dit | neutral_832 | 2 (partial) | 832x480 | 1 | 7.1 | - | 22.2 | - | - | 7.00 | 80.0 | 72.68 | 69.6 | 5.95 | 0.167 | - | - | - |
| ours_sf24c1hyb_cloud_neutral_832 | neutral_832 | 84 | 832x480 | 2 | 5.5 | - | 29.4 | - | 62.1 (dev) | 7.77 | <u>35.0</u> | 74.65 | 69.5 | **3.58** | 0.732 | 0.14 (12) | 2.2 (12) | 2 |
| *ours_sf24c1hyb_cloud_neutral_832 excl. black (n=82)* | neutral_832 | 82 | 832x480 | 2 | 5.5 | - | 29.4 | - | 62.1 (dev) | 7.79 | 33.9 | 74.81 | 69.9 | 3.42 | 0.735 | 0.14 (12) | 2.2 (12) | 0 |
| ours_sf24c1hyb_dev_neutral_832 | neutral_832 | 84 | 832x480 | 2 | 5.6 | - | 28.7 | - | 62.2 (dev) | 7.71 | 36.0 | <u>74.80</u> | 69.2 | 3.74 | 0.825 | 0.16 (11) | 2.6 (11) | 2 |
| *ours_sf24c1hyb_dev_neutral_832 excl. black (n=82)* | neutral_832 | 82 | 832x480 | 2 | 5.6 | - | 28.7 | - | 62.2 (dev) | 7.73 | 35.1 | 74.96 | 69.7 | 3.57 | 0.820 | 0.16 (11) | 2.6 (11) | 0 |
| ours_student1p3b_neutral_832 | neutral_832 | 84 | 832x480 | 1 | 11.2 | 9.0 | 14.2 | 6.86 | - | 7.04 | 41.1 | **74.95** | 63.8 | 4.48 | 1.409 | <u>0.11 (12)</u> | <u>1.8 (12)</u> | 4 |
| *ours_student1p3b_neutral_832 excl. black (n=80)* | neutral_832 | 80 | 832x480 | 1 | 11.2 | 9.0 | 14.2 | 6.86 | - | 7.04 | 39.7 | 75.20 | 64.6 | 4.06 | 1.402 | 0.11 (12) | 1.8 (12) | 0 |
| ours_teacher_neutral_832 | neutral_832 | 84 | 832x480 | 1 | 5.3 | 4.1 | 29.4 | 8.75 | 62.6 | 7.73 | 38.9 | 74.02 | 69.2 | 4.11 | 0.757 | 0.30 (12) | 4.8 (12) | 2 |
| *ours_teacher_neutral_832 excl. black (n=82)* | neutral_832 | 82 | 832x480 | 1 | 5.3 | 4.1 | 29.4 | 8.69 | 62.6 | 7.74 | 38.0 | 74.15 | 69.5 | 3.98 | 0.765 | 0.30 (12) | 4.8 (12) | 0 |

- `astra` (84 of 84 clips): timing.json flags a shared-GPU measurement (speed columns are a lower bound there).
- `astra` (6 of 84), `ours_adapt25_neutral_832` (1 of 84), `ours_adapt25c1hyb_cloud_neutral_832` (3 of 84), `ours_adapt25c1hyb_dev_neutral_832` (3 of 84), `ours_rt2_neutral_832` (1 of 84), `ours_rt2c1hyb_cloud_neutral_832` (3 of 84), `ours_rt2c1hyb_dev_neutral_832` (3 of 84), `ours_sf1p3b_cal_neutral_832` (1 of 84), `ours_sf24_neutral_832` (1 of 84), `ours_sf24c1hyb_cloud_neutral_832` (2 of 84), `ours_sf24c1hyb_dev_neutral_832` (2 of 84), `ours_student1p3b_neutral_832` (4 of 84), `ours_teacher_neutral_832` (2 of 84): clips go black (see the 'excl. black' rows).
- `ours_adapt25c1hyb_cloud_neutral_832`, `ours_adapt25c1hyb_dev_neutral_832`, `ours_rt2c1hyb24_cloud_neutral_832`, `ours_rt2c1hyb24_dev_neutral_832`, `ours_rt2c1hyb32_cloud_neutral_832`, `ours_rt2c1hyb32_dev_neutral_832`, `ours_rt2c1hyb_cloud_neutral_832`, `ours_rt2c1hyb_dev_neutral_832`, `ours_sf24c1hyb_cloud_neutral_832`, `ours_sf24c1hyb_dev_neutral_832`: 2 GPUs (device + cloud): VRAM = device GPU peak only (the cloud GPU's peak is timing.json `cloud_peak_alloc_GiB`); TTFF / fps e2e are not measured in the two-process mode.
- `ours_sf1p3b_cal_neutral_832` (84 of 84 clips), `ours_sf1p3b_pub_neutral_832` (84 of 84 clips), `ours_sf24_neutral_832` (84 of 84 clips), `ours_sf24_neutral_832_dit` (2 of 2 clips), `ours_student1p3b_neutral_832` (84 of 84 clips): fps gen / gen s are DiT-only, corrected for the side-stream decode overlap (audit 10/19).
- '(partial)' rows: n below the 84 clips of tasks/neutral_832.json (generation or scoring still running).

### set `neutral_832_long` (n expected 4 per arm = 4 scenes x P6)

| model_arm | set | n | res | GPUs | fps gen ↑ | fps e2e ↑ | gen s ↓ | TTFF s ↓ | VRAM GiB ↓ | judge Q ↑ | judge drift ↓ | VBench Q ↑ | imaging ↑ | MUSIQ drift ↓ | RPE Rot ° ↓ | react s ↓ | react fr ↓ | black |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ours_adapt25_neutral_832_long | neutral_832_long | 4 | 832x480 | 1 | 9.5 | 6.3 | 103.9 | 40.88 | 49.5 | <u>7.25</u> | 83.8 | 69.22 | 70.0 | 5.97 | 0.825 | - | - | - |
| ours_adapt25c1hyb_cloud_neutral_832_long | neutral_832_long | 4 | 832x480 | 2 | 6.8 | - | 142.9 | - | 47.3 (dev) | 7.00 | 72.5 | 69.84 | 68.3 | 6.95 | 0.517 | - | - | - |
| ours_adapt25c1hyb_dev_neutral_832_long | neutral_832_long | 4 | 832x480 | 2 | 6.8 | - | 142.2 | - | 47.5 (dev) | <u>7.25</u> | 72.5 | 69.81 | 67.7 | 7.33 | **0.457** | - | - | - |
| ours_mx24_neutral_832_long | neutral_832_long | 4 | 832x480 | 1 | 7.4 | - | 130.0 | - | **22.9** | 7.00 | 83.8 | 69.42 | 67.9 | 6.06 | 0.999 | - | - | - |
| ours_mx32_neutral_832_long | neutral_832_long | 4 | 832x480 | 1 | 10.8 | - | 89.5 | - | 31.2 | 6.75 | 70.0 | 68.95 | 67.0 | 5.96 | 0.880 | - | - | - |
| ours_mxK8_neutral_832_long | neutral_832_long | 4 | 832x480 | 1 | 9.1 | 6.1 | 104.7 | <u>39.80</u> | 38.4 | 6.50 | 83.8 | 67.96 | 65.9 | 5.43 | 0.786 | - | - | - |
| ours_mxK8c1hyb_cloud_neutral_832_long | neutral_832_long | 4 | 832x480 | 2 | 6.5 | - | 146.8 | - | 45.7 (dev) | <u>7.25</u> | 72.5 | 69.89 | 69.5 | 5.54 | 0.495 | - | - | - |
| ours_mxK8c1hyb_dev_neutral_832_long | neutral_832_long | 4 | 832x480 | 2 | 6.5 | - | 146.1 | - | 32.2 (dev) | 7.00 | 72.5 | 69.71 | 68.5 | 5.72 | 0.622 | - | - | - |
| ours_mxQ4_neutral_832_long | neutral_832_long | 4 | 832x480 | 1 | 6.9 | 5.0 | 140.0 | 41.16 | 40.8 | <u>7.25</u> | 81.2 | 68.40 | 67.3 | 7.15 | 0.745 | - | - | - |
| ours_mxQ4c1hyb_cloud_neutral_832_long | neutral_832_long | 4 | 832x480 | 2 | 4.9 | - | 196.4 | - | 45.7 (dev) | <u>7.25</u> | 86.2 | 70.22 | **71.1** | 5.90 | 0.475 | - | - | - |
| ours_mxQ4c1hyb_dev_neutral_832_long | neutral_832_long | 4 | 832x480 | 2 | 4.9 | - | 195.7 | - | 34.3 (dev) | <u>7.25</u> | 86.2 | 70.23 | <u>70.5</u> | 6.36 | 0.576 | - | - | - |
| ours_mxW12_neutral_832_long | neutral_832_long | 4 | 832x480 | 1 | 11.3 | <u>7.0</u> | 85.7 | 39.91 | 42.2 | <u>7.25</u> | 81.2 | 69.30 | 68.0 | 6.28 | 0.718 | - | - | - |
| ours_mxW12c1hyb_cloud_neutral_832_long | neutral_832_long | 4 | 832x480 | 2 | 7.9 | - | 121.1 | - | 47.3 (dev) | <u>7.25</u> | 81.2 | 70.90 | 70.3 | 5.47 | 0.822 | - | - | - |
| ours_mxW12c1hyb_dev_neutral_832_long | neutral_832_long | 4 | 832x480 | 2 | 7.9 | - | 120.4 | - | 40.2 (dev) | 7.00 | 83.8 | 70.96 | 69.7 | 5.85 | 0.557 | - | - | - |
| ours_mxW9_neutral_832_long | neutral_832_long | 4 | 832x480 | 1 | <u>12.3</u> | **7.4** | <u>78.6</u> | 40.70 | 38.6 | 7.00 | <u>50.0</u> | <u>71.95</u> | 63.9 | 3.93 | 4.478 | - | - | - |
| ours_mxW9c1hyb_cloud_neutral_832_long | neutral_832_long | 4 | 832x480 | 2 | **12.4** | - | 79.1 | - | 47.3 (dev) | <u>7.25</u> | **47.5** | **72.71** | 66.8 | **3.06** | 4.181 | - | - | - |
| ours_mxW9c1hyb_dev_neutral_832_long | neutral_832_long | 4 | 832x480 | 2 | **12.4** | - | **78.4** | - | 36.6 (dev) | 7.00 | **47.5** | 71.67 | 63.8 | <u>3.64</u> | 4.407 | - | - | - |
| ours_rt2_neutral_832_long | neutral_832_long | 4 | 832x480 | 1 | 9.5 | 6.3 | 100.4 | 39.82 | 49.3 | <u>7.25</u> | 81.2 | 67.98 | 67.3 | 8.59 | 0.735 | - | - | - |
| ours_rt2c1hyb24_cloud_neutral_832_long | neutral_832_long | 4 | 832x480 | 2 | 5.3 | - | 181.2 | - | 45.7 (dev) | 7.00 | 80.0 | 71.07 | 69.4 | 6.48 | 0.482 | - | - | - |
| ours_rt2c1hyb24_dev_neutral_832_long | neutral_832_long | 4 | 832x480 | 2 | 5.3 | - | 180.5 | - | **22.9 (dev)** | 7.00 | 70.0 | 71.01 | 67.7 | 6.94 | 0.467 | - | - | - |
| ours_rt2c1hyb32_cloud_neutral_832_long | neutral_832_long | 4 | 832x480 | 2 | 7.7 | - | 125.5 | - | 45.7 (dev) | **7.50** | 70.0 | 70.38 | 68.9 | 5.45 | 0.525 | - | - | - |
| ours_rt2c1hyb32_dev_neutral_832_long | neutral_832_long | 4 | 832x480 | 2 | 7.7 | - | 124.8 | - | <u>31.1 (dev)</u> | <u>7.25</u> | 70.0 | 70.24 | 67.6 | 5.88 | <u>0.460</u> | - | - | - |
| ours_rt2c1hyb_cloud_neutral_832_long | neutral_832_long | 4 | 832x480 | 2 | 6.8 | - | 140.6 | - | 47.3 (dev) | 7.00 | 86.2 | 69.80 | 69.6 | 6.60 | 0.551 | - | - | - |
| ours_rt2c1hyb_dev_neutral_832_long | neutral_832_long | 4 | 832x480 | 2 | 6.8 | - | 139.9 | - | 47.3 (dev) | 7.00 | 83.8 | 69.73 | 69.0 | 6.86 | 0.568 | - | - | - |
| ours_sf24_neutral_832_long | neutral_832_long | 4 | 832x480 | 1 | 5.7 | 5.1 | 173.0 | **35.04** | - | 7.00 | 75.0 | 69.27 | 68.4 | 6.16 | 0.899 | - | - | 0 (1) |
| ours_sf24c1hyb_cloud_neutral_832_long | neutral_832_long | 4 | 832x480 | 2 | 5.3 | - | 185.8 | - | 47.3 (dev) | 7.00 | 68.8 | 70.35 | 68.9 | 6.27 | 0.771 | - | - | - |
| ours_sf24c1hyb_dev_neutral_832_long | neutral_832_long | 4 | 832x480 | 2 | 5.3 | - | 185.1 | - | 47.4 (dev) | 7.00 | 71.2 | 70.30 | 68.3 | 6.66 | 0.829 | - | - | - |
| ours_teacher_neutral_832_long | neutral_832_long | 4 | 832x480 | 1 | 5.3 | 4.1 | 182.2 | 40.05 | 49.2 | 7.00 | 86.2 | 67.95 | 69.8 | 6.09 | 0.828 | - | - | 0 (1) |

- `ours_adapt25c1hyb_cloud_neutral_832_long`, `ours_adapt25c1hyb_dev_neutral_832_long`, `ours_mxK8c1hyb_cloud_neutral_832_long`, `ours_mxK8c1hyb_dev_neutral_832_long`, `ours_mxQ4c1hyb_cloud_neutral_832_long`, `ours_mxQ4c1hyb_dev_neutral_832_long`, `ours_mxW12c1hyb_cloud_neutral_832_long`, `ours_mxW12c1hyb_dev_neutral_832_long`, `ours_mxW9c1hyb_cloud_neutral_832_long`, `ours_mxW9c1hyb_dev_neutral_832_long`, `ours_rt2c1hyb24_cloud_neutral_832_long`, `ours_rt2c1hyb24_dev_neutral_832_long`, `ours_rt2c1hyb32_cloud_neutral_832_long`, `ours_rt2c1hyb32_dev_neutral_832_long`, `ours_rt2c1hyb_cloud_neutral_832_long`, `ours_rt2c1hyb_dev_neutral_832_long`, `ours_sf24c1hyb_cloud_neutral_832_long`, `ours_sf24c1hyb_dev_neutral_832_long`: 2 GPUs (device + cloud): VRAM = device GPU peak only (the cloud GPU's peak is timing.json `cloud_peak_alloc_GiB`); TTFF / fps e2e are not measured in the two-process mode.
- `ours_sf24_neutral_832_long` (4 of 4 clips): fps gen / gen s are DiT-only, corrected for the side-stream decode overlap (audit 10/19).

### set `neutral_832_nocap` (n expected 84 per arm)

| model_arm | set | n | res | GPUs | fps gen ↑ | fps e2e ↑ | gen s ↓ | TTFF s ↓ | VRAM GiB ↓ | judge Q ↑ | judge drift ↓ | VBench Q ↑ | imaging ↑ | MUSIQ drift ↓ | RPE Rot ° ↓ | react s ↓ | react fr ↓ | black |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| astra_nocap (Astra on neutral scenes, no caption) | neutral_832_nocap | 84 | 832x480 | 1 | 1.3 | 1.1 | 173.1 | 25.80 | **14.9** | 4.58 | 89.2 | 68.74 | 57.6 | 6.81 | 2.115 | 1.10 (1) | 22.0 (1) | 7 |
| *astra_nocap (Astra on neutral scenes, no caption) excl. black (n=77)* | neutral_832_nocap | 77 | 832x480 | 1 | 1.3 | 1.1 | 173.2 | 25.81 | 14.9 | 4.57 | 88.6 | 69.15 | 59.2 | 6.52 | 2.024 | 1.10 (1) | 22.0 (1) | 0 |
| ours_adapt25_neutral_832_nocap | neutral_832_nocap | 84 | 832x480 | 1 | 9.5 | 6.2 | 17.6 | 8.02 | 62.8 | 7.54 | 35.9 | 75.33 | <u>69.2</u> | <u>4.23</u> | 1.083 | 0.13 (12) | 2.1 (12) | 4 |
| *ours_adapt25_neutral_832_nocap excl. black (n=80)* | neutral_832_nocap | 80 | 832x480 | 1 | 9.5 | 6.2 | 17.6 | 8.02 | 62.8 | 7.55 | 33.9 | 75.65 | 69.9 | 3.90 | 1.070 | 0.13 (12) | 2.1 (12) | 0 |
| ours_adapt25c1hyb_cloud_neutral_832_nocap | neutral_832_nocap | 84 | 832x480 | 2 | 6.7 | - | 24.0 | - | 62.1 (dev) | 7.69 | 32.1 | 75.43 | 68.5 | 4.37 | <u>0.762</u> | 0.15 (12) | 2.4 (12) | 9 |
| *ours_adapt25c1hyb_cloud_neutral_832_nocap excl. black (n=75)* | neutral_832_nocap | 75 | 832x480 | 2 | 6.7 | - | 24.0 | - | 62.1 (dev) | 7.81 | 28.8 | 75.98 | 70.2 | 3.43 | 0.719 | 0.15 (12) | 2.4 (12) | 0 |
| ours_adapt25c1hyb_dev_neutral_832_nocap | neutral_832_nocap | 84 | 832x480 | 2 | 6.7 | - | 23.2 | - | 62.3 (dev) | <u>7.70</u> | 33.3 | <u>75.48</u> | 68.3 | 4.40 | 0.775 | 0.14 (12) | 2.2 (12) | 9 |
| *ours_adapt25c1hyb_dev_neutral_832_nocap excl. black (n=75)* | neutral_832_nocap | 75 | 832x480 | 2 | 6.7 | - | 23.3 | - | 62.3 (dev) | 7.83 | 31.1 | 76.04 | 70.0 | 3.47 | 0.734 | 0.14 (12) | 2.2 (12) | 0 |
| ours_rt2_neutral_832_nocap | neutral_832_nocap | 84 | 832x480 | 1 | 9.6 | 6.3 | 16.1 | 7.69 | 62.6 | 7.57 | <u>31.6</u> | 74.89 | 67.0 | 4.87 | 0.991 | **0.10 (12)** | **1.6 (12)** | 3 |
| *ours_rt2_neutral_832_nocap excl. black (n=81)* | neutral_832_nocap | 81 | 832x480 | 1 | 9.6 | 6.3 | 16.1 | 7.68 | 62.6 | 7.59 | 30.7 | 75.15 | 67.9 | 4.46 | 0.993 | 0.10 (12) | 1.6 (12) | 0 |
| ours_sf1p3b_cal_neutral_832_nocap | neutral_832_nocap | 84 | 832x480 | 1 | 15.3 | 11.7 | 10.8 | 6.77 | <u>28.7</u> | 7.02 | 41.8 | 73.21 | 63.5 | 4.67 | 1.291 | <u>0.12 (12)</u> | 1.9 (12) | 0 |
| ours_sf1p3b_cal_neutral_832_nocap_idle | neutral_832_nocap | 22 (partial) | 832x480 | 1 | 16.2 | 12.3 | 10.3 | **6.73** | <u>28.7</u> | 6.95 | 34.1 | 73.19 | 63.1 | 4.36 | 1.400 | 0.26 (3) | 4.2 (3) | - |
| ours_sf1p3b_pub_neutral_832_nocap | neutral_832_nocap | 84 | 832x480 | 1 | <u>16.7</u> | <u>12.4</u> | <u>10.0</u> | <u>6.76</u> | <u>28.7</u> | 6.05 | 48.5 | 71.37 | 60.1 | 5.10 | 1.358 | <u>0.12 (12)</u> | 2.0 (12) | 0 |
| ours_sf1p3b_pub_neutral_832_nocap_idle | neutral_832_nocap | 22 (partial) | 832x480 | 1 | **17.9** | **13.1** | **9.5** | 6.77 | <u>28.7</u> | 6.18 | 49.3 | 70.53 | 60.1 | 4.63 | 1.490 | **0.10 (3)** | <u>1.7 (3)</u> | - |
| ours_sf24_neutral_832_nocap | neutral_832_nocap | 84 | 832x480 | 1 | 6.6 | 5.8 | 25.0 | 8.00 | - | **7.88** | **25.7** | **76.57** | 68.9 | 4.75 | **0.713** | **0.10 (12)** | **1.6 (12)** | 7 |
| *ours_sf24_neutral_832_nocap excl. black (n=77)* | neutral_832_nocap | 77 | 832x480 | 1 | 6.6 | 5.9 | 24.9 | 8.00 | - | 7.96 | 22.6 | 77.06 | 70.0 | 4.33 | 0.697 | 0.10 (12) | 1.6 (12) | 0 |
| ours_student1p3b_neutral_832_nocap | neutral_832_nocap | 84 | 832x480 | 1 | 10.2 | 8.2 | 16.2 | 7.70 | - | 7.35 | 39.6 | 74.71 | 64.4 | 4.50 | 1.272 | <u>0.12 (12)</u> | 2.0 (12) | 5 |
| *ours_student1p3b_neutral_832_nocap excl. black (n=79)* | neutral_832_nocap | 79 | 832x480 | 1 | 10.1 | 8.2 | 16.4 | 7.75 | - | 7.41 | 36.9 | 75.19 | 65.7 | 3.95 | 1.212 | 0.12 (12) | 2.0 (12) | 0 |
| ours_teacher_neutral_832_nocap | neutral_832_nocap | 84 | 832x480 | 1 | 5.3 | 4.1 | 29.4 | 8.34 | 62.6 | 7.56 | 35.9 | 75.14 | **69.8** | **4.16** | 1.104 | <u>0.12 (12)</u> | 2.0 (12) | 3 |
| *ours_teacher_neutral_832_nocap excl. black (n=81)* | neutral_832_nocap | 81 | 832x480 | 1 | 5.3 | 4.1 | 29.4 | 8.34 | 62.6 | 7.58 | 34.3 | 75.41 | 70.7 | 3.74 | 1.114 | 0.12 (12) | 2.0 (12) | 0 |

- `astra_nocap` = Astra run on the 12 neutral_832 scenes with an empty prompt (NOT the Astra scenes without captions); it belongs to this group only and is ranked here.
- `astra_nocap` (84 of 84 clips): timing.json flags a shared-GPU measurement (speed columns are a lower bound there).
- `astra_nocap` (7 of 84), `ours_adapt25_neutral_832_nocap` (4 of 84), `ours_adapt25c1hyb_cloud_neutral_832_nocap` (9 of 84), `ours_adapt25c1hyb_dev_neutral_832_nocap` (9 of 84), `ours_rt2_neutral_832_nocap` (3 of 84), `ours_sf24_neutral_832_nocap` (7 of 84), `ours_student1p3b_neutral_832_nocap` (5 of 84), `ours_teacher_neutral_832_nocap` (3 of 84): clips go black (see the 'excl. black' rows).
- `ours_adapt25c1hyb_cloud_neutral_832_nocap`, `ours_adapt25c1hyb_dev_neutral_832_nocap`: 2 GPUs (device + cloud): VRAM = device GPU peak only (the cloud GPU's peak is timing.json `cloud_peak_alloc_GiB`); TTFF / fps e2e are not measured in the two-process mode.
- `ours_sf1p3b_cal_neutral_832_nocap` (22 of 84 clips): timing columns (steady_fps_gen, steady_fps_gen_dit, gen_s, gen_s_dit, e2e_s, ttff_s, steady_fps_e2e) from an exclusive-GPU re-measure (/home/hk28965/specwm/baselines/runs/ours_sf1p3b_cal_neutral_832_nocap_idle).
- `ours_sf1p3b_cal_neutral_832_nocap` (84 of 84 clips), `ours_sf1p3b_cal_neutral_832_nocap_idle` (22 of 22 clips), `ours_sf1p3b_pub_neutral_832_nocap` (84 of 84 clips), `ours_sf1p3b_pub_neutral_832_nocap_idle` (22 of 22 clips), `ours_sf24_neutral_832_nocap` (84 of 84 clips), `ours_student1p3b_neutral_832_nocap` (84 of 84 clips): fps gen / gen s are DiT-only, corrected for the side-stream decode overlap (audit 10/19).
- `ours_sf1p3b_pub_neutral_832_nocap` (22 of 84 clips): timing columns (steady_fps_gen, steady_fps_gen_dit, gen_s, gen_s_dit, e2e_s, ttff_s, steady_fps_e2e) from an exclusive-GPU re-measure (/home/hk28965/specwm/baselines/runs/ours_sf1p3b_pub_neutral_832_nocap_idle).
- '(partial)' rows: n below the 84 clips of tasks/neutral_832_nocap.json (generation or scoring still running).

### set `neutral_abot` (n expected 60 per arm = 5 programs P1-P5)

| model_arm | set | n | res | GPUs | fps gen ↑ | fps e2e ↑ | gen s ↓ | TTFF s ↓ | VRAM GiB ↓ | judge Q ↑ | judge drift ↓ | VBench Q ↑ | imaging ↑ | MUSIQ drift ↓ | RPE Rot ° ↓ | react s ↓ | react fr ↓ | black |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| abot | neutral_abot | 60 | 1280x704 | 1 | **15.4** | **12.7** | **7.9** | **1.06** | **32.1** | 7.07 | 43.5 | 73.81 | 62.6 | 6.16 | **0.631** | 0.42 (12) | 5.1 (12) | 3 |
| *abot excl. black (n=57)* | neutral_abot | 57 | 1280x704 | 1 | 15.4 | 12.7 | 7.9 | 1.06 | 32.1 | 7.07 | 41.7 | 74.30 | 64.0 | 5.60 | 0.634 | 0.42 (12) | 5.1 (12) | 0 |
| ours_adapt25_neutral_abot | neutral_abot | 60 | 1280x704 | 1 | 3.6 | 2.5 | 47.4 | 17.23 | 85.1 | 7.62 | 32.2 | 75.70 | 70.6 | 4.02 | 0.993 | 0.13 (11) | 2.1 (11) | 2 |
| *ours_adapt25_neutral_abot (P2p+P4p only, n=24)* | neutral_abot | 24 | 1280x704 | 1 | 3.6 | 2.5 | 47.0 | 17.28 | 85.1 | 7.75 | 36.0 | 75.09 | 71.1 | 3.93 | 1.438 | - | - | 0 |
| *ours_adapt25_neutral_abot excl. black (n=58)* | neutral_abot | 58 | 1280x704 | 1 | 3.6 | 2.5 | 47.4 | 17.22 | 85.1 | 7.64 | 32.7 | 75.90 | 71.1 | 3.82 | 0.997 | 0.13 (11) | 2.1 (11) | 0 |
| ours_adapt25c1hyb_cloud_neutral_abot | neutral_abot | 60 | 1280x704 | 2 | 2.4 | - | 72.9 | - | <u>84.3 (dev)</u> | <u>7.87</u> | 26.3 | **76.73** | <u>71.2</u> | <u>3.44</u> | 0.727 | <u>0.12 (12)</u> | <u>1.9 (12)</u> | 3 |
| *ours_adapt25c1hyb_cloud_neutral_abot (P2p+P4p only, n=24)* | neutral_abot | 24 | 1280x704 | 2 | 2.5 | - | 70.1 | - | 84.3 (dev) | 8.00 | 27.5 | 76.50 | 72.1 | 3.25 | 0.925 | - | - | 1 |
| *ours_adapt25c1hyb_cloud_neutral_abot excl. black (n=57)* | neutral_abot | 57 | 1280x704 | 2 | 2.4 | - | 73.4 | - | 84.3 (dev) | 7.93 | 26.1 | 77.09 | 72.2 | 3.09 | 0.666 | 0.12 (12) | 1.9 (12) | 0 |
| ours_adapt25c1hyb_dev_neutral_abot | neutral_abot | 60 | 1280x704 | 2 | 2.4 | - | 71.0 | - | 84.6 (dev) | **7.88** | **23.4** | <u>76.66</u> | 70.9 | 3.57 | 0.711 | <u>0.12 (12)</u> | 2.0 (12) | 3 |
| *ours_adapt25c1hyb_dev_neutral_abot (P2p+P4p only, n=24)* | neutral_abot | 24 | 1280x704 | 2 | 2.5 | - | 68.2 | - | 84.6 (dev) | 7.88 | 25.2 | 76.54 | 72.0 | 3.26 | 0.928 | - | - | 1 |
| *ours_adapt25c1hyb_dev_neutral_abot excl. black (n=57)* | neutral_abot | 57 | 1280x704 | 2 | 2.4 | - | 71.5 | - | 84.6 (dev) | 7.95 | 23.3 | 77.01 | 71.9 | 3.22 | 0.649 | 0.12 (12) | 2.0 (12) | 0 |
| ours_rt2_neutral_abot | neutral_abot | 60 | 1280x704 | 1 | 3.7 | 2.5 | 43.5 | <u>16.30</u> | 84.8 | 7.72 | 27.8 | 76.08 | 70.5 | 3.72 | 0.923 | 0.20 (12) | 3.3 (12) | 3 |
| *ours_rt2_neutral_abot (P2p+P4p only, n=24)* | neutral_abot | 24 | 1280x704 | 1 | 3.7 | 2.5 | 43.4 | 16.33 | 84.8 | 7.50 | 34.8 | 74.78 | 70.4 | 4.16 | 1.333 | - | - | 0 |
| *ours_rt2_neutral_abot excl. black (n=57)* | neutral_abot | 57 | 1280x704 | 1 | 3.7 | 2.5 | 43.5 | 16.30 | 84.8 | 7.70 | 27.5 | 76.14 | 70.8 | 3.62 | 0.945 | 0.20 (12) | 3.3 (12) | 0 |
| ours_sf24_neutral_abot | neutral_abot | 60 | 1280x704 | 1 | 2.4 | 2.2 | 70.9 | 25.02 | - | **7.88** | 26.0 | <u>76.66</u> | 70.9 | 3.51 | <u>0.677</u> | **0.11 (12)** | **1.8 (12)** | 2 |
| *ours_sf24_neutral_abot (P2p+P4p only, n=24)* | neutral_abot | 24 | 1280x704 | 1 | 2.4 | 2.1 | 70.5 | 24.40 | - | 8.12 | 26.0 | 76.77 | 72.5 | 3.27 | 0.963 | - | - | 0 |
| *ours_sf24_neutral_abot excl. black (n=58)* | neutral_abot | 58 | 1280x704 | 1 | 2.4 | 2.2 | 71.0 | 25.05 | - | 7.88 | 26.2 | 76.82 | 71.4 | 3.27 | 0.684 | 0.11 (12) | 1.8 (12) | 0 |
| ours_student1p3b_neutral_abot | neutral_abot | 60 | 1280x704 | 1 | <u>4.3</u> | <u>3.6</u> | <u>39.5</u> | 22.51 | - | 7.13 | 52.0 | 75.57 | 65.6 | 4.72 | 1.019 | 0.13 (12) | 2.1 (12) | 1 |
| *ours_student1p3b_neutral_abot (P2p+P4p only, n=24)* | neutral_abot | 24 | 1280x704 | 1 | 4.4 | 3.6 | 38.0 | 21.77 | - | 7.38 | 37.1 | 76.44 | 67.0 | 3.98 | 1.714 | - | - | 0 |
| *ours_student1p3b_neutral_abot excl. black (n=59)* | neutral_abot | 59 | 1280x704 | 1 | 4.3 | 3.5 | 39.5 | 22.53 | - | 7.14 | 51.4 | 75.73 | 66.0 | 4.56 | 1.032 | 0.13 (12) | 2.1 (12) | 0 |
| ours_teacher_neutral_abot | neutral_abot | 60 | 1280x704 | 1 | 2.0 | 1.6 | 79.3 | 18.07 | 84.8 | 7.83 | <u>25.4</u> | 76.37 | **71.6** | **3.34** | 1.068 | 0.27 (11) | 4.3 (11) | 2 |
| *ours_teacher_neutral_abot (P2p+P4p only, n=24)* | neutral_abot | 24 | 1280x704 | 1 | 2.0 | 1.6 | 79.3 | 18.00 | 84.8 | 7.79 | 35.0 | 75.06 | 72.1 | 3.30 | 1.331 | - | - | 0 |
| *ours_teacher_neutral_abot excl. black (n=58)* | neutral_abot | 58 | 1280x704 | 1 | 2.0 | 1.6 | 79.3 | 18.06 | 84.8 | 7.83 | 25.9 | 76.52 | 72.2 | 3.07 | 1.082 | 0.27 (11) | 4.3 (11) | 0 |

- Set means use programs P1-P5 only for EVERY arm (ABot has no key magnitude, so its P2p = P2 and P4p = P4 videos are identical); ours' P2p/P4p clips are listed in separate '(P2p+P4p only)' rows and are not ranked.
- `abot` (24 clips): P2p/P4p clips exist but are byte-identical to the P2/P4 clips (no key magnitude) and are not counted.
- `abot` (3 of 60), `ours_adapt25_neutral_abot` (2 of 60), `ours_adapt25c1hyb_cloud_neutral_abot` (3 of 60), `ours_adapt25c1hyb_dev_neutral_abot` (3 of 60), `ours_rt2_neutral_abot` (3 of 60), `ours_sf24_neutral_abot` (2 of 60), `ours_student1p3b_neutral_abot` (1 of 60), `ours_teacher_neutral_abot` (2 of 60): clips go black (see the 'excl. black' rows).
- `ours_adapt25c1hyb_cloud_neutral_abot`, `ours_adapt25c1hyb_dev_neutral_abot`: 2 GPUs (device + cloud): VRAM = device GPU peak only (the cloud GPU's peak is timing.json `cloud_peak_alloc_GiB`); TTFF / fps e2e are not measured in the two-process mode.
- `ours_sf24_neutral_abot` (60 of 60 clips), `ours_student1p3b_neutral_abot` (60 of 60 clips): fps gen / gen s are DiT-only, corrected for the side-stream decode overlap (audit 10/19).

### set `neutral_mg2` (n expected 84 per arm)

| model_arm | set | n | res | GPUs | fps gen ↑ | fps e2e ↑ | gen s ↓ | TTFF s ↓ | VRAM GiB ↓ | judge Q ↑ | judge drift ↓ | VBench Q ↑ | imaging ↑ | MUSIQ drift ↓ | RPE Rot ° ↓ | react s ↓ | react fr ↓ | black |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mg2 | neutral_mg2 | 84 | 640x352 | 1 | **34.3** | **21.0** | **3.8** | **2.15** | **11.5** | 6.79 | 48.8 | 74.62 | 65.3 | 6.26 | **0.450** | 0.45 (12) | 5.4 (12) | 10 |
| *mg2 excl. black (n=74)* | neutral_mg2 | 74 | 640x352 | 1 | 34.3 | 21.0 | 3.8 | 2.15 | 11.5 | 7.01 | 45.8 | 75.18 | 67.5 | 5.24 | 0.413 | 0.45 (12) | 5.4 (12) | 0 |
| ours_adapt25_neutral_mg2 | neutral_mg2 | 84 | 640x352 | 1 | 19.3 | 12.1 | 8.9 | 4.73 | 43.9 | 7.32 | 38.5 | 74.78 | 66.9 | 4.31 | 1.095 | <u>0.11 (11)</u> | 1.8 (11) | 2 |
| *ours_adapt25_neutral_mg2 excl. black (n=82)* | neutral_mg2 | 82 | 640x352 | 1 | 19.3 | 12.1 | 8.9 | 4.74 | 43.9 | 7.33 | 38.1 | 74.89 | 67.2 | 4.19 | 1.086 | 0.11 (11) | 1.8 (11) | 0 |
| ours_adapt25c1hyb_cloud_neutral_mg2 | neutral_mg2 | 84 | 640x352 | 2 | 12.3 | - | 14.1 | - | <u>43.3 (dev)</u> | <u>7.61</u> | <u>30.1</u> | 75.61 | **68.3** | **3.99** | 0.709 | 0.19 (12) | 3.0 (12) | 0 |
| ours_adapt25c1hyb_dev_neutral_mg2 | neutral_mg2 | 84 | 640x352 | 2 | 12.4 | - | 13.8 | - | 43.4 (dev) | <u>7.61</u> | 30.8 | <u>75.64</u> | <u>68.0</u> | 4.10 | 0.702 | 0.15 (12) | 2.4 (12) | 0 |
| ours_rt2_neutral_mg2 | neutral_mg2 | 84 | 640x352 | 1 | 19.4 | 12.2 | 8.1 | 4.55 | 43.7 | 7.27 | 38.3 | 74.27 | 64.9 | 4.83 | 1.204 | 0.17 (12) | 2.7 (12) | 1 |
| *ours_rt2_neutral_mg2 excl. black (n=83)* | neutral_mg2 | 83 | 640x352 | 1 | 19.4 | 12.2 | 8.1 | 4.54 | 43.7 | 7.33 | 38.6 | 74.28 | 65.2 | 4.85 | 1.217 | 0.17 (12) | 2.7 (12) | 0 |
| ours_sf24_neutral_mg2 | neutral_mg2 | 84 | 640x352 | 1 | 10.6 | 9.3 | 17.3 | 6.30 | - | **7.69** | **28.3** | **76.01** | 67.2 | <u>4.03</u> | <u>0.665</u> | **0.10 (12)** | <u>1.7 (12)</u> | 1 |
| *ours_sf24_neutral_mg2 excl. black (n=83)* | neutral_mg2 | 83 | 640x352 | 1 | 10.7 | 9.3 | 17.1 | 6.26 | - | 7.67 | 28.4 | 76.01 | 67.2 | 3.89 | 0.657 | 0.10 (12) | 1.7 (12) | 0 |
| ours_student1p3b_neutral_mg2 | neutral_mg2 | 84 | 640x352 | 1 | <u>21.9</u> | <u>17.3</u> | <u>7.4</u> | <u>4.05</u> | - | 7.12 | 40.5 | 74.94 | 62.9 | 4.31 | 1.186 | **0.10 (12)** | **1.6 (12)** | 2 |
| *ours_student1p3b_neutral_mg2 excl. black (n=82)* | neutral_mg2 | 82 | 640x352 | 1 | 21.9 | 17.3 | 7.4 | 4.05 | - | 7.12 | 39.3 | 75.11 | 63.3 | 4.06 | 1.193 | 0.10 (12) | 1.6 (12) | 0 |
| ours_teacher_neutral_mg2 | neutral_mg2 | 84 | 640x352 | 1 | 10.6 | 8.0 | 14.9 | 4.92 | 43.7 | 7.40 | 37.0 | 74.84 | 67.8 | 4.27 | 1.129 | 0.15 (12) | 2.4 (12) | 2 |
| *ours_teacher_neutral_mg2 excl. black (n=82)* | neutral_mg2 | 82 | 640x352 | 1 | 10.6 | 8.0 | 14.9 | 4.91 | 43.7 | 7.44 | 36.1 | 74.89 | 68.2 | 4.26 | 1.124 | 0.15 (12) | 2.4 (12) | 0 |

- `mg2` (10 of 84), `ours_adapt25_neutral_mg2` (2 of 84), `ours_rt2_neutral_mg2` (1 of 84), `ours_sf24_neutral_mg2` (1 of 84), `ours_student1p3b_neutral_mg2` (2 of 84), `ours_teacher_neutral_mg2` (2 of 84): clips go black (see the 'excl. black' rows).
- `ours_adapt25c1hyb_cloud_neutral_mg2`, `ours_adapt25c1hyb_dev_neutral_mg2`: 2 GPUs (device + cloud): VRAM = device GPU peak only (the cloud GPU's peak is timing.json `cloud_peak_alloc_GiB`); TTFF / fps e2e are not measured in the two-process mode.
- `ours_sf24_neutral_mg2` (84 of 84 clips), `ours_student1p3b_neutral_mg2` (84 of 84 clips): fps gen / gen s are DiT-only, corrected for the side-stream decode overlap (audit 10/19).

### set `neutral_wp` (n expected 84 per arm)

| model_arm | set | n | res | GPUs | fps gen ↑ | fps e2e ↑ | gen s ↓ | TTFF s ↓ | VRAM GiB ↓ | judge Q ↑ | judge drift ↓ | VBench Q ↑ | imaging ↑ | MUSIQ drift ↓ | RPE Rot ° ↓ | react s ↓ | react fr ↓ | black |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ours_adapt25_neutral_wp | neutral_wp | 84 | 1024x512 | 1 | 6.3 | 4.3 | 26.2 | 10.44 | 76.4 | 7.76 | 33.2 | 75.02 | 68.4 | 4.12 | 1.005 | 0.22 (12) | 3.5 (12) | 8 |
| *ours_adapt25_neutral_wp excl. black (n=76)* | neutral_wp | 76 | 1024x512 | 1 | 6.3 | 4.3 | 26.2 | 10.44 | 76.4 | 7.86 | 30.0 | 75.52 | 69.7 | 3.48 | 0.923 | 0.22 (12) | 3.5 (12) | 0 |
| ours_adapt25c1hyb_cloud_neutral_wp | neutral_wp | 84 | 1024x512 | 2 | 4.1 | - | 44.3 | - | <u>75.7 (dev)</u> | 7.81 | <u>27.4</u> | <u>75.92</u> | 69.1 | <u>3.98</u> | <u>0.763</u> | 0.12 (12) | 1.9 (12) | 12 |
| *ours_adapt25c1hyb_cloud_neutral_wp excl. black (n=72)* | neutral_wp | 72 | 1024x512 | 2 | 4.0 | - | 45.3 | - | 75.7 (dev) | 7.97 | 21.2 | 77.01 | 72.0 | 2.79 | 0.677 | 0.12 (12) | 1.9 (12) | 0 |
| ours_adapt25c1hyb_dev_neutral_wp | neutral_wp | 84 | 1024x512 | 2 | 4.1 | - | 43.2 | - | 75.9 (dev) | <u>7.87</u> | 27.5 | 75.81 | 68.8 | 4.07 | 0.810 | 0.11 (12) | <u>1.8 (12)</u> | 12 |
| *ours_adapt25c1hyb_dev_neutral_wp excl. black (n=72)* | neutral_wp | 72 | 1024x512 | 2 | 4.0 | - | 44.2 | - | 75.9 (dev) | 8.03 | 21.1 | 76.89 | 71.8 | 2.91 | 0.682 | 0.11 (12) | 1.8 (12) | 0 |
| ours_rt2_neutral_wp | neutral_wp | 84 | 1024x512 | 1 | 6.4 | 4.4 | <u>23.9</u> | <u>9.91</u> | 76.2 | 7.69 | 33.4 | 75.41 | 68.3 | 4.68 | 0.940 | <u>0.09 (12)</u> | **1.4 (12)** | 5 |
| *ours_rt2_neutral_wp excl. black (n=79)* | neutral_wp | 79 | 1024x512 | 1 | 6.4 | 4.4 | 23.9 | 9.90 | 76.2 | 7.72 | 31.3 | 75.73 | 69.1 | 4.35 | 0.921 | 0.09 (12) | 1.4 (12) | 0 |
| ours_sf24_neutral_wp | neutral_wp | 84 | 1024x512 | 1 | 4.3 | 3.8 | 39.7 | 11.34 | - | **8.06** | **20.4** | **76.79** | **69.6** | **3.62** | **0.672** | 0.12 (12) | 2.0 (12) | 3 |
| *ours_sf24_neutral_wp excl. black (n=81)* | neutral_wp | 81 | 1024x512 | 1 | 4.3 | 3.8 | 39.6 | 11.36 | - | 8.09 | 19.0 | 77.05 | 70.0 | 3.45 | 0.677 | 0.12 (12) | 2.0 (12) | 0 |
| ours_student1p3b_neutral_wp | neutral_wp | 84 | 1024x512 | 1 | <u>7.2</u> | <u>5.8</u> | <u>23.9</u> | 10.47 | - | 7.13 | 44.2 | 74.78 | 62.6 | 5.23 | 1.269 | 0.15 (12) | 2.4 (12) | 9 |
| *ours_student1p3b_neutral_wp excl. black (n=75)* | neutral_wp | 75 | 1024x512 | 1 | 7.1 | 5.8 | 24.4 | 10.67 | - | 7.27 | 43.3 | 75.51 | 65.0 | 4.42 | 1.277 | 0.15 (12) | 2.4 (12) | 0 |
| ours_teacher_neutral_wp | neutral_wp | 84 | 1024x512 | 1 | 3.5 | 2.8 | 43.6 | 10.90 | 76.2 | 7.76 | 28.8 | 75.83 | <u>69.4</u> | 4.22 | 1.037 | 0.21 (12) | 3.3 (12) | 8 |
| *ours_teacher_neutral_wp excl. black (n=76)* | neutral_wp | 76 | 1024x512 | 1 | 3.5 | 2.8 | 43.6 | 10.90 | 76.2 | 7.89 | 26.6 | 76.47 | 71.3 | 3.31 | 0.973 | 0.21 (12) | 3.3 (12) | 0 |
| waypoint | neutral_wp | 84 | 1024x512 | 1 | **48.9** | **46.0** | **13.0** | **0.49** | **6.6** | 5.81 | 69.0 | 71.36 | **69.6** | 4.37 | 0.914 | **0.05 (12)** | 2.9 (12) | 0 |
| waypoint_idle | neutral_wp | 84 | 1024x512 | 1 | **48.9** | **46.0** | **13.0** | **0.49** | **6.6** | - | - | - | - | - | - | - | - | - |

- `ours_adapt25_neutral_wp` (8 of 84), `ours_adapt25c1hyb_cloud_neutral_wp` (12 of 84), `ours_adapt25c1hyb_dev_neutral_wp` (12 of 84), `ours_rt2_neutral_wp` (5 of 84), `ours_sf24_neutral_wp` (3 of 84), `ours_student1p3b_neutral_wp` (9 of 84), `ours_teacher_neutral_wp` (8 of 84): clips go black (see the 'excl. black' rows).
- `ours_adapt25c1hyb_cloud_neutral_wp`, `ours_adapt25c1hyb_dev_neutral_wp`: 2 GPUs (device + cloud): VRAM = device GPU peak only (the cloud GPU's peak is timing.json `cloud_peak_alloc_GiB`); TTFF / fps e2e are not measured in the two-process mode.
- `ours_sf24_neutral_wp` (84 of 84 clips), `ours_student1p3b_neutral_wp` (84 of 84 clips): fps gen / gen s are DiT-only, corrected for the side-stream decode overlap (audit 10/19).
- `waypoint` (84 of 84 clips): timing columns (steady_fps_gen, steady_fps_e2e, gen_s, e2e_s, ttff_s) from an exclusive-GPU re-measure (/home/hk28965/specwm/baselines/runs/waypoint_idle).
- `waypoint_idle` (84 of 84 clips): speed columns (fps gen / fps e2e / TTFF / gen s) are a lower bound, shared-GPU measurement (timing.json `timing_caveat`); the `_idle` exclusive-GPU re-measure is pending.

### set `wp` (n expected 84 per arm)

| model_arm | set | n | res | GPUs | fps gen ↑ | fps e2e ↑ | gen s ↓ | TTFF s ↓ | VRAM GiB ↓ | judge Q ↑ | judge drift ↓ | VBench Q ↑ | imaging ↑ | MUSIQ drift ↓ | RPE Rot ° ↓ | react s ↓ | react fr ↓ | black |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ours_adapt25_wp | wp | 84 | 1024x512 | 1 | 6.4 | 4.4 | 25.9 | 10.40 | 76.4 | 7.76 | 10.4 | 80.52 | 69.6 | 3.18 | 0.752 | 0.09 (12) | 1.5 (12) | 2 |
| *ours_adapt25_wp excl. black (n=82)* | wp | 82 | 1024x512 | 1 | 6.4 | 4.4 | 25.8 | 10.40 | 76.4 | 7.76 | 10.0 | 80.63 | 69.7 | 2.97 | 0.738 | 0.09 (12) | 1.5 (12) | 0 |
| ours_adapt25c1hyb_cloud_wp | wp | 84 | 1024x512 | 2 | 4.1 | - | 42.8 | - | 75.7 (dev) | 7.93 | 9.0 | 81.00 | 70.1 | 3.02 | 0.571 | **0.07 (12)** | <u>1.2 (12)</u> | 2 |
| *ours_adapt25c1hyb_cloud_wp excl. black (n=82)* | wp | 82 | 1024x512 | 2 | 4.2 | - | 42.4 | - | 75.7 (dev) | 7.94 | 8.7 | 81.09 | 70.3 | 2.78 | 0.564 | 0.07 (12) | 1.2 (12) | 0 |
| ours_adapt25c1hyb_dev_wp | wp | 84 | 1024x512 | 2 | 4.2 | - | 41.7 | - | 75.9 (dev) | 7.98 | 9.3 | 81.03 | 69.9 | 3.09 | 0.574 | **0.07 (12)** | <u>1.2 (12)</u> | 2 |
| *ours_adapt25c1hyb_dev_wp excl. black (n=82)* | wp | 82 | 1024x512 | 2 | 4.2 | - | 41.3 | - | 75.9 (dev) | 7.99 | 9.0 | 81.12 | 70.0 | 2.86 | 0.567 | 0.07 (12) | 1.2 (12) | 0 |
| ours_rt2_wp | wp | 84 | 1024x512 | 1 | 6.4 | 4.4 | <u>23.9</u> | <u>9.91</u> | 76.2 | 7.76 | 12.9 | 80.37 | 69.1 | 3.03 | 0.840 | 0.09 (12) | 1.5 (12) | 2 |
| *ours_rt2_wp excl. black (n=82)* | wp | 82 | 1024x512 | 1 | 6.4 | 4.4 | 23.9 | 9.92 | 76.2 | 7.78 | 12.7 | 80.41 | 69.3 | 2.96 | 0.840 | 0.09 (12) | 1.5 (12) | 0 |
| ours_rt2c1hyb32_cloud_wp | wp | 84 | 1024x512 | 2 | 5.3 | - | 30.3 | - | 75.7 (dev) | 7.80 | 10.7 | 80.95 | 69.4 | 3.12 | 0.549 | **0.07 (12)** | **1.1 (12)** | - |
| ours_rt2c1hyb32_dev_wp | wp | 84 | 1024x512 | 2 | 5.3 | - | 29.2 | - | <u>30.1 (dev)</u> | 7.71 | 10.7 | 80.92 | 68.8 | 3.10 | 0.547 | 0.11 (12) | 1.8 (12) | - |
| ours_rt2c1hyb_cloud_wp | wp | 84 | 1024x512 | 2 | 4.5 | - | 33.4 | - | 75.7 (dev) | 7.86 | 10.2 | 80.98 | 69.6 | 2.97 | 0.587 | 0.09 (12) | 1.4 (12) | 1 |
| *ours_rt2c1hyb_cloud_wp excl. black (n=83)* | wp | 83 | 1024x512 | 2 | 4.5 | - | 33.4 | - | 75.7 (dev) | 7.87 | 10.1 | 81.00 | 69.6 | 2.91 | 0.579 | 0.09 (12) | 1.4 (12) | 0 |
| ours_rt2c1hyb_dev_wp | wp | 84 | 1024x512 | 2 | 4.5 | - | 32.3 | - | 75.7 (dev) | 7.85 | 10.8 | 80.96 | 69.3 | 3.05 | 0.605 | <u>0.08 (12)</u> | 1.3 (12) | 1 |
| *ours_rt2c1hyb_dev_wp excl. black (n=83)* | wp | 83 | 1024x512 | 2 | 4.5 | - | 32.3 | - | 75.7 (dev) | 7.86 | 10.7 | 80.98 | 69.4 | 3.00 | 0.597 | 0.08 (12) | 1.3 (12) | 0 |
| ours_sf24_wp | wp | 84 | 1024x512 | 1 | 4.6 | 4.1 | 35.5 | 10.56 | - | <u>8.01</u> | <u>8.5</u> | **81.31** | 69.6 | 2.93 | <u>0.530</u> | <u>0.08 (12)</u> | 1.3 (12) | 0 |
| ours_sf24_wp_dit | wp | 2 (partial) | 1024x512 | 1 | 6.3 | - | 26.2 | - | - | **9.00** | **0.0** | 80.61 | **72.8** | **1.95** | **0.388** | - | - | - |
| ours_sf24c1hyb_cloud_wp | wp | 84 | 1024x512 | 2 | 4.2 | - | 38.4 | - | 75.7 (dev) | 7.98 | 10.4 | <u>81.13</u> | 70.1 | 2.87 | 0.549 | 0.16 (12) | 2.5 (12) | 0 |
| ours_sf24c1hyb_dev_wp | wp | 84 | 1024x512 | 2 | 4.2 | - | 37.3 | - | 75.7 (dev) | 7.93 | 9.3 | 81.04 | 70.0 | 3.04 | 0.542 | 0.16 (12) | 2.6 (12) | 0 |
| ours_student1p3b_wp | wp | 84 | 1024x512 | 1 | <u>7.1</u> | <u>5.8</u> | 24.2 | 10.60 | - | 7.40 | 21.7 | 76.60 | 58.8 | 5.72 | 0.896 | 0.10 (12) | 1.7 (12) | 2 |
| *ours_student1p3b_wp excl. black (n=82)* | wp | 82 | 1024x512 | 1 | 7.2 | 5.8 | 24.0 | 10.53 | - | 7.40 | 21.7 | 76.61 | 59.1 | 5.58 | 0.902 | 0.10 (12) | 1.7 (12) | 0 |
| ours_teacher_wp | wp | 84 | 1024x512 | 1 | 3.5 | 2.8 | 43.6 | 10.85 | 76.2 | 7.68 | 11.5 | 80.49 | 70.3 | 2.97 | 0.791 | 0.17 (12) | 2.7 (12) | 2 |
| *ours_teacher_wp excl. black (n=82)* | wp | 82 | 1024x512 | 1 | 3.5 | 2.8 | 43.6 | 10.85 | 76.2 | 7.70 | 11.0 | 80.59 | 70.5 | 2.86 | 0.781 | 0.17 (12) | 2.7 (12) | 0 |
| waypoint | wp | 84 | 1024x512 | 1 | **48.6** | **45.7** | **13.0** | **0.47** | **6.6** | 6.94 | 22.0 | 76.98 | <u>71.3</u> | <u>2.30</u> | 0.654 | **0.07 (11)** | 4.1 (11) | 0 |
| waypoint_idle | wp | 84 | 1024x512 | 1 | **48.6** | **45.7** | **13.0** | **0.47** | **6.6** | - | - | - | - | - | - | - | - | - |

- `ours_adapt25_wp` (2 of 84), `ours_adapt25c1hyb_cloud_wp` (2 of 84), `ours_adapt25c1hyb_dev_wp` (2 of 84), `ours_rt2_wp` (2 of 84), `ours_rt2c1hyb_cloud_wp` (1 of 84), `ours_rt2c1hyb_dev_wp` (1 of 84), `ours_student1p3b_wp` (2 of 84), `ours_teacher_wp` (2 of 84): clips go black (see the 'excl. black' rows).
- `ours_adapt25c1hyb_cloud_wp`, `ours_adapt25c1hyb_dev_wp`, `ours_rt2c1hyb32_cloud_wp`, `ours_rt2c1hyb32_dev_wp`, `ours_rt2c1hyb_cloud_wp`, `ours_rt2c1hyb_dev_wp`, `ours_sf24c1hyb_cloud_wp`, `ours_sf24c1hyb_dev_wp`: 2 GPUs (device + cloud): VRAM = device GPU peak only (the cloud GPU's peak is timing.json `cloud_peak_alloc_GiB`); TTFF / fps e2e are not measured in the two-process mode.
- `ours_sf24_wp` (84 of 84 clips), `ours_sf24_wp_dit` (2 of 2 clips), `ours_student1p3b_wp` (84 of 84 clips): fps gen / gen s are DiT-only, corrected for the side-stream decode overlap (audit 10/19).
- `waypoint` (84 of 84 clips): timing columns (steady_fps_gen, steady_fps_e2e, gen_s, e2e_s, ttff_s) from an exclusive-GPU re-measure (/home/hk28965/specwm/baselines/runs/waypoint_idle).
- `waypoint_idle` (84 of 84 clips): speed columns (fps gen / fps e2e / TTFF / gen s) are a lower bound, shared-GPU measurement (timing.json `timing_caveat`); the `_idle` exclusive-GPU re-measure is pending.
- '(partial)' rows: n below the 84 clips of tasks/wp.json (generation or scoring still running).

### set `wp_long` (n expected 4 per arm = 4 scenes x P6)

| model_arm | set | n | res | GPUs | fps gen ↑ | fps e2e ↑ | gen s ↓ | TTFF s ↓ | VRAM GiB ↓ | judge Q ↑ | judge drift ↓ | VBench Q ↑ | imaging ↑ | MUSIQ drift ↓ | RPE Rot ° ↓ | react s ↓ | react fr ↓ | black |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ours_adapt25_wp_long | wp_long | 4 | 1024x512 | 1 | <u>6.4</u> | <u>4.4</u> | 155.0 | 47.97 | 59.0 | **7.25** | 58.8 | 75.23 | <u>71.4</u> | 5.40 | 0.505 | - | - | - |
| ours_adapt25c1hyb_cloud_wp_long | wp_long | 4 | 1024x512 | 2 | 4.5 | - | 213.6 | - | <u>56.4 (dev)</u> | **7.25** | <u>52.5</u> | 75.49 | 65.9 | 8.61 | 0.381 | - | - | - |
| ours_adapt25c1hyb_dev_wp_long | wp_long | 4 | 1024x512 | 2 | 4.5 | - | 212.5 | - | 56.5 (dev) | **7.25** | <u>52.5</u> | 75.60 | 65.8 | 8.67 | 0.385 | - | - | - |
| ours_rt2_wp_long | wp_long | 4 | 1024x512 | 1 | <u>6.4</u> | <u>4.4</u> | <u>149.9</u> | 47.41 | 58.8 | **7.25** | **50.0** | 75.29 | 69.3 | <u>5.31</u> | 0.473 | - | - | - |
| ours_rt2c1hyb_cloud_wp_long | wp_long | 4 | 1024x512 | 2 | 4.5 | - | 209.9 | - | <u>56.4 (dev)</u> | **7.25** | 63.8 | **76.87** | 69.9 | 6.50 | <u>0.265</u> | - | - | - |
| ours_rt2c1hyb_dev_wp_long | wp_long | 4 | 1024x512 | 2 | 4.5 | - | 208.8 | - | <u>56.4 (dev)</u> | **7.25** | 63.8 | **76.87** | 69.7 | 6.69 | **0.261** | - | - | - |
| ours_sf24_wp_long | wp_long | 4 | 1024x512 | 1 | 4.7 | 4.3 | 209.4 | <u>45.96</u> | - | <u>7.00</u> | 80.0 | <u>76.48</u> | 70.1 | 8.26 | 0.421 | - | - | - |
| ours_sf24c1hyb_cloud_wp_long | wp_long | 4 | 1024x512 | 2 | 4.2 | - | 240.9 | - | <u>56.4 (dev)</u> | **7.25** | <u>52.5</u> | 76.21 | 69.3 | 6.87 | 0.361 | - | - | - |
| ours_sf24c1hyb_dev_wp_long | wp_long | 4 | 1024x512 | 2 | 4.2 | - | 239.8 | - | <u>56.4 (dev)</u> | **7.25** | <u>52.5</u> | 76.24 | 69.2 | 6.84 | 0.346 | - | - | - |
| ours_teacher_wp_long | wp_long | 4 | 1024x512 | 1 | 3.5 | 2.8 | 271.6 | 48.22 | 58.8 | **7.25** | 75.0 | 74.88 | **72.6** | 5.42 | 0.423 | - | - | - |
| waypoint | wp_long | 4 | 1024x512 | 1 | **39.7** | **37.1** | **92.3** | **0.49** | **6.6** | 5.75 | 92.5 | 72.16 | 69.7 | **4.67** | 0.705 | - | - | 0 (1) |

- `ours_adapt25c1hyb_cloud_wp_long`, `ours_adapt25c1hyb_dev_wp_long`, `ours_rt2c1hyb_cloud_wp_long`, `ours_rt2c1hyb_dev_wp_long`, `ours_sf24c1hyb_cloud_wp_long`, `ours_sf24c1hyb_dev_wp_long`: 2 GPUs (device + cloud): VRAM = device GPU peak only (the cloud GPU's peak is timing.json `cloud_peak_alloc_GiB`); TTFF / fps e2e are not measured in the two-process mode.
- `ours_sf24_wp_long` (4 of 4 clips): fps gen / gen s are DiT-only, corrected for the side-stream decode overlap (audit 10/19).
- `waypoint` (4 of 4 clips): speed columns (fps gen / fps e2e / TTFF / gen s) are a lower bound, shared-GPU measurement (timing.json `timing_caveat`); the `_idle` exclusive-GPU re-measure is pending.


## 2. Paired delta vs `ours_teacher_<set><suffix>` on the same (set, scene, program, suffix)

delta = mean(arm - ref) over paired clips; b/e/w = clips where the arm is better / equal / worse (direction-aware; equal within tolerance: judge Q exact, judge drift exact, VBench Q ±0.5, imaging ±0.5, MUSIQ drift ±0.1, RPE Rot ° ±0.1, react s ±0.05); fps e2e x = mean per-clip ratio arm / ref. [10/19 audit] abot / neutral_abot pairs use programs P1-P5 only (ABot's P2p/P4p duplicate P2/P4).

| model_arm | set | paired n | fps e2e x | Δ judge Q ↑ (b/e/w) | Δ judge drift ↓ (b/e/w) | Δ VBench Q ↑ (b/e/w) | Δ imaging ↑ (b/e/w) | Δ MUSIQ drift ↓ (b/e/w) | Δ RPE Rot ° ↓ (b/e/w) | Δ react s ↓ (b/e/w) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| abot | abot | 60 | 8.11x | -0.80 (3/20/37) | +12.8 (7/31/22) | -2.35 (13/5/42) | -3.1 (11/6/43) | +1.35 (14/2/44) | -0.184 (31/21/8) | +0.30 (1/0/9) n=10 |
| abot | abot_long | 4 | 7.97x | -0.50 (1/1/2) | +20.0 (0/2/2) | -3.76 (0/0/4) | -7.7 (0/0/4) | +4.52 (0/0/4) | +0.098 (1/0/3) | - |
| abot | neutral_abot | 60 | 7.88x | -0.77 (5/21/34) | +18.1 (10/21/29) | -2.56 (16/4/40) | -9.0 (10/4/46) | +2.83 (19/3/38) | -0.437 (39/14/7) | +0.16 (2/0/9) n=11 |
| abot_t5offload | abot | 15 | 7.77x | -0.80 (0/7/8) | +12.7 (1/9/5) | -1.77 (3/2/10) | -3.3 (1/1/13) | +1.27 (3/1/11) | -0.004 (5/5/5) | +0.39 (0/0/3) n=3 |
| astra | astra | 21 | 0.27x | -1.33 (0/6/15) | +37.9 (0/4/17) | -1.42 (7/2/12) | -3.2 (5/1/15) | +0.45 (9/0/12) | +0.429 (3/3/15) | - |
| astra | neutral_832 | 84 | 0.27x | -2.77 (0/6/78) | +38.3 (3/14/67) | -3.83 (16/10/58) | -13.9 (11/2/71) | +2.05 (30/0/54) | +0.901 (9/5/70) | +0.30 (1/1/3) n=5 |
| astra_nocap (Astra on neutral scenes, no caption) | neutral_832_nocap | 84 | 0.27x | -2.98 (0/5/79) | +53.3 (3/4/77) | -6.40 (11/4/69) | -12.2 (6/2/76) | +2.66 (20/1/63) | +1.011 (2/3/79) | +1.04 (0/0/1) n=1 |
| mg2 | mg2 | 84 | 2.63x | +0.02 (21/26/37) | +7.4 (26/25/33) | +1.48 (45/10/29) | -0.9 (35/4/45) | -0.62 (53/2/29) | -0.751 (62/18/4) | +0.10 (1/1/10) n=12 |
| mg2 | mg2_long | 4 | 2.66x | -1.00 (0/1/3) | -10.0 (2/0/2) | -3.05 (0/1/3) | -6.6 (1/0/3) | +3.27 (0/0/4) | +0.334 (3/0/1) | - |
| mg2 | neutral_mg2 | 84 | 2.63x | -0.62 (6/41/37) | +11.7 (17/29/38) | -0.22 (39/6/39) | -2.5 (37/6/41) | +1.98 (34/3/47) | -0.679 (68/15/1) | +0.30 (0/1/11) n=12 |
| ours_adapt25_abot | abot | 60 | 1.62x | -0.08 (2/51/7) | -0.7 (8/46/6) | +0.10 (15/33/12) | +0.3 (16/31/13) | -0.16 (26/10/24) | +0.002 (9/44/7) | +0.27 (0/6/4) n=10 |
| ours_adapt25_abot_long | abot_long | 4 | 1.57x | +0.50 (2/2/0) | -8.8 (1/2/1) | -1.70 (1/0/3) | -1.5 (1/2/1) | +0.89 (1/2/1) | -0.041 (1/3/0) | - |
| ours_adapt25_astra | astra | 21 | 1.53x | +0.05 (4/13/4) | -0.5 (3/16/2) | +0.18 (4/12/5) | -1.0 (4/7/10) | +0.35 (7/4/10) | -0.041 (6/13/2) | -0.01 (1/1/1) n=3 |
| ours_adapt25_mg2 | mg2 | 84 | 1.51x | -0.30 (14/43/27) | +3.9 (17/48/19) | -0.03 (25/28/31) | -1.5 (22/9/53) | +0.46 (32/5/47) | -0.028 (29/34/21) | +0.04 (3/5/4) n=12 |
| ours_adapt25_mg2_long | mg2_long | 4 | 1.52x | +0.25 (2/1/1) | -1.2 (2/1/1) | -0.39 (1/1/2) | -1.9 (1/0/3) | +1.38 (1/1/2) | +0.052 (2/1/1) | - |
| ours_adapt25_neutral_832 | neutral_832 | 84 | 1.51x | -0.19 (7/55/22) | +2.0 (14/47/23) | +0.11 (28/27/29) | -0.7 (24/20/40) | -0.20 (42/8/34) | -0.076 (25/43/16) | -0.03 (4/5/3) n=12 |
| ours_adapt25_neutral_832_long | neutral_832_long | 4 | 1.53x | +0.25 (1/3/0) | -2.5 (1/3/0) | +1.27 (3/1/0) | +0.2 (1/2/1) | -0.12 (2/0/2) | -0.003 (1/2/1) | - |
| ours_adapt25_neutral_832_nocap | neutral_832_nocap | 84 | 1.53x | -0.02 (13/55/16) | +0.0 (11/57/16) | +0.19 (33/15/36) | -0.6 (26/19/39) | +0.08 (35/6/43) | -0.021 (31/36/17) | +0.01 (3/6/3) n=12 |
| ours_adapt25_neutral_abot | neutral_abot | 60 | 1.56x | -0.22 (6/37/17) | +6.8 (7/35/18) | -0.67 (19/10/31) | -1.0 (17/13/30) | +0.68 (22/6/32) | -0.075 (18/22/20) | -0.14 (6/3/2) n=11 |
| ours_adapt25_neutral_mg2 | neutral_mg2 | 84 | 1.51x | -0.08 (12/56/16) | +1.4 (18/44/22) | -0.06 (37/13/34) | -0.9 (25/17/42) | +0.04 (38/5/41) | -0.034 (28/35/21) | -0.04 (4/5/2) n=11 |
| ours_adapt25_neutral_wp | neutral_wp | 84 | 1.54x | +0.00 (14/54/16) | +4.5 (8/59/17) | -0.81 (18/20/46) | -1.0 (28/12/44) | -0.10 (33/8/43) | -0.032 (22/36/26) | +0.01 (3/5/4) n=12 |
| ours_adapt25_wp | wp | 84 | 1.56x | +0.08 (17/58/9) | -1.1 (15/60/9) | +0.02 (23/35/26) | -0.7 (19/20/45) | +0.21 (35/9/40) | -0.038 (25/46/13) | -0.08 (4/6/2) n=12 |
| ours_adapt25_wp_long | wp_long | 4 | 1.55x | +0.00 (0/4/0) | -16.2 (3/1/0) | +0.34 (2/2/0) | -1.2 (1/0/3) | -0.02 (1/1/2) | +0.082 (0/2/2) | - |
| ours_adapt25c1hyb_cloud_abot | abot | 60 | - | +0.02 (6/48/6) | -1.4 (9/45/6) | +0.71 (19/29/12) | +0.4 (20/20/20) | -0.12 (24/15/21) | -0.163 (35/24/1) | -0.04 (2/8/1) n=11 |
| ours_adapt25c1hyb_cloud_abot_long | abot_long | 4 | - | +0.75 (2/2/0) | -2.5 (2/1/1) | +0.73 (1/0/3) | -0.7 (1/1/2) | +0.15 (2/1/1) | -0.219 (1/3/0) | - |
| ours_adapt25c1hyb_cloud_astra | astra | 21 | - | +0.33 (6/15/0) | -2.4 (4/16/1) | +0.53 (6/8/7) | -0.7 (5/7/9) | +0.03 (10/3/8) | -0.149 (12/9/0) | -0.01 (1/1/1) n=3 |
| ours_adapt25c1hyb_cloud_mg2 | mg2 | 84 | - | -0.10 (16/52/16) | -4.5 (24/48/12) | +0.90 (40/23/21) | -0.6 (26/11/47) | -0.34 (44/3/37) | -0.424 (58/23/3) | -0.04 (2/10/0) n=12 |
| ours_adapt25c1hyb_cloud_mg2_long | mg2_long | 4 | - | +0.00 (1/2/1) | -15.0 (3/0/1) | +2.76 (3/0/1) | +0.4 (2/0/2) | -0.29 (2/0/2) | -0.371 (4/0/0) | - |
| ours_adapt25c1hyb_cloud_neutral_832 | neutral_832 | 84 | - | -0.02 (12/59/13) | +1.1 (15/55/14) | +0.19 (31/25/28) | -0.5 (36/12/36) | -0.19 (43/6/35) | -0.067 (34/31/19) | -0.13 (6/4/2) n=12 |
| ours_adapt25c1hyb_cloud_neutral_832_long | neutral_832_long | 4 | - | +0.00 (1/2/1) | -13.8 (2/2/0) | +1.88 (4/0/0) | -1.5 (2/0/2) | +0.86 (1/0/3) | -0.311 (2/1/1) | - |
| ours_adapt25c1hyb_cloud_neutral_832_nocap | neutral_832_nocap | 84 | - | +0.13 (21/51/12) | -3.8 (17/54/13) | +0.29 (30/22/32) | -1.3 (25/15/44) | +0.21 (37/9/38) | -0.342 (52/22/10) | +0.03 (3/5/4) n=12 |
| ours_adapt25c1hyb_cloud_neutral_abot | neutral_abot | 60 | - | +0.03 (9/45/6) | +0.9 (13/34/13) | +0.36 (27/16/17) | -0.4 (19/20/21) | +0.10 (35/3/22) | -0.342 (39/17/4) | -0.16 (6/4/1) n=11 |
| ours_adapt25c1hyb_cloud_neutral_mg2 | neutral_mg2 | 84 | - | +0.20 (20/58/6) | -7.0 (26/46/12) | +0.77 (48/12/24) | +0.5 (38/13/33) | -0.28 (44/4/36) | -0.420 (59/21/4) | +0.03 (2/8/2) n=12 |
| ours_adapt25c1hyb_cloud_neutral_wp | neutral_wp | 84 | - | +0.05 (17/57/10) | -1.4 (22/52/10) | +0.09 (27/21/36) | -0.3 (27/21/36) | -0.24 (40/4/40) | -0.274 (53/25/6) | -0.09 (4/5/3) n=12 |
| ours_adapt25c1hyb_cloud_wp | wp | 84 | - | +0.25 (27/50/7) | -2.5 (20/58/6) | +0.50 (41/24/19) | -0.2 (30/17/37) | +0.05 (43/5/36) | -0.220 (53/31/0) | -0.10 (5/6/1) n=12 |
| ours_adapt25c1hyb_cloud_wp_long | wp_long | 4 | - | +0.00 (0/4/0) | -22.5 (3/1/0) | +0.61 (2/1/1) | -6.6 (1/0/3) | +3.18 (2/0/2) | -0.042 (0/4/0) | - |
| ours_adapt25c1hyb_dev_abot | abot | 60 | - | -0.03 (5/48/7) | -0.8 (10/43/7) | +0.77 (21/28/11) | +0.4 (18/25/17) | -0.16 (26/10/24) | -0.165 (36/23/1) | +0.01 (1/8/1) n=10 |
| ours_adapt25c1hyb_dev_abot_long | abot_long | 4 | - | +0.75 (2/1/1) | -25.0 (3/0/1) | +0.88 (1/1/2) | -0.8 (1/1/2) | +0.18 (2/1/1) | -0.230 (1/3/0) | - |
| ours_adapt25c1hyb_dev_astra | astra | 21 | - | +0.33 (6/15/0) | +0.5 (2/17/2) | +0.62 (9/6/6) | -0.8 (4/6/11) | +0.04 (11/1/9) | -0.151 (13/8/0) | +0.05 (0/1/2) n=3 |
| ours_adapt25c1hyb_dev_mg2 | mg2 | 84 | - | +0.00 (22/46/16) | -4.5 (22/53/9) | +1.01 (40/27/17) | -0.9 (24/10/50) | -0.32 (45/4/35) | -0.433 (58/24/2) | -0.03 (2/9/1) n=12 |
| ours_adapt25c1hyb_dev_mg2_long | mg2_long | 4 | - | +0.00 (1/2/1) | -1.2 (2/1/1) | +3.19 (3/0/1) | +0.1 (2/0/2) | -0.09 (2/0/2) | -0.375 (4/0/0) | - |
| ours_adapt25c1hyb_dev_neutral_832 | neutral_832 | 84 | - | -0.07 (9/61/14) | -0.2 (17/50/17) | +0.20 (34/21/29) | -1.0 (29/15/40) | -0.04 (42/3/39) | -0.008 (34/33/17) | -0.06 (5/4/3) n=12 |
| ours_adapt25c1hyb_dev_neutral_832_long | neutral_832_long | 4 | - | +0.25 (1/2/1) | -13.8 (2/2/0) | +1.86 (4/0/0) | -2.0 (1/1/2) | +1.24 (1/0/3) | -0.372 (2/2/0) | - |
| ours_adapt25c1hyb_dev_neutral_832_nocap | neutral_832_nocap | 84 | - | +0.14 (19/54/11) | -2.6 (16/55/13) | +0.34 (31/23/30) | -1.5 (24/17/43) | +0.24 (36/9/39) | -0.329 (52/22/10) | +0.01 (3/6/3) n=12 |
| ours_adapt25c1hyb_dev_neutral_abot | neutral_abot | 60 | - | +0.05 (11/42/7) | -2.0 (16/34/10) | +0.29 (26/15/19) | -0.7 (19/21/20) | +0.23 (33/3/24) | -0.357 (40/18/2) | -0.14 (5/4/2) n=11 |
| ours_adapt25c1hyb_dev_neutral_mg2 | neutral_mg2 | 84 | - | +0.20 (23/53/8) | -6.2 (24/47/13) | +0.81 (46/16/22) | +0.2 (34/12/38) | -0.17 (41/6/37) | -0.427 (59/23/2) | -0.00 (2/9/1) n=12 |
| ours_adapt25c1hyb_dev_neutral_wp | neutral_wp | 84 | - | +0.11 (19/55/10) | -1.2 (20/50/14) | -0.02 (29/23/32) | -0.6 (26/22/36) | -0.15 (39/3/42) | -0.227 (50/23/11) | -0.10 (4/5/3) n=12 |
| ours_adapt25c1hyb_dev_wp | wp | 84 | - | +0.30 (30/48/6) | -2.2 (18/59/7) | +0.54 (43/19/22) | -0.4 (26/15/43) | +0.12 (42/6/36) | -0.217 (54/30/0) | -0.10 (5/6/1) n=12 |
| ours_adapt25c1hyb_dev_wp_long | wp_long | 4 | - | +0.00 (0/4/0) | -22.5 (3/1/0) | +0.72 (2/2/0) | -6.8 (1/0/3) | +3.24 (2/0/2) | -0.038 (0/4/0) | - |
| ours_mx24_mg2_idle5 | mg2 | 6 | - | +0.00 (1/4/1) | +0.0 (0/6/0) | +2.41 (3/0/3) | -1.4 (2/1/3) | +0.41 (1/0/5) | +0.032 (0/6/0) | - |
| ours_mx24_mg2_long | mg2_long | 4 | - | +0.25 (2/0/2) | -12.5 (2/1/1) | -0.97 (1/1/2) | -2.5 (1/0/3) | +1.90 (0/0/4) | +0.342 (0/1/3) | - |
| ours_mx24_neutral_832_idle5 | neutral_832 | 6 | - | +0.17 (1/5/0) | -5.0 (1/4/1) | -0.31 (1/3/2) | -3.2 (2/0/4) | +0.92 (3/1/2) | -0.370 (2/2/2) | - |
| ours_mx24_neutral_832_long | neutral_832_long | 4 | - | +0.00 (1/2/1) | -2.5 (1/3/0) | +1.47 (3/1/0) | -1.9 (1/0/3) | -0.03 (1/0/3) | +0.171 (1/0/3) | - |
| ours_mx32_mg2_idle5 | mg2 | 6 | - | -0.17 (0/5/1) | -3.3 (1/5/0) | +0.45 (1/1/4) | -3.0 (0/0/6) | +0.38 (2/0/4) | +0.017 (0/5/1) | - |
| ours_mx32_mg2_long | mg2_long | 4 | - | +0.50 (2/2/0) | +1.2 (1/2/1) | -1.00 (1/2/1) | -3.4 (1/0/3) | +2.62 (0/1/3) | +0.607 (0/1/3) | - |
| ours_mx32_neutral_832_idle5 | neutral_832 | 6 | - | +0.00 (0/6/0) | -6.7 (2/3/1) | -1.49 (1/1/4) | -6.0 (1/1/4) | +2.75 (2/0/4) | -0.367 (4/1/1) | - |
| ours_mx32_neutral_832_long | neutral_832_long | 4 | - | -0.25 (1/1/2) | -16.2 (3/1/0) | +0.99 (2/1/1) | -2.8 (1/0/3) | -0.13 (1/0/3) | +0.052 (1/0/3) | - |
| ours_mxK8_mg2_long | mg2_long | 4 | 1.47x | -1.00 (1/1/2) | +7.5 (1/1/2) | -2.53 (1/1/2) | -2.8 (1/1/2) | +1.85 (1/0/3) | +0.358 (1/0/3) | - |
| ours_mxK8_neutral_832_long | neutral_832_long | 4 | 1.49x | -0.50 (1/1/2) | -2.5 (1/3/0) | +0.00 (2/0/2) | -3.9 (1/0/3) | -0.66 (1/0/3) | -0.042 (1/2/1) | - |
| ours_mxK8c1hyb_cloud_mg2_long | mg2_long | 4 | - | +0.00 (1/2/1) | -25.0 (3/0/1) | +2.22 (3/1/0) | +0.1 (2/0/2) | -0.63 (2/0/2) | -0.371 (4/0/0) | - |
| ours_mxK8c1hyb_cloud_neutral_832_long | neutral_832_long | 4 | - | +0.25 (1/2/1) | -13.8 (1/3/0) | +1.94 (4/0/0) | -0.3 (1/1/2) | -0.55 (2/0/2) | -0.334 (2/2/0) | - |
| ours_mxK8c1hyb_dev_mg2_long | mg2_long | 4 | - | +0.00 (1/2/1) | -25.0 (3/0/1) | +2.13 (3/1/0) | -0.5 (2/0/2) | -0.39 (2/0/2) | -0.370 (4/0/0) | - |
| ours_mxK8c1hyb_dev_neutral_832_long | neutral_832_long | 4 | - | +0.00 (1/1/2) | -13.8 (2/2/0) | +1.75 (3/1/0) | -1.3 (1/1/2) | -0.37 (2/0/2) | -0.206 (2/1/1) | - |
| ours_mxQ4_mg2_long | mg2_long | 4 | 1.09x | +0.25 (2/1/1) | -5.0 (2/2/0) | -1.07 (0/2/2) | -1.3 (2/0/2) | +2.43 (0/0/4) | +0.097 (0/2/2) | - |
| ours_mxQ4_neutral_832_long | neutral_832_long | 4 | 1.22x | +0.25 (1/3/0) | -5.0 (2/2/0) | +0.45 (2/1/1) | -2.4 (1/0/3) | +1.06 (1/0/3) | -0.083 (1/1/2) | - |
| ours_mxQ4c1hyb_cloud_mg2_long | mg2_long | 4 | - | +0.50 (2/1/1) | -15.0 (3/0/1) | +2.81 (3/1/0) | +2.0 (2/0/2) | -0.76 (3/0/1) | -0.348 (4/0/0) | - |
| ours_mxQ4c1hyb_cloud_neutral_832_long | neutral_832_long | 4 | - | +0.25 (1/3/0) | +0.0 (0/4/0) | +2.27 (3/1/0) | +1.4 (2/1/1) | -0.19 (1/1/2) | -0.353 (2/1/1) | - |
| ours_mxQ4c1hyb_dev_mg2_long | mg2_long | 4 | - | +0.50 (2/1/1) | -15.0 (3/0/1) | +3.12 (3/1/0) | +1.8 (2/0/2) | -0.72 (3/0/1) | -0.347 (4/0/0) | - |
| ours_mxQ4c1hyb_dev_neutral_832_long | neutral_832_long | 4 | - | +0.25 (1/3/0) | +0.0 (0/4/0) | +2.28 (4/0/0) | +0.7 (2/1/1) | +0.27 (1/0/3) | -0.252 (2/1/1) | - |
| ours_mxW12_mg2_long | mg2_long | 4 | 1.64x | +0.25 (2/1/1) | -1.2 (2/1/1) | -1.56 (0/1/3) | -7.1 (0/0/4) | +3.98 (0/0/4) | +0.329 (0/2/2) | - |
| ours_mxW12_neutral_832_long | neutral_832_long | 4 | 1.71x | +0.25 (1/3/0) | -5.0 (2/2/0) | +1.35 (4/0/0) | -1.8 (1/0/3) | +0.19 (1/1/2) | -0.110 (1/1/2) | - |
| ours_mxW12c1hyb_cloud_mg2_long | mg2_long | 4 | - | +0.25 (1/3/0) | -26.2 (3/0/1) | +1.20 (4/0/0) | -0.8 (1/0/3) | -1.27 (3/0/1) | -0.304 (4/0/0) | - |
| ours_mxW12c1hyb_cloud_neutral_832_long | neutral_832_long | 4 | - | +0.25 (1/3/0) | -5.0 (2/2/0) | +2.94 (4/0/0) | +0.5 (2/1/1) | -0.63 (2/0/2) | -0.006 (1/0/3) | - |
| ours_mxW12c1hyb_dev_mg2_long | mg2_long | 4 | - | +0.50 (2/2/0) | -23.8 (3/0/1) | +1.23 (4/0/0) | -1.5 (1/0/3) | -0.96 (3/0/1) | -0.309 (4/0/0) | - |
| ours_mxW12c1hyb_dev_neutral_832_long | neutral_832_long | 4 | - | +0.00 (1/2/1) | -2.5 (1/3/0) | +3.00 (4/0/0) | -0.1 (1/1/2) | -0.24 (1/0/3) | -0.271 (2/0/2) | - |
| ours_mxW9_mg2_long | mg2_long | 4 | 1.71x | +0.25 (2/0/2) | -7.5 (3/1/0) | -1.09 (1/0/3) | -8.9 (1/0/3) | -0.70 (2/0/2) | +3.217 (0/0/4) | - |
| ours_mxW9_neutral_832_long | neutral_832_long | 4 | 1.80x | +0.00 (1/2/1) | -36.2 (3/1/0) | +3.99 (4/0/0) | -5.9 (1/0/3) | -2.16 (3/0/1) | +3.650 (0/0/4) | - |
| ours_mxW9c1hyb_cloud_mg2_long | mg2_long | 4 | - | +0.25 (2/0/2) | -7.5 (3/1/0) | -0.42 (1/0/3) | -5.7 (1/0/3) | -1.04 (2/1/1) | +3.334 (0/0/4) | - |
| ours_mxW9c1hyb_cloud_neutral_832_long | neutral_832_long | 4 | - | +0.25 (1/3/0) | -38.8 (4/0/0) | +4.76 (4/0/0) | -2.9 (1/0/3) | -3.03 (3/0/1) | +3.352 (0/0/4) | - |
| ours_mxW9c1hyb_dev_mg2_long | mg2_long | 4 | - | +0.25 (2/0/2) | -7.5 (3/1/0) | -1.09 (1/0/3) | -8.9 (1/0/3) | -0.70 (2/0/2) | +3.217 (0/0/4) | - |
| ours_mxW9c1hyb_dev_neutral_832_long | neutral_832_long | 4 | - | +0.00 (1/2/1) | -38.8 (4/0/0) | +3.72 (4/0/0) | -5.9 (1/0/3) | -2.45 (3/0/1) | +3.579 (0/0/4) | - |
| ours_rs576_astra | astra | 21 | 2.37x | -4.19 (0/0/21) | +4.8 (1/13/7) | -4.99 (0/1/20) | -14.5 (0/0/21) | +0.82 (4/0/17) | -0.111 (8/8/5) | +0.01 (1/0/2) n=3 |
| ours_rs576_neutral_832 | neutral_832 | 84 | 2.37x | -2.31 (4/6/74) | +6.1 (13/40/31) | -3.25 (4/3/77) | -17.5 (1/1/82) | +1.57 (25/3/56) | -0.001 (29/29/26) | -0.18 (6/4/1) n=11 |
| ours_rs576r_astra | astra | 21 | 1.83x | +0.14 (5/12/4) | +3.1 (2/16/3) | -0.98 (3/5/13) | +0.3 (6/10/5) | -0.33 (11/4/6) | -0.092 (9/6/6) | +0.04 (1/1/1) n=3 |
| ours_rs576r_neutral_832 | neutral_832 | 84 | 1.83x | -0.19 (8/55/21) | -2.5 (23/43/18) | +0.23 (37/13/34) | -2.1 (27/10/47) | +0.59 (32/5/47) | -0.001 (37/30/17) | -0.21 (7/5/0) n=12 |
| ours_rt2_abot | abot | 60 | 1.63x | -0.05 (5/47/8) | -2.8 (9/45/6) | +0.12 (15/34/11) | -0.3 (9/23/28) | -0.15 (30/8/22) | +0.025 (8/38/14) | +0.15 (2/6/3) n=11 |
| ours_rt2_abot_long | abot_long | 4 | 1.59x | +0.50 (2/2/0) | -13.8 (1/3/0) | +0.40 (2/2/0) | -0.5 (2/0/2) | +0.43 (1/1/2) | +0.013 (0/4/0) | - |
| ours_rt2_astra | astra | 21 | 1.54x | -0.10 (3/13/5) | +1.4 (1/18/2) | -0.23 (2/12/7) | -1.1 (2/6/13) | -0.01 (7/4/10) | -0.047 (5/14/2) | +0.01 (1/0/2) n=3 |
| ours_rt2_mg2 | mg2 | 84 | 1.52x | -0.45 (6/50/28) | +1.5 (20/44/20) | -0.20 (24/25/35) | -2.2 (16/9/59) | +0.25 (32/4/48) | -0.012 (23/42/19) | -0.03 (2/7/3) n=12 |
| ours_rt2_mg2_idle5 | mg2 | 6 | 1.52x | +0.33 (3/2/1) | +0.0 (1/4/1) | -0.58 (2/0/4) | -1.3 (2/2/2) | +0.47 (2/0/4) | +0.009 (0/6/0) | - |
| ours_rt2_mg2_long | mg2_long | 4 | 1.52x | +0.00 (1/2/1) | -1.2 (2/1/1) | -0.85 (1/1/2) | -4.4 (1/0/3) | +3.41 (0/0/4) | +0.132 (0/3/1) | - |
| ours_rt2_neutral_832 | neutral_832 | 84 | 1.54x | -0.13 (9/57/18) | +0.3 (14/54/16) | +0.12 (35/20/29) | -1.2 (24/17/43) | -0.01 (43/4/37) | +0.057 (20/37/27) | -0.11 (4/7/0) n=11 |
| ours_rt2_neutral_832_idle5 | neutral_832 | 6 | 1.53x | +0.00 (0/6/0) | +5.8 (1/4/1) | -1.64 (0/1/5) | -3.9 (1/1/4) | +1.19 (3/1/2) | -0.487 (2/4/0) | - |
| ours_rt2_neutral_832_long | neutral_832_long | 4 | 1.53x | +0.25 (1/2/1) | -5.0 (2/2/0) | +0.03 (1/2/1) | -2.4 (1/0/3) | +2.50 (1/0/3) | -0.093 (1/1/2) | - |
| ours_rt2_neutral_832_nocap | neutral_832_nocap | 84 | 1.54x | +0.01 (12/59/13) | -4.3 (19/52/13) | -0.25 (26/14/44) | -2.8 (19/13/52) | +0.71 (28/2/54) | -0.113 (40/27/17) | -0.03 (3/7/2) n=12 |
| ours_rt2_neutral_abot | neutral_abot | 60 | 1.57x | -0.12 (5/43/12) | +2.3 (13/32/15) | -0.29 (19/10/31) | -1.1 (14/12/34) | +0.38 (23/7/30) | -0.145 (16/33/11) | -0.08 (5/3/3) n=11 |
| ours_rt2_neutral_mg2 | neutral_mg2 | 84 | 1.52x | -0.13 (9/53/22) | +1.3 (14/52/18) | -0.56 (28/8/48) | -2.9 (12/10/62) | +0.55 (31/3/50) | +0.075 (18/29/37) | +0.02 (3/6/3) n=12 |
| ours_rt2_neutral_wp | neutral_wp | 84 | 1.56x | -0.07 (8/62/14) | +4.6 (11/50/23) | -0.42 (23/15/46) | -1.0 (24/10/50) | +0.47 (26/8/50) | -0.097 (24/37/23) | -0.12 (5/7/0) n=12 |
| ours_rt2_wp | wp | 84 | 1.57x | +0.08 (16/59/9) | +1.4 (12/54/18) | -0.13 (25/25/34) | -1.2 (16/20/48) | +0.06 (37/6/41) | +0.049 (18/45/21) | -0.08 (3/8/1) n=12 |
| ours_rt2_wp_long | wp_long | 4 | 1.55x | +0.00 (0/4/0) | -25.0 (3/1/0) | +0.41 (2/1/1) | -3.3 (1/0/3) | -0.12 (2/0/2) | +0.050 (0/3/1) | - |
| ours_rt2c1hyb24_cloud_mg2 | mg2 | 1 | - | +0.00 (0/1/0) | +0.0 (0/1/0) | -0.02 (0/1/0) | -2.5 (0/0/1) | +0.03 (0/1/0) | -0.003 (0/1/0) | - |
| ours_rt2c1hyb24_cloud_mg2_long | mg2_long | 4 | - | +0.50 (2/2/0) | -16.2 (3/1/0) | +3.10 (4/0/0) | +1.7 (1/0/3) | -0.31 (1/0/3) | -0.252 (3/1/0) | - |
| ours_rt2c1hyb24_cloud_neutral_832 | neutral_832 | 1 | - | +0.00 (0/1/0) | +0.0 (0/1/0) | -0.01 (0/1/0) | +1.5 (1/0/0) | +0.86 (0/0/1) | -2.166 (1/0/0) | - |
| ours_rt2c1hyb24_cloud_neutral_832_long | neutral_832_long | 4 | - | +0.00 (1/2/1) | -6.2 (2/2/0) | +3.12 (4/0/0) | -0.4 (2/0/2) | +0.39 (2/0/2) | -0.346 (2/1/1) | - |
| ours_rt2c1hyb24_dev_mg2 | mg2 | 1 | - | +0.00 (0/1/0) | +0.0 (0/1/0) | +0.23 (0/1/0) | -2.1 (0/0/1) | -0.28 (1/0/0) | -0.004 (0/1/0) | - |
| ours_rt2c1hyb24_dev_mg2_long | mg2_long | 4 | - | +0.50 (2/2/0) | -18.8 (2/2/0) | +3.21 (4/0/0) | +0.5 (1/0/3) | -0.17 (1/1/2) | -0.229 (3/0/1) | - |
| ours_rt2c1hyb24_dev_neutral_832 | neutral_832 | 1 | - | +0.00 (0/1/0) | +0.0 (0/1/0) | -0.12 (0/1/0) | +0.9 (1/0/0) | +1.05 (0/0/1) | -2.175 (1/0/0) | - |
| ours_rt2c1hyb24_dev_neutral_832_long | neutral_832_long | 4 | - | +0.00 (1/2/1) | -16.2 (2/2/0) | +3.06 (4/0/0) | -2.1 (1/1/2) | +0.85 (2/0/2) | -0.362 (2/1/1) | - |
| ours_rt2c1hyb32_cloud_astra | astra | 21 | - | +0.29 (6/14/1) | +4.5 (2/16/3) | +0.37 (7/6/8) | -0.9 (4/5/12) | -0.12 (10/2/9) | -0.129 (13/7/1) | +0.08 (1/0/2) n=3 |
| ours_rt2c1hyb32_cloud_mg2 | mg2 | 84 | - | -0.24 (15/49/20) | -4.8 (23/50/11) | +0.59 (36/16/32) | -1.8 (18/4/62) | -0.19 (43/3/38) | -0.445 (60/23/1) | -0.01 (1/9/2) n=12 |
| ours_rt2c1hyb32_cloud_mg2_long | mg2_long | 4 | - | +0.50 (2/2/0) | -27.5 (3/1/0) | +3.05 (3/1/0) | -0.0 (1/0/3) | -0.48 (1/0/3) | -0.341 (4/0/0) | - |
| ours_rt2c1hyb32_cloud_neutral_832 | neutral_832 | 84 | - | +0.10 (14/64/6) | -4.3 (19/50/15) | +0.69 (46/13/25) | -0.4 (31/15/38) | -0.38 (43/5/36) | -0.057 (37/30/17) | -0.09 (5/5/2) n=12 |
| ours_rt2c1hyb32_cloud_neutral_832_long | neutral_832_long | 4 | - | +0.50 (1/3/0) | -16.2 (2/2/0) | +2.42 (3/1/0) | -0.9 (1/1/2) | -0.65 (1/0/3) | -0.303 (2/1/1) | - |
| ours_rt2c1hyb32_cloud_wp | wp | 84 | - | +0.12 (18/57/9) | -0.8 (17/55/12) | +0.46 (42/23/19) | -0.9 (24/12/48) | +0.15 (38/7/39) | -0.241 (55/28/1) | -0.11 (5/7/0) n=12 |
| ours_rt2c1hyb32_dev_astra | astra | 21 | - | -0.05 (4/12/5) | +4.5 (2/16/3) | +0.43 (8/5/8) | -1.2 (3/5/13) | -0.13 (9/4/8) | -0.131 (13/7/1) | +0.08 (1/0/2) n=3 |
| ours_rt2c1hyb32_dev_mg2 | mg2 | 84 | - | -0.17 (18/46/20) | -4.6 (23/50/11) | +0.64 (36/17/31) | -2.8 (16/3/65) | -0.14 (36/6/42) | -0.442 (60/23/1) | -0.03 (1/8/3) n=12 |
| ours_rt2c1hyb32_dev_mg2_long | mg2_long | 4 | - | -0.25 (1/2/1) | -23.8 (3/0/1) | +3.04 (3/1/0) | -1.1 (1/0/3) | -0.25 (1/0/3) | -0.348 (4/0/0) | - |
| ours_rt2c1hyb32_dev_neutral_832 | neutral_832 | 84 | - | -0.04 (9/63/12) | -2.9 (22/46/16) | +0.61 (42/19/23) | -1.4 (24/12/48) | -0.10 (38/8/38) | -0.003 (33/30/21) | -0.10 (5/5/2) n=12 |
| ours_rt2c1hyb32_dev_neutral_832_long | neutral_832_long | 4 | - | +0.25 (1/2/1) | -16.2 (2/2/0) | +2.29 (3/1/0) | -2.2 (1/0/3) | -0.21 (1/0/3) | -0.368 (3/0/1) | - |
| ours_rt2c1hyb32_dev_wp | wp | 84 | - | +0.04 (16/55/13) | -0.8 (17/56/11) | +0.42 (38/24/22) | -1.5 (18/11/55) | +0.13 (36/6/42) | -0.244 (57/26/1) | -0.06 (5/6/1) n=12 |
| ours_rt2c1hyb_cloud_abot | abot | 60 | - | +0.03 (7/47/6) | -0.9 (11/39/10) | +0.92 (25/24/11) | +0.0 (16/28/16) | -0.27 (30/11/19) | -0.142 (30/29/1) | -0.04 (2/8/1) n=11 |
| ours_rt2c1hyb_cloud_abot_long | abot_long | 4 | - | +0.50 (2/0/2) | -16.2 (2/1/1) | +0.99 (1/1/2) | -0.1 (2/1/1) | -0.01 (3/0/1) | -0.230 (1/3/0) | - |
| ours_rt2c1hyb_cloud_astra | astra | 21 | - | +0.24 (6/13/2) | +0.5 (1/19/1) | -0.12 (3/9/9) | -0.7 (3/6/12) | +0.00 (7/3/11) | -0.110 (11/10/0) | +0.01 (1/1/1) n=3 |
| ours_rt2c1hyb_cloud_mg2 | mg2 | 84 | - | -0.19 (12/54/18) | -0.9 (19/51/14) | +0.61 (38/19/27) | -1.2 (22/8/54) | -0.31 (42/7/35) | -0.404 (58/23/3) | -0.05 (2/8/2) n=12 |
| ours_rt2c1hyb_cloud_mg2_long | mg2_long | 4 | - | +0.50 (2/2/0) | -7.5 (3/1/0) | +1.86 (3/1/0) | +1.0 (1/0/3) | -0.30 (1/0/3) | -0.347 (4/0/0) | - |
| ours_rt2c1hyb_cloud_neutral_832 | neutral_832 | 84 | - | -0.05 (12/58/14) | -1.3 (15/57/12) | +0.36 (39/16/29) | -0.9 (31/13/40) | -0.34 (46/3/35) | +0.078 (27/39/18) | -0.15 (5/6/0) n=11 |
| ours_rt2c1hyb_cloud_neutral_832_long | neutral_832_long | 4 | - | +0.00 (1/2/1) | +0.0 (0/4/0) | +1.85 (3/1/0) | -0.1 (1/0/3) | +0.51 (2/0/2) | -0.277 (2/0/2) | - |
| ours_rt2c1hyb_cloud_wp | wp | 84 | - | +0.18 (20/59/5) | -1.2 (12/67/5) | +0.48 (41/27/16) | -0.7 (25/9/50) | +0.00 (42/6/36) | -0.204 (52/29/3) | -0.09 (4/7/1) n=12 |
| ours_rt2c1hyb_cloud_wp_long | wp_long | 4 | - | +0.00 (0/4/0) | -11.2 (3/1/0) | +1.99 (4/0/0) | -2.6 (1/0/3) | +1.08 (2/0/2) | -0.158 (2/2/0) | - |
| ours_rt2c1hyb_dev_abot | abot | 60 | - | +0.05 (6/50/4) | -0.5 (10/41/9) | +0.96 (26/21/13) | -0.1 (13/26/21) | -0.25 (28/14/18) | -0.145 (31/28/1) | -0.04 (2/8/1) n=11 |
| ours_rt2c1hyb_dev_abot_long | abot_long | 4 | - | +0.25 (2/0/2) | -13.8 (2/1/1) | +1.11 (1/1/2) | -0.2 (2/1/1) | -0.03 (3/0/1) | -0.231 (1/3/0) | - |
| ours_rt2c1hyb_dev_astra | astra | 21 | - | +0.05 (3/16/2) | +0.5 (1/19/1) | +0.06 (6/7/8) | -0.8 (3/5/13) | +0.04 (6/3/12) | -0.113 (11/10/0) | +0.03 (1/0/2) n=3 |
| ours_rt2c1hyb_dev_mg2 | mg2 | 84 | - | -0.14 (14/53/17) | -1.9 (21/48/15) | +0.71 (38/18/28) | -1.5 (21/6/57) | -0.23 (41/7/36) | -0.419 (58/23/3) | -0.05 (2/8/2) n=12 |
| ours_rt2c1hyb_dev_mg2_long | mg2_long | 4 | - | -0.25 (1/2/1) | -2.5 (2/1/1) | +1.97 (3/1/0) | +0.6 (1/0/3) | -0.07 (1/0/3) | -0.350 (4/0/0) | - |
| ours_rt2c1hyb_dev_neutral_832 | neutral_832 | 84 | - | -0.05 (12/57/15) | -1.0 (16/54/14) | +0.38 (38/18/28) | -1.3 (29/13/42) | -0.26 (43/5/36) | +0.047 (29/36/19) | -0.17 (7/5/0) n=12 |
| ours_rt2c1hyb_dev_neutral_832_long | neutral_832_long | 4 | - | +0.00 (1/2/1) | -2.5 (1/3/0) | +1.78 (3/1/0) | -0.8 (1/0/3) | +0.77 (1/0/3) | -0.260 (2/0/2) | - |
| ours_rt2c1hyb_dev_wp | wp | 84 | - | +0.17 (22/53/9) | -0.7 (15/60/9) | +0.46 (40/28/16) | -1.0 (21/13/50) | +0.08 (37/7/40) | -0.186 (52/30/2) | -0.09 (4/7/1) n=12 |
| ours_rt2c1hyb_dev_wp_long | wp_long | 4 | - | +0.00 (0/4/0) | -11.2 (3/1/0) | +1.98 (4/0/0) | -2.8 (1/0/3) | +1.27 (1/1/2) | -0.162 (3/1/0) | - |
| ours_sf1p3b_cal_astra | astra | 21 | 3.07x | -0.38 (3/10/8) | +12.1 (4/10/7) | +2.43 (12/3/6) | +1.0 (9/6/6) | -0.48 (10/2/9) | +0.342 (5/8/8) | -0.04 (2/0/1) n=3 |
| ours_sf1p3b_cal_neutral_832 | neutral_832 | 84 | 2.74x | -1.12 (7/25/52) | +7.7 (16/38/30) | -0.02 (35/13/36) | -5.4 (17/5/62) | -0.20 (38/7/39) | +0.618 (13/16/55) | -0.17 (6/4/2) n=12 |
| ours_sf1p3b_cal_neutral_832_nocap | neutral_832_nocap | 84 | 2.85x | -0.54 (10/30/44) | +6.0 (21/30/33) | -1.93 (22/8/54) | -6.3 (18/12/54) | +0.52 (28/4/52) | +0.187 (23/22/39) | -0.00 (3/4/5) n=12 |
| ours_sf1p3b_cal_neutral_832_nocap_idle | neutral_832_nocap | 22 | 3.01x | -0.23 (5/9/8) | +0.7 (6/10/6) | -0.88 (9/3/10) | -5.9 (7/1/14) | -0.13 (12/2/8) | +0.233 (6/4/12) | +0.19 (0/0/3) n=3 |
| ours_sf1p3b_pub_astra | astra | 21 | 3.27x | -0.57 (2/7/12) | +14.8 (4/8/9) | +1.76 (10/1/10) | +0.1 (8/1/12) | -0.23 (11/1/9) | +0.332 (6/7/8) | -0.02 (1/2/0) n=3 |
| ours_sf1p3b_pub_neutral_832 | neutral_832 | 84 | 2.98x | -2.08 (3/18/63) | +16.4 (14/28/42) | -2.54 (18/12/54) | -9.4 (14/7/63) | +0.50 (32/5/47) | +0.652 (16/15/53) | -0.17 (6/3/3) n=12 |
| ours_sf1p3b_pub_neutral_832_nocap | neutral_832_nocap | 84 | 3.04x | -1.51 (7/21/56) | +12.6 (20/16/48) | -3.77 (11/6/67) | -9.6 (4/7/73) | +0.94 (28/5/51) | +0.254 (21/24/39) | +0.00 (2/4/6) n=12 |
| ours_sf1p3b_pub_neutral_832_nocap_idle | neutral_832_nocap | 22 | 3.22x | -1.00 (3/6/13) | +15.9 (4/8/10) | -3.54 (5/1/16) | -9.0 (3/1/18) | +0.14 (8/0/14) | +0.323 (6/3/13) | +0.03 (0/1/2) n=3 |
| ours_sf24_abot | abot | 60 | 1.53x | +0.02 (7/46/7) | +0.2 (12/38/10) | +0.72 (25/21/14) | -0.0 (15/16/29) | -0.29 (29/5/26) | -0.194 (39/20/1) | -0.03 (2/7/2) n=11 |
| ours_sf24_abot_dit | abot | 2 | - | +0.00 (0/2/0) | +0.0 (0/2/0) | +0.30 (0/2/0) | -0.3 (0/1/1) | -0.28 (2/0/0) | -0.087 (1/1/0) | - |
| ours_sf24_abot_long | abot_long | 4 | 1.43x | +1.00 (2/2/0) | -11.2 (3/0/1) | +0.95 (1/1/2) | -1.4 (1/0/3) | +0.70 (1/0/3) | -0.158 (1/2/1) | - |
| ours_sf24_astra | astra | 21 | 1.54x | +0.24 (6/12/3) | +0.5 (1/19/1) | +0.34 (10/7/4) | -1.6 (4/4/13) | -0.11 (10/2/9) | -0.118 (8/13/0) | +0.04 (1/0/2) n=3 |
| ours_sf24_astra_dit | astra | 2 | - | +0.00 (0/2/0) | +10.0 (0/1/1) | +5.47 (2/0/0) | -0.3 (0/2/0) | -0.35 (1/1/0) | -0.067 (1/1/0) | - |
| ours_sf24_mg2 | mg2 | 84 | 1.47x | +0.00 (18/50/16) | -4.3 (18/59/7) | +0.92 (39/19/26) | -1.8 (22/9/53) | -0.50 (45/8/31) | -0.515 (63/18/3) | -0.06 (3/9/0) n=12 |
| ours_sf24_mg2_dit | mg2 | 2 | - | -0.50 (0/1/1) | +0.0 (0/2/0) | -0.26 (1/0/1) | -4.2 (0/0/2) | +0.03 (1/0/1) | -0.649 (1/1/0) | - |
| ours_sf24_mg2_long | mg2_long | 4 | 1.51x | +0.75 (2/2/0) | -28.8 (3/0/1) | +3.68 (4/0/0) | +0.0 (2/0/2) | -0.84 (3/0/1) | -0.380 (4/0/0) | - |
| ours_sf24_neutral_832 | neutral_832 | 84 | 1.13x | +0.06 (17/56/11) | -3.6 (20/53/11) | +0.67 (45/19/20) | +0.4 (35/19/30) | -0.37 (42/7/35) | -0.046 (29/37/18) | -0.11 (6/4/2) n=12 |
| ours_sf24_neutral_832_dit | neutral_832 | 2 | - | +0.00 (0/2/0) | +0.0 (0/2/0) | -0.11 (1/0/1) | -1.4 (1/0/1) | +1.84 (1/0/1) | -1.190 (2/0/0) | - |
| ours_sf24_neutral_832_long | neutral_832_long | 4 | 1.25x | +0.00 (1/2/1) | -11.2 (1/3/0) | +1.32 (3/0/1) | -1.4 (1/2/1) | +0.07 (3/0/1) | +0.071 (1/2/1) | - |
| ours_sf24_neutral_832_nocap | neutral_832_nocap | 84 | 1.42x | +0.32 (27/52/5) | -10.2 (28/49/7) | +1.43 (54/12/18) | -0.9 (32/13/39) | +0.59 (33/6/45) | -0.391 (51/31/2) | -0.03 (3/8/1) n=12 |
| ours_sf24_neutral_abot | neutral_abot | 60 | 1.34x | +0.05 (12/40/8) | +0.6 (13/35/12) | +0.28 (31/11/18) | -0.7 (23/10/27) | +0.17 (31/5/24) | -0.391 (39/18/3) | -0.15 (7/3/1) n=11 |
| ours_sf24_neutral_mg2 | neutral_mg2 | 84 | 1.16x | +0.29 (29/49/6) | -8.8 (30/41/13) | +1.17 (46/16/22) | -0.6 (28/11/45) | -0.25 (40/9/35) | -0.463 (64/15/5) | -0.05 (4/7/1) n=12 |
| ours_sf24_neutral_wp | neutral_wp | 84 | 1.37x | +0.30 (24/56/4) | -8.4 (29/46/9) | +0.95 (43/25/16) | +0.3 (36/13/35) | -0.60 (40/3/41) | -0.365 (58/23/3) | -0.09 (4/6/2) n=12 |
| ours_sf24_wp | wp | 84 | 1.47x | +0.33 (28/53/3) | -3.0 (20/57/7) | +0.81 (45/28/11) | -0.7 (23/20/41) | -0.04 (37/5/42) | -0.261 (53/30/1) | -0.09 (5/6/1) n=12 |
| ours_sf24_wp_dit | wp | 2 | - | +0.50 (1/1/0) | -10.0 (1/1/0) | -1.19 (0/0/2) | -2.3 (0/0/2) | -0.62 (1/0/1) | -0.435 (1/1/0) | - |
| ours_sf24_wp_long | wp_long | 4 | 1.52x | -0.25 (0/3/1) | +5.0 (2/1/1) | +1.60 (3/0/1) | -2.4 (1/0/3) | +2.83 (1/0/3) | -0.002 (1/2/1) | - |
| ours_sf24c1hyb_cloud_abot | abot | 60 | - | -0.02 (6/48/6) | -0.2 (13/37/10) | +0.67 (22/21/17) | +0.3 (18/23/19) | -0.24 (32/6/22) | -0.196 (40/20/0) | -0.04 (2/8/1) n=11 |
| ours_sf24c1hyb_cloud_abot_long | abot_long | 4 | - | +0.75 (2/1/1) | -11.2 (2/1/1) | +2.56 (1/3/0) | -0.4 (1/1/2) | +0.39 (1/1/2) | -0.227 (1/3/0) | - |
| ours_sf24c1hyb_cloud_astra | astra | 21 | - | +0.19 (6/11/4) | -1.4 (3/17/1) | -0.04 (5/9/7) | -1.7 (3/5/13) | -0.09 (7/5/9) | -0.125 (11/10/0) | +0.05 (1/0/2) n=3 |
| ours_sf24c1hyb_cloud_mg2 | mg2 | 84 | - | +0.14 (20/53/11) | -5.7 (24/54/6) | +0.58 (37/18/29) | -2.0 (21/6/57) | -0.62 (51/5/28) | -0.494 (59/22/3) | -0.05 (2/10/0) n=12 |
| ours_sf24c1hyb_cloud_mg2_long | mg2_long | 4 | - | +0.75 (3/0/1) | -30.0 (3/1/0) | +2.75 (4/0/0) | -0.6 (1/1/2) | +0.17 (2/0/2) | -0.312 (4/0/0) | - |
| ours_sf24c1hyb_cloud_neutral_832 | neutral_832 | 84 | - | +0.05 (10/67/7) | -3.9 (20/53/11) | +0.63 (44/19/21) | +0.3 (36/19/29) | -0.53 (47/3/34) | -0.025 (27/39/18) | -0.16 (6/5/1) n=12 |
| ours_sf24c1hyb_cloud_neutral_832_long | neutral_832_long | 4 | - | +0.00 (1/2/1) | -17.5 (3/1/0) | +2.39 (3/0/1) | -0.9 (1/0/3) | +0.17 (1/0/3) | -0.057 (2/0/2) | - |
| ours_sf24c1hyb_cloud_wp | wp | 84 | - | +0.30 (27/53/4) | -1.1 (13/64/7) | +0.64 (39/28/17) | -0.2 (30/18/36) | -0.10 (44/4/36) | -0.242 (52/30/2) | -0.01 (3/8/1) n=12 |
| ours_sf24c1hyb_cloud_wp_long | wp_long | 4 | - | +0.00 (0/4/0) | -22.5 (3/1/0) | +1.32 (2/2/0) | -3.3 (0/2/2) | +1.44 (1/1/2) | -0.062 (1/3/0) | - |
| ours_sf24c1hyb_dev_abot | abot | 60 | - | -0.05 (5/47/8) | -0.4 (13/38/9) | +0.68 (26/17/17) | +0.2 (20/17/23) | -0.28 (30/9/21) | -0.200 (40/20/0) | +0.07 (1/8/2) n=11 |
| ours_sf24c1hyb_dev_abot_long | abot_long | 4 | - | +1.00 (2/2/0) | -11.2 (2/1/1) | +2.81 (2/2/0) | -0.6 (1/1/2) | +0.46 (1/0/3) | -0.236 (1/3/0) | - |
| ours_sf24c1hyb_dev_astra | astra | 21 | - | +0.19 (4/15/2) | -3.3 (5/15/1) | +0.18 (7/8/6) | -2.0 (1/7/13) | -0.12 (9/3/9) | -0.123 (11/10/0) | +0.03 (1/0/2) n=3 |
| ours_sf24c1hyb_dev_mg2 | mg2 | 84 | - | +0.18 (20/56/8) | -7.0 (26/52/6) | +0.55 (36/21/27) | -2.6 (16/6/62) | -0.45 (51/1/32) | -0.501 (59/23/2) | -0.06 (3/9/0) n=12 |
| ours_sf24c1hyb_dev_mg2_long | mg2_long | 4 | - | +0.50 (2/1/1) | -26.2 (3/0/1) | +3.05 (4/0/0) | -0.9 (1/1/2) | +0.44 (2/0/2) | -0.314 (4/0/0) | - |
| ours_sf24c1hyb_dev_neutral_832 | neutral_832 | 84 | - | -0.01 (10/62/12) | -2.9 (18/55/11) | +0.78 (47/20/17) | +0.1 (35/16/33) | -0.36 (43/6/35) | +0.069 (27/39/18) | -0.10 (4/6/1) n=11 |
| ours_sf24c1hyb_dev_neutral_832_long | neutral_832_long | 4 | - | +0.00 (1/2/1) | -15.0 (2/2/0) | +2.35 (3/0/1) | -1.4 (1/0/3) | +0.57 (1/0/3) | +0.001 (2/0/2) | - |
| ours_sf24c1hyb_dev_wp | wp | 84 | - | +0.25 (22/58/4) | -2.2 (20/57/7) | +0.55 (38/27/19) | -0.3 (27/16/41) | +0.07 (38/6/40) | -0.248 (53/30/1) | -0.01 (3/7/2) n=12 |
| ours_sf24c1hyb_dev_wp_long | wp_long | 4 | - | +0.00 (0/4/0) | -22.5 (3/1/0) | +1.36 (2/2/0) | -3.4 (0/1/3) | +1.41 (1/1/2) | -0.077 (1/3/0) | - |
| ours_student1p3b_abot | abot | 60 | 2.36x | -0.30 (4/35/21) | +10.5 (9/22/29) | -0.79 (22/4/34) | -5.1 (8/3/49) | +1.86 (19/3/38) | -0.057 (22/25/13) | -0.01 (2/4/4) n=10 |
| ours_student1p3b_astra | astra | 21 | 2.19x | +0.24 (7/11/3) | +0.2 (4/15/2) | +4.47 (17/0/4) | +0.0 (6/7/8) | +0.62 (6/4/11) | +0.220 (4/8/9) | -0.01 (1/1/1) n=3 |
| ours_student1p3b_mg2 | mg2 | 84 | 2.16x | -0.75 (10/30/44) | +6.0 (23/26/35) | +0.46 (44/8/32) | -4.8 (16/5/63) | -0.25 (46/3/35) | -0.090 (30/26/28) | -0.06 (2/10/0) n=12 |
| ours_student1p3b_neutral_832 | neutral_832 | 84 | 2.19x | -0.69 (9/32/43) | +2.2 (24/29/31) | +0.93 (42/11/31) | -5.3 (14/7/63) | +0.37 (36/6/42) | +0.652 (11/16/57) | -0.19 (6/4/2) n=12 |
| ours_student1p3b_neutral_832_nocap | neutral_832_nocap | 84 | 2.01x | -0.21 (16/40/28) | +3.8 (24/31/29) | -0.43 (30/10/44) | -5.4 (16/7/61) | +0.34 (36/2/46) | +0.168 (24/15/45) | -0.00 (1/8/3) n=12 |
| ours_student1p3b_neutral_abot | neutral_abot | 60 | 2.20x | -0.70 (4/26/30) | +26.6 (8/13/39) | -0.80 (23/5/32) | -6.1 (8/5/47) | +1.38 (16/2/42) | -0.050 (24/14/22) | -0.14 (4/4/3) n=11 |
| ours_student1p3b_neutral_mg2 | neutral_mg2 | 84 | 2.16x | -0.29 (15/36/33) | +3.5 (27/26/31) | +0.10 (39/5/40) | -4.9 (26/7/51) | +0.03 (46/6/32) | +0.057 (27/19/38) | -0.05 (4/7/1) n=12 |
| ours_student1p3b_neutral_wp | neutral_wp | 84 | 2.07x | -0.63 (4/37/43) | +15.4 (17/30/37) | -1.06 (33/9/42) | -6.8 (15/8/61) | +1.01 (26/2/56) | +0.232 (22/24/38) | -0.06 (4/5/3) n=12 |
| ours_student1p3b_wp | wp | 84 | 2.07x | -0.27 (11/42/31) | +10.2 (8/39/37) | -3.89 (9/5/70) | -11.5 (4/4/76) | +2.75 (18/5/61) | +0.105 (22/30/32) | -0.07 (5/5/2) n=12 |
| waypoint | neutral_wp | 84 | 16.34x | -1.95 (3/16/65) | +40.3 (6/10/68) | -4.48 (14/5/65) | +0.2 (29/6/49) | +0.15 (32/2/50) | -0.123 (39/16/29) | -0.16 (6/6/0) n=12 |
| waypoint | wp | 84 | 16.28x | -0.74 (2/33/49) | +10.5 (4/36/44) | -3.51 (5/6/73) | +1.0 (50/7/27) | -0.67 (48/5/31) | -0.137 (44/25/15) | -0.11 (5/6/0) n=11 |
| waypoint | wp_long | 4 | 13.15x | -1.50 (0/1/3) | +17.5 (0/3/1) | -2.72 (0/0/4) | -2.8 (2/0/2) | -0.75 (2/0/2) | +0.282 (0/1/3) | - |
| waypoint_idle | neutral_wp | 84 | 16.34x | - | - | - | - | - | - | - |
| waypoint_idle | wp | 84 | 16.28x | - | - | - | - | - | - | - |

## 3. Control following per program (RPE Rot ° ↓ mean over sets; P5 reaction s ↓)

[10/19 audit] ABot's P2p / P4p columns repeat its P2 / P4 (identical videos). 10-s programs only (P6 = `_long` groups are excluded).

| model_arm | RPE P1 | RPE P2 | RPE P2p | RPE P3 | RPE P4 | RPE P4p | RPE P5 | react P5 s | react P5 switches found/valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| abot | 0.130 (n=24) | 0.802 (n=24) | 0.802 (n=24) | 0.810 (n=24) | 1.130 (n=24) | 1.114 (n=24) | 0.063 (n=24) | 0.44 (n=23) | 63/65 |
| abot_t5offload | 0.156 (n=3) | 0.723 (n=3) | - | 0.861 (n=3) | 1.107 (n=3) | - | 0.047 (n=3) | 0.49 (n=3) | 9/9 |
| astra | 0.554 (n=15) | 1.782 (n=15) | 1.780 (n=15) | 1.586 (n=15) | 2.491 (n=15) | 2.039 (n=15) | 0.440 (n=15) | 0.75 (n=5) | 6/13 |
| astra_nocap (Astra on neutral scenes, no caption) | 0.797 (n=12) | 2.533 (n=12) | 2.352 (n=12) | 2.235 (n=12) | 3.542 (n=12) | 2.693 (n=12) | 0.653 (n=12) | 1.10 (n=1) | 1/7 |
| mg2 | 0.110 (n=24) | 0.555 (n=24) | 0.478 (n=24) | 0.454 (n=24) | 0.695 (n=24) | 0.582 (n=24) | 0.071 (n=24) | 0.35 (n=24) | 68/69 |
| ours_adapt25_abot | 0.127 (n=12) | 1.209 (n=12) | 1.167 (n=12) | 1.140 (n=12) | 0.992 (n=12) | 0.973 (n=12) | 0.176 (n=12) | 0.34 (n=11) | 24/24 |
| ours_adapt25_astra | 0.096 (n=3) | 0.315 (n=3) | 0.277 (n=3) | 1.115 (n=3) | 0.712 (n=3) | 0.962 (n=3) | 0.176 (n=3) | 0.14 (n=3) | 9/9 |
| ours_adapt25_mg2 | 0.223 (n=12) | 1.180 (n=12) | 1.196 (n=12) | 1.278 (n=12) | 1.873 (n=12) | 1.859 (n=12) | 0.189 (n=12) | 0.19 (n=12) | 30/30 |
| ours_adapt25_neutral_832 | 0.303 (n=12) | 0.399 (n=12) | 0.471 (n=12) | 1.343 (n=12) | 0.976 (n=12) | 1.064 (n=12) | 0.209 (n=12) | 0.27 (n=12) | 28/28 |
| ours_adapt25_neutral_832_nocap | 0.394 (n=12) | 1.027 (n=12) | 0.991 (n=12) | 1.569 (n=12) | 1.865 (n=12) | 1.548 (n=12) | 0.187 (n=12) | 0.13 (n=12) | 33/33 |
| ours_adapt25_neutral_abot | 0.405 (n=12) | 1.413 (n=12) | 1.407 (n=12) | 1.340 (n=12) | 1.231 (n=12) | 1.470 (n=12) | 0.575 (n=12) | 0.13 (n=11) | 23/24 |
| ours_adapt25_neutral_mg2 | 0.305 (n=12) | 1.122 (n=12) | 1.101 (n=12) | 1.380 (n=12) | 1.632 (n=12) | 1.726 (n=12) | 0.397 (n=12) | 0.11 (n=11) | 24/25 |
| ours_adapt25_neutral_wp | 0.542 (n=12) | 1.177 (n=12) | 1.067 (n=12) | 1.333 (n=12) | 1.510 (n=12) | 1.168 (n=12) | 0.241 (n=12) | 0.22 (n=12) | 34/34 |
| ours_adapt25_wp | 0.074 (n=12) | 0.977 (n=12) | 0.957 (n=12) | 1.257 (n=12) | 0.985 (n=12) | 0.899 (n=12) | 0.119 (n=12) | 0.09 (n=12) | 32/32 |
| ours_adapt25c1hyb_cloud_abot | 0.089 (n=12) | 0.926 (n=12) | 0.877 (n=12) | 0.974 (n=12) | 0.756 (n=12) | 0.728 (n=12) | 0.075 (n=12) | 0.09 (n=12) | 31/31 |
| ours_adapt25c1hyb_cloud_astra | 0.072 (n=3) | 0.241 (n=3) | 0.219 (n=3) | 0.885 (n=3) | 0.638 (n=3) | 0.791 (n=3) | 0.048 (n=3) | 0.15 (n=3) | 9/9 |
| ours_adapt25c1hyb_cloud_mg2 | 0.143 (n=12) | 0.540 (n=12) | 0.555 (n=12) | 1.135 (n=12) | 1.340 (n=12) | 1.243 (n=12) | 0.067 (n=12) | 0.11 (n=12) | 30/30 |
| ours_adapt25c1hyb_cloud_neutral_832 | 0.332 (n=12) | 0.354 (n=12) | 0.380 (n=12) | 1.277 (n=12) | 1.195 (n=12) | 0.998 (n=12) | 0.292 (n=12) | 0.18 (n=12) | 31/31 |
| ours_adapt25c1hyb_cloud_neutral_832_nocap | 0.347 (n=12) | 0.670 (n=12) | 0.662 (n=12) | 1.335 (n=12) | 1.083 (n=12) | 1.143 (n=12) | 0.090 (n=12) | 0.15 (n=12) | 32/32 |
| ours_adapt25c1hyb_cloud_neutral_abot | 0.398 (n=12) | 0.956 (n=12) | 0.911 (n=12) | 1.142 (n=12) | 0.992 (n=12) | 0.939 (n=12) | 0.145 (n=12) | 0.12 (n=12) | 32/32 |
| ours_adapt25c1hyb_cloud_neutral_mg2 | 0.165 (n=12) | 0.531 (n=12) | 0.561 (n=12) | 1.279 (n=12) | 1.090 (n=12) | 1.209 (n=12) | 0.129 (n=12) | 0.19 (n=12) | 32/33 |
| ours_adapt25c1hyb_cloud_neutral_wp | 0.358 (n=12) | 0.743 (n=12) | 0.760 (n=12) | 1.174 (n=12) | 1.161 (n=12) | 1.052 (n=12) | 0.093 (n=12) | 0.12 (n=12) | 34/34 |
| ours_adapt25c1hyb_cloud_wp | 0.065 (n=12) | 0.686 (n=12) | 0.697 (n=12) | 1.077 (n=12) | 0.712 (n=12) | 0.690 (n=12) | 0.069 (n=12) | 0.07 (n=12) | 36/36 |
| ours_adapt25c1hyb_dev_abot | 0.094 (n=12) | 0.924 (n=12) | 0.878 (n=12) | 0.958 (n=12) | 0.762 (n=12) | 0.731 (n=12) | 0.075 (n=12) | 0.10 (n=11) | 29/29 |
| ours_adapt25c1hyb_dev_astra | 0.072 (n=3) | 0.241 (n=3) | 0.223 (n=3) | 0.885 (n=3) | 0.622 (n=3) | 0.788 (n=3) | 0.050 (n=3) | 0.20 (n=3) | 9/9 |
| ours_adapt25c1hyb_dev_mg2 | 0.136 (n=12) | 0.535 (n=12) | 0.551 (n=12) | 1.131 (n=12) | 1.303 (n=12) | 1.244 (n=12) | 0.064 (n=12) | 0.12 (n=12) | 32/32 |
| ours_adapt25c1hyb_dev_neutral_832 | 0.315 (n=12) | 0.359 (n=12) | 0.381 (n=12) | 1.264 (n=12) | 1.587 (n=12) | 0.990 (n=12) | 0.344 (n=12) | 0.24 (n=12) | 32/33 |
| ours_adapt25c1hyb_dev_neutral_832_nocap | 0.395 (n=12) | 0.669 (n=12) | 0.662 (n=12) | 1.294 (n=12) | 1.063 (n=12) | 1.220 (n=12) | 0.119 (n=12) | 0.14 (n=12) | 33/33 |
| ours_adapt25c1hyb_dev_neutral_abot | 0.420 (n=12) | 0.953 (n=12) | 0.910 (n=12) | 1.097 (n=12) | 0.947 (n=12) | 0.946 (n=12) | 0.138 (n=12) | 0.12 (n=12) | 33/33 |
| ours_adapt25c1hyb_dev_neutral_mg2 | 0.127 (n=12) | 0.531 (n=12) | 0.562 (n=12) | 1.305 (n=12) | 1.074 (n=12) | 1.204 (n=12) | 0.111 (n=12) | 0.15 (n=12) | 32/33 |
| ours_adapt25c1hyb_dev_neutral_wp | 0.447 (n=12) | 0.745 (n=12) | 0.757 (n=12) | 1.268 (n=12) | 1.184 (n=12) | 1.174 (n=12) | 0.093 (n=12) | 0.11 (n=12) | 34/34 |
| ours_adapt25c1hyb_dev_wp | 0.069 (n=12) | 0.691 (n=12) | 0.696 (n=12) | 1.074 (n=12) | 0.725 (n=12) | 0.694 (n=12) | 0.067 (n=12) | 0.07 (n=12) | 36/36 |
| ours_mx24_mg2_idle5 | 0.082 (n=3) | - | - | 1.475 (n=3) | - | - | - | - | - |
| ours_mx24_neutral_832_idle5 | 0.458 (n=3) | - | - | 1.163 (n=3) | - | - | - | - | - |
| ours_mx32_mg2_idle5 | 0.047 (n=3) | - | - | 1.480 (n=3) | - | - | - | - | - |
| ours_mx32_neutral_832_idle5 | 0.660 (n=3) | - | - | 0.967 (n=3) | - | - | - | - | - |
| ours_rs576_astra | 0.183 (n=3) | 0.616 (n=3) | 0.541 (n=3) | 0.644 (n=3) | 0.542 (n=3) | 0.531 (n=3) | 0.108 (n=3) | 0.16 (n=3) | 7/7 |
| ours_rs576_neutral_832 | 0.484 (n=12) | 0.475 (n=12) | 0.429 (n=12) | 1.189 (n=12) | 1.167 (n=12) | 1.347 (n=12) | 0.201 (n=12) | 0.09 (n=11) | 31/32 |
| ours_rs576r_astra | 0.259 (n=3) | 0.489 (n=3) | 0.359 (n=3) | 0.560 (n=3) | 0.780 (n=3) | 0.704 (n=3) | 0.142 (n=3) | 0.19 (n=3) | 8/8 |
| ours_rs576r_neutral_832 | 0.546 (n=12) | 0.375 (n=12) | 0.317 (n=12) | 1.212 (n=12) | 1.217 (n=12) | 1.396 (n=12) | 0.227 (n=12) | 0.09 (n=12) | 32/32 |
| ours_rt2_abot | 0.122 (n=12) | 1.312 (n=12) | 1.240 (n=12) | 1.206 (n=12) | 0.968 (n=12) | 0.906 (n=12) | 0.154 (n=12) | 0.27 (n=12) | 29/29 |
| ours_rt2_astra | 0.081 (n=3) | 0.405 (n=3) | 0.449 (n=3) | 1.172 (n=3) | 0.604 (n=3) | 0.823 (n=3) | 0.077 (n=3) | 0.16 (n=3) | 9/9 |
| ours_rt2_mg2 | 0.130 (n=12) | 1.392 (n=12) | 1.327 (n=12) | 1.304 (n=12) | 1.895 (n=12) | 1.733 (n=12) | 0.129 (n=12) | 0.12 (n=12) | 31/33 |
| ours_rt2_mg2_idle5 | 0.054 (n=3) | - | - | 1.459 (n=3) | - | - | - | - | - |
| ours_rt2_neutral_832 | 0.488 (n=12) | 0.532 (n=12) | 0.497 (n=12) | 1.529 (n=12) | 1.181 (n=12) | 1.271 (n=12) | 0.202 (n=12) | 0.16 (n=11) | 26/27 |
| ours_rt2_neutral_832_idle5 | 0.253 (n=3) | - | - | 1.134 (n=3) | - | - | - | - | - |
| ours_rt2_neutral_832_nocap | 0.212 (n=12) | 1.013 (n=12) | 1.049 (n=12) | 1.542 (n=12) | 1.387 (n=12) | 1.570 (n=12) | 0.165 (n=12) | 0.10 (n=12) | 34/34 |
| ours_rt2_neutral_abot | 0.242 (n=12) | 1.535 (n=12) | 1.468 (n=12) | 1.219 (n=12) | 1.253 (n=12) | 1.198 (n=12) | 0.364 (n=12) | 0.20 (n=12) | 29/31 |
| ours_rt2_neutral_mg2 | 0.246 (n=12) | 1.319 (n=12) | 1.164 (n=12) | 1.467 (n=12) | 1.974 (n=12) | 1.898 (n=12) | 0.360 (n=12) | 0.17 (n=12) | 25/26 |
| ours_rt2_neutral_wp | 0.170 (n=12) | 1.202 (n=12) | 1.132 (n=12) | 1.408 (n=12) | 1.354 (n=12) | 1.175 (n=12) | 0.138 (n=12) | 0.09 (n=12) | 33/33 |
| ours_rt2_wp | 0.071 (n=12) | 1.070 (n=12) | 1.037 (n=12) | 1.292 (n=12) | 0.955 (n=12) | 1.332 (n=12) | 0.120 (n=12) | 0.09 (n=12) | 30/31 |
| ours_rt2c1hyb24_cloud_mg2 | 0.034 (n=1) | - | - | - | - | - | - | - | - |
| ours_rt2c1hyb24_cloud_neutral_832 | 0.115 (n=1) | - | - | - | - | - | - | - | - |
| ours_rt2c1hyb24_dev_mg2 | 0.033 (n=1) | - | - | - | - | - | - | - | - |
| ours_rt2c1hyb24_dev_neutral_832 | 0.107 (n=1) | - | - | - | - | - | - | - | - |
| ours_rt2c1hyb32_cloud_astra | 0.061 (n=3) | 0.333 (n=3) | 0.268 (n=3) | 0.919 (n=3) | 0.656 (n=3) | 0.743 (n=3) | 0.056 (n=3) | 0.24 (n=3) | 9/9 |
| ours_rt2c1hyb32_cloud_mg2 | 0.151 (n=12) | 0.521 (n=12) | 0.520 (n=12) | 1.072 (n=12) | 1.294 (n=12) | 1.252 (n=12) | 0.067 (n=12) | 0.14 (n=12) | 33/33 |
| ours_rt2c1hyb32_cloud_neutral_832 | 0.527 (n=12) | 0.400 (n=12) | 0.327 (n=12) | 1.468 (n=12) | 0.876 (n=12) | 1.165 (n=12) | 0.138 (n=12) | 0.21 (n=12) | 28/28 |
| ours_rt2c1hyb32_cloud_wp | 0.069 (n=12) | 0.572 (n=12) | 0.562 (n=12) | 0.967 (n=12) | 0.720 (n=12) | 0.881 (n=12) | 0.075 (n=12) | 0.07 (n=12) | 33/33 |
| ours_rt2c1hyb32_dev_astra | 0.063 (n=3) | 0.324 (n=3) | 0.268 (n=3) | 0.921 (n=3) | 0.671 (n=3) | 0.726 (n=3) | 0.053 (n=3) | 0.23 (n=3) | 9/9 |
| ours_rt2c1hyb32_dev_mg2 | 0.225 (n=12) | 0.515 (n=12) | 0.507 (n=12) | 1.069 (n=12) | 1.280 (n=12) | 1.238 (n=12) | 0.065 (n=12) | 0.12 (n=12) | 32/32 |
| ours_rt2c1hyb32_dev_neutral_832 | 0.544 (n=12) | 0.405 (n=12) | 0.337 (n=12) | 1.478 (n=12) | 1.223 (n=12) | 1.125 (n=12) | 0.163 (n=12) | 0.20 (n=12) | 27/27 |
| ours_rt2c1hyb32_dev_wp | 0.068 (n=12) | 0.571 (n=12) | 0.561 (n=12) | 0.972 (n=12) | 0.699 (n=12) | 0.882 (n=12) | 0.074 (n=12) | 0.11 (n=12) | 36/36 |
| ours_rt2c1hyb_cloud_abot | 0.083 (n=12) | 0.945 (n=12) | 0.899 (n=12) | 1.002 (n=12) | 0.833 (n=12) | 0.721 (n=12) | 0.067 (n=12) | 0.09 (n=12) | 31/31 |
| ours_rt2c1hyb_cloud_astra | 0.077 (n=3) | 0.312 (n=3) | 0.271 (n=3) | 1.015 (n=3) | 0.644 (n=3) | 0.797 (n=3) | 0.055 (n=3) | 0.16 (n=3) | 9/9 |
| ours_rt2c1hyb_cloud_mg2 | 0.189 (n=12) | 0.572 (n=12) | 0.603 (n=12) | 1.186 (n=12) | 1.280 (n=12) | 1.275 (n=12) | 0.058 (n=12) | 0.10 (n=12) | 34/34 |
| ours_rt2c1hyb_cloud_neutral_832 | 0.712 (n=12) | 0.373 (n=12) | 0.404 (n=12) | 1.655 (n=12) | 1.560 (n=12) | 0.990 (n=12) | 0.151 (n=12) | 0.13 (n=11) | 26/27 |
| ours_rt2c1hyb_cloud_wp | 0.065 (n=12) | 0.695 (n=12) | 0.705 (n=12) | 1.127 (n=12) | 0.697 (n=12) | 0.744 (n=12) | 0.074 (n=12) | 0.09 (n=12) | 34/34 |
| ours_rt2c1hyb_dev_abot | 0.081 (n=12) | 0.948 (n=12) | 0.894 (n=12) | 0.994 (n=12) | 0.821 (n=12) | 0.723 (n=12) | 0.066 (n=12) | 0.09 (n=12) | 31/31 |
| ours_rt2c1hyb_dev_astra | 0.078 (n=3) | 0.307 (n=3) | 0.265 (n=3) | 1.013 (n=3) | 0.638 (n=3) | 0.794 (n=3) | 0.056 (n=3) | 0.18 (n=3) | 9/9 |
| ours_rt2c1hyb_dev_mg2 | 0.111 (n=12) | 0.570 (n=12) | 0.595 (n=12) | 1.181 (n=12) | 1.277 (n=12) | 1.273 (n=12) | 0.057 (n=12) | 0.11 (n=12) | 34/34 |
| ours_rt2c1hyb_dev_neutral_832 | 0.526 (n=12) | 0.365 (n=12) | 0.400 (n=12) | 1.510 (n=12) | 1.675 (n=12) | 0.998 (n=12) | 0.150 (n=12) | 0.13 (n=12) | 28/28 |
| ours_rt2c1hyb_dev_wp | 0.065 (n=12) | 0.703 (n=12) | 0.709 (n=12) | 1.132 (n=12) | 0.696 (n=12) | 0.856 (n=12) | 0.075 (n=12) | 0.08 (n=12) | 33/33 |
| ours_sf1p3b_cal_astra | 0.063 (n=3) | 1.811 (n=3) | 1.689 (n=3) | 0.773 (n=3) | 1.038 (n=3) | 0.910 (n=3) | 0.053 (n=3) | 0.11 (n=3) | 9/9 |
| ours_sf1p3b_cal_neutral_832 | 0.780 (n=12) | 1.910 (n=12) | 1.803 (n=12) | 1.496 (n=12) | 1.836 (n=12) | 1.622 (n=12) | 0.179 (n=12) | 0.13 (n=12) | 35/35 |
| ours_sf1p3b_cal_neutral_832_nocap | 0.222 (n=12) | 2.020 (n=12) | 1.868 (n=12) | 1.318 (n=12) | 2.042 (n=12) | 1.436 (n=12) | 0.131 (n=12) | 0.12 (n=12) | 33/34 |
| ours_sf1p3b_cal_neutral_832_nocap_idle | 0.704 (n=3) | 2.041 (n=3) | 1.827 (n=4) | 1.480 (n=3) | 1.937 (n=3) | 1.431 (n=3) | 0.239 (n=3) | 0.26 (n=3) | 7/8 |
| ours_sf1p3b_pub_astra | 0.061 (n=3) | 1.823 (n=3) | 1.706 (n=3) | 0.771 (n=3) | 1.062 (n=3) | 0.792 (n=3) | 0.050 (n=3) | 0.13 (n=3) | 9/9 |
| ours_sf1p3b_pub_neutral_832 | 0.505 (n=12) | 2.016 (n=12) | 1.816 (n=12) | 1.754 (n=12) | 1.906 (n=12) | 1.662 (n=12) | 0.206 (n=12) | 0.13 (n=12) | 33/33 |
| ours_sf1p3b_pub_neutral_832_nocap | 0.343 (n=12) | 2.067 (n=12) | 1.883 (n=12) | 1.263 (n=12) | 2.063 (n=12) | 1.697 (n=12) | 0.190 (n=12) | 0.12 (n=12) | 34/34 |
| ours_sf1p3b_pub_neutral_832_nocap_idle | 0.733 (n=3) | 2.078 (n=3) | 1.861 (n=4) | 1.427 (n=3) | 2.561 (n=3) | 1.371 (n=3) | 0.277 (n=3) | 0.10 (n=3) | 7/8 |
| ours_sf24_abot | 0.075 (n=12) | 0.900 (n=12) | 0.855 (n=12) | 0.889 (n=12) | 0.746 (n=12) | 0.675 (n=12) | 0.056 (n=12) | 0.09 (n=12) | 31/32 |
| ours_sf24_abot_dit | 0.040 (n=1) | 0.628 (n=1) | - | - | - | - | - | - | - |
| ours_sf24_astra | 0.086 (n=3) | 0.236 (n=3) | 0.260 (n=3) | 0.977 (n=3) | 0.691 (n=3) | 0.820 (n=3) | 0.043 (n=3) | 0.19 (n=3) | 9/9 |
| ours_sf24_astra_dit | 0.100 (n=1) | 0.216 (n=1) | - | - | - | - | - | - | - |
| ours_sf24_mg2 | 0.088 (n=12) | 0.414 (n=12) | 0.426 (n=12) | 1.049 (n=12) | 1.200 (n=12) | 1.156 (n=12) | 0.057 (n=12) | 0.09 (n=12) | 31/31 |
| ours_sf24_mg2_dit | 0.030 (n=1) | 0.628 (n=1) | - | - | - | - | - | - | - |
| ours_sf24_neutral_832 | 0.539 (n=12) | 0.368 (n=12) | 0.375 (n=12) | 1.239 (n=12) | 1.287 (n=12) | 0.957 (n=12) | 0.215 (n=12) | 0.19 (n=12) | 30/31 |
| ours_sf24_neutral_832_dit | 0.105 (n=1) | 0.228 (n=1) | - | - | - | - | - | - | - |
| ours_sf24_neutral_832_nocap | 0.217 (n=12) | 0.633 (n=12) | 0.632 (n=12) | 1.292 (n=12) | 1.036 (n=12) | 1.081 (n=12) | 0.100 (n=12) | 0.10 (n=12) | 35/35 |
| ours_sf24_neutral_abot | 0.246 (n=12) | 0.966 (n=12) | 0.907 (n=12) | 1.168 (n=12) | 0.891 (n=12) | 1.018 (n=12) | 0.114 (n=12) | 0.11 (n=12) | 34/34 |
| ours_sf24_neutral_mg2 | 0.215 (n=12) | 0.441 (n=12) | 0.453 (n=12) | 1.264 (n=12) | 1.104 (n=12) | 1.112 (n=12) | 0.068 (n=12) | 0.10 (n=12) | 34/34 |
| ours_sf24_neutral_wp | 0.172 (n=12) | 0.658 (n=12) | 0.686 (n=12) | 1.159 (n=12) | 0.935 (n=12) | 1.009 (n=12) | 0.086 (n=12) | 0.12 (n=12) | 33/33 |
| ours_sf24_wp | 0.059 (n=12) | 0.596 (n=12) | 0.636 (n=12) | 1.029 (n=12) | 0.661 (n=12) | 0.662 (n=12) | 0.068 (n=12) | 0.08 (n=12) | 36/36 |
| ours_sf24_wp_dit | 0.035 (n=1) | 0.741 (n=1) | - | - | - | - | - | - | - |
| ours_sf24c1hyb_cloud_abot | 0.078 (n=12) | 0.897 (n=12) | 0.840 (n=12) | 0.902 (n=12) | 0.713 (n=12) | 0.669 (n=12) | 0.068 (n=12) | 0.09 (n=12) | 32/33 |
| ours_sf24c1hyb_cloud_astra | 0.095 (n=3) | 0.265 (n=3) | 0.237 (n=3) | 0.966 (n=3) | 0.653 (n=3) | 0.793 (n=3) | 0.056 (n=3) | 0.20 (n=3) | 9/9 |
| ours_sf24c1hyb_cloud_mg2 | 0.079 (n=12) | 0.456 (n=12) | 0.462 (n=12) | 1.093 (n=12) | 1.242 (n=12) | 1.148 (n=12) | 0.054 (n=12) | 0.10 (n=12) | 33/33 |
| ours_sf24c1hyb_cloud_neutral_832 | 0.790 (n=12) | 0.384 (n=12) | 0.384 (n=12) | 1.094 (n=12) | 1.278 (n=12) | 1.007 (n=12) | 0.184 (n=12) | 0.14 (n=12) | 29/29 |
| ours_sf24c1hyb_cloud_wp | 0.063 (n=12) | 0.622 (n=12) | 0.652 (n=12) | 1.014 (n=12) | 0.739 (n=12) | 0.677 (n=12) | 0.075 (n=12) | 0.16 (n=12) | 35/35 |
| ours_sf24c1hyb_dev_abot | 0.072 (n=12) | 0.889 (n=12) | 0.840 (n=12) | 0.895 (n=12) | 0.717 (n=12) | 0.666 (n=12) | 0.062 (n=12) | 0.18 (n=12) | 30/31 |
| ours_sf24c1hyb_dev_astra | 0.098 (n=3) | 0.261 (n=3) | 0.242 (n=3) | 0.964 (n=3) | 0.667 (n=3) | 0.784 (n=3) | 0.058 (n=3) | 0.19 (n=3) | 9/9 |
| ours_sf24c1hyb_dev_mg2 | 0.080 (n=12) | 0.447 (n=12) | 0.452 (n=12) | 1.088 (n=12) | 1.231 (n=12) | 1.131 (n=12) | 0.056 (n=12) | 0.09 (n=12) | 34/34 |
| ours_sf24c1hyb_dev_neutral_832 | 1.270 (n=12) | 0.379 (n=12) | 0.385 (n=12) | 1.186 (n=12) | 1.324 (n=12) | 1.050 (n=12) | 0.182 (n=12) | 0.16 (n=11) | 26/28 |
| ours_sf24c1hyb_dev_wp | 0.063 (n=12) | 0.630 (n=12) | 0.659 (n=12) | 1.012 (n=12) | 0.698 (n=12) | 0.660 (n=12) | 0.074 (n=12) | 0.16 (n=12) | 35/35 |
| ours_student1p3b_abot | 0.128 (n=12) | 1.132 (n=12) | 1.133 (n=12) | 0.849 (n=12) | 1.166 (n=12) | 1.124 (n=12) | 0.078 (n=12) | 0.11 (n=11) | 33/34 |
| ours_student1p3b_astra | 0.075 (n=3) | 1.276 (n=3) | 1.154 (n=3) | 0.841 (n=3) | 1.098 (n=3) | 0.983 (n=3) | 0.056 (n=3) | 0.14 (n=3) | 9/9 |
| ours_student1p3b_mg2 | 0.134 (n=12) | 1.737 (n=12) | 1.662 (n=12) | 1.111 (n=12) | 1.429 (n=12) | 1.209 (n=12) | 0.080 (n=12) | 0.09 (n=12) | 33/33 |
| ours_student1p3b_neutral_832 | 1.087 (n=12) | 1.745 (n=12) | 1.599 (n=12) | 1.763 (n=12) | 1.897 (n=12) | 1.600 (n=12) | 0.172 (n=12) | 0.11 (n=12) | 32/32 |
| ours_student1p3b_neutral_832_nocap | 0.636 (n=12) | 1.817 (n=12) | 1.763 (n=12) | 1.321 (n=12) | 1.696 (n=12) | 1.519 (n=12) | 0.153 (n=12) | 0.12 (n=12) | 36/36 |
| ours_student1p3b_neutral_abot | 0.361 (n=12) | 1.562 (n=12) | 1.561 (n=12) | 1.270 (n=12) | 1.757 (n=12) | 1.867 (n=12) | 0.143 (n=12) | 0.13 (n=12) | 36/36 |
| ours_student1p3b_neutral_mg2 | 0.282 (n=12) | 1.927 (n=12) | 1.794 (n=12) | 1.250 (n=12) | 1.460 (n=12) | 1.445 (n=12) | 0.141 (n=12) | 0.10 (n=12) | 34/35 |
| ours_student1p3b_neutral_wp | 0.390 (n=12) | 1.509 (n=12) | 1.525 (n=12) | 1.736 (n=12) | 1.811 (n=12) | 1.751 (n=12) | 0.161 (n=12) | 0.15 (n=12) | 35/35 |
| ours_student1p3b_wp | 0.143 (n=12) | 1.116 (n=12) | 1.166 (n=12) | 0.895 (n=12) | 1.382 (n=12) | 1.487 (n=12) | 0.080 (n=12) | 0.10 (n=12) | 34/34 |
| ours_teacher_abot | 0.133 (n=12) | 1.269 (n=12) | 1.205 (n=12) | 1.145 (n=12) | 0.901 (n=12) | 0.876 (n=12) | 0.189 (n=12) | 0.12 (n=11) | 27/28 |
| ours_teacher_astra | 0.138 (n=3) | 0.319 (n=3) | 0.426 (n=3) | 1.205 (n=3) | 0.814 (n=3) | 0.958 (n=3) | 0.080 (n=3) | 0.15 (n=3) | 9/9 |
| ours_teacher_mg2 | 0.226 (n=12) | 1.415 (n=12) | 1.252 (n=12) | 1.291 (n=12) | 1.861 (n=12) | 1.764 (n=12) | 0.184 (n=12) | 0.15 (n=12) | 25/28 |
| ours_teacher_neutral_832 | 0.447 (n=12) | 0.529 (n=12) | 0.458 (n=12) | 1.383 (n=12) | 1.135 (n=12) | 1.082 (n=12) | 0.264 (n=12) | 0.30 (n=12) | 29/29 |
| ours_teacher_neutral_832_nocap | 0.318 (n=12) | 1.214 (n=12) | 1.200 (n=12) | 1.547 (n=12) | 1.766 (n=12) | 1.531 (n=12) | 0.150 (n=12) | 0.12 (n=12) | 32/32 |
| ours_teacher_neutral_abot | 0.577 (n=12) | 1.534 (n=12) | 1.464 (n=12) | 1.247 (n=12) | 1.599 (n=12) | 1.198 (n=12) | 0.383 (n=12) | 0.27 (n=11) | 28/30 |
| ours_teacher_neutral_mg2 | 0.336 (n=12) | 1.263 (n=12) | 1.187 (n=12) | 1.330 (n=12) | 1.785 (n=12) | 1.777 (n=12) | 0.222 (n=12) | 0.15 (n=12) | 28/29 |
| ours_teacher_neutral_wp | 0.599 (n=12) | 1.178 (n=12) | 1.146 (n=12) | 1.459 (n=12) | 1.366 (n=12) | 1.301 (n=12) | 0.213 (n=12) | 0.21 (n=12) | 31/31 |
| ours_teacher_wp | 0.067 (n=12) | 1.166 (n=12) | 1.104 (n=12) | 1.184 (n=12) | 1.000 (n=12) | 0.884 (n=12) | 0.130 (n=12) | 0.17 (n=12) | 33/33 |
| waypoint | 0.572 (n=24) | 0.809 (n=24) | 0.805 (n=24) | 1.010 (n=24) | 1.100 (n=24) | 0.999 (n=24) | 0.193 (n=24) | 0.06 (n=23) | 59/61 |
