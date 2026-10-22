# Frozen baseline table (generated 2026-10-19 19:20 CDT by baselines/report.py)

runs: `/home/hk28965/specwm/baselines/runs` (94 arms, 7185 clips); metrics: `/home/hk28965/specwm/baselines/metrics, /home/hk28965/specwm/baselines/metrics_g7` (judge 4449, VBench 4414, RPE 4365, reaction 594, luminance 7119 clips with values).

Columns: GPUs = GPUs the arm runs on (timing.json `gpus`, default 1); fps gen = steady DiT-only fps, fps e2e = steady fps incl. decode, gen s = DiT-only generation time of the 10-s clip, TTFF = input to first decoded frame (s), VRAM = peak allocated GiB (timing.json, PROTOCOL section 4; '(dev)' = device GPU only for 2-GPU hybrid arms); judge Q (1-10) / drift (0-100) = Qwen3-VL clip judge on 8 evenly spaced frames; VBench Q = 6-dim quality score (0-100, clips resampled to 16 fps), imaging = MUSIQ (0-100), MUSIQ drift = std of the 2 s segment means; RPE Rot = mean relative rotation error (deg per 1/8 s step) of DA3-recovered poses vs the commanded program (Sim(3)-aligned per fifth); react = P5 reaction time from the model's effective input switch until the optical-flow sign follows (s / frames at native fps, mean over valid switches); black = number of the row's clips that go black (last-second mean luminance of the 16x9 gray downscale < 10 or < 30 % of the first second; metrics/luminance.json). set = <scene set><suffix> of the clip names (suffix _nocap = empty-prompt arm, _long = 60-s P6 clips: each ranked and paired only within its own suffix). n = clips in the row; '(partial)' = fewer than the frozen task list tasks/<set>.json expects (generation or scoring still running). Arrows give the direction; per scene set, **bold** = best arm, <u>underline</u> = second among the main rows; '(k)' = metric present on k of n clips. Means over clips. Extra rows (split by scene type, P2p+P4p only, excl. black) are shown in italics and are never ranked.

## 1. Per model arm x scene set (one table per set, footnotes below each)

### set `abot` (n expected 60 per arm = 5 programs P1-P5)

| model_arm | set | n | res | GPUs | fps gen ↑ | fps e2e ↑ | gen s ↓ | TTFF s ↓ | VRAM GiB ↓ | judge Q ↑ | judge drift ↓ | VBench Q ↑ | imaging ↑ | MUSIQ drift ↓ | RPE Rot ° ↓ | react s ↓ | react fr ↓ | black |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| abot | abot | 60 | 1280x704 | 1 | **15.4** | **12.6** | **7.9** | **0.95** | <u>32.1</u> | 8.03 | 26.5 | 80.50 | 71.5 | 3.08 | <u>0.543</u> | 0.46 (11) | 5.5 (11) | 0 |
| *abot (_1st scenes only, n=35)* | abot | 35 | 1280x704 | 1 | 15.5 | 12.7 | 7.9 | 0.94 | 32.1 | 7.94 | 27.0 | 78.65 | 70.1 | 3.78 | 0.538 | 0.32 (6) | 3.9 (6) | 0 |
| *abot (_3rd scenes, n=25)* | abot | 25 | 1280x704 | 1 | 15.4 | 12.4 | 7.9 | 0.96 | 32.1 | 8.16 | 25.8 | 83.11 | 73.6 | 2.11 | 0.550 | 0.62 (5) | 7.5 (5) | 0 |
| abot_t5offload | abot | 15 (partial) | 1280x704 | 1 | **15.4** | <u>12.5</u> | <u>8.0</u> | <u>0.97</u> | **22.5** | 8.07 | 22.7 | 80.23 | 72.5 | 3.17 | 0.579 | 0.49 (3) | 5.9 (3) | 0 |
| *abot_t5offload (_1st scenes only, n=5)* | abot | 5 (partial) | 1280x704 | 1 | 15.3 | 12.4 | 8.0 | 1.01 | 22.5 | 7.40 | 16.0 | 75.42 | 67.8 | 6.29 | 0.623 | 0.36 (1) | 4.3 (1) | 0 |
| *abot_t5offload (_3rd scenes, n=10)* | abot | 10 (partial) | 1280x704 | 1 | 15.4 | 12.6 | 7.9 | 0.95 | 22.5 | 8.40 | 26.0 | 82.64 | 74.8 | 1.61 | 0.557 | 0.56 (2) | 6.7 (2) | 0 |
| ours_adapt25_abot | abot | 60 | 1280x704 | 1 | 3.6 | 2.5 | 47.0 | 17.37 | 85.1 | 8.75 | 13.0 | 82.96 | <u>74.9</u> | <u>1.57</u> | 0.729 | 0.34 (11) | 5.4 (11) | 0 |
| *ours_adapt25_abot (_1st scenes only, n=35)* | abot | 35 | 1280x704 | 1 | 3.6 | 2.5 | 47.1 | 17.31 | 85.1 | 8.77 | 12.0 | 81.71 | 74.2 | 1.79 | 0.732 | 0.15 (6) | 2.4 (6) | 0 |
| *ours_adapt25_abot (_3rd scenes, n=25)* | abot | 25 | 1280x704 | 1 | 3.6 | 2.5 | 46.8 | 17.45 | 85.1 | 8.72 | 14.4 | 84.71 | 75.8 | 1.26 | 0.726 | 0.56 (5) | 9.0 (5) | 0 |
| *ours_adapt25_abot (P2p+P4p only, n=24)* | abot | 24 | 1280x704 | 1 | 3.7 | 2.5 | 47.4 | 17.20 | 85.1 | 8.62 | 24.8 | 82.38 | 74.7 | 1.93 | 1.070 | - | - | 0 |
| ours_adapt25c1hyb_cloud_abot | abot | 60 | 1280x704 | 2 | 2.3 | - | 81.2 | - | 84.3 (dev) | **8.85** | <u>12.2</u> | 83.56 | **75.0** | 1.61 | 0.564 | **0.09 (12)** | **1.5 (12)** | 0 |
| *ours_adapt25c1hyb_cloud_abot (_1st scenes only, n=35)* | abot | 35 | 1280x704 | 2 | 2.3 | - | 83.0 | - | 84.3 (dev) | 8.91 | 8.6 | 82.59 | 74.6 | 1.77 | 0.550 | 0.09 (7) | 1.5 (7) | 0 |
| *ours_adapt25c1hyb_cloud_abot (_3rd scenes, n=25)* | abot | 25 | 1280x704 | 2 | 2.4 | - | 78.7 | - | 84.3 (dev) | 8.76 | 17.4 | 84.93 | 75.6 | 1.38 | 0.583 | 0.09 (5) | 1.5 (5) | 0 |
| *ours_adapt25c1hyb_cloud_abot (P2p+P4p only, n=24)* | abot | 24 | 1280x704 | 2 | 2.3 | - | 83.3 | - | 84.3 (dev) | 8.75 | 16.2 | 82.92 | 75.0 | 1.63 | 0.803 | - | - | 0 |
| ours_adapt25c1hyb_dev_abot | abot | 60 | 1280x704 | 2 | 2.3 | - | 79.3 | - | 84.6 (dev) | 8.80 | 12.8 | **83.62** | **75.0** | <u>1.57</u> | 0.563 | <u>0.10 (11)</u> | <u>1.6 (11)</u> | 0 |
| *ours_adapt25c1hyb_dev_abot (_1st scenes only, n=35)* | abot | 35 | 1280x704 | 2 | 2.3 | - | 81.0 | - | 84.6 (dev) | 8.86 | 9.3 | 82.63 | 74.5 | 1.75 | 0.549 | 0.11 (6) | 1.8 (6) | 0 |
| *ours_adapt25c1hyb_dev_abot (_3rd scenes, n=25)* | abot | 25 | 1280x704 | 2 | 2.4 | - | 76.8 | - | 84.6 (dev) | 8.72 | 17.8 | 85.02 | 75.7 | 1.31 | 0.582 | 0.09 (5) | 1.4 (5) | 0 |
| *ours_adapt25c1hyb_dev_abot (P2p+P4p only, n=24)* | abot | 24 | 1280x704 | 2 | 2.3 | - | 81.4 | - | 84.6 (dev) | 8.71 | 20.0 | 82.95 | 75.0 | 1.63 | 0.804 | - | - | 0 |
| ours_rt2_abot | abot | 60 | 1280x704 | 1 | 3.7 | 2.5 | 43.4 | 16.50 | 84.8 | 8.78 | **10.9** | 82.98 | 74.3 | 1.58 | 0.752 | 0.27 (12) | 4.4 (12) | 0 |
| *ours_rt2_abot (_1st scenes only, n=35)* | abot | 35 | 1280x704 | 1 | 3.7 | 2.5 | 43.5 | 16.44 | 84.8 | 8.77 | 9.6 | 82.24 | 73.5 | 1.86 | 0.732 | 0.08 (7) | 1.2 (7) | 0 |
| *ours_rt2_abot (_3rd scenes, n=25)* | abot | 25 | 1280x704 | 1 | 3.7 | 2.5 | 43.3 | 16.58 | 84.8 | 8.80 | 12.8 | 84.01 | 75.5 | 1.20 | 0.781 | 0.55 (5) | 8.8 (5) | 0 |
| *ours_rt2_abot (P2p+P4p only, n=24)* | abot | 24 | 1280x704 | 1 | 3.7 | 2.5 | 43.5 | 16.38 | 84.8 | 8.79 | 17.7 | 81.24 | 74.7 | 1.65 | 1.073 | - | - | 0 |
| ours_rt2c1hyb_cloud_abot | abot | 24 (partial) | 1280x704 | 2 | 2.6 | - | 60.9 | - | 84.3 (dev) | - | - | - | - | - | - | - | - | - |
| *ours_rt2c1hyb_cloud_abot (_1st scenes only, n=14)* | abot | 14 (partial) | 1280x704 | 2 | 2.6 | - | 60.8 | - | 84.3 (dev) | - | - | - | - | - | - | - | - | - |
| *ours_rt2c1hyb_cloud_abot (_3rd scenes, n=10)* | abot | 10 (partial) | 1280x704 | 2 | 2.6 | - | 61.2 | - | 84.3 (dev) | - | - | - | - | - | - | - | - | - |
| *ours_rt2c1hyb_cloud_abot (P2p+P4p only, n=9)* | abot | 9 (partial) | 1280x704 | 2 | 2.6 | - | 60.8 | - | 84.3 (dev) | - | - | - | - | - | - | - | - | - |
| ours_rt2c1hyb_dev_abot | abot | 24 (partial) | 1280x704 | 2 | 2.6 | - | 59.7 (12) | - | 84.3 (dev) | - | - | - | - | - | - | - | - | - |
| *ours_rt2c1hyb_dev_abot (_1st scenes only, n=14)* | abot | 14 (partial) | 1280x704 | 2 | 2.6 | - | 59.7 (5) | - | 84.3 (dev) | - | - | - | - | - | - | - | - | - |
| *ours_rt2c1hyb_dev_abot (_3rd scenes, n=10)* | abot | 10 (partial) | 1280x704 | 2 | 2.6 | - | 59.7 (7) | - | 84.3 (dev) | - | - | - | - | - | - | - | - | - |
| *ours_rt2c1hyb_dev_abot (P2p+P4p only, n=9)* | abot | 9 (partial) | 1280x704 | 2 | 2.6 | - | 59.7 (4) | - | 84.3 (dev) | - | - | - | - | - | - | - | - | - |
| ours_sf24_abot | abot | 60 | 1280x704 | 1 | 2.6 | 2.4 | 62.5 | 23.54 | - | **8.85** | 13.9 | <u>83.58</u> | 74.5 | **1.45** | **0.533** | **0.09 (12)** | **1.5 (12)** | 0 |
| *ours_sf24_abot (_1st scenes only, n=35)* | abot | 35 | 1280x704 | 1 | 2.7 | 2.4 | 61.7 | 23.36 | - | 8.91 | 12.4 | 82.69 | 74.0 | 1.65 | 0.530 | 0.11 (7) | 1.8 (7) | 0 |
| *ours_sf24_abot (_3rd scenes, n=25)* | abot | 25 | 1280x704 | 1 | 2.5 | 2.3 | 63.8 | 23.79 | - | 8.76 | 16.0 | 84.83 | 75.3 | 1.16 | 0.538 | 0.07 (5) | 1.1 (5) | 0 |
| *ours_sf24_abot (P2p+P4p only, n=24)* | abot | 24 | 1280x704 | 1 | 2.5 | 2.3 | 64.6 | 23.54 | - | 8.75 | 20.8 | 83.06 | 74.6 | 1.70 | 0.765 | - | - | 0 |
| ours_sf24c1hyb_cloud_abot | abot | 60 | 1280x704 | 2 | 2.4 | - | 67.9 | - | 84.3 (dev) | - | - | - | - | - | - | - | - | 0 |
| *ours_sf24c1hyb_cloud_abot (_1st scenes only, n=35)* | abot | 35 | 1280x704 | 2 | 2.4 | - | 66.8 | - | 84.3 (dev) | - | - | - | - | - | - | - | - | 0 |
| *ours_sf24c1hyb_cloud_abot (_3rd scenes, n=25)* | abot | 25 | 1280x704 | 2 | 2.3 | - | 69.3 | - | 84.3 (dev) | - | - | - | - | - | - | - | - | 0 |
| *ours_sf24c1hyb_cloud_abot (P2p+P4p only, n=24)* | abot | 24 | 1280x704 | 2 | 2.3 | - | 69.9 | - | 84.3 (dev) | - | - | - | - | - | - | - | - | 0 |
| ours_sf24c1hyb_dev_abot | abot | 60 | 1280x704 | 2 | 2.4 | - | 65.9 | - | 84.4 (dev) | - | - | - | - | - | - | - | - | 0 |
| *ours_sf24c1hyb_dev_abot (_1st scenes only, n=35)* | abot | 35 | 1280x704 | 2 | 2.4 | - | 64.9 | - | 84.4 (dev) | - | - | - | - | - | - | - | - | 0 |
| *ours_sf24c1hyb_dev_abot (_3rd scenes, n=25)* | abot | 25 | 1280x704 | 2 | 2.3 | - | 67.4 | - | 84.4 (dev) | - | - | - | - | - | - | - | - | 0 |
| *ours_sf24c1hyb_dev_abot (P2p+P4p only, n=24)* | abot | 24 | 1280x704 | 2 | 2.3 | - | 68.0 | - | 84.4 (dev) | - | - | - | - | - | - | - | - | 0 |
| ours_student1p3b_abot | abot | 60 | 1280x704 | 1 | <u>4.5</u> | 3.7 | 36.8 | 21.03 | - | - | - | - | - | - | - | - | - | 3 |
| *ours_student1p3b_abot (_1st scenes only, n=35)* | abot | 35 | 1280x704 | 1 | 4.5 | 3.7 | 36.8 | 21.00 | - | - | - | - | - | - | - | - | - | 3 |
| *ours_student1p3b_abot (_3rd scenes, n=25)* | abot | 25 | 1280x704 | 1 | 4.5 | 3.6 | 36.8 | 21.07 | - | - | - | - | - | - | - | - | - | 0 |
| *ours_student1p3b_abot (P2p+P4p only, n=24)* | abot | 24 | 1280x704 | 1 | 4.5 | 3.7 | 36.3 | 20.58 | - | - | - | - | - | - | - | - | - | 2 |
| *ours_student1p3b_abot excl. black (n=57)* | abot | 57 | 1280x704 | 1 | 4.5 | 3.6 | 37.0 | 21.10 | - | - | - | - | - | - | - | - | - | 0 |
| ours_teacher_abot | abot | 60 | 1280x704 | 1 | 2.0 | 1.6 | 81.5 | 18.53 | 84.8 | <u>8.83</u> | 13.7 | 82.86 | 74.6 | 1.73 | 0.727 | 0.12 (11) | 2.0 (11) | 0 |
| *ours_teacher_abot (_1st scenes only, n=35)* | abot | 35 | 1280x704 | 1 | 2.0 | 1.6 | 79.2 | 18.16 | 84.8 | 8.86 | 11.0 | 81.77 | 73.7 | 1.97 | 0.699 | 0.14 (7) | 2.2 (7) | 0 |
| *ours_teacher_abot (_3rd scenes, n=25)* | abot | 25 | 1280x704 | 1 | 1.9 | 1.5 | 84.7 | 19.05 | 84.8 | 8.80 | 17.4 | 84.37 | 75.8 | 1.40 | 0.767 | 0.10 (4) | 1.6 (4) | 0 |
| *ours_teacher_abot (P2p+P4p only, n=24)* | abot | 24 | 1280x704 | 1 | 2.0 | 1.6 | 79.9 | 18.80 | 84.8 | 8.58 | 18.1 | 82.30 | 74.7 | 2.00 | 1.040 | - | - | 0 |

- Set means use programs P1-P5 only for EVERY arm (ABot has no key magnitude, so its P2p = P2 and P4p = P4 videos are identical); ours' P2p/P4p clips are listed in separate '(P2p+P4p only)' rows and are not ranked.
- 5 of the 12 ABot scenes are third-person (`_3rd`: aurora_ice, case4_desert_caravan, desert_camel, desert_valley, ruins_explorer), where WASD moves the character and not the camera: RPE Rot / reaction on them is not a camera comparison; see the '_1st scenes only' (7 scenes) and '_3rd scenes' (5 scenes) split rows (not ranked).
- `abot` (24 clips): P2p/P4p clips exist but are byte-identical to the P2/P4 clips (no key magnitude) and are not counted.
- `ours_adapt25c1hyb_cloud_abot`, `ours_adapt25c1hyb_dev_abot`, `ours_rt2c1hyb_cloud_abot`, `ours_rt2c1hyb_dev_abot`, `ours_sf24c1hyb_cloud_abot`, `ours_sf24c1hyb_dev_abot`: 2 GPUs (device + cloud): VRAM = device GPU peak only (the cloud GPU's peak is timing.json `cloud_peak_alloc_GiB`); TTFF / fps e2e are not measured in the two-process mode.
- `ours_sf24_abot` (60 of 60 clips), `ours_student1p3b_abot` (60 of 60 clips): fps gen / gen s are DiT-only, corrected for the side-stream decode overlap (audit 10/19).
- `ours_student1p3b_abot` (3 of 60): clips go black (see the 'excl. black' rows).
- '(partial)' rows: n below the 60 clips of tasks/abot.json (generation or scoring still running).

### set `abot_long` (n expected 4 per arm = 4 scenes x P6)

| model_arm | set | n | res | GPUs | fps gen ↑ | fps e2e ↑ | gen s ↓ | TTFF s ↓ | VRAM GiB ↓ | judge Q ↑ | judge drift ↓ | VBench Q ↑ | imaging ↑ | MUSIQ drift ↓ | RPE Rot ° ↓ | react s ↓ | react fr ↓ | black |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| abot | abot_long | 1 (partial) | 1280x704 | 1 | **15.3** | **12.2** | **47.1** | **0.99** | **32.1** | 7.00 | 100.0 | 72.82 | 59.7 | 11.76 | 0.445 | - | - | 0 |
| ours_teacher_abot_long | abot_long | 1 (partial) | 1280x704 | 1 | <u>2.0</u> | <u>1.6</u> | <u>483.7</u> | <u>81.58</u> | <u>70.8</u> | - | - | - | - | - | - | - | - | 0 |

- '(partial)' rows: n below the 4 clips of tasks/abot_long.json (generation or scoring still running).

### set `astra` (n expected 21 per arm)

| model_arm | set | n | res | GPUs | fps gen ↑ | fps e2e ↑ | gen s ↓ | TTFF s ↓ | VRAM GiB ↓ | judge Q ↑ | judge drift ↓ | VBench Q ↑ | imaging ↑ | MUSIQ drift ↓ | RPE Rot ° ↓ | react s ↓ | react fr ↓ | black |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| astra | astra | 21 | 832x480 | 1 | 1.3 | 1.1 | 173.0 | 25.80 | **14.9** | 6.52 | 48.8 | 77.27 | 70.8 | 2.81 | 0.992 | - | - | 0 |
| ours_adapt25_astra | astra | 21 | 832x480 | 1 | 9.5 | 6.3 | 17.5 | 8.76 | 62.8 | 7.90 | <u>10.5</u> | 78.87 | 73.0 | 2.71 | 0.522 | **0.14 (3)** | **2.2 (3)** | 0 |
| ours_adapt25c1hyb_cloud_astra | astra | 21 | 832x480 | 2 | 6.8 | - | 23.8 | - | 62.1 (dev) | **8.19** | **8.6** | <u>79.22</u> | <u>73.3</u> | 2.38 | <u>0.413</u> | <u>0.15 (3)</u> | <u>2.3 (3)</u> | 0 |
| ours_adapt25c1hyb_dev_astra | astra | 21 | 832x480 | 2 | 6.8 | - | 23.0 | - | 62.3 (dev) | **8.19** | 11.4 | **79.31** | 73.1 | 2.40 | **0.412** | 0.20 (3) | 3.2 (3) | 0 |
| ours_rt2_astra | astra | 21 | 832x480 | 1 | 9.6 | 6.3 | 16.1 | 8.29 | 62.6 | 7.76 | 12.4 | 78.46 | 72.9 | <u>2.34</u> | 0.516 | 0.16 (3) | 2.6 (3) | 0 |
| ours_sf1p3b_cal_astra | astra | 21 | 832x480 | 1 | <u>16.9</u> | <u>12.6</u> | <u>9.7</u> | <u>6.77</u> | <u>28.7</u> | - | - | - | - | - | - | - | - | 0 |
| ours_sf1p3b_pub_astra | astra | 21 | 832x480 | 1 | **18.6** | **13.4** | **8.9** | **6.66** | <u>28.7</u> | - | - | - | - | - | - | - | - | 0 |
| ours_sf24_astra | astra | 21 | 832x480 | 1 | 7.2 | 6.3 | 22.4 | 7.86 | - | <u>8.10</u> | 11.4 | 79.03 | 72.4 | **2.25** | 0.445 | 0.19 (3) | 3.1 (3) | 0 |
| ours_student1p3b_astra | astra | 21 | 832x480 | 1 | 11.2 | 9.0 | 14.2 | 6.87 | - | - | - | - | - | - | - | - | - | 0 |
| ours_teacher_astra | astra | 21 | 832x480 | 1 | 5.3 | 4.1 | 30.0 | 9.06 | 62.6 | 7.86 | 11.0 | 78.69 | **74.0** | 2.36 | 0.563 | <u>0.15 (3)</u> | 2.4 (3) | 0 |

- Astra's own-scene set = 3 scenes x 7 programs = 21 clips per arm (n expected 21).
- `astra` (21 of 21 clips): timing.json flags a shared-GPU measurement (speed columns are a lower bound there).
- `ours_adapt25c1hyb_cloud_astra`, `ours_adapt25c1hyb_dev_astra`: 2 GPUs (device + cloud): VRAM = device GPU peak only (the cloud GPU's peak is timing.json `cloud_peak_alloc_GiB`); TTFF / fps e2e are not measured in the two-process mode.
- `ours_sf1p3b_cal_astra` (21 of 21 clips), `ours_sf1p3b_pub_astra` (21 of 21 clips), `ours_sf24_astra` (21 of 21 clips), `ours_student1p3b_astra` (21 of 21 clips): fps gen / gen s are DiT-only, corrected for the side-stream decode overlap (audit 10/19).

### set `mg2` (n expected 84 per arm)

| model_arm | set | n | res | GPUs | fps gen ↑ | fps e2e ↑ | gen s ↓ | TTFF s ↓ | VRAM GiB ↓ | judge Q ↑ | judge drift ↓ | VBench Q ↑ | imaging ↑ | MUSIQ drift ↓ | RPE Rot ° ↓ | react s ↓ | react fr ↓ | black |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mg2 | mg2 | 84 | 640x352 | 1 | **34.3** | **21.1** | **3.8** | **2.11** | **11.5** | **8.00** | 27.0 | **78.88** | 68.7 | **3.28** | **0.391** | 0.26 (12) | 3.1 (12) | 0 |
| ours_adapt25_mg2 | mg2 | 84 | 640x352 | 1 | 19.2 | 12.1 | 8.8 | 4.76 | 43.9 | 7.68 | 23.4 | 77.37 | 68.1 | 4.36 | 1.114 | 0.19 (12) | 3.1 (12) | 0 |
| ours_adapt25c1hyb_cloud_mg2 | mg2 | 84 | 640x352 | 2 | 12.4 | - | 14.1 | - | <u>43.3 (dev)</u> | 7.88 | **15.1** | 78.29 | <u>68.9</u> | 3.56 | 0.717 | <u>0.11 (12)</u> | <u>1.8 (12)</u> | 1 |
| *ours_adapt25c1hyb_cloud_mg2 excl. black (n=83)* | mg2 | 83 | 640x352 | 2 | 12.4 | - | 14.1 | - | 43.3 (dev) | 7.89 | 14.3 | 78.30 | 69.1 | 3.56 | 0.724 | 0.11 (12) | 1.8 (12) | 0 |
| ours_adapt25c1hyb_dev_mg2 | mg2 | 84 | 640x352 | 2 | 12.4 | - | 13.7 | - | 43.4 (dev) | <u>7.98</u> | **15.1** | <u>78.40</u> | 68.7 | 3.59 | 0.709 | 0.12 (12) | 1.9 (12) | 1 |
| *ours_adapt25c1hyb_dev_mg2 excl. black (n=83)* | mg2 | 83 | 640x352 | 2 | 12.4 | - | 13.7 | - | 43.4 (dev) | 7.98 | 14.3 | 78.41 | 68.9 | 3.58 | 0.716 | 0.12 (12) | 1.9 (12) | 0 |
| ours_rt2_mg2 | mg2 | 84 | 640x352 | 1 | 19.3 | 12.1 | 8.2 | 4.56 | 43.7 | 7.52 | 21.1 | 77.20 | 67.4 | 4.15 | 1.130 | 0.12 (12) | 2.0 (12) | 0 |
| ours_rt2c1hyb_cloud_mg2 | mg2 | 84 | 640x352 | 2 | 13.8 | - | 11.3 | - | <u>43.3 (dev)</u> | - | - | - | - | - | - | - | - | 0 |
| ours_rt2c1hyb_dev_mg2 | mg2 | 84 | 640x352 | 2 | 13.8 | - | 10.9 | - | <u>43.3 (dev)</u> | - | - | - | - | - | - | - | - | 0 |
| ours_sf24_mg2 | mg2 | 84 | 640x352 | 1 | 13.6 | 11.8 | 12.2 | 4.69 | - | <u>7.98</u> | <u>15.2</u> | 78.31 | 67.8 | <u>3.40</u> | <u>0.627</u> | **0.09 (12)** | **1.4 (12)** | 0 |
| ours_sf24c1hyb_cloud_mg2 | mg2 | 84 | 640x352 | 2 | 12.9 | - | 12.6 | - | <u>43.3 (dev)</u> | - | - | - | - | - | - | - | - | 0 |
| ours_sf24c1hyb_dev_mg2 | mg2 | 84 | 640x352 | 2 | 13.0 | - | 12.2 | - | <u>43.3 (dev)</u> | - | - | - | - | - | - | - | - | 0 |
| ours_student1p3b_mg2 | mg2 | 84 | 640x352 | 1 | <u>21.8</u> | <u>17.3</u> | <u>7.4</u> | <u>4.11</u> | - | - | - | - | - | - | - | - | - | 1 |
| *ours_student1p3b_mg2 excl. black (n=83)* | mg2 | 83 | 640x352 | 1 | 21.8 | 17.3 | 7.4 | 4.11 | - | - | - | - | - | - | - | - | - | 0 |
| ours_teacher_mg2 | mg2 | 84 | 640x352 | 1 | 10.6 | 8.0 | 14.9 | 4.92 | 43.7 | <u>7.98</u> | 19.5 | 77.40 | **69.6** | 3.90 | 1.142 | 0.15 (12) | 2.4 (12) | 0 |

- `ours_adapt25c1hyb_cloud_mg2`, `ours_adapt25c1hyb_dev_mg2`, `ours_rt2c1hyb_cloud_mg2`, `ours_rt2c1hyb_dev_mg2`, `ours_sf24c1hyb_cloud_mg2`, `ours_sf24c1hyb_dev_mg2`: 2 GPUs (device + cloud): VRAM = device GPU peak only (the cloud GPU's peak is timing.json `cloud_peak_alloc_GiB`); TTFF / fps e2e are not measured in the two-process mode.
- `ours_adapt25c1hyb_cloud_mg2` (1 of 84), `ours_adapt25c1hyb_dev_mg2` (1 of 84), `ours_student1p3b_mg2` (1 of 84): clips go black (see the 'excl. black' rows).
- `ours_sf24_mg2` (84 of 84 clips), `ours_student1p3b_mg2` (84 of 84 clips): fps gen / gen s are DiT-only, corrected for the side-stream decode overlap (audit 10/19).

### set `mg2_long` (n expected 4 per arm = 4 scenes x P6)

| model_arm | set | n | res | GPUs | fps gen ↑ | fps e2e ↑ | gen s ↓ | TTFF s ↓ | VRAM GiB ↓ | judge Q ↑ | judge drift ↓ | VBench Q ↑ | imaging ↑ | MUSIQ drift ↓ | RPE Rot ° ↓ | react s ↓ | react fr ↓ | black |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mg2 | mg2_long | 1 (partial) | 640x352 | 1 | 34.8 | 21.3 | 21.1 | 9.32 | 16.2 | 3.00 | 100.0 | 60.93 | 36.3 | 16.95 | 2.980 | - | - | 1 |

- `mg2` (1 of 1): clips go black (see the 'excl. black' rows).
- '(partial)' rows: n below the 4 clips of tasks/mg2_long.json (generation or scoring still running).

### set `neutral_832` (n expected 84 per arm)

| model_arm | set | n | res | GPUs | fps gen ↑ | fps e2e ↑ | gen s ↓ | TTFF s ↓ | VRAM GiB ↓ | judge Q ↑ | judge drift ↓ | VBench Q ↑ | imaging ↑ | MUSIQ drift ↓ | RPE Rot ° ↓ | react s ↓ | react fr ↓ | black |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| astra | neutral_832 | 84 | 832x480 | 1 | 1.3 | 1.1 | 173.0 | 25.80 | **14.9** | 4.95 | 77.2 | 70.18 | 55.3 | 6.16 | 1.658 | 0.75 (5) | 15.0 (5) | 6 |
| *astra excl. black (n=78)* | neutral_832 | 78 | 832x480 | 1 | 1.3 | 1.1 | 173.0 | 25.80 | 14.9 | 4.88 | 76.8 | 70.61 | 56.4 | 5.59 | 1.626 | 0.75 (5) | 15.0 (5) | 0 |
| ours_adapt25_neutral_832 | neutral_832 | 84 | 832x480 | 1 | 9.4 | 6.2 | 17.8 | 8.83 | 62.8 | 7.54 | 40.9 | 74.12 | 68.5 | **3.91** | **0.681** | 0.27 (12) | 4.3 (12) | 1 |
| *ours_adapt25_neutral_832 excl. black (n=83)* | neutral_832 | 83 | 832x480 | 1 | 9.4 | 6.2 | 17.8 | 8.75 | 62.8 | 7.54 | 40.4 | 74.18 | 68.8 | 3.76 | 0.688 | 0.27 (12) | 4.3 (12) | 0 |
| ours_adapt25c1hyb_cloud_neutral_832 | neutral_832 | 84 | 832x480 | 2 | 6.7 | - | 24.0 | - | 62.1 (dev) | <u>7.70</u> | 39.9 | <u>74.21</u> | <u>68.7</u> | <u>3.92</u> | <u>0.690</u> | 0.18 (12) | 2.8 (12) | 3 |
| *ours_adapt25c1hyb_cloud_neutral_832 excl. black (n=81)* | neutral_832 | 81 | 832x480 | 2 | 6.7 | - | 24.0 | - | 62.1 (dev) | 7.73 | 38.6 | 74.45 | 69.2 | 3.59 | 0.676 | 0.18 (12) | 2.8 (12) | 0 |
| ours_adapt25c1hyb_dev_neutral_832 | neutral_832 | 84 | 832x480 | 2 | 6.7 | - | 23.3 | - | 62.3 (dev) | 7.65 | **38.6** | **74.22** | 68.2 | 4.07 | 0.749 | 0.24 (12) | 3.9 (12) | 3 |
| *ours_adapt25c1hyb_dev_neutral_832 excl. black (n=81)* | neutral_832 | 81 | 832x480 | 2 | 6.7 | - | 23.3 | - | 62.3 (dev) | 7.68 | 37.3 | 74.47 | 68.8 | 3.74 | 0.740 | 0.24 (12) | 3.9 (12) | 0 |
| ours_rt2_neutral_832 | neutral_832 | 84 | 832x480 | 1 | 9.6 | 6.3 | 16.1 | 8.55 | 62.6 | 7.60 | 39.2 | 74.14 | 68.0 | 4.10 | 0.814 | <u>0.16 (11)</u> | <u>2.6 (11)</u> | 1 |
| *ours_rt2_neutral_832 excl. black (n=83)* | neutral_832 | 83 | 832x480 | 1 | 9.6 | 6.3 | 16.1 | 8.47 | 62.6 | 7.60 | 38.6 | 74.20 | 68.3 | 3.98 | 0.822 | 0.16 (11) | 2.6 (11) | 0 |
| ours_rt2c1hyb_cloud_neutral_832 | neutral_832 | 84 | 832x480 | 2 | 6.8 | - | 22.5 | - | 62.1 (dev) | - | - | - | - | - | - | - | - | 3 |
| *ours_rt2c1hyb_cloud_neutral_832 excl. black (n=81)* | neutral_832 | 81 | 832x480 | 2 | 6.8 | - | 22.5 | - | 62.1 (dev) | - | - | - | - | - | - | - | - | 0 |
| ours_rt2c1hyb_dev_neutral_832 | neutral_832 | 84 | 832x480 | 2 | 6.8 | - | 21.8 | - | 62.1 (dev) | - | - | - | - | - | - | - | - | 3 |
| *ours_rt2c1hyb_dev_neutral_832 excl. black (n=81)* | neutral_832 | 81 | 832x480 | 2 | 6.8 | - | 21.8 | - | 62.1 (dev) | - | - | - | - | - | - | - | - | 0 |
| ours_sf1p3b_cal_neutral_832 | neutral_832 | 84 | 832x480 | 1 | <u>14.6</u> | <u>11.2</u> | <u>11.2</u> | **6.78** | <u>28.7</u> | 6.61 | 46.5 | 74.00 | 63.8 | **3.91** | 1.375 | **0.13 (12)** | **2.1 (12)** | 1 |
| *ours_sf1p3b_cal_neutral_832 excl. black (n=83)* | neutral_832 | 83 | 832x480 | 1 | 14.7 | 11.2 | 11.2 | 6.78 | 28.7 | 6.59 | 47.1 | 73.91 | 63.7 | 3.92 | 1.390 | 0.13 (12) | 2.1 (12) | 0 |
| ours_sf1p3b_pub_neutral_832 | neutral_832 | 84 | 832x480 | 1 | **16.2** | **12.2** | **10.2** | **6.78** | <u>28.7</u> | - | - | - | - | - | - | - | - | 0 |
| ours_sf24_neutral_832 | neutral_832 | 84 | 832x480 | 1 | 5.1 | 4.6 | 33.1 | 9.44 | - | - | - | - | - | - | - | - | - | 1 |
| *ours_sf24_neutral_832 excl. black (n=83)* | neutral_832 | 83 | 832x480 | 1 | 5.1 | 4.6 | 33.2 | 9.45 | - | - | - | - | - | - | - | - | - | 0 |
| ours_sf24c1hyb_cloud_neutral_832 | neutral_832 | 84 | 832x480 | 2 | 5.5 | - | 29.4 | - | 62.1 (dev) | - | - | - | - | - | - | - | - | 2 |
| *ours_sf24c1hyb_cloud_neutral_832 excl. black (n=82)* | neutral_832 | 82 | 832x480 | 2 | 5.5 | - | 29.4 | - | 62.1 (dev) | - | - | - | - | - | - | - | - | 0 |
| ours_sf24c1hyb_dev_neutral_832 | neutral_832 | 84 | 832x480 | 2 | 5.6 | - | 28.7 | - | 62.2 (dev) | - | - | - | - | - | - | - | - | 2 |
| *ours_sf24c1hyb_dev_neutral_832 excl. black (n=82)* | neutral_832 | 82 | 832x480 | 2 | 5.6 | - | 28.7 | - | 62.2 (dev) | - | - | - | - | - | - | - | - | 0 |
| ours_student1p3b_neutral_832 | neutral_832 | 84 | 832x480 | 1 | 11.2 | 9.0 | 14.2 | <u>6.86</u> | - | - | - | - | - | - | - | - | - | 4 |
| *ours_student1p3b_neutral_832 excl. black (n=80)* | neutral_832 | 80 | 832x480 | 1 | 11.2 | 9.0 | 14.2 | 6.86 | - | - | - | - | - | - | - | - | - | 0 |
| ours_teacher_neutral_832 | neutral_832 | 84 | 832x480 | 1 | 5.3 | 4.1 | 29.4 | 8.75 | 62.6 | **7.73** | <u>38.9</u> | 74.02 | **69.2** | 4.11 | 0.757 | 0.30 (12) | 4.8 (12) | 2 |
| *ours_teacher_neutral_832 excl. black (n=82)* | neutral_832 | 82 | 832x480 | 1 | 5.3 | 4.1 | 29.4 | 8.69 | 62.6 | 7.74 | 38.0 | 74.15 | 69.5 | 3.98 | 0.765 | 0.30 (12) | 4.8 (12) | 0 |

- `astra` (84 of 84 clips): timing.json flags a shared-GPU measurement (speed columns are a lower bound there).
- `astra` (6 of 84), `ours_adapt25_neutral_832` (1 of 84), `ours_adapt25c1hyb_cloud_neutral_832` (3 of 84), `ours_adapt25c1hyb_dev_neutral_832` (3 of 84), `ours_rt2_neutral_832` (1 of 84), `ours_rt2c1hyb_cloud_neutral_832` (3 of 84), `ours_rt2c1hyb_dev_neutral_832` (3 of 84), `ours_sf1p3b_cal_neutral_832` (1 of 84), `ours_sf24_neutral_832` (1 of 84), `ours_sf24c1hyb_cloud_neutral_832` (2 of 84), `ours_sf24c1hyb_dev_neutral_832` (2 of 84), `ours_student1p3b_neutral_832` (4 of 84), `ours_teacher_neutral_832` (2 of 84): clips go black (see the 'excl. black' rows).
- `ours_adapt25c1hyb_cloud_neutral_832`, `ours_adapt25c1hyb_dev_neutral_832`, `ours_rt2c1hyb_cloud_neutral_832`, `ours_rt2c1hyb_dev_neutral_832`, `ours_sf24c1hyb_cloud_neutral_832`, `ours_sf24c1hyb_dev_neutral_832`: 2 GPUs (device + cloud): VRAM = device GPU peak only (the cloud GPU's peak is timing.json `cloud_peak_alloc_GiB`); TTFF / fps e2e are not measured in the two-process mode.
- `ours_sf1p3b_cal_neutral_832` (84 of 84 clips), `ours_sf1p3b_pub_neutral_832` (84 of 84 clips), `ours_sf24_neutral_832` (84 of 84 clips), `ours_student1p3b_neutral_832` (84 of 84 clips): fps gen / gen s are DiT-only, corrected for the side-stream decode overlap (audit 10/19).

### set `neutral_832_long` (n expected 4 per arm = 4 scenes x P6)

| model_arm | set | n | res | GPUs | fps gen ↑ | fps e2e ↑ | gen s ↓ | TTFF s ↓ | VRAM GiB ↓ | judge Q ↑ | judge drift ↓ | VBench Q ↑ | imaging ↑ | MUSIQ drift ↓ | RPE Rot ° ↓ | react s ↓ | react fr ↓ | black |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ours_sf24_neutral_832_long | neutral_832_long | 1 (partial) | 832x480 | 1 | **5.9** | **5.3** | **164.4** | **35.42** | - | - | - | - | - | - | - | - | - | 0 |
| ours_teacher_neutral_832_long | neutral_832_long | 1 (partial) | 832x480 | 1 | <u>5.3</u> | <u>4.1</u> | <u>181.2</u> | <u>36.59</u> | 49.1 | - | - | - | - | - | - | - | - | 0 |

- `ours_sf24_neutral_832_long` (1 of 1 clips): fps gen / gen s are DiT-only, corrected for the side-stream decode overlap (audit 10/19).
- '(partial)' rows: n below the 4 clips of tasks/neutral_832_long.json (generation or scoring still running).

### set `neutral_832_nocap` (n expected 84 per arm)

| model_arm | set | n | res | GPUs | fps gen ↑ | fps e2e ↑ | gen s ↓ | TTFF s ↓ | VRAM GiB ↓ | judge Q ↑ | judge drift ↓ | VBench Q ↑ | imaging ↑ | MUSIQ drift ↓ | RPE Rot ° ↓ | react s ↓ | react fr ↓ | black |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| astra_nocap (Astra on neutral scenes, no caption) | neutral_832_nocap | 84 | 832x480 | 1 | 1.3 | 1.1 | 173.1 | 25.80 | **14.9** | 4.58 | 89.2 | 68.74 | 57.6 | 6.81 | 2.115 | 1.10 (1) | 22.0 (1) | 7 |
| *astra_nocap (Astra on neutral scenes, no caption) excl. black (n=77)* | neutral_832_nocap | 77 | 832x480 | 1 | 1.3 | 1.1 | 173.2 | 25.81 | 14.9 | 4.57 | 88.6 | 69.15 | 59.2 | 6.52 | 2.024 | 1.10 (1) | 22.0 (1) | 0 |
| ours_adapt25_neutral_832_nocap | neutral_832_nocap | 84 | 832x480 | 1 | 9.5 | 6.2 | 17.6 | 8.02 | 62.8 | 7.54 | 35.9 | 75.33 | <u>69.2</u> | <u>4.23</u> | 1.083 | 0.13 (12) | 2.1 (12) | 4 |
| *ours_adapt25_neutral_832_nocap excl. black (n=80)* | neutral_832_nocap | 80 | 832x480 | 1 | 9.5 | 6.2 | 17.6 | 8.02 | 62.8 | 7.55 | 33.9 | 75.65 | 69.9 | 3.90 | 1.070 | 0.13 (12) | 2.1 (12) | 0 |
| ours_adapt25c1hyb_cloud_neutral_832_nocap | neutral_832_nocap | 84 | 832x480 | 2 | 6.7 | - | 24.0 | - | 62.1 (dev) | <u>7.69</u> | <u>32.1</u> | <u>75.43</u> | 68.5 | 4.37 | **0.762** | 0.15 (12) | 2.4 (12) | 9 |
| *ours_adapt25c1hyb_cloud_neutral_832_nocap excl. black (n=75)* | neutral_832_nocap | 75 | 832x480 | 2 | 6.7 | - | 24.0 | - | 62.1 (dev) | 7.81 | 28.8 | 75.98 | 70.2 | 3.43 | 0.719 | 0.15 (12) | 2.4 (12) | 0 |
| ours_adapt25c1hyb_dev_neutral_832_nocap | neutral_832_nocap | 84 | 832x480 | 2 | 6.7 | - | 23.2 | - | 62.3 (dev) | **7.70** | 33.3 | **75.48** | 68.3 | 4.40 | <u>0.775</u> | 0.14 (12) | 2.2 (12) | 9 |
| *ours_adapt25c1hyb_dev_neutral_832_nocap excl. black (n=75)* | neutral_832_nocap | 75 | 832x480 | 2 | 6.7 | - | 23.3 | - | 62.3 (dev) | 7.83 | 31.1 | 76.04 | 70.0 | 3.47 | 0.734 | 0.14 (12) | 2.2 (12) | 0 |
| ours_rt2_neutral_832_nocap | neutral_832_nocap | 84 | 832x480 | 1 | 9.6 | 6.3 | 16.1 | **7.69** | 62.6 | 7.57 | **31.6** | 74.89 | 67.0 | 4.87 | 0.991 | **0.10 (12)** | **1.6 (12)** | 3 |
| *ours_rt2_neutral_832_nocap excl. black (n=81)* | neutral_832_nocap | 81 | 832x480 | 1 | 9.6 | 6.3 | 16.1 | 7.68 | 62.6 | 7.59 | 30.7 | 75.15 | 67.9 | 4.46 | 0.993 | 0.10 (12) | 1.6 (12) | 0 |
| ours_sf1p3b_cal_neutral_832_nocap | neutral_832_nocap | 84 | 832x480 | 1 | <u>13.0</u> | <u>9.9</u> | <u>14.1</u> | 8.69 | <u>28.7</u> | - | - | - | - | - | - | - | - | 0 |
| ours_sf1p3b_pub_neutral_832_nocap | neutral_832_nocap | 84 | 832x480 | 1 | **14.2** | **10.6** | **13.0** | 8.70 | <u>28.7</u> | - | - | - | - | - | - | - | - | 0 |
| ours_sf24_neutral_832_nocap | neutral_832_nocap | 84 | 832x480 | 1 | 6.6 | 5.8 | 25.0 | 8.00 | - | - | - | - | - | - | - | - | - | 7 |
| *ours_sf24_neutral_832_nocap excl. black (n=77)* | neutral_832_nocap | 77 | 832x480 | 1 | 6.6 | 5.9 | 24.9 | 8.00 | - | - | - | - | - | - | - | - | - | 0 |
| ours_student1p3b_neutral_832_nocap | neutral_832_nocap | 84 | 832x480 | 1 | 10.2 | 8.2 | 16.2 | <u>7.70</u> | - | - | - | - | - | - | - | - | - | 5 |
| *ours_student1p3b_neutral_832_nocap excl. black (n=79)* | neutral_832_nocap | 79 | 832x480 | 1 | 10.1 | 8.2 | 16.4 | 7.75 | - | - | - | - | - | - | - | - | - | 0 |
| ours_teacher_neutral_832_nocap | neutral_832_nocap | 84 | 832x480 | 1 | 5.3 | 4.1 | 29.4 | 8.34 | 62.6 | 7.56 | 35.9 | 75.14 | **69.8** | **4.16** | 1.104 | <u>0.12 (12)</u> | <u>2.0 (12)</u> | 3 |
| *ours_teacher_neutral_832_nocap excl. black (n=81)* | neutral_832_nocap | 81 | 832x480 | 1 | 5.3 | 4.1 | 29.4 | 8.34 | 62.6 | 7.58 | 34.3 | 75.41 | 70.7 | 3.74 | 1.114 | 0.12 (12) | 2.0 (12) | 0 |

- `astra_nocap` = Astra run on the 12 neutral_832 scenes with an empty prompt (NOT the Astra scenes without captions); it belongs to this group only and is ranked here.
- `astra_nocap` (84 of 84 clips): timing.json flags a shared-GPU measurement (speed columns are a lower bound there).
- `astra_nocap` (7 of 84), `ours_adapt25_neutral_832_nocap` (4 of 84), `ours_adapt25c1hyb_cloud_neutral_832_nocap` (9 of 84), `ours_adapt25c1hyb_dev_neutral_832_nocap` (9 of 84), `ours_rt2_neutral_832_nocap` (3 of 84), `ours_sf24_neutral_832_nocap` (7 of 84), `ours_student1p3b_neutral_832_nocap` (5 of 84), `ours_teacher_neutral_832_nocap` (3 of 84): clips go black (see the 'excl. black' rows).
- `ours_adapt25c1hyb_cloud_neutral_832_nocap`, `ours_adapt25c1hyb_dev_neutral_832_nocap`: 2 GPUs (device + cloud): VRAM = device GPU peak only (the cloud GPU's peak is timing.json `cloud_peak_alloc_GiB`); TTFF / fps e2e are not measured in the two-process mode.
- `ours_sf1p3b_cal_neutral_832_nocap` (84 of 84 clips), `ours_sf1p3b_pub_neutral_832_nocap` (84 of 84 clips), `ours_sf24_neutral_832_nocap` (84 of 84 clips), `ours_student1p3b_neutral_832_nocap` (84 of 84 clips): fps gen / gen s are DiT-only, corrected for the side-stream decode overlap (audit 10/19).

### set `neutral_abot` (n expected 60 per arm = 5 programs P1-P5)

| model_arm | set | n | res | GPUs | fps gen ↑ | fps e2e ↑ | gen s ↓ | TTFF s ↓ | VRAM GiB ↓ | judge Q ↑ | judge drift ↓ | VBench Q ↑ | imaging ↑ | MUSIQ drift ↓ | RPE Rot ° ↓ | react s ↓ | react fr ↓ | black |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| abot | neutral_abot | 60 | 1280x704 | 1 | **15.4** | **12.7** | **7.9** | **1.06** | **32.1** | 7.07 | 43.5 | 73.81 | 62.6 | 6.16 | **0.631** | 0.42 (12) | 5.1 (12) | 3 |
| *abot excl. black (n=57)* | neutral_abot | 57 | 1280x704 | 1 | 15.4 | 12.7 | 7.9 | 1.06 | 32.1 | 7.07 | 41.7 | 74.30 | 64.0 | 5.60 | 0.634 | 0.42 (12) | 5.1 (12) | 0 |
| ours_adapt25_neutral_abot | neutral_abot | 60 | 1280x704 | 1 | 3.6 | 2.5 | 47.4 | 17.23 | 85.1 | 7.62 | 32.2 | 75.70 | 70.6 | 4.02 | 0.993 | <u>0.13 (11)</u> | 2.1 (11) | 2 |
| *ours_adapt25_neutral_abot (P2p+P4p only, n=24)* | neutral_abot | 24 | 1280x704 | 1 | 3.6 | 2.5 | 47.0 | 17.28 | 85.1 | 7.75 | 36.0 | 75.09 | 71.1 | 3.93 | 1.438 | - | - | 0 |
| *ours_adapt25_neutral_abot excl. black (n=58)* | neutral_abot | 58 | 1280x704 | 1 | 3.6 | 2.5 | 47.4 | 17.22 | 85.1 | 7.64 | 32.7 | 75.90 | 71.1 | 3.82 | 0.997 | 0.13 (11) | 2.1 (11) | 0 |
| ours_adapt25c1hyb_cloud_neutral_abot | neutral_abot | 60 | 1280x704 | 2 | 2.4 | - | 72.9 | - | <u>84.3 (dev)</u> | <u>7.87</u> | 26.3 | **76.73** | <u>71.2</u> | <u>3.44</u> | 0.727 | **0.12 (12)** | **1.9 (12)** | 3 |
| *ours_adapt25c1hyb_cloud_neutral_abot (P2p+P4p only, n=24)* | neutral_abot | 24 | 1280x704 | 2 | 2.5 | - | 70.1 | - | 84.3 (dev) | 8.00 | 27.5 | 76.50 | 72.1 | 3.25 | 0.925 | - | - | 1 |
| *ours_adapt25c1hyb_cloud_neutral_abot excl. black (n=57)* | neutral_abot | 57 | 1280x704 | 2 | 2.4 | - | 73.4 | - | 84.3 (dev) | 7.93 | 26.1 | 77.09 | 72.2 | 3.09 | 0.666 | 0.12 (12) | 1.9 (12) | 0 |
| ours_adapt25c1hyb_dev_neutral_abot | neutral_abot | 60 | 1280x704 | 2 | 2.4 | - | 71.0 | - | 84.6 (dev) | **7.88** | **23.4** | <u>76.66</u> | 70.9 | 3.57 | <u>0.711</u> | **0.12 (12)** | <u>2.0 (12)</u> | 3 |
| *ours_adapt25c1hyb_dev_neutral_abot (P2p+P4p only, n=24)* | neutral_abot | 24 | 1280x704 | 2 | 2.5 | - | 68.2 | - | 84.6 (dev) | 7.88 | 25.2 | 76.54 | 72.0 | 3.26 | 0.928 | - | - | 1 |
| *ours_adapt25c1hyb_dev_neutral_abot excl. black (n=57)* | neutral_abot | 57 | 1280x704 | 2 | 2.4 | - | 71.5 | - | 84.6 (dev) | 7.95 | 23.3 | 77.01 | 71.9 | 3.22 | 0.649 | 0.12 (12) | 2.0 (12) | 0 |
| ours_rt2_neutral_abot | neutral_abot | 60 | 1280x704 | 1 | 3.7 | 2.5 | 43.5 | <u>16.30</u> | 84.8 | 7.72 | 27.8 | 76.08 | 70.5 | 3.72 | 0.923 | 0.20 (12) | 3.3 (12) | 3 |
| *ours_rt2_neutral_abot (P2p+P4p only, n=24)* | neutral_abot | 24 | 1280x704 | 1 | 3.7 | 2.5 | 43.4 | 16.33 | 84.8 | 7.50 | 34.8 | 74.78 | 70.4 | 4.16 | 1.333 | - | - | 0 |
| *ours_rt2_neutral_abot excl. black (n=57)* | neutral_abot | 57 | 1280x704 | 1 | 3.7 | 2.5 | 43.5 | 16.30 | 84.8 | 7.70 | 27.5 | 76.14 | 70.8 | 3.62 | 0.945 | 0.20 (12) | 3.3 (12) | 0 |
| ours_sf24_neutral_abot | neutral_abot | 60 | 1280x704 | 1 | 2.4 | 2.2 | 70.9 | 25.02 | - | - | - | - | - | - | - | - | - | 2 |
| *ours_sf24_neutral_abot (P2p+P4p only, n=24)* | neutral_abot | 24 | 1280x704 | 1 | 2.4 | 2.1 | 70.5 | 24.40 | - | - | - | - | - | - | - | - | - | 0 |
| *ours_sf24_neutral_abot excl. black (n=58)* | neutral_abot | 58 | 1280x704 | 1 | 2.4 | 2.2 | 71.0 | 25.05 | - | - | - | - | - | - | - | - | - | 0 |
| ours_student1p3b_neutral_abot | neutral_abot | 60 | 1280x704 | 1 | <u>4.3</u> | <u>3.6</u> | <u>39.5</u> | 22.51 | - | - | - | - | - | - | - | - | - | 1 |
| *ours_student1p3b_neutral_abot (P2p+P4p only, n=24)* | neutral_abot | 24 | 1280x704 | 1 | 4.4 | 3.6 | 38.0 | 21.77 | - | - | - | - | - | - | - | - | - | 0 |
| *ours_student1p3b_neutral_abot excl. black (n=59)* | neutral_abot | 59 | 1280x704 | 1 | 4.3 | 3.5 | 39.5 | 22.53 | - | - | - | - | - | - | - | - | - | 0 |
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
| ours_adapt25_neutral_mg2 | neutral_mg2 | 84 | 640x352 | 1 | 19.3 | 12.1 | 8.9 | 4.73 | 43.9 | 7.32 | 38.5 | 74.78 | 66.9 | 4.31 | 1.095 | **0.11 (11)** | **1.8 (11)** | 2 |
| *ours_adapt25_neutral_mg2 excl. black (n=82)* | neutral_mg2 | 82 | 640x352 | 1 | 19.3 | 12.1 | 8.9 | 4.74 | 43.9 | 7.33 | 38.1 | 74.89 | 67.2 | 4.19 | 1.086 | 0.11 (11) | 1.8 (11) | 0 |
| ours_adapt25c1hyb_cloud_neutral_mg2 | neutral_mg2 | 84 | 640x352 | 2 | 12.3 | - | 14.1 | - | <u>43.3 (dev)</u> | **7.61** | **30.1** | <u>75.61</u> | <u>68.3</u> | <u>3.99</u> | <u>0.709</u> | 0.19 (12) | 3.0 (12) | 0 |
| ours_adapt25c1hyb_dev_neutral_mg2 | neutral_mg2 | 84 | 640x352 | 2 | 12.4 | - | 13.8 | - | 43.4 (dev) | **7.61** | <u>30.8</u> | **77.00 (49)** | **69.7 (49)** | **3.17 (49)** | - | - | - | 0 |
| ours_rt2_neutral_mg2 | neutral_mg2 | 84 | 640x352 | 1 | 19.4 | 12.2 | 8.1 | 4.55 | 43.7 | 7.27 | 38.3 | 74.27 | 64.9 | 4.83 | 1.204 | 0.17 (12) | 2.7 (12) | 1 |
| *ours_rt2_neutral_mg2 excl. black (n=83)* | neutral_mg2 | 83 | 640x352 | 1 | 19.4 | 12.2 | 8.1 | 4.54 | 43.7 | 7.33 | 38.6 | 74.28 | 65.2 | 4.85 | 1.217 | 0.17 (12) | 2.7 (12) | 0 |
| ours_sf24_neutral_mg2 | neutral_mg2 | 84 | 640x352 | 1 | 10.6 | 9.3 | 17.3 | 6.30 | - | - | - | - | - | - | - | - | - | 1 |
| *ours_sf24_neutral_mg2 excl. black (n=83)* | neutral_mg2 | 83 | 640x352 | 1 | 10.7 | 9.3 | 17.1 | 6.26 | - | - | - | - | - | - | - | - | - | 0 |
| ours_student1p3b_neutral_mg2 | neutral_mg2 | 84 | 640x352 | 1 | <u>21.9</u> | <u>17.3</u> | <u>7.4</u> | <u>4.05</u> | - | - | - | - | - | - | - | - | - | 2 |
| *ours_student1p3b_neutral_mg2 excl. black (n=82)* | neutral_mg2 | 82 | 640x352 | 1 | 21.9 | 17.3 | 7.4 | 4.05 | - | - | - | - | - | - | - | - | - | 0 |
| ours_teacher_neutral_mg2 | neutral_mg2 | 84 | 640x352 | 1 | 10.6 | 8.0 | 14.9 | 4.92 | 43.7 | <u>7.40</u> | 37.0 | 74.84 | 67.8 | 4.27 | 1.129 | <u>0.15 (12)</u> | <u>2.4 (12)</u> | 2 |
| *ours_teacher_neutral_mg2 excl. black (n=82)* | neutral_mg2 | 82 | 640x352 | 1 | 10.6 | 8.0 | 14.9 | 4.91 | 43.7 | 7.44 | 36.1 | 74.89 | 68.2 | 4.26 | 1.124 | 0.15 (12) | 2.4 (12) | 0 |

- `mg2` (10 of 84), `ours_adapt25_neutral_mg2` (2 of 84), `ours_rt2_neutral_mg2` (1 of 84), `ours_sf24_neutral_mg2` (1 of 84), `ours_student1p3b_neutral_mg2` (2 of 84), `ours_teacher_neutral_mg2` (2 of 84): clips go black (see the 'excl. black' rows).
- `ours_adapt25c1hyb_cloud_neutral_mg2`, `ours_adapt25c1hyb_dev_neutral_mg2`: 2 GPUs (device + cloud): VRAM = device GPU peak only (the cloud GPU's peak is timing.json `cloud_peak_alloc_GiB`); TTFF / fps e2e are not measured in the two-process mode.
- `ours_sf24_neutral_mg2` (84 of 84 clips), `ours_student1p3b_neutral_mg2` (84 of 84 clips): fps gen / gen s are DiT-only, corrected for the side-stream decode overlap (audit 10/19).

### set `neutral_wp` (n expected 84 per arm)

| model_arm | set | n | res | GPUs | fps gen ↑ | fps e2e ↑ | gen s ↓ | TTFF s ↓ | VRAM GiB ↓ | judge Q ↑ | judge drift ↓ | VBench Q ↑ | imaging ↑ | MUSIQ drift ↓ | RPE Rot ° ↓ | react s ↓ | react fr ↓ | black |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ours_adapt25_neutral_wp | neutral_wp | 84 | 1024x512 | 1 | 6.3 | 4.3 | 26.2 | 10.44 | 76.4 | 7.76 | 33.2 | 75.02 | 68.4 | 4.12 | 1.005 | 0.22 (12) | 3.5 (12) | 8 |
| *ours_adapt25_neutral_wp excl. black (n=76)* | neutral_wp | 76 | 1024x512 | 1 | 6.3 | 4.3 | 26.2 | 10.44 | 76.4 | 7.86 | 30.0 | 75.52 | 69.7 | 3.48 | 0.923 | 0.22 (12) | 3.5 (12) | 0 |
| ours_adapt25c1hyb_cloud_neutral_wp | neutral_wp | 84 | 1024x512 | 2 | 4.1 | - | 44.3 | - | <u>75.7 (dev)</u> | <u>7.81</u> | <u>27.4</u> | <u>75.92</u> | 69.1 | <u>3.98</u> | <u>0.763</u> | 0.12 (12) | <u>1.9 (12)</u> | 12 |
| *ours_adapt25c1hyb_cloud_neutral_wp excl. black (n=72)* | neutral_wp | 72 | 1024x512 | 2 | 4.0 | - | 45.3 | - | 75.7 (dev) | 7.97 | 21.2 | 77.01 | 72.0 | 2.79 | 0.677 | 0.12 (12) | 1.9 (12) | 0 |
| ours_adapt25c1hyb_dev_neutral_wp | neutral_wp | 84 | 1024x512 | 2 | 4.1 | - | 43.2 | - | 75.9 (dev) | - | - | - | - | - | - | - | - | 12 |
| *ours_adapt25c1hyb_dev_neutral_wp excl. black (n=72)* | neutral_wp | 72 | 1024x512 | 2 | 4.0 | - | 44.2 | - | 75.9 (dev) | - | - | - | - | - | - | - | - | 0 |
| ours_rt2_neutral_wp | neutral_wp | 84 | 1024x512 | 1 | 6.4 | 4.4 | <u>23.9</u> | <u>9.91</u> | 76.2 | 7.69 | 33.4 | 75.41 | 68.3 | 4.68 | 0.940 | <u>0.09 (12)</u> | **1.4 (12)** | 5 |
| *ours_rt2_neutral_wp excl. black (n=79)* | neutral_wp | 79 | 1024x512 | 1 | 6.4 | 4.4 | 23.9 | 9.90 | 76.2 | 7.72 | 31.3 | 75.73 | 69.1 | 4.35 | 0.921 | 0.09 (12) | 1.4 (12) | 0 |
| ours_sf24_neutral_wp | neutral_wp | 84 | 1024x512 | 1 | 4.3 | 3.8 | 39.7 | 11.34 | - | **8.06** | **20.4** | **76.79** | **69.6** | **3.62** | **0.672** | 0.12 (12) | 2.0 (12) | 3 |
| *ours_sf24_neutral_wp excl. black (n=81)* | neutral_wp | 81 | 1024x512 | 1 | 4.3 | 3.8 | 39.6 | 11.36 | - | 8.09 | 19.0 | 77.05 | 70.0 | 3.45 | 0.677 | 0.12 (12) | 2.0 (12) | 0 |
| ours_student1p3b_neutral_wp | neutral_wp | 84 | 1024x512 | 1 | <u>7.2</u> | <u>5.8</u> | <u>23.9</u> | 10.47 | - | - | - | - | - | - | - | - | - | 9 |
| *ours_student1p3b_neutral_wp excl. black (n=75)* | neutral_wp | 75 | 1024x512 | 1 | 7.1 | 5.8 | 24.4 | 10.67 | - | - | - | - | - | - | - | - | - | 0 |
| ours_teacher_neutral_wp | neutral_wp | 84 | 1024x512 | 1 | 3.5 | 2.8 | 43.6 | 10.90 | 76.2 | 7.76 | 28.8 | 75.83 | <u>69.4</u> | 4.22 | 1.037 | 0.21 (12) | 3.3 (12) | 8 |
| *ours_teacher_neutral_wp excl. black (n=76)* | neutral_wp | 76 | 1024x512 | 1 | 3.5 | 2.8 | 43.6 | 10.90 | 76.2 | 7.89 | 26.6 | 76.47 | 71.3 | 3.31 | 0.973 | 0.21 (12) | 3.3 (12) | 0 |
| waypoint | neutral_wp | 84 | 1024x512 | 1 | **40.4** | **37.8** | **15.4** | **0.50** | **6.6** | 5.81 | 69.0 | 71.36 | **69.6** | 4.37 | 0.914 | **0.05 (12)** | 2.9 (12) | 0 |

- `ours_adapt25_neutral_wp` (8 of 84), `ours_adapt25c1hyb_cloud_neutral_wp` (12 of 84), `ours_adapt25c1hyb_dev_neutral_wp` (12 of 84), `ours_rt2_neutral_wp` (5 of 84), `ours_sf24_neutral_wp` (3 of 84), `ours_student1p3b_neutral_wp` (9 of 84), `ours_teacher_neutral_wp` (8 of 84): clips go black (see the 'excl. black' rows).
- `ours_adapt25c1hyb_cloud_neutral_wp`, `ours_adapt25c1hyb_dev_neutral_wp`: 2 GPUs (device + cloud): VRAM = device GPU peak only (the cloud GPU's peak is timing.json `cloud_peak_alloc_GiB`); TTFF / fps e2e are not measured in the two-process mode.
- `ours_sf24_neutral_wp` (84 of 84 clips), `ours_student1p3b_neutral_wp` (84 of 84 clips): fps gen / gen s are DiT-only, corrected for the side-stream decode overlap (audit 10/19).
- `waypoint` (84 of 84 clips): speed columns (fps gen / fps e2e / TTFF / gen s) are a lower bound, shared-GPU measurement (timing.json `timing_caveat`); the `_idle` exclusive-GPU re-measure is pending.

### set `wp` (n expected 84 per arm)

| model_arm | set | n | res | GPUs | fps gen ↑ | fps e2e ↑ | gen s ↓ | TTFF s ↓ | VRAM GiB ↓ | judge Q ↑ | judge drift ↓ | VBench Q ↑ | imaging ↑ | MUSIQ drift ↓ | RPE Rot ° ↓ | react s ↓ | react fr ↓ | black |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ours_adapt25_wp | wp | 84 | 1024x512 | 1 | 6.4 | 4.4 | 25.9 | 10.40 | 76.4 | 7.76 | 10.4 | 80.52 | 69.6 | 3.18 | 0.752 | 0.09 (12) | 1.5 (12) | 2 |
| *ours_adapt25_wp excl. black (n=82)* | wp | 82 | 1024x512 | 1 | 6.4 | 4.4 | 25.8 | 10.40 | 76.4 | 7.76 | 10.0 | 80.63 | 69.7 | 2.97 | 0.738 | 0.09 (12) | 1.5 (12) | 0 |
| ours_adapt25c1hyb_cloud_wp | wp | 84 | 1024x512 | 2 | 4.1 | - | 42.8 | - | <u>75.7 (dev)</u> | <u>7.93</u> | <u>9.0</u> | <u>81.00</u> | 70.1 | 3.02 | <u>0.571</u> | **0.07 (12)** | **1.2 (12)** | 2 |
| *ours_adapt25c1hyb_cloud_wp excl. black (n=82)* | wp | 82 | 1024x512 | 2 | 4.2 | - | 42.4 | - | 75.7 (dev) | 7.94 | 8.7 | 81.09 | 70.3 | 2.78 | 0.564 | 0.07 (12) | 1.2 (12) | 0 |
| ours_adapt25c1hyb_dev_wp | wp | 84 | 1024x512 | 2 | 4.2 | - | 41.7 | - | 75.9 (dev) | - | - | - | - | - | - | - | - | 2 |
| *ours_adapt25c1hyb_dev_wp excl. black (n=82)* | wp | 82 | 1024x512 | 2 | 4.2 | - | 41.3 | - | 75.9 (dev) | - | - | - | - | - | - | - | - | 0 |
| ours_rt2_wp | wp | 84 | 1024x512 | 1 | 6.4 | 4.4 | <u>23.9</u> | <u>9.91</u> | 76.2 | 7.76 | 12.9 | 80.37 | 69.1 | 3.03 | 0.840 | 0.09 (12) | 1.5 (12) | 2 |
| *ours_rt2_wp excl. black (n=82)* | wp | 82 | 1024x512 | 1 | 6.4 | 4.4 | 23.9 | 9.92 | 76.2 | 7.78 | 12.7 | 80.41 | 69.3 | 2.96 | 0.840 | 0.09 (12) | 1.5 (12) | 0 |
| ours_rt2c1hyb_cloud_wp | wp | 84 | 1024x512 | 2 | 4.5 | - | 33.4 | - | <u>75.7 (dev)</u> | - | - | - | - | - | - | - | - | 1 |
| *ours_rt2c1hyb_cloud_wp excl. black (n=83)* | wp | 83 | 1024x512 | 2 | 4.5 | - | 33.4 | - | 75.7 (dev) | - | - | - | - | - | - | - | - | 0 |
| ours_rt2c1hyb_dev_wp | wp | 84 | 1024x512 | 2 | 4.5 | - | 32.3 | - | <u>75.7 (dev)</u> | - | - | - | - | - | - | - | - | 1 |
| *ours_rt2c1hyb_dev_wp excl. black (n=83)* | wp | 83 | 1024x512 | 2 | 4.5 | - | 32.3 | - | 75.7 (dev) | - | - | - | - | - | - | - | - | 0 |
| ours_sf24_wp | wp | 84 | 1024x512 | 1 | 4.6 | 4.1 | 35.5 | 10.56 | - | **8.01** | **8.5** | **81.31** | 69.6 | <u>2.93</u> | **0.530** | <u>0.08 (12)</u> | <u>1.3 (12)</u> | 0 |
| ours_sf24c1hyb_cloud_wp | wp | 84 | 1024x512 | 2 | 4.2 | - | 38.4 | - | <u>75.7 (dev)</u> | - | - | - | - | - | - | - | - | 0 |
| ours_sf24c1hyb_dev_wp | wp | 84 | 1024x512 | 2 | 4.2 | - | 37.3 | - | <u>75.7 (dev)</u> | - | - | - | - | - | - | - | - | 0 |
| ours_student1p3b_wp | wp | 84 | 1024x512 | 1 | <u>7.1</u> | <u>5.8</u> | 24.2 | 10.60 | - | - | - | - | - | - | - | - | - | 2 |
| *ours_student1p3b_wp excl. black (n=82)* | wp | 82 | 1024x512 | 1 | 7.2 | 5.8 | 24.0 | 10.53 | - | - | - | - | - | - | - | - | - | 0 |
| ours_teacher_wp | wp | 84 | 1024x512 | 1 | 3.5 | 2.8 | 43.6 | 10.85 | 76.2 | 7.68 | 11.5 | 80.49 | <u>70.3</u> | 2.97 | 0.791 | 0.17 (12) | 2.7 (12) | 2 |
| *ours_teacher_wp excl. black (n=82)* | wp | 82 | 1024x512 | 1 | 3.5 | 2.8 | 43.6 | 10.85 | 76.2 | 7.70 | 11.0 | 80.59 | 70.5 | 2.86 | 0.781 | 0.17 (12) | 2.7 (12) | 0 |
| waypoint | wp | 84 | 1024x512 | 1 | **39.3** | **36.7** | **15.8** | **0.52** | **6.6** | 6.94 | 22.0 | 76.98 | **71.3** | **2.30** | 0.654 | **0.07 (11)** | 4.1 (11) | 0 |

- `ours_adapt25_wp` (2 of 84), `ours_adapt25c1hyb_cloud_wp` (2 of 84), `ours_adapt25c1hyb_dev_wp` (2 of 84), `ours_rt2_wp` (2 of 84), `ours_rt2c1hyb_cloud_wp` (1 of 84), `ours_rt2c1hyb_dev_wp` (1 of 84), `ours_student1p3b_wp` (2 of 84), `ours_teacher_wp` (2 of 84): clips go black (see the 'excl. black' rows).
- `ours_adapt25c1hyb_cloud_wp`, `ours_adapt25c1hyb_dev_wp`, `ours_rt2c1hyb_cloud_wp`, `ours_rt2c1hyb_dev_wp`, `ours_sf24c1hyb_cloud_wp`, `ours_sf24c1hyb_dev_wp`: 2 GPUs (device + cloud): VRAM = device GPU peak only (the cloud GPU's peak is timing.json `cloud_peak_alloc_GiB`); TTFF / fps e2e are not measured in the two-process mode.
- `ours_sf24_wp` (84 of 84 clips), `ours_student1p3b_wp` (84 of 84 clips): fps gen / gen s are DiT-only, corrected for the side-stream decode overlap (audit 10/19).
- `waypoint` (84 of 84 clips): speed columns (fps gen / fps e2e / TTFF / gen s) are a lower bound, shared-GPU measurement (timing.json `timing_caveat`); the `_idle` exclusive-GPU re-measure is pending.

### set `wp_long` (n expected 4 per arm = 4 scenes x P6)

| model_arm | set | n | res | GPUs | fps gen ↑ | fps e2e ↑ | gen s ↓ | TTFF s ↓ | VRAM GiB ↓ | judge Q ↑ | judge drift ↓ | VBench Q ↑ | imaging ↑ | MUSIQ drift ↓ | RPE Rot ° ↓ | react s ↓ | react fr ↓ | black |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| waypoint | wp_long | 1 (partial) | 1024x512 | 1 | 38.8 | 36.4 | 93.2 | 0.56 | 6.6 | 4.00 | 100.0 | 70.40 | 63.9 | 5.88 | 0.769 | - | - | 0 |

- `waypoint` (1 of 1 clips): speed columns (fps gen / fps e2e / TTFF / gen s) are a lower bound, shared-GPU measurement (timing.json `timing_caveat`); the `_idle` exclusive-GPU re-measure is pending.
- '(partial)' rows: n below the 4 clips of tasks/wp_long.json (generation or scoring still running).


## 2. Paired delta vs `ours_teacher_<set><suffix>` on the same (set, scene, program, suffix)

delta = mean(arm - ref) over paired clips; b/e/w = clips where the arm is better / equal / worse (direction-aware; equal within tolerance: judge Q exact, judge drift exact, VBench Q ±0.5, imaging ±0.5, MUSIQ drift ±0.1, RPE Rot ° ±0.1, react s ±0.05); fps e2e x = mean per-clip ratio arm / ref. [10/19 audit] abot / neutral_abot pairs use programs P1-P5 only (ABot's P2p/P4p duplicate P2/P4).

| model_arm | set | paired n | fps e2e x | Δ judge Q ↑ (b/e/w) | Δ judge drift ↓ (b/e/w) | Δ VBench Q ↑ (b/e/w) | Δ imaging ↑ (b/e/w) | Δ MUSIQ drift ↓ (b/e/w) | Δ RPE Rot ° ↓ (b/e/w) | Δ react s ↓ (b/e/w) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| abot | abot | 60 | 8.11x | -0.80 (3/20/37) | +12.8 (7/31/22) | -2.35 (13/5/42) | -3.1 (11/6/43) | +1.35 (14/2/44) | -0.184 (31/21/8) | +0.30 (1/0/9) n=10 |
| abot | abot_long | 1 | 7.63x | - | - | - | - | - | - | - |
| abot | neutral_abot | 60 | 7.88x | -0.77 (5/21/34) | +18.1 (10/21/29) | -2.56 (16/4/40) | -9.0 (10/4/46) | +2.83 (19/3/38) | -0.437 (39/14/7) | +0.16 (2/0/9) n=11 |
| abot_t5offload | abot | 15 | 7.77x | -0.80 (0/7/8) | +12.7 (1/9/5) | -1.77 (3/2/10) | -3.3 (1/1/13) | +1.27 (3/1/11) | -0.004 (5/5/5) | +0.39 (0/0/3) n=3 |
| astra | astra | 21 | 0.27x | -1.33 (0/6/15) | +37.9 (0/4/17) | -1.42 (7/2/12) | -3.2 (5/1/15) | +0.45 (9/0/12) | +0.429 (3/3/15) | - |
| astra | neutral_832 | 84 | 0.27x | -2.77 (0/6/78) | +38.3 (3/14/67) | -3.83 (16/10/58) | -13.9 (11/2/71) | +2.05 (30/0/54) | +0.901 (9/5/70) | +0.30 (1/1/3) n=5 |
| astra_nocap (Astra on neutral scenes, no caption) | neutral_832_nocap | 84 | 0.27x | -2.98 (0/5/79) | +53.3 (3/4/77) | -6.40 (11/4/69) | -12.2 (6/2/76) | +2.66 (20/1/63) | +1.011 (2/3/79) | +1.04 (0/0/1) n=1 |
| mg2 | mg2 | 84 | 2.63x | +0.02 (21/26/37) | +7.4 (26/25/33) | +1.48 (45/10/29) | -0.9 (35/4/45) | -0.62 (53/2/29) | -0.751 (62/18/4) | +0.10 (1/1/10) n=12 |
| mg2 | neutral_mg2 | 84 | 2.63x | -0.62 (6/41/37) | +11.7 (17/29/38) | -0.22 (39/6/39) | -2.5 (37/6/41) | +1.98 (34/3/47) | -0.679 (68/15/1) | +0.30 (0/1/11) n=12 |
| ours_adapt25_abot | abot | 60 | 1.62x | -0.08 (2/51/7) | -0.7 (8/46/6) | +0.10 (15/33/12) | +0.3 (16/31/13) | -0.16 (26/10/24) | +0.002 (9/44/7) | +0.27 (0/6/4) n=10 |
| ours_adapt25_astra | astra | 21 | 1.53x | +0.05 (4/13/4) | -0.5 (3/16/2) | +0.18 (4/12/5) | -1.0 (4/7/10) | +0.35 (7/4/10) | -0.041 (6/13/2) | -0.01 (1/1/1) n=3 |
| ours_adapt25_mg2 | mg2 | 84 | 1.51x | -0.30 (14/43/27) | +3.9 (17/48/19) | -0.03 (25/28/31) | -1.5 (22/9/53) | +0.46 (32/5/47) | -0.028 (29/34/21) | +0.04 (3/5/4) n=12 |
| ours_adapt25_neutral_832 | neutral_832 | 84 | 1.51x | -0.19 (7/55/22) | +2.0 (14/47/23) | +0.11 (28/27/29) | -0.7 (24/20/40) | -0.20 (42/8/34) | -0.076 (25/43/16) | -0.03 (4/5/3) n=12 |
| ours_adapt25_neutral_832_nocap | neutral_832_nocap | 84 | 1.53x | -0.02 (13/55/16) | +0.0 (11/57/16) | +0.19 (33/15/36) | -0.6 (26/19/39) | +0.08 (35/6/43) | -0.021 (31/36/17) | +0.01 (3/6/3) n=12 |
| ours_adapt25_neutral_abot | neutral_abot | 60 | 1.56x | -0.22 (6/37/17) | +6.8 (7/35/18) | -0.67 (19/10/31) | -1.0 (17/13/30) | +0.68 (22/6/32) | -0.075 (18/22/20) | -0.14 (6/3/2) n=11 |
| ours_adapt25_neutral_mg2 | neutral_mg2 | 84 | 1.51x | -0.08 (12/56/16) | +1.4 (18/44/22) | -0.06 (37/13/34) | -0.9 (25/17/42) | +0.04 (38/5/41) | -0.034 (28/35/21) | -0.04 (4/5/2) n=11 |
| ours_adapt25_neutral_wp | neutral_wp | 84 | 1.54x | +0.00 (14/54/16) | +4.5 (8/59/17) | -0.81 (18/20/46) | -1.0 (28/12/44) | -0.10 (33/8/43) | -0.032 (22/36/26) | +0.01 (3/5/4) n=12 |
| ours_adapt25_wp | wp | 84 | 1.56x | +0.08 (17/58/9) | -1.1 (15/60/9) | +0.02 (23/35/26) | -0.7 (19/20/45) | +0.21 (35/9/40) | -0.038 (25/46/13) | -0.08 (4/6/2) n=12 |
| ours_adapt25c1hyb_cloud_abot | abot | 60 | - | +0.02 (6/48/6) | -1.4 (9/45/6) | +0.71 (19/29/12) | +0.4 (20/20/20) | -0.12 (24/15/21) | -0.163 (35/24/1) | -0.04 (2/8/1) n=11 |
| ours_adapt25c1hyb_cloud_astra | astra | 21 | - | +0.33 (6/15/0) | -2.4 (4/16/1) | +0.53 (6/8/7) | -0.7 (5/7/9) | +0.03 (10/3/8) | -0.149 (12/9/0) | -0.01 (1/1/1) n=3 |
| ours_adapt25c1hyb_cloud_mg2 | mg2 | 84 | - | -0.10 (16/52/16) | -4.5 (24/48/12) | +0.90 (40/23/21) | -0.6 (26/11/47) | -0.34 (44/3/37) | -0.424 (58/23/3) | -0.04 (2/10/0) n=12 |
| ours_adapt25c1hyb_cloud_neutral_832 | neutral_832 | 84 | - | -0.02 (12/59/13) | +1.1 (15/55/14) | +0.19 (31/25/28) | -0.5 (36/12/36) | -0.19 (43/6/35) | -0.067 (34/31/19) | -0.13 (6/4/2) n=12 |
| ours_adapt25c1hyb_cloud_neutral_832_nocap | neutral_832_nocap | 84 | - | +0.13 (21/51/12) | -3.8 (17/54/13) | +0.29 (30/22/32) | -1.3 (25/15/44) | +0.21 (37/9/38) | -0.342 (52/22/10) | +0.03 (3/5/4) n=12 |
| ours_adapt25c1hyb_cloud_neutral_abot | neutral_abot | 60 | - | +0.03 (9/45/6) | +0.9 (13/34/13) | +0.36 (27/16/17) | -0.4 (19/20/21) | +0.10 (35/3/22) | -0.342 (39/17/4) | -0.16 (6/4/1) n=11 |
| ours_adapt25c1hyb_cloud_neutral_mg2 | neutral_mg2 | 84 | - | +0.20 (20/58/6) | -7.0 (26/46/12) | +0.77 (48/12/24) | +0.5 (38/13/33) | -0.28 (44/4/36) | -0.420 (59/21/4) | +0.03 (2/8/2) n=12 |
| ours_adapt25c1hyb_cloud_neutral_wp | neutral_wp | 84 | - | +0.05 (17/57/10) | -1.4 (22/52/10) | +0.09 (27/21/36) | -0.3 (27/21/36) | -0.24 (40/4/40) | -0.274 (53/25/6) | -0.09 (4/5/3) n=12 |
| ours_adapt25c1hyb_cloud_wp | wp | 84 | - | +0.25 (27/50/7) | -2.5 (20/58/6) | +0.50 (41/24/19) | -0.2 (30/17/37) | +0.05 (43/5/36) | -0.220 (53/31/0) | -0.10 (5/6/1) n=12 |
| ours_adapt25c1hyb_dev_abot | abot | 60 | - | -0.03 (5/48/7) | -0.8 (10/43/7) | +0.77 (21/28/11) | +0.4 (18/25/17) | -0.16 (26/10/24) | -0.165 (36/23/1) | +0.01 (1/8/1) n=10 |
| ours_adapt25c1hyb_dev_astra | astra | 21 | - | +0.33 (6/15/0) | +0.5 (2/17/2) | +0.62 (9/6/6) | -0.8 (4/6/11) | +0.04 (11/1/9) | -0.151 (13/8/0) | +0.05 (0/1/2) n=3 |
| ours_adapt25c1hyb_dev_mg2 | mg2 | 84 | - | +0.00 (22/46/16) | -4.5 (22/53/9) | +1.01 (40/27/17) | -0.9 (24/10/50) | -0.32 (45/4/35) | -0.433 (58/24/2) | -0.03 (2/9/1) n=12 |
| ours_adapt25c1hyb_dev_neutral_832 | neutral_832 | 84 | - | -0.07 (9/61/14) | -0.2 (17/50/17) | +0.20 (34/21/29) | -1.0 (29/15/40) | -0.04 (42/3/39) | -0.008 (34/33/17) | -0.06 (5/4/3) n=12 |
| ours_adapt25c1hyb_dev_neutral_832_nocap | neutral_832_nocap | 84 | - | +0.14 (19/54/11) | -2.6 (16/55/13) | +0.34 (31/23/30) | -1.5 (24/17/43) | +0.24 (36/9/39) | -0.329 (52/22/10) | +0.01 (3/6/3) n=12 |
| ours_adapt25c1hyb_dev_neutral_abot | neutral_abot | 60 | - | +0.05 (11/42/7) | -2.0 (16/34/10) | +0.29 (26/15/19) | -0.7 (19/21/20) | +0.23 (33/3/24) | -0.357 (40/18/2) | -0.14 (5/4/2) n=11 |
| ours_adapt25c1hyb_dev_neutral_mg2 | neutral_mg2 | 84 | - | +0.20 (23/53/8) | -6.2 (24/47/13) | +0.85 (27/13/9) n=49 | +0.4 (21/8/20) n=49 | -0.45 (24/5/20) n=49 | - | - |
| ours_adapt25c1hyb_dev_neutral_wp | neutral_wp | 84 | - | - | - | - | - | - | - | - |
| ours_adapt25c1hyb_dev_wp | wp | 84 | - | - | - | - | - | - | - | - |
| ours_rt2_abot | abot | 60 | 1.63x | -0.05 (5/47/8) | -2.8 (9/45/6) | +0.12 (15/34/11) | -0.3 (9/23/28) | -0.15 (30/8/22) | +0.025 (8/38/14) | +0.15 (2/6/3) n=11 |
| ours_rt2_astra | astra | 21 | 1.54x | -0.10 (3/13/5) | +1.4 (1/18/2) | -0.23 (2/12/7) | -1.1 (2/6/13) | -0.01 (7/4/10) | -0.047 (5/14/2) | +0.01 (1/0/2) n=3 |
| ours_rt2_mg2 | mg2 | 84 | 1.52x | -0.45 (6/50/28) | +1.5 (20/44/20) | -0.20 (24/25/35) | -2.2 (16/9/59) | +0.25 (32/4/48) | -0.012 (23/42/19) | -0.03 (2/7/3) n=12 |
| ours_rt2_neutral_832 | neutral_832 | 84 | 1.54x | -0.13 (9/57/18) | +0.3 (14/54/16) | +0.12 (35/20/29) | -1.2 (24/17/43) | -0.01 (43/4/37) | +0.057 (20/37/27) | -0.11 (4/7/0) n=11 |
| ours_rt2_neutral_832_nocap | neutral_832_nocap | 84 | 1.54x | +0.01 (12/59/13) | -4.3 (19/52/13) | -0.25 (26/14/44) | -2.8 (19/13/52) | +0.71 (28/2/54) | -0.113 (40/27/17) | -0.03 (3/7/2) n=12 |
| ours_rt2_neutral_abot | neutral_abot | 60 | 1.57x | -0.12 (5/43/12) | +2.3 (13/32/15) | -0.29 (19/10/31) | -1.1 (14/12/34) | +0.38 (23/7/30) | -0.145 (16/33/11) | -0.08 (5/3/3) n=11 |
| ours_rt2_neutral_mg2 | neutral_mg2 | 84 | 1.52x | -0.13 (9/53/22) | +1.3 (14/52/18) | -0.56 (28/8/48) | -2.9 (12/10/62) | +0.55 (31/3/50) | +0.075 (18/29/37) | +0.02 (3/6/3) n=12 |
| ours_rt2_neutral_wp | neutral_wp | 84 | 1.56x | -0.07 (8/62/14) | +4.6 (11/50/23) | -0.42 (23/15/46) | -1.0 (24/10/50) | +0.47 (26/8/50) | -0.097 (24/37/23) | -0.12 (5/7/0) n=12 |
| ours_rt2_wp | wp | 84 | 1.57x | +0.08 (16/59/9) | +1.4 (12/54/18) | -0.13 (25/25/34) | -1.2 (16/20/48) | +0.06 (37/6/41) | +0.049 (18/45/21) | -0.08 (3/8/1) n=12 |
| ours_rt2c1hyb_cloud_abot | abot | 24 | - | - | - | - | - | - | - | - |
| ours_rt2c1hyb_cloud_mg2 | mg2 | 84 | - | - | - | - | - | - | - | - |
| ours_rt2c1hyb_cloud_neutral_832 | neutral_832 | 84 | - | - | - | - | - | - | - | - |
| ours_rt2c1hyb_cloud_wp | wp | 84 | - | - | - | - | - | - | - | - |
| ours_rt2c1hyb_dev_abot | abot | 24 | - | - | - | - | - | - | - | - |
| ours_rt2c1hyb_dev_mg2 | mg2 | 84 | - | - | - | - | - | - | - | - |
| ours_rt2c1hyb_dev_neutral_832 | neutral_832 | 84 | - | - | - | - | - | - | - | - |
| ours_rt2c1hyb_dev_wp | wp | 84 | - | - | - | - | - | - | - | - |
| ours_sf1p3b_cal_astra | astra | 21 | 3.07x | - | - | - | - | - | - | - |
| ours_sf1p3b_cal_neutral_832 | neutral_832 | 84 | 2.74x | -1.12 (7/25/52) | +7.7 (16/38/30) | -0.02 (35/13/36) | -5.4 (17/5/62) | -0.20 (38/7/39) | +0.618 (13/16/55) | -0.17 (6/4/2) n=12 |
| ours_sf1p3b_cal_neutral_832_nocap | neutral_832_nocap | 84 | 2.42x | - | - | - | - | - | - | - |
| ours_sf1p3b_pub_astra | astra | 21 | 3.27x | - | - | - | - | - | - | - |
| ours_sf1p3b_pub_neutral_832 | neutral_832 | 84 | 2.98x | - | - | - | - | - | - | - |
| ours_sf1p3b_pub_neutral_832_nocap | neutral_832_nocap | 84 | 2.58x | - | - | - | - | - | - | - |
| ours_sf24_abot | abot | 60 | 1.53x | +0.02 (7/46/7) | +0.2 (12/38/10) | +0.72 (25/21/14) | -0.0 (15/16/29) | -0.29 (29/5/26) | -0.194 (39/20/1) | -0.03 (2/7/2) n=11 |
| ours_sf24_astra | astra | 21 | 1.54x | +0.24 (6/12/3) | +0.5 (1/19/1) | +0.34 (10/7/4) | -1.6 (4/4/13) | -0.11 (10/2/9) | -0.118 (8/13/0) | +0.04 (1/0/2) n=3 |
| ours_sf24_mg2 | mg2 | 84 | 1.47x | +0.00 (18/50/16) | -4.3 (18/59/7) | +0.92 (39/19/26) | -1.8 (22/9/53) | -0.50 (45/8/31) | -0.515 (63/18/3) | -0.06 (3/9/0) n=12 |
| ours_sf24_neutral_832 | neutral_832 | 84 | 1.13x | - | - | - | - | - | - | - |
| ours_sf24_neutral_832_long | neutral_832_long | 1 | 1.29x | - | - | - | - | - | - | - |
| ours_sf24_neutral_832_nocap | neutral_832_nocap | 84 | 1.42x | - | - | - | - | - | - | - |
| ours_sf24_neutral_abot | neutral_abot | 60 | 1.34x | - | - | - | - | - | - | - |
| ours_sf24_neutral_mg2 | neutral_mg2 | 84 | 1.16x | - | - | - | - | - | - | - |
| ours_sf24_neutral_wp | neutral_wp | 84 | 1.37x | +0.30 (24/56/4) | -8.4 (29/46/9) | +0.95 (43/25/16) | +0.3 (36/13/35) | -0.60 (40/3/41) | -0.365 (58/23/3) | -0.09 (4/6/2) n=12 |
| ours_sf24_wp | wp | 84 | 1.47x | +0.33 (28/53/3) | -3.0 (20/57/7) | +0.81 (45/28/11) | -0.7 (23/20/41) | -0.04 (37/5/42) | -0.261 (53/30/1) | -0.09 (5/6/1) n=12 |
| ours_sf24c1hyb_cloud_abot | abot | 60 | - | - | - | - | - | - | - | - |
| ours_sf24c1hyb_cloud_mg2 | mg2 | 84 | - | - | - | - | - | - | - | - |
| ours_sf24c1hyb_cloud_neutral_832 | neutral_832 | 84 | - | - | - | - | - | - | - | - |
| ours_sf24c1hyb_cloud_wp | wp | 84 | - | - | - | - | - | - | - | - |
| ours_sf24c1hyb_dev_abot | abot | 60 | - | - | - | - | - | - | - | - |
| ours_sf24c1hyb_dev_mg2 | mg2 | 84 | - | - | - | - | - | - | - | - |
| ours_sf24c1hyb_dev_neutral_832 | neutral_832 | 84 | - | - | - | - | - | - | - | - |
| ours_sf24c1hyb_dev_wp | wp | 84 | - | - | - | - | - | - | - | - |
| ours_student1p3b_abot | abot | 60 | 2.36x | - | - | - | - | - | - | - |
| ours_student1p3b_astra | astra | 21 | 2.19x | - | - | - | - | - | - | - |
| ours_student1p3b_mg2 | mg2 | 84 | 2.16x | - | - | - | - | - | - | - |
| ours_student1p3b_neutral_832 | neutral_832 | 84 | 2.19x | - | - | - | - | - | - | - |
| ours_student1p3b_neutral_832_nocap | neutral_832_nocap | 84 | 2.01x | - | - | - | - | - | - | - |
| ours_student1p3b_neutral_abot | neutral_abot | 60 | 2.20x | - | - | - | - | - | - | - |
| ours_student1p3b_neutral_mg2 | neutral_mg2 | 84 | 2.16x | - | - | - | - | - | - | - |
| ours_student1p3b_neutral_wp | neutral_wp | 84 | 2.07x | - | - | - | - | - | - | - |
| ours_student1p3b_wp | wp | 84 | 2.07x | - | - | - | - | - | - | - |
| waypoint | neutral_wp | 84 | 13.42x | -1.95 (3/16/65) | +40.3 (6/10/68) | -4.48 (14/5/65) | +0.2 (29/6/49) | +0.15 (32/2/50) | -0.123 (39/16/29) | -0.16 (6/6/0) n=12 |
| waypoint | wp | 84 | 13.08x | -0.74 (2/33/49) | +10.5 (4/36/44) | -3.51 (5/6/73) | +1.0 (50/7/27) | -0.67 (48/5/31) | -0.137 (44/25/15) | -0.11 (5/6/0) n=11 |

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
| ours_rt2_abot | 0.122 (n=12) | 1.312 (n=12) | 1.240 (n=12) | 1.206 (n=12) | 0.968 (n=12) | 0.906 (n=12) | 0.154 (n=12) | 0.27 (n=12) | 29/29 |
| ours_rt2_astra | 0.081 (n=3) | 0.405 (n=3) | 0.449 (n=3) | 1.172 (n=3) | 0.604 (n=3) | 0.823 (n=3) | 0.077 (n=3) | 0.16 (n=3) | 9/9 |
| ours_rt2_mg2 | 0.130 (n=12) | 1.392 (n=12) | 1.327 (n=12) | 1.304 (n=12) | 1.895 (n=12) | 1.733 (n=12) | 0.129 (n=12) | 0.12 (n=12) | 31/33 |
| ours_rt2_neutral_832 | 0.488 (n=12) | 0.532 (n=12) | 0.497 (n=12) | 1.529 (n=12) | 1.181 (n=12) | 1.271 (n=12) | 0.202 (n=12) | 0.16 (n=11) | 26/27 |
| ours_rt2_neutral_832_nocap | 0.212 (n=12) | 1.013 (n=12) | 1.049 (n=12) | 1.542 (n=12) | 1.387 (n=12) | 1.570 (n=12) | 0.165 (n=12) | 0.10 (n=12) | 34/34 |
| ours_rt2_neutral_abot | 0.242 (n=12) | 1.535 (n=12) | 1.468 (n=12) | 1.219 (n=12) | 1.253 (n=12) | 1.198 (n=12) | 0.364 (n=12) | 0.20 (n=12) | 29/31 |
| ours_rt2_neutral_mg2 | 0.246 (n=12) | 1.319 (n=12) | 1.164 (n=12) | 1.467 (n=12) | 1.974 (n=12) | 1.898 (n=12) | 0.360 (n=12) | 0.17 (n=12) | 25/26 |
| ours_rt2_neutral_wp | 0.170 (n=12) | 1.202 (n=12) | 1.132 (n=12) | 1.408 (n=12) | 1.354 (n=12) | 1.175 (n=12) | 0.138 (n=12) | 0.09 (n=12) | 33/33 |
| ours_rt2_wp | 0.071 (n=12) | 1.070 (n=12) | 1.037 (n=12) | 1.292 (n=12) | 0.955 (n=12) | 1.332 (n=12) | 0.120 (n=12) | 0.09 (n=12) | 30/31 |
| ours_sf1p3b_cal_neutral_832 | 0.780 (n=12) | 1.910 (n=12) | 1.803 (n=12) | 1.496 (n=12) | 1.836 (n=12) | 1.622 (n=12) | 0.179 (n=12) | 0.13 (n=12) | 35/35 |
| ours_sf24_abot | 0.075 (n=12) | 0.900 (n=12) | 0.855 (n=12) | 0.889 (n=12) | 0.746 (n=12) | 0.675 (n=12) | 0.056 (n=12) | 0.09 (n=12) | 31/32 |
| ours_sf24_astra | 0.086 (n=3) | 0.236 (n=3) | 0.260 (n=3) | 0.977 (n=3) | 0.691 (n=3) | 0.820 (n=3) | 0.043 (n=3) | 0.19 (n=3) | 9/9 |
| ours_sf24_mg2 | 0.088 (n=12) | 0.414 (n=12) | 0.426 (n=12) | 1.049 (n=12) | 1.200 (n=12) | 1.156 (n=12) | 0.057 (n=12) | 0.09 (n=12) | 31/31 |
| ours_sf24_neutral_wp | 0.172 (n=12) | 0.658 (n=12) | 0.686 (n=12) | 1.159 (n=12) | 0.935 (n=12) | 1.009 (n=12) | 0.086 (n=12) | 0.12 (n=12) | 33/33 |
| ours_sf24_wp | 0.059 (n=12) | 0.596 (n=12) | 0.636 (n=12) | 1.029 (n=12) | 0.661 (n=12) | 0.662 (n=12) | 0.068 (n=12) | 0.08 (n=12) | 36/36 |
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
