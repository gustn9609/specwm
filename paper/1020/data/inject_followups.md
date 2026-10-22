# Inject follow-ups (10/20): does the 14B context lift the 1.3B's frames, and what does the re-anchor boundary look like?

Setup: the divergence-injection latents (README, facts §52), 832x480, 13 chunks x 3 latent frames, 5 scenes x 10 seeds. Clips decoded VAE-only to 16 fps mp4 (crf 23) in `lingbot-world-v2/specwm_out/inject/judge/<arm>_<scene>_s<seed>.mp4` (~0.9 GB total). Judge = Qwen3-VL-32B on :30100, 8 evenly spaced frames, the baselines/score_runs.sh prompts (`data/inject_vlm.json`). Handoff arms (`hand@kmM` = 14B generates chunks k..k+M−1, 1.3B continues; sandcastle + stonehenge only) were added on request. Scripts: `inject_followups.py` (decode + boundary stats -> `data/inject_boundary.json`), `hand_decode.py`, `inject_followups_report.py` (tables + sheets).

## Conclusions

1. **The 14B context does not lift the 1.3B's per-frame quality or drift (Test 1).** Pooled over 50 clips: quality T 7.62 > alone 7.28 = once@6 7.28 > every 6.92; drift T 46 < alone 57 < every 61 < once@6 64. Paired once@6 − alone: quality +0.00 ± 0.14 (17 better / 11 worse / 22 tie), drift +6.7 ± 4.7 (12 / 16 / 22). `every` (teacher forcing) is *worse* than alone on quality (−0.36 ± 0.16, 12 / 18 / 20) and no better on drift (+4.2 ± 4.6). Only T itself is clearly better (quality +0.34 ± 0.17, drift −10.5 ± 5.7, 29 / 11 / 10). Per scene the picture flips: on lake the judge prefers the 1.3B's own rollout (8.9 / 31) over the 14B's (7.6 / 70) and every injected arm inherits the 14B's drift there; on cup/jar/stonehenge T is better but neither once@6 nor every closes the gap. The judge is coarse (whole scenes pinned at 75 drift, quality steps of 1), so these are "no detectable effect" statements, not precise effect sizes.
2. **Handoff arms**: quality identical to alone (paired 0.00–0.05 ± 0.09, 17–20/20 ties); drift hand@3m3 −6.0 ± 4.0 vs alone (4 better / 2 worse / 14 tie; stonehenge 36 → 24), hand@3m1 −2.8 ± 3.0, hand@6m2 +8.8 ± 4.5 (worse, 1 / 5 / 14). Versus once@6 all three are slightly worse on quality (−0.30…−0.35 ± 0.12, 0–1 better / 7 worse) and hand@6m2 also on drift (+10.8 ± 4.6). A short early 14B segment buys a small drift reduction at best; nothing lifts per-frame quality.
3. **The re-anchor boundary is a hard cut (Test 2).** The frame-to-frame L1 at the spliced boundary k=6 is 45.1 ± 1.4 (8-bit) vs 15.8 (alone) / 21.3 (T) at the same ordinary boundary: **ratio 2.55 pooled, 2.0–3.7 per scene, 48/50 seeds > 1.5**. The unspliced once@6 clip's own boundary (20.7) equals an ordinary boundary, i.e. the 1.3B continues the 14B context smoothly; the whole jump is the content mismatch between what the device had shown (its own rollout) and the cloud-anchored continuation. Ordinary boundaries are not special (alone 16.9 at boundaries vs 16.4 within chunks; T 21.7 vs 21.6); `every` shows the teacher-forcing seam (29.7 vs 20.3). 1−SSIM is motion-dominated at 16 fps (0.45–0.55 even within chunks) and only rises 1.17x; it is reported but not diagnostic.
4. **Contact sheets** (`figs/boundary_<scene>_k6_s1234.png`, frames 72–88, boundary between f80 and f81, rows: device view / alone / once@6 / T): both are scene cuts, not smooth transitions. Sandcastle: the device view shows the 1.3B's castle on dry sand with bushes and hills to f80, then f81 is the 14B world — a crouching person on a wet shoreline with blue water, a different composition; f81 itself is a half-blended frame (VAE temporal receptive field), from f82 on it is cleanly the once@6 content. Stonehenge: the 1.3B's sunlit trilithons in dry grass to f80, then f81–f82 a ghosted blend and from f83 a low, close-up aerial over a large grey stone on green grass (the 14B's camera position). In the unspliced once@6 row the same boundary is invisible.

What this means for the device/cloud design: cloud re-anchoring moves the device onto the 14B's trajectory but, by itself, gives the viewer a visible cut every time it fires, and the frames after it are not sharper or less drifty by the judge. A blend / transition (soft KV update, crossfade, or re-anchoring only at scene-consistent moments) is needed before this is user-visible.

Not done / caveats: k = 3 and 9 boundary statistics have n = 1 (jar s11) — phase 4 of `inject_followups.py` is still decoding on GPU 5 (~50 s per seed-scene under contention) and appends to `data/inject_boundary.json`; rerun `inject_followups_report.py` later to fill those rows. VBench6 / MUSIQ-drift skipped (≈ 20 s per clip -> > 60 min for 200 clips). Judge quantisation means per-scene s.e. of 0.00 is a tie artefact, not precision.


## Test 1 — Qwen3-VL judge (8 frames/clip): quality 1–10 ↑, drift 0–100 ↓; mean ± s.e. over seeds (n)

| quality ↑ | T | alone | every | once@6 | hand@3m1 | hand@3m3 | hand@6m2 |
|---|---|---|---|---|---|---|---|
| jar | **8.30 ± 0.21** | 7.40 ± 0.22 | 7.40 ± 0.16 | 7.40 ± 0.16 | – | – | – |
| sandcastle | 6.90 ± 0.10 | 6.60 ± 0.16 | 6.80 ± 0.13 | **7.00 ± 0.00** | 6.60 ± 0.16 | 6.60 ± 0.16 | 6.60 ± 0.16 |
| lake | 7.60 ± 0.22 | **8.90 ± 0.10** | 6.80 ± 0.13 | 7.60 ± 0.22 | – | – | – |
| stonehenge | **7.60 ± 0.16** | 7.00 ± 0.00 | 6.70 ± 0.15 | 7.30 ± 0.15 | 7.00 ± 0.00 | 7.10 ± 0.10 | 7.00 ± 0.00 |
| cup | **7.70 ± 0.15** | 6.50 ± 0.17 | 6.90 ± 0.10 | 7.10 ± 0.10 | – | – | – |
| pooled | **7.62 ± 0.10** (n=50) | 7.28 ± 0.14 (n=50) | 6.92 ± 0.07 (n=50) | 7.28 ± 0.07 (n=50) | 6.80 ± 0.09 (n=20) | 6.85 ± 0.11 (n=20) | 6.80 ± 0.09 (n=20) |

| drift ↓ | T | alone | every | once@6 | hand@3m1 | hand@3m3 | hand@6m2 |
|---|---|---|---|---|---|---|---|
| jar | **30.00 ± 8.60** | 61.00 ± 10.82 | 43.00 ± 9.43 | 64.50 ± 9.76 | – | – | – |
| sandcastle | 76.00 ± 1.00 | 81.00 ± 1.63 | 77.00 ± 1.33 | **73.50 ± 5.06** | 79.00 ± 1.63 | 81.00 ± 1.63 | 81.00 ± 1.63 |
| lake | 69.50 ± 5.50 | **31.00 ± 7.33** | 75.00 ± 0.00 | 65.00 ± 7.56 | – | – | – |
| stonehenge | **11.50 ± 7.53** | 36.50 ± 8.40 | 37.50 ± 8.24 | 40.00 ± 9.72 | 33.00 ± 8.67 | 24.50 ± 7.01 | 54.00 ± 9.30 |
| cup | **45.00 ± 6.19** | 75.00 ± 0.00 | 73.00 ± 1.33 | 75.00 ± 0.00 | – | – | – |
| pooled | **46.40 ± 4.38** (n=50) | 56.90 ± 4.15 (n=50) | 61.10 ± 3.45 (n=50) | 63.60 ± 3.64 (n=50) | 56.00 ± 6.80 (n=20) | 52.75 ± 7.37 (n=20) | 67.50 ± 5.54 (n=20) |

### Paired differences (same scene + seed), mean ± s.e.; (+/−/0) = seeds better / worse / tie in the metric's own direction

| quality ↑ | every − alone | once@6 − alone | T − alone | hand@3m1 − alone | hand@3m3 − alone | hand@6m2 − alone | hand@3m1 − once@6 | hand@3m3 − once@6 | hand@6m2 − once@6 |
|---|---|---|---|---|---|---|---|---|---|
| jar | +0.00 ± 0.33 (n=10; 4/3/3) | +0.00 ± 0.30 (n=10; 3/2/5) | +0.90 ± 0.38 (n=10; 8/1/1) | – | – | – | – | – | – |
| sandcastle | +0.20 ± 0.25 (n=10; 4/2/4) | +0.40 ± 0.16 (n=10; 4/0/6) | +0.30 ± 0.21 (n=10; 4/1/5) | +0.00 ± 0.15 (n=10; 1/1/8) | +0.00 ± 0.15 (n=10; 1/1/8) | +0.00 ± 0.00 (n=10; 0/0/10) | -0.40 ± 0.16 (n=10; 0/4/6) | -0.40 ± 0.16 (n=10; 0/4/6) | -0.40 ± 0.16 (n=10; 0/4/6) |
| lake | -2.10 ± 0.18 (n=10; 0/10/0) | -1.30 ± 0.30 (n=10; 1/9/0) | -1.30 ± 0.30 (n=10; 1/9/0) | – | – | – | – | – | – |
| stonehenge | -0.30 ± 0.15 (n=10; 0/3/7) | +0.30 ± 0.15 (n=10; 3/0/7) | +0.60 ± 0.16 (n=10; 6/0/4) | +0.00 ± 0.00 (n=10; 0/0/10) | +0.10 ± 0.10 (n=10; 1/0/9) | +0.00 ± 0.00 (n=10; 0/0/10) | -0.30 ± 0.15 (n=10; 0/3/7) | -0.20 ± 0.20 (n=10; 1/3/6) | -0.30 ± 0.15 (n=10; 0/3/7) |
| cup | +0.40 ± 0.16 (n=10; 4/0/6) | +0.60 ± 0.16 (n=10; 6/0/4) | +1.20 ± 0.25 (n=10; 8/0/2) | – | – | – | – | – | – |
| pooled | -0.36 ± 0.16 (n=50; 12/18/20) | +0.00 ± 0.14 (n=50; 17/11/22) | +0.34 ± 0.17 (n=50; 27/11/12) | +0.00 ± 0.07 (n=20; 1/1/18) | +0.05 ± 0.09 (n=20; 2/1/17) | +0.00 ± 0.00 (n=20; 0/0/20) | -0.35 ± 0.11 (n=20; 0/7/13) | -0.30 ± 0.13 (n=20; 1/7/12) | -0.35 ± 0.11 (n=20; 0/7/13) |

| drift ↓ | every − alone | once@6 − alone | T − alone | hand@3m1 − alone | hand@3m3 − alone | hand@6m2 − alone | hand@3m1 − once@6 | hand@3m3 − once@6 | hand@6m2 − once@6 |
|---|---|---|---|---|---|---|---|---|---|
| jar | -18.00 ± 14.63 (n=10; 5/2/3) | +3.50 ± 14.10 (n=10; 4/4/2) | -31.00 ± 16.00 (n=10; 6/2/2) | – | – | – | – | – | – |
| sandcastle | -4.00 ± 2.21 (n=10; 5/1/4) | -7.50 ± 5.83 (n=10; 5/2/3) | -5.00 ± 2.24 (n=10; 6/1/3) | -2.00 ± 2.00 (n=10; 3/1/6) | +0.00 ± 1.49 (n=10; 1/1/8) | +0.00 ± 1.49 (n=10; 1/1/8) | +5.50 ± 6.08 (n=10; 3/4/3) | +7.50 ± 5.83 (n=10; 2/5/3) | +7.50 ± 5.83 (n=10; 2/5/3) |
| lake | +44.00 ± 7.33 (n=10; 0/8/2) | +34.00 ± 11.90 (n=10; 1/8/1) | +38.50 ± 11.74 (n=10; 1/8/1) | – | – | – | – | – | – |
| stonehenge | +1.00 ± 8.26 (n=10; 1/2/7) | +3.50 ± 10.19 (n=10; 2/2/6) | -25.00 ± 8.82 (n=10; 7/0/3) | -3.50 ± 5.87 (n=10; 1/2/7) | -12.00 ± 7.54 (n=10; 3/1/6) | +17.50 ± 8.24 (n=10; 0/4/6) | -7.00 ± 8.27 (n=10; 2/3/5) | -15.50 ± 10.42 (n=10; 3/2/5) | +14.00 ± 7.14 (n=10; 0/4/6) |
| cup | -2.00 ± 1.33 (n=10; 2/0/8) | +0.00 ± 0.00 (n=10; 0/0/10) | -30.00 ± 6.19 (n=10; 9/0/1) | – | – | – | – | – | – |
| pooled | +4.20 ± 4.64 (n=50; 13/13/24) | +6.70 ± 4.66 (n=50; 12/16/22) | -10.50 ± 5.74 (n=50; 29/11/10) | -2.75 ± 3.02 (n=20; 4/3/13) | -6.00 ± 3.98 (n=20; 4/2/14) | +8.75 ± 4.54 (n=20; 1/5/14) | -0.75 ± 5.20 (n=20; 5/7/8) | -4.00 ± 6.38 (n=20; 5/7/8) | +10.75 ± 4.55 (n=20; 2/9/9) |


## Test 2 — re-anchor boundary jump (frame-to-frame change at the chunk boundary k|k+1; pixel L1 on 8-bit ↓, 1−SSIM ↓)

Splice = alone chunks 0..k + once@k chunks k+1.. (what a device user sees). Ordinary = the same boundary index k|k+1 in alone and in T (unspliced). Ratio = splice / mean(alone, T) at that boundary. n = seeds.

| L1 | k | splice | alone (ordinary) | T (ordinary) | once@k own boundary | ratio splice/ordinary | seeds ratio>1.5 | within-chunk alone |
|---|---|---|---|---|---|---|---|---|
| jar | 3 | 39.82 ± 0.00 | 27.05 ± 0.00 | 26.39 ± 0.00 | 28.06 ± 0.00 | **1.49** (med 1.49) | 0/1 | 21.23 |
| jar | 6 | 37.66 ± 2.16 | 17.67 ± 0.99 | 21.96 ± 1.77 | 21.93 ± 1.59 | **1.96** ⚠ (med 2.04) | 8/10 | 19.21 |
| jar | 9 | 47.18 ± 0.00 | 21.83 ± 0.00 | 20.32 ± 0.00 | 20.71 ± 0.00 | **2.24** ⚠ (med 2.24) | 1/1 | 21.23 |
| sandcastle | 6 | 49.87 ± 3.48 | 23.92 ± 1.95 | 19.93 ± 2.39 | 18.78 ± 2.09 | **2.36** ⚠ (med 2.21) | 10/10 | 22.05 |
| lake | 6 | 43.30 ± 1.70 | 8.49 ± 0.34 | 23.69 ± 1.52 | 22.69 ± 1.45 | **2.75** ⚠ (med 2.66) | 10/10 | 10.44 |
| stonehenge | 6 | 38.10 ± 0.88 | 12.52 ± 0.69 | 25.97 ± 1.39 | 26.32 ± 1.55 | **2.00** ⚠ (med 1.91) | 10/10 | 12.80 |
| cup | 6 | 56.49 ± 1.53 | 16.46 ± 1.05 | 15.03 ± 1.12 | 13.84 ± 1.07 | **3.69** ⚠ (med 3.65) | 10/10 | 17.38 |
| pooled | 3 | 39.82 ± 0.00 | 27.05 ± 0.00 | 26.39 ± 0.00 | 28.06 ± 0.00 | **1.49** (med 1.49) | 0/1 | 21.23 |
| pooled | 6 | 45.08 ± 1.38 | 15.81 ± 0.89 | 21.32 ± 0.90 | 20.71 ± 0.90 | **2.55** ⚠ (med 2.40) | 48/50 | 16.38 |
| pooled | 9 | 47.18 ± 0.00 | 21.83 ± 0.00 | 20.32 ± 0.00 | 20.71 ± 0.00 | **2.24** ⚠ (med 2.24) | 1/1 | 21.23 |

| 1−SSIM | k | splice | alone (ordinary) | T (ordinary) | once@k own boundary | ratio splice/ordinary | seeds ratio>1.5 | within-chunk alone |
|---|---|---|---|---|---|---|---|---|
| jar | 3 | 0.74 ± 0.00 | 0.72 ± 0.00 | 0.74 ± 0.00 | 0.73 ± 0.00 | **1.02** (med 1.02) | 0/1 | 0.71 |
| jar | 6 | 0.77 ± 0.02 | 0.67 ± 0.03 | 0.75 ± 0.04 | 0.75 ± 0.04 | **1.09** (med 1.08) | 0/10 | 0.63 |
| jar | 9 | 0.78 ± 0.00 | 0.81 ± 0.00 | 0.61 ± 0.00 | 0.60 ± 0.00 | **1.10** (med 1.10) | 0/1 | 0.71 |
| sandcastle | 6 | 0.74 ± 0.03 | 0.70 ± 0.04 | 0.56 ± 0.02 | 0.55 ± 0.02 | **1.18** (med 1.19) | 0/10 | 0.58 |
| lake | 6 | 0.55 ± 0.02 | 0.28 ± 0.01 | 0.64 ± 0.03 | 0.65 ± 0.03 | **1.20** (med 1.20) | 0/10 | 0.30 |
| stonehenge | 6 | 0.65 ± 0.01 | 0.48 ± 0.02 | 0.84 ± 0.03 | 0.84 ± 0.03 | **1.00** (med 0.98) | 0/10 | 0.37 |
| cup | 6 | 0.56 ± 0.02 | 0.47 ± 0.03 | 0.33 ± 0.02 | 0.31 ± 0.02 | **1.39** (med 1.38) | 1/10 | 0.38 |
| pooled | 3 | 0.74 ± 0.00 | 0.72 ± 0.00 | 0.74 ± 0.00 | 0.73 ± 0.00 | **1.02** (med 1.02) | 0/1 | 0.71 |
| pooled | 6 | 0.65 ± 0.02 | 0.52 ± 0.02 | 0.63 ± 0.03 | 0.62 ± 0.03 | **1.17** (med 1.14) | 1/50 | 0.45 |
| pooled | 9 | 0.78 ± 0.00 | 0.81 ± 0.00 | 0.61 ± 0.00 | 0.60 ± 0.00 | **1.10** (med 1.10) | 0/1 | 0.71 |

Ordinary boundaries averaged over ALL 12 chunk boundaries (pooled over scenes and seeds), to show the boundary itself is not special:

| arm | L1 at boundaries | L1 within chunk | 1−SSIM at boundaries | 1−SSIM within chunk |
|---|---|---|---|---|
| alone | 16.87 | 16.38 | 0.473 | 0.451 | n=50 |
| T | 21.68 | 21.61 | 0.560 | 0.534 | n=50 |
| every | 29.70 | 20.27 | 0.603 | 0.506 | n=50 |
| once@6 | 20.43 | 20.25 | 0.567 | 0.547 | n=50 |
