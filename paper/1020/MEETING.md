# 10/20 meeting notes — divergence injection (1.3B ← 14B)

**Terms (ours, not standard names).** "envelope ratio" = normalized distance to the 14B: |seed-mean path of the arm − seed-mean path of the 14B| ÷ (mean distance of the 14B's own seeds from their mean), i.e. a z-score-like distance in units of the 14B's seed-to-seed spread; < 1 = within the range the 14B itself produces across seeds (distribution level). "follow ratio" = same-seed following: d_own / (d_14B + d_own) with same-seed, same-noise latent distances; 1 = on the 14B's own trajectory, 0 = on the generator's own trajectory (trajectory level; its ceiling is set by rollout chaos, see the `every` arm).

**Question.** Generate with the 1.3B; replace its context with the 14B 4-step latents at chunk k (re-committed through the 1.3B's own forward); does the 1.3B then move like the 14B or go back to its own path?

**Setup.** Key-press stream tau2_s0 (13 chunks × 0.75 s, 832×480), 5 scenes × 10 seeds, same noise per chunk in every arm. Arms: `alone` (1.3B), `once@k` (14B context for chunks ≤ k, k = 1/3/6/9), `every` (14B context after every chunk = ceiling), `self@3` (re-commit its own latents = floor). Reference T = 14B 4-step.

**Metrics.** Envelope ratio ↓ = |seed-mean path − 14B seed-mean path| / 14B seed spread, in the scene's latent PCA (< 1 = inside the 14B seed spread). Follow ratio ↑ = d_G / (d_T + d_G) with same-seed latent distances (1 = on the 14B same-seed path, 0 = on its own path).

| arm | envelope ratio after injection ↓ (5 scenes) | follow ratio ↑ (mean over scenes) |
|---|---|---|
| 1.3B alone | 0.9 / 1.4–2.0 / 2.2–2.9 / 4.2–7.5 / 2.0–2.6 | — |
| once@1 | 0.69 / 1.23 / 0.59 / 2.24 / 1.10 | 0.49 |
| once@3 | 0.36 / 1.22 / 0.29 / 0.96 / 0.87 | 0.51 |
| once@6 | <u>0.18</u> / 0.55 / **0.23** / **0.51** / 0.84 | 0.57 |
| once@9 | 0.17 / **0.24** / 0.46 / 0.89 / **0.31** | <u>0.61</u> |
| every (ceiling) | **0.14** / <u>0.29</u> / <u>0.28</u> / <u>0.76</u> / <u>0.53</u> | **0.67** |
| self@3 (floor) | 0.90 / 1.42 / 2.36 / 5.84 / 1.95 | 0.15 |

(scene order: jar / sandcastle / lake / stonehenge / pouring)

**Findings.**
1. After the injection the 1.3B does not return to its own path: it stays inside the 14B seed spread (envelope 0.1–0.5 vs 1.4–7.5 alone); distance to its own alone path 0.4–6.8 (self re-commit: 0.01–0.05).
2. It does not reproduce the 14B's same-seed trajectory: follow ≤ 0.67 even with the 14B context every chunk — the AR rollout is chaotic (10/3: numerically identical 14B runs diverge after 3–4 chunks). "Like the 14B" holds at the distribution level, not the trajectory level.
3. A single injection fades after ~4–6 chunks (3–4.5 s); later injections hold better (0.49 → 0.61) because the injected context then already contains the 14B's committed content (sandcastle: the person on the shore), which the 1.3B keeps; an early context (castle + wave only) is continued with the 1.3B's own prior (sand flats, bushes).
4. The 1.3B mean path is outside the 14B envelope from chunk 0 on 4/5 scenes (jar: chunk 3): there is no "similar first, then diverge" phase at the distribution level; the f20 impression came from the shared first frame.

**Frames (sandcastle s1234).** 14B: castle → wave → wet sand → a person sitting in the water → shoreline. 1.3B alone: castle → wave → dry sand flats, bushes, mountains, lake. once@1/@3: the 1.3B's own landscape. once@6/@9: keep the person, continue on the shore (@6 ends with the person walking; @9 ends with the 14B's shoreline shot). every: ≈ 14B.

**Caveats.** 5 scenes, 1 stream, 10 seeds; the injection is a re-commit through the 1.3B's own forward (no cross-model KV copy); "following the 14B" ≠ "correct" (the 14B hallucinates the person too). Per-frame quality / drift of the injected 1.3B and the visual jump at a re-anchor boundary were not measured in this probe (follow-ups running 10/20 PM: judge on alone/every/once@6; boundary jump; handoff arms 1.3B → 14B-on-1.3B-context → 1.3B; same-model control 14B 2-step ← 14B 4-step).

**Same-model control (14B 2-step ← 14B 4-step, partial: jar / castle n=10, lake n=5).** Follow ratio once@1/3/6/9 = 0.55/0.60/0.66/0.74, every 0.76 (vs 1.3B 0.49/0.51/0.57/0.61, 0.67); envelope distance after injection 0.01–0.08 for k ≥ 3 (1.3B 0.2–0.9) and it STAYS inside. First post-injection chunk distance to T: 14B 2-step 0.41–0.44 vs 1.3B 0.66–0.90 (1 = independent seeds). → Same-model re-anchoring is a different regime: the 2-step 14B really follows the 4-step 14B; the 1.3B keeps the injected content and continues with its own dynamics, re-leaving the envelope within ~the KV window (4 chunks) after an early injection.

**Device tiers video** (`videos/tiers832_ALL.mp4`, 7 bench clips, 14B 2-step, 832×480, same seed/prompt/camera per clip, 10/13 fit24 harness): original FP8 weights + bf16 KV **peak 40 GiB** | 32 GB class FP8 KV **31 GiB** | 24 GB class GPTQ-4 + FP8 KV **22 GiB** (generation peaks, window 18; fp32 decode adds ~7 GiB, the device's bf16 2×2 tiled decode +1.5). Judge 10/13: GPTQ-4 at 832 −0.58 vs FP8. The three tiles diverge in content within a few seconds (chaotic rollout) — compare quality class, not content. Not the frozen baseline tasks; the queued rt2c1hyb24/32 arms (window 12) are the frozen versions.

**Idea.** Device-tier ladder under one cloud: 14B FP8 (96 GB) / 14B W4 (24–32 GB) / 1.3B (~8 GB); the 1.3B is the only device faster than real time (1.7× @832, 3.4× @704) → predict-ahead design (device shows look-ahead chunks, cloud chunk arrives 0.5–0.7 s later and re-anchors the KV; only un-shown chunks are regenerated → no visible jump). Risks: 1.3B per-frame quality (judge pending), cloud must generate full 14B chunks (2-step for real time), re-anchor cadence ≤ 2–3 chunks.

Files: figs/f21_inject_clean_1p3b_4step.png, videos/inject_1p3b_4step_*_s1234.mp4 (red border = shown chunk's context is the 14B's), data/inject_tables.md, README.md.

## Follow-ups (10/20 PM, GPU 5; details in data/inject_followups.md, data/handoff_tables.md, figs/f22_handoff_clean.png)
- **Judge (5 scenes × 10 seeds):** T 7.62 / 46.4, 1.3B alone 7.28 / 56.9, once@6 7.28 / 63.6 (paired vs alone +0.00), every 6.92 / 61.1. The 14B context does not lift the 1.3B's per-frame quality or drift.
- **Re-anchor boundary (k=6, 50 seeds):** device-view splice L1 45.1 vs ordinary chunk boundary 15.8–21.3 → 2.55× (48/50 seeds > 1.5×) = a hard scene cut each time the KV is re-anchored; the same boundary inside the un-spliced once@6 clip is ordinary (20.7). Needs a transition (soft KV update / replace only un-shown chunks).
- **Handoff (1.3B → 14B on the 1.3B's context → 1.3B; sandcastle + stonehenge, 10 seeds):** the 14B continues the 1.3B's scene (its segment sits at 0.33 / 8.8 from the 14B mean = the 1.3B's world), hands over discrete content (a person, a wave) which the 1.3B keeps; trajectory returns to the 1.3B's own path within ~2 chunks (follow 0.30–0.37 vs once@3/6 0.45–0.63, floor 0.09–0.14); judge quality unchanged (ties), drift −6 ± 4 for the early 3-chunk segment only.
- **Verifier-driven handoff (hand@reject, drift head on 14B features):** at the calibrated tau −3.25 the head flags 0 % (sandcastle) / 19 % (stonehenge, late chunks only) of the 1.3B's chunks — the 1.3B's chunks are not "drift" by the 14B's own standard, they are a different world; at a 25 % quantile tau the replaced chunks (8–11) sit at 1.45 / 4.40 from the 14B mean, follow 0.36–0.39: same picture as the fixed handoffs.
- **Same-model control (14B 2-step ← 14B 4-step):** goes to 0.01–0.08 from the 14B mean after k ≥ 3 and stays — the regime where cloud re-anchoring works.
- **Conclusion:** for a 1.3B device the cloud's KV re-anchoring buys content for ~a window length, no quality, and a visible cut; the untested lever is pushing the 1.3B's *sampling* toward the 14B (chunk-wise autoguidance with one 14B forward per chunk, variant B). For a squeezed-14B device the same re-anchoring is a real anchor (thesis).

## Same-model control, complete (10/20 23:50 CDT; 5 scenes x 10 seeds; data/inject_tables.md, figs/f21_inject_clean_14b_2step.png)
14B 2-step <- 14B 4-step latents. Follow ratio ↑ once@1/3/6/9 = 0.48–0.55 / 0.57–0.60 / 0.61–0.66 / 0.67–0.74, every 0.73–0.76 (1.3B: 0.44–0.54 / 0.45–0.58 / 0.53–0.63 / 0.56–0.64, every 0.62–0.71); floor self@3 0.27–0.37 (1.3B 0.09–0.23). Envelope distance after the injection ↓ (once@6 / once@9 / every): jar 0.05 / 0.01 / 0.05, sandcastle 0.05 / 0.03 / 0.03, lake 0.07 / 0.04 / 0.04, stonehenge 0.09 / 0.08 / 0.14, cup 0.19 / 0.05 / 0.07 — vs the 1.3B's 0.18–0.86 / 0.17–0.89 / 0.14–0.75. Note the 14B 2-step is itself inside the 14B 4-step seed spread when alone (0.20–0.60): its "divergence" is a seed-level spread, not a different world, and one injection pins it to the 4-step mean path (0.01–0.09) for the rest of the clip. First post-injection chunk d_T 0.25–0.43 (once@6/9) vs the 1.3B's 0.42–0.76. Same picture on all 5 scenes → the squeezed-14B device is the regime where cloud KV re-anchoring is a real anchor; the 1.3B is not.

## Multi-resolution denoising steps, training-free pilot (10/20 23:40 CDT; data/multires_pilot.md, figs/multires_rel_l2.png, videos/multires_<scene>_s1234.mp4)
Advisor's "x3 -> x2 -> x1 -> x0 at different resolutions" tried without training on the stock 14B FP8 (half-res K/V cache rebuilt per chunk, x0 upsampled + re-noised with the reference noise), 5 scenes x 3 seeds. Speed (steady s/chunk ↓ / fps ↑): full4 2.55 / 4.7, full2 (nodes 0,500) 1.53 / 7.9, mr4 1.80 / 6.6, mr4b 2.22 / 5.4, **mr2 1.20 / 10.0 (2.12x full4, 1.27x full2)**; a half-res step costs 0.09 s vs 0.51 s full (5.5x cheaper), the clean commit pass (0.51 s) is 42 % of mr2's chunk. Quality: judge quality ties (pooled 7.47 for every arm, mr2 7.27); judge drift ↓ pooled T 46 / full4 49 / full2 42 / mr4 55 / mr4b 54 / **mr2 64** (paired mr2 − T +18 ± 5, 0 better / 10 worse / 5 ties); sharpness (Laplacian var ↑) full4 669, full2 641, mr4 472, mr4b 541, mr2 552 (paired mr4 − full4 −197 ± 62, 5/10/0); rel-L2 to T at chunk 0: full4 0.15, full2 0.34, mr4b 0.52, mr4 0.60, mr2 0.62. Frames (sandcastle, lake, seed 1234): the half-res first step changes the CONTENT from chunk 1 on — the sandcastle is gone in mr4/mr2 (open sea, mr4 later a jet-ski handlebar POV), the lake's willow is gone and returns elsewhere; full2 keeps the composition. The content is decided at the first denoising step (10/13 action-speculation finding) and the stock model's half-res x0 at sigma 1 is a different scene. Verdict: training-free multi-res is CLOSED as a quality-neutral speed-up; the 2.1x is real only with a fine-tune that teaches the model the half-res-then-full schedule (config-aware distillation on Vista), and the commit pass (fused commit, 10/11) is the cheaper 1.3x lever.

## Soft KV re-anchor sweep (10/21 00:03 CDT; same-model pair, device 14B 2-step <- cloud 14B 4-step, k = 6, 3 scenes x 10 seeds; data/soft_reanchor_tables.md, data/soft_boundary.json, data/soft_vlm.json, videos/soft_reanchor_<scene>_s1234.mp4)
Arms: context 0..6 = own + alpha (cloud − own), alpha 0 (alone) / 0.25 / 0.5 / 0.75 / 1 (hard once@6), and progressive 0.25 → 0.5 → 1 over chunks 6/7/8. Three-way trade-off, mean of 3 scenes:

| arm | follow ↑ | envelope ↓ | boundary jump 6\|7 ratio ↓ (1 = ordinary) | judge quality vs alone | judge drift vs alone |
|---|---|---|---|---|---|
| alpha 0.25 | 0.35 | 0.32 | **1.02** | −0.17 | +1.0 |
| alpha 0.5 | 0.48 | 0.25 | 1.17 | −0.33 | +15.2 |
| alpha 0.75 | <u>0.60</u> | 0.15 | 1.42 | −0.13 | +1.3 |
| alpha 1 (hard) | **0.65** | **0.06** | 1.73 | **+0.07** | **+0.7** |
| progressive | 0.58 | <u>0.12</u> | <u>1.02</u> (then 1.15 / 1.10 at 7\|8, 8\|9) | −0.17 | +2.0 |

Frames (sandcastle s1234, chunks 6 → 7 → 11): alpha 0.25 keeps the device's castle (no anchoring); alpha 0.5 decodes as a DOUBLE EXPOSURE — the cloud's person appears ghosted inside the castle and the clip continues with both (drift +15); alpha 1 is the hard cut to the person; progressive spreads the cut over three boundaries and ends on the cloud's content. The linear latent blend is not a transition: below alpha 0.75 it does not anchor, at 0.5 it ghosts, at 1 it cuts (1.73x an ordinary boundary, down from the 1.3B's 2.55x because the 2-step device is already inside the 4-step spread). Progressive is the best of these (no single jump > 1.15x, follow 0.58, envelope 0.12) at −0.17 quality. The clean fix remains to re-anchor only chunks the viewer has not seen (predict-ahead) or to time the re-anchor to a natural cut; a learned transition is a training item.

## Device-tier profile on an idle GPU (10/21 00:55 CDT; data/profile24.md; 6 clips of 10 s per cell, medians; 14B 2-step, frozen-protocol run_ours.sh with RUNS_SUFFIX=_idle5)
| res | device tier | steady fps gen ↑ | chunk s ↓ | TTFF s ↓ | fps e2e ↑ | peak reserved GiB ↓ |
|---|---|---|---|---|---|---|
| 832x480 | original (FP8 w, bf16 KV, window 18) | <u>9.60</u> | <u>1.249</u> | 7.57 | 6.30 | 72.5 |
| 832x480 | 32 GB class (FP8 KV, window 12, cap 32) | **10.82** | **1.109** | n/a | n/a | <u>27.9</u> |
| 832x480 | 24 GB class (GPTQ-4 + FP8 KV, window 12, cap 24) | 7.42 | 1.616 | n/a | n/a | **19.7** |
| 640x352 | original | <u>19.35</u> | <u>0.620</u> | 4.50 | 12.14 | 49.1 |
| 640x352 | 32 GB class | **21.05** | **0.570** | n/a | n/a | <u>24.0</u> |
| 640x352 | 24 GB class | 12.50 | 0.960 | n/a | n/a | **15.8** |

Readings: the 32 GB class is FASTER than the original (window 12 < 18 = less KV to attend; FP8 KV) at 28 / 24 GiB reserved. The 24 GB class pays for the INT4 dequant: 0.69x the 32 GB class at 832 (7.4 fps) and 0.59x at 640 (12.5 fps) — the packed GPTQ-4 kernel (10/13: no speed-up at our token counts) is a memory trick only; 19.7 / 15.8 GiB reserved leaves ~4–8 GiB on a 24 GB card for the bf16 tiled decode. Neither capped arm reaches 16 fps at 832; at 640 the 32 GB class does (21 fps) and the 24 GB class does not (12.5). TTFF / e2e are n/a for the capped arms (no timed side-stream decode in those runs; the 10/13 harness measured the bf16 2x2 tiled decode at +1.5 GiB). The 10/13 "11 fps at 24 GB" was the in-process GPTQ at window 12 on the fit24 harness; this is the frozen-protocol number. GPTQ-4 file: specwm_out/gptq4/14b_g128_832.pt (calibration on 2 non-benchmark StEvo scenes, 560 layers, 0 non-finite; sentinel gen/sentinels/gptq4_save.ok) — the tier-9 mx24 / rt2c1hyb24 / mxQ4 chunks are now unblocked.


## 10/21 13:20 CDT — 60-s degradation x repair matrix scored, res-split baseline scored

- Queue tier 9 (48 chunks) complete; `mxW6` was degenerate (window 6 = sink only → modulo-by-zero) and became `mxW9` (sink + 1 chunk), PROTOCOL.md note 10/21.
- Table + readings: `data/matrix60_1021.md`, frames `figs/m60/montage_*.png`. Headline: the cloud re-anchoring recovers camera following (RPE rot 3–4x) and VBench on every repairable tier and keeps the mg2 scene for 60 s while every device-only tier (teacher included) drifts to sky by 20 s; at 832 the scene drift is the base model's own (teacher judge drift 86) and the cloud does not lower it. Window 9 is unusable and unrepairable (0/324 corrections applied: the corrected chunk is already out of the ring); the smallest repairable window is 12. n = 4 clips per cell — judge drift is noise-level between arms; need more clips + a scene-persistence metric.
- Res-split baseline scored (`data/ressplit_port.md`): rs576r (cloud 576 + 1.3B refiner) = full-res 2-step quality at 1.3x gen fps; bilinear-only rs576 unusable.
- Scoring now split over GPUs 0–3 + 5 (metrics_g0..g3; report.py `--metrics metrics,metrics_g7,metrics_g0,metrics_g1,metrics_g2,metrics_g3`); full report `data/report_1021.md`.
