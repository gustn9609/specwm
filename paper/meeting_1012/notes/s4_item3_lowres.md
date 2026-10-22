# Action item 3: Isolating the low-resolution loss

**Headline:** We removed everything on the device side and ran the 14B alone at native resolution (12-prompt × 3-seed, 2-step, separate commit). Over the full curve 832 / 704 / 576 / 448 / 384, VLM quality ↑ is 7.53 / 7.22 / 6.83 / 6.17 / 5.06 and VLM drift ↓ is 50.8 / 54.2 / 56.9 / 58.2 / 66.9. Every quality step down is significant (paired), and the steps grow below 576 (−0.67 to 448, −1.11 to 384); drift is clearly worse than 832 only at 384 (+16.1 ± 5.3 paired). At 448 the 14B ties the edge-only 1.3B student and at 384 it falls below it; the 14B stays clearly above the student only down to 704 (+0.67 ± 0.25), and 576 (+0.28 ± 0.25) is the last size before the steep drop. The realtime mode stays at 704 (15.7 fps quiet steady, generation only, 3.85× the teacher's 4.08 fps). Dropping to 576 (fused + partial) gives 25.7 fps quiet gen-only (17.1 fps whole-clip / ≈16.1 steady streaming on one GPU incl. decode). Its quality is not measured; the resolution step costs −0.39 ± 0.11 on native separate-commit output and, with fused commit, −0.50 ± 0.19 quality / +11.0 ± 3.5 drift (paired, 576 fused vs 704 fused). At 704 the old 1.3B refiner had cost about as much quality as the drop from 832 (paired −0.50 ± 0.11 vs −0.31 ± 0.10).

**Setup (same for every row unless noted)**
- Model: 14B causal-fast, 2 steps (σ 1.0 → 0.833), separate clean commit. No draft and no refiner.
- Output: the cloud's own latents, decoded at native resolution with the full Wan2.1 VAE.
- Prompt set: 12-prompt × 3-seed (cmp10 + jar + castle × seeds 1234/4321/9999), 36 clips per arm.
- Judge: Qwen3-VL-32B, 8 frames per clip at 448 px. It gives drift on a 0–100 scale (↓) and visual quality on a 1–10 scale (↑).
- Timing: steady seconds per chunk on one RTX PRO 6000 GPU, from the quiet timing pass (facts §15: jar + sandcastle, generation only, no decode) unless a column says otherwise.
- Why we ran this: "ours 704" (the realtime mode) runs below 832. We need to know how much of its gap to the teacher comes from resolution and how much from the pipeline around it.

## 1. The resolution curve: 14B alone (figure `f08_lowres_curve`)

| cloud resolution (14B 2-step, separate commit, native output) | fps ↑ quiet steady, gen only (timing pass) | fps ↑ earlier runs (README §N / §U, diag-run loop.json; basis per footnote, load not recorded) | VLM drift ↓ | VLM quality ↑ |
|---|---|---|---|---|
| 832×480 | <u>7.98</u> | 8.3¹ | **50.8 ± 5.0** | **7.53 ± 0.14** |
| 704 (≈704×384) | **12.50** | 12.7 | <u>54.2 ± 5.0</u> | <u>7.22 ± 0.15</u> |
| 576 (≈576×304) | n/a² | 21.5 | 56.9 ± 4.7 | 6.83 ± 0.14 |
| 448 (≈448×256) | n/a (not in the quiet timing pass) | <u>32.7</u>³ | 58.2 ± 4.8 | 6.17 ± 0.18 |
| 384 (≈384×224) | n/a (not in the quiet timing pass) | **48.0**³ | 66.9 ± 4.2 | 5.06 ± 0.25 |
| *reference: 14B 4-step teacher, 832* | *4.08* | *4.9 (quiet, whole-clip average⁴)* | *45.4 ± 5.2* | *7.64 ± 0.16* |
| *reference: 1.3B student 4-step, 832 (edge only)* | *–⁵* | *–⁵* | *58.6 ± 5.9* | *6.56 ± 0.25* |

Bold marks the best of the five 2-step resolution arms and underline the second, per column (in the earlier-runs column, ranked among the steady cloud values 12.7–48.0; 8.3 is on a different basis). The reference rows are not ranked. Values are mean ± SE over 36 clips. The 704–384 rows are labels `dg704_cloud` … `dg384_cloud` in `paper/1012/vlm_scores_diag.json`; the 832 row and both reference rows come from the 3-seed matrix on the same 12 prompts × 3 seeds (`vlm_scores_3seed.json`, labels `14B 2-step 832`, `teacher 14B 4-step 832`, `edge 1.3B 4-step 832`). Sizes are the nominal pixel budget (`lo_area` in each run's `loop.json`); the exact W×H follows the start image's aspect (facts §1), e.g. 448×240 for 33 of 36 clips in the 448 class and 560×304 for 21 of 36 clips in the 576 class (576×304 = `lo_area` 175104). Quiet fps (facts §15) is the median of `cloud_chunk_s[6:]` for the 2-step arms and of `diff(chunk_start)[5:]` for the teacher, on jar + sandcastle (5–6 clips per arm), not on the 12 VLM prompts. Where both fps columns are steady (704: 12.7 vs 12.50) they agree within 2 %; the 832 value 8.3 is a whole-clip average (footnote ¹) and is not directly comparable.

¹ Whole-clip average, generation only: 153 frames / 18.5 s (README §N, 10 prompts, seed 1234), the same basis as the teacher's 4.9, not steady. README §Q lists it as 8.3 with TAEHV and 5.8 with the full VAE (= 12 / (18.5/13 + 0.631)). The 704 and 576 values (12.7, 21.5) are README §U's 12 / max(cloud s/chunk, device + TAEHV), cloud-bound (0.95 / 0.56 s/chunk), from seed-1234 runs with the refiner.
² 576 with separate commit was not in the quiet timing pass. 576 fused + partial was: 0.468 s/chunk → 25.66 fps (section 6).
³ Diag-run `loop.json` (`lingbot-world-v2/specwm_out/fused/*/dg448_s*/loop.json`, `dg384_s*`, next to the videos of facts §4): median over 36 clips of each clip's median `cloud_chunk_s[6:]` = 0.367 s/chunk (448) and 0.250 s/chunk (384); cloud only, load not recorded. The same method on dg704 / dg576 gives 12.5 / 21.4 fps, matching 12.7 / 21.5 in this column.
⁴ 4.9 = 153 frames / 31.2 s generation for the whole clip (13 chunks, README §N), including the cheap early chunks before the KV window fills. On the steady basis the teacher runs at 4.08 fps quiet (3.9 under load on bench-43) (facts §15).
⁵ The student is not in the quiet timing pass (17.2 fps under load on bench-43, facts §2). Its 12×3 VLM scores come from an older run without torch.compile (facts §3).

**Paired steps along the curve** (same prompt and seed, 36 pairs; recomputed on CPU from `vlm_scores_diag.json` + `vlm_scores_3seed.json`; better / equal / worse clip counts in parentheses):

| resolution | Δ quality vs next-higher res ↑ | Δ drift vs next-higher res ↓ | Δ quality vs 1.3B student ↑ | Δ drift vs 1.3B student ↓ |
|---|---|---|---|---|
| 832 | – | – | **+0.97 ± 0.22** (23 / 11 / 2) | **−7.8 ± 5.0** |
| 704 | **−0.31 ± 0.10** (2 / 22 / 12) | +3.3 ± 3.0 | <u>+0.67 ± 0.25</u> (19 / 10 / 7) | <u>−4.4 ± 4.7</u> |
| 576 | <u>−0.39 ± 0.11</u> (1 / 23 / 12) | <u>+2.8 ± 4.5</u> | +0.28 ± 0.25 (13 / 13 / 10) | −1.7 ± 3.6 |
| 448 | −0.67 ± 0.14 (1 / 16 / 19) | **+1.2 ± 4.7** | −0.39 ± 0.25 (7 / 13 / 16) | −0.4 ± 4.0 |
| 384 | −1.11 ± 0.21 (2 / 10 / 24) | +8.8 ± 4.7 | −1.50 ± 0.22 (1 / 6 / 29) | +8.3 ± 4.6 |

Bold = best, underline = second per column. Paired means equal the differences of the unrounded means (e.g. 6.833 → 6.167 = −0.67).

- **Quality.** Every step down is significant (2.9–5.4 SE), and the steps grow below 576: −0.31, −0.39, then −0.67 and −1.11. The knee is at 576. Cumulative from 832 (paired): −0.69 ± 0.12 at 576, −1.36 ± 0.16 at 448, −2.47 ± 0.20 at 384 (worse on 36/36 clips).
- **Drift.** No adjacent step reaches 2 SE; the largest is 448 → 384 (+8.8 ± 4.7, 1.9 SE). Against 832 (paired) drift is +3.3 ± 3.0 at 704, +6.1 ± 4.6 at 576, +7.4 ± 4.1 at 448 and +16.1 ± 5.3 at 384, so only 384 is clearly worse.
- **Against the edge-only 1.3B student (same 36 clips).** The 14B's advantage shrinks with resolution: +0.97 at 832, +0.67 at 704 (both > 2 SE), +0.28 ± 0.25 at 576 (within noise), a tie at 448 (−0.39 ± 0.25, drift −0.4 ± 4.0) and a clear loss at 384 (−1.50 ± 0.22). Below 576 a 14B cloud adds nothing over running the 1.3B alone.
- This fits Guandao's 10/5 remark that below 400 px is outside the comfortable range.
- The 832 vs 704 gap also shows on bench-43, our real operating mode (43-prompt × 2-seed, 86 clips; fused + partial commit; bench-43 fps under load, quiet steady fps from the timing pass in brackets):
  - ours 832 (quality mode): 9.8 fps [9.74 quiet], quality 7.21 ± 0.08, drift 53.5 ± 3.0
  - ours 704 (realtime mode): 15.8 fps [15.73 quiet], quality 6.90 ± 0.08, drift 56.9 ± 2.7
  - Paired Δ quality vs the teacher: −0.30 ± 0.08 at 832 and −0.62 ± 0.09 at 704.
- Earlier partial evidence on jar/castle points the same way for 448:
  - 448 + Real-ESRGAN had 73 % of chunks flagged by the latent drift critic.
  - A person was injected into 1 of 4 clips.

## 2. Supported sizes

- The official `wan/configs` `SUPPORTED_SIZES` lists only 480×832 and 720×1280, in both orientations. This holds for `i2v-A14B` and for `i2v-1.3B`.
- 832×480 is the smallest supported size. Every low-res arm runs the 14B outside the official size list, including the 704 realtime mode.
- So the curve shows how the model behaves below its supported sizes; none of these sizes is an operating point the model was built for. A base model trained at low res could behave differently. We have not tested this.

## 3. Device-side upsampler (refiner / ESRGAN) vs resolution

Native and refined rows at the same resolution come from the same runs. The 832 reference rows are separate runs on the same 12 prompts × 3 seeds. 36 clips per arm. Scores from different judge inputs are not comparable, so best and second are ranked within each judge block.

| judge input | output | VLM drift ↓ | VLM quality ↑ |
|---|---|---|---|
| 448 px | 704 cloud, native | **54.2 ± 5.0** | **7.22 ± 0.15** |
| 448 px | 704 cloud + 1.3B refiner → 832 | 57.1 ± 4.8 | 6.72 ± 0.14 |
| 448 px | 576 cloud, native | <u>56.9 ± 4.7</u> | <u>6.83 ± 0.14</u> |
| 448 px | 576 cloud + 1.3B refiner → 832 | 60.3 ± 4.7 | 6.58 ± 0.18 |
| 832 px | 14B 4-step teacher, 832 | <u>43.2 ± 5.0</u> | **7.78 ± 0.14** |
| 832 px | 14B 2-step, 832 | **42.4 ± 4.9** | <u>7.22 ± 0.18</u> |
| 832 px | 704 cloud, bicubic upsample | 48.3 ± 5.2 | 7.11 ± 0.21 |
| 832 px | 704 cloud + 1.3B refiner → 832 | 53.5 ± 4.9 | 6.58 ± 0.23 |

- **Refiner cost, paired by clip.** At 704 the refiner costs −0.50 ± 0.11 quality: of 36 clips, 1 got better, 18 stayed equal and 17 got worse. At 576 it costs −0.25 ± 0.12. Drift is about 3 points higher with the refiner (54.2 → 57.1, 56.9 → 60.3; paired +2.9 ± 2.1 and +3.3 ± 3.0), within noise.
- **Refiner vs resolution.** At 704 the 1.3B refiner cost about as much quality as the drop from 832 did (paired: refiner −0.50 ± 0.11; 832→704 −0.31 ± 0.10; per-clip difference −0.19 ± 0.18, not significant). The other run of the same 704 + refiner config (3-seed matrix, label `704 cloud + refiner`, 6.89 ± 0.13) gives a refiner cost of −0.33 ± 0.13, the same size as the resolution cost. At 576, resolution is the larger loss: 832 → 576 native is −0.69 ± 0.12 (paired, 7.53 → 6.83), while the refiner adds −0.25 ± 0.12.
- **832-px judge.** Upsampling blur is visible at this input size. Even so, 704 with a plain bicubic upsample (7.11) is close to 14B 2-step at 832 (7.22). Both are well above 704 + refiner (6.58).
- **Real-ESRGAN.** The earlier realtime super-resolution also scored worse than bicubic in VLM quality.
- **Decision:** both learned upsamplers are retired. The device now does only the VAE decode and a bicubic upsample.
- **Decoder (`f09_decoder`).** A cheaper device-side decoder also costs quality. At 704, the tiny TAEHV decoder vs the full VAE (paired, 36 clips):
  - quality −0.56 ± 0.13 with the 448-px judge and −1.06 ± 0.14 with the 832-px judge
  - drift +7.4 and +11.1
  - All earlier VLM scores of the LingBot arms (teacher, student, SDVG, ours) were computed on full-VAE videos (Matrix-Game 2.0 and Waypoint use their own decoders), so TAEHV is not part of the loss above; it would add this cost if it were used.
- **Bottom line:** at 704 the refiner (−0.50 ± 0.11 paired) and the drop from 832 (−0.31 ± 0.10 paired) cost about the same. At 576 most of the loss is resolution (−0.69 ± 0.12 from 832), and the refiner adds −0.25 ± 0.12. Below 576 the 14B's native output degrades steeply (6.17 at 448, 5.06 at 384): 448 ties the 1.3B student and 384 is below it.

## 4. More steps do not help at low res

- More denoising steps at 576 do not reduce drift. At 576, 2, 3 and 4 steps all flag 65–77 % of chunks; at 832 the range is 40–55 %.
  - Data: jar/castle × 3 seeds, scored by the latent drift critic.
  - The critic is applied after Real-ESRGAN → 832, a route that the degradation control showed adds no domain shift.
- Resolution, not step count, sets the drift level.
- Early-anchor test (README §P): we generated the first K chunks at 832 and then switched the cloud to 576. Anchoring the first 1–3 chunks at 832 did not carry over. DINO min rises roughly in proportion to the 832 share (0.113 at K = 0 → 0.194 at K = 5). VLM drift 57 / 64 / 58 / 60 / 51 and quality 6.8 / 6.7 / 6.8 / 6.6 / 7.0 for K = 0 / 1 / 2 / 3 / 5 (576 + pixel refiner, 12 prompts, single seed): no consistent gain for K = 1–3; only K = 5 (38 % of the clip at 832) is better.
- Adaptive 576→832 switching turned into scene routing, and its gains were proportional to the time spent at 832.
- Conclusion: no change to the sampler, the step count or the schedule at inference recovers the low-res drift.

## 5. Post-training at 576 failed

- Resolution distillation: a LoRA on the 14B at 576, trained with an x0 loss on 144 downscaled 832 2-step rollouts (36 scenes × 4 seeds); evaluated as 576 cloud + LoRA + pixel refiner on 12 prompts, seed 1234 (base 58 drift / 6.8 quality; README §S).
  - With σ = 1 in the loss: drift 68–86 and collapse; the few-step generator collapses and tabletop scenes turn flat grey.
  - With σ 0.833 only: drift 60–62 vs base 58, i.e. no change. The model stays intact, but drift does not move.
- RFT and Diffusion-DPO: we kept the best of 4 candidates scored by the drift head, then ran LoRA self-distillation toward them; DPO ran on the same candidates (README N3). RFT did not reduce drift; DPO diverged (twice).
- Conclusion: post-training on generated data does not fix low-res drift in this model. Training on real video at low res, or a base model that is robust there, would be the next step. We have tried neither.

## 6. What this means for the realtime mode (`f05_pareto_bench43`, `f02_partial_commit`, `f13_latency`)

Speed of the three fused + partial operating points (14B 2-step) against the teacher. Timing: quiet timing pass, jar + sandcastle, one GPU (facts §13, §15; raw `lingbot-world-v2/specwm_out/timing_pass/`). Quality: bench-43 (86 clips).

| config | steady s/chunk ↓ (gen only) | fps ↑ (gen only) | speed-up vs teacher ↑ (quiet steady) | streaming output fps ↑, whole-clip (full-VAE decode on the same GPU) | streaming steady fps ↑ (s/chunk) | action→frame p50 ↓ | frames > 750 ms ↓ | bench-43 quality ↑ / drift ↓ |
|---|---|---|---|---|---|---|---|---|
| ours quality mode (832 fused + partial) | 1.232 | 9.74 | 2.39× | 6.9 | 6.4 (1.877) | 1995 ms | 100 % | **7.21** / **53.5** |
| ours realtime mode (704 fused + partial) | <u>0.763</u> | <u>15.73</u> | <u>3.85×</u> | <u>10.8</u> | <u>10.0 (1.203)</u> | <u>1296 ms</u> | <u>96.7 %</u> | <u>6.90</u> / <u>56.9</u> |
| 576 fused + partial | **0.468** | **25.66** | **6.28×** | **17.1** | **16.1 (0.747)** | **818 ms** | **60 %** | not scored |
| *reference: teacher 14B 4-step* | *2.939* | *4.08* | *1.00×* | *4.2⁶* | *3.80 (3.155)⁶* | *3237 ms⁶* | *100 %⁶* | *7.51 / 51.0* |

Bold = best, underline = second per column; the teacher row is not ranked. Streaming columns follow Chenfeng's method (`scripts/wm_latency.py`): output fps is a whole-clip mean, (frames − 1) / (last − first ready), and runs above steady (facts §13), so compare it only with other whole-clip values; action→frame = frame ready − start of its chunk. Streaming steady fps uses the facts §13 steady method (`scripts/e_steady_fps.py`: per clip, median of the diff of each chunk's last-frame ready time [5:], median over clips) on `timing_pass/*/{ours832,ours704,c576_pc1}_s*/loop.json`; 16 fps playback = 0.750 s/chunk. Streaming n = 5 clips for the 832 / 704 modes (sandcastle s1234 excluded: cold recompile) and 6 for 576 (warmed per task). Raw: `timing_pass/latency/{ours832,ours704,c576_pc1}/summary.json`.
⁶ Teacher streaming values come from the optimized target-only row of the latency profile (facts §13; same protocol, separate run); its steady streaming rate (3.80 fps, 3.155 s/chunk) is from `data/e_steady_fps.json`.

- **Full curve → 704.** From 832 to 704 the 14B loses −0.31 ± 0.10 quality, from 704 to 576 another −0.39 ± 0.11, and below 576 the steps grow to −0.67 and −1.11 while the 14B falls to the 1.3B student's level (448) and below it (384). 448 and 384 are not in the quiet timing pass (their diag runs logged 0.367 / 0.250 s/chunk cloud-only, load not recorded); at those sizes a cloud 14B gives no quality over running the 1.3B alone on the device. The usable range is 832 → 576.
- **Why 704 and not 576.** 704 is the lowest resolution where the 14B is still clearly above the student (+0.67 ± 0.25 paired); at 576 the margin is within noise (+0.28 ± 0.25). The realtime mode therefore keeps the cloud at 704: 15.73 fps quiet, 3.85× the teacher; 6.90 quality / 56.9 drift on bench-43. Its 0.763 s/chunk is slightly longer than the 0.750 s a 12-frame chunk plays at 16 fps, so it is just below real time.
- **What 576 would buy.** 576 fused + partial runs 1.6× faster than the realtime mode (25.66 vs 15.73 fps, generation only). Of the configurations streamed with this protocol (facts §13, §15), it is the only one at or above the 16 fps playback rate on one GPU with the full VAE: 17.1 fps whole-clip, 0.747 s/chunk ≈ 16.1 fps steady, i.e. right at real time with no margin. Its action→frame p50 is 818 ms, and 40 % of its frames meet the 750 ms deadline (the realtime mode: 3.3 %). The cost is the 704 → 576 resolution step (−0.39 ± 0.11 quality on native separate-commit output); with fused commit, 576 fused vs 704 fused costs −0.50 ± 0.19 quality and +11.0 ± 3.5 drift (paired, 36 clips, `fc576_cloud` − `fc704_cloud` in `vlm_scores_diag.json`; the drift cost is 3.2 SE), and 576 fused has 18 % boundary jumps. 576 fused + partial itself has no VLM score and no boundary-jump count.
- **Lower resolution makes the fused-commit jumps worse.** Share of boundaries with a jump ratio above 1.5× (fused commit, no partial commit, 12-prompt × 3-seed):
  - 832: 7 % (seed 1234 only)
  - 704: 12 %
  - 672: 12 %
  - 576: 18 %
  - Partial commit brings 704 back to 4 % (separate commit 3 %); at 576 the effect of partial commit on jumps was not measured.
- 576 fused without partial commit runs at 0.375 s/chunk → 32.0 fps (fc576 `loop.json`, load not recorded; corrected from "≈ 29", facts §5), but at drift 60.3, quality 6.58 and 18 % jumps.

### Caveats
- **Below 576 the curve has only two points.** 448 and 384 show the steep fall (6.17 and 5.06) and the drift jump at 384 (66.9), but neither is in the quiet timing pass (diag-run cloud times 0.367 / 0.250 s/chunk, load not recorded), and there is no point between 576 and 448.
- **Drift differences on the curve are small down to 448.** No adjacent drift step reaches 2 SE (paired); against 832 only 384 is clearly worse (+16.1 ± 5.3). Quality is the metric that separates the resolutions.
- **The judge may favour low res.** The VLM quality prompt has not been validated against humans; the drift prompt has (AUROC 0.82). The judge sees 8 frames at 448 px, which hides upsampling blur. On the resolution curve, only 832 and 704 were re-judged at 832 px. 576 (separate), 448 and 384 were judged at 448 px only. (576 fused was re-judged: 6.28 ± 0.18.)
- **The 1.3B student on this set scores high.** On 12×3 the student (older run without compile) gets 6.56 quality; ours realtime mode (704 fused + partial, 6.86 on the same 36 clips) is only +0.31 ± 0.28 above it, paired. On bench-43 (86 clips) the gap is large (6.90 vs 5.83). The "ties the student at 448" reading uses the 12×3 student and may understate the 14B's margin on harder prompts.
- **The step-count result is narrow.** It covers jar/castle only, with 6 clips per arm, scored by a latent critic trained at 832. On the 5 easy prompts, the extra drift at 576 did not show by eye, so the penalty depends on the scene.
- **Timing comes from two prompts.** The quiet timing pass ran jar + sandcastle only (5–6 clips per arm), not the 12 VLM prompts; cost depends on the token count, so this matters little, but the exact W×H differs by start image. 576 separate, 448 and 384 are not in it; the 576 separate value (21.5) is README §U's 12 / max(cloud s/chunk, device + TAEHV) = 12 / 0.56 s cloud (seed-1234 run with the refiner, load not recorded), and 448 / 384 have only diag-run cloud times (0.367 / 0.250 s/chunk, load not recorded). The teacher's often-quoted 4.9 fps is a whole-clip average; on the steady basis it is 4.08 quiet / 3.9 under load (facts §15). The 576 fused + partial speed has no quality measurement behind it yet.
- **Gen-only vs streaming.** The gen-only fps counts the cloud GPU only (decode assumed on a separate device). Streamed on one GPU with the full-VAE decode competing, the output rates are 6.9 / 10.8 / 17.1 fps for 832 / 704 / 576 as whole-clip output fps, or 6.4 / 10.0 / 16.1 fps steady (1.877 / 1.203 / 0.747 s/chunk; teacher 3.80). Only 576 reaches 16 fps, with no margin. With the decode on a separate device the cloud rate would be the bound; this was not measured end to end.
- **The post-training negatives are narrow.** They cover only fine-tunes on generated data (distillation, RFT; DPO diverged), evaluated on a single seed. Training on real video at low res is untested.
- **"Unsupported" only means the size is not on the official list.** We do not know which resolutions LingBot-World v2 saw during training.

### 발표 메모 (Korean)
- 디바이스 쪽을 다 빼고 14B만 native 해상도로 돌린 곡선입니다(12 prompt × 3 seed, 2-step). 832 / 704 / 576 / 448 / 384에서 품질은 7.53 / 7.22 / 6.83 / 6.17 / 5.06, 드리프트는 50.8 / 54.2 / 56.9 / 58.2 / 66.9입니다.
- paired로 보면 품질은 한 단계 내려갈 때마다 모두 유의하게 떨어지고(−0.31, −0.39, −0.67, −1.11), 576 아래에서 낙폭이 확 커집니다. 드리프트는 832 대비 384에서만 확실히 나빠집니다(+16.1 ± 5.3). Guandao 교수님이 10/5에 말씀하신 대로 400 px 아래는 편한 범위 밖입니다.
- 같은 36 clip에서 1.3B student와 비교하면, 14B의 우위가 해상도가 내려갈수록 줄어듭니다: 832 +0.97, 704 +0.67(유의), 576 +0.28(노이즈 안), 448은 동률, 384는 student보다 나쁩니다(−1.50). 14B가 student보다 확실히 나은 건 704까지이고(+0.67 ± 0.25), 576(+0.28 ± 0.25)은 급락 직전 마지막 크기입니다. 448 이하에서는 14B 클라우드를 쓸 이유가 없습니다. 단, 이 세트의 student 점수가 높게 나온 편이라(6.56; bench-43에서는 5.83) 이 비교는 보수적입니다.
- 그래서 realtime 모드는 704로 유지합니다: quiet steady 15.73 fps(생성만, teacher 4.08 fps 대비 3.85×), bench-43 품질 6.90 / 드리프트 56.9. teacher의 '4.9 fps'는 clip 전체 평균이라 steady 기준으로는 4.08입니다.
- 576 fused + partial은 quiet steady 25.66 fps(생성만, teacher 4.08 fps 대비 6.28×)입니다. 한 GPU에서 full VAE 디코드까지 스트리밍하면 clip 전체 평균 17.1 fps, steady 기준으로는 0.747 s/chunk ≈ 16.1 fps라서 16 fps 재생 속도에 딱 걸치고 여유는 없습니다(action→frame p50 818 ms). 같은 프로토콜로 스트리밍한 구성 중 16 fps에 닿는 건 이것뿐입니다(steady: 832 6.4, 704 10.0, teacher 3.80).
- 576의 품질 비용: −0.39 ± 0.11은 native separate-commit 출력에서 704→576 한 단계 손실입니다. fused commit 기준(576 fused vs 704 fused, paired 36 clip)으로는 품질 −0.50 ± 0.19, 드리프트 +11.0 ± 3.5(유의, 3.2 SE)이고, 576 fused는 boundary jump도 18 %입니다. 576 fused + partial 자체의 VLM 점수와 jump 수는 아직 없습니다. "한 GPU에서 디코드 포함 실시간"이 꼭 필요하면 576이 선택지이지만, 드리프트 비용이 크다는 점을 같이 말해야 합니다.
- 704에서는 예전 1.3B 리파이너 손실(paired −0.50 ± 0.11)이 832→704 해상도 손실(−0.31 ± 0.10)과 비슷했습니다. 576에서는 해상도 손실(832 대비 −0.69 ± 0.12)이 더 크고 리파이너가 −0.25를 더합니다. 리파이너와 ESRGAN은 은퇴했고, 디바이스는 full VAE 디코드 + bicubic만 합니다. TAEHV를 쓰면 −0.56(448 judge) / −1.06(832 judge)이 추가로 깎입니다.
- 576에서는 스텝을 늘려도, 포스트트레이닝(distillation, RFT)을 해도 드리프트가 줄지 않았고(DPO는 두 번 발산), 처음 1–3 청크를 832로 앵커해도 효과가 이어지지 않았습니다. 공식 지원 해상도는 480×832와 720×1280뿐이라 704 이하는 모두 지원 범위 밖입니다.
