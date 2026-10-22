# Overview: what changed since 10/5

**Headline:** We now have three operating modes of the 14B. Speed comes from the quiet timing pass (steady s/chunk, jar + sandcastle, no other generation jobs; facts.md §15). Quality comes from bench-43 (86 clips per arm, paired vs the teacher; §12). Against the teacher's **4.08 fps quiet steady** (2.939 s/chunk):
- **Teacher-parity mode: 1.67× at teacher-level quality** (Δ quality −0.07 ± 0.06, Δ drift −3.3 ± 3.1).
- **Quality mode: 2.39× at −0.30** (± 0.08).
- **Realtime mode: 3.85× at −0.62** (± 0.09). That is Matrix-Game 2.0's quality with **13 points less VLM drift** (paired ours − MG2: Δ quality −0.01 ± 0.10, Δ drift −13.1 ± 3.6).

The bench-43 under-load ratios (1.74× / 2.5× / 4.1×) agree with these within ~5 %. With Chenfeng's per-frame method (one GPU, decode streamed on the same GPU), **the realtime mode has the lowest action→frame p50 among the teacher, student, SDVG and our modes**: 1296 ms, vs student 1434, teacher 3237 and SDVG 4156–4463 ms. No arm meets the 750 ms per-frame budget. On **StEvo-38** (38 tasks, 1 seed, local Qwen3-VL-32B judge; not comparable to the paper's Gemini leaderboard; differences < ~0.15 not significant), the 14B-based arms, ours included, are flagged for physical inaccuracy less often (0.55–0.71) than the student (1.00; paired p < 0.001 for every 14B arm). Against Waypoint (0.92) the teacher and both our modes separate (paired p ≤ 0.035); SDVG is borderline (p = 0.057). Against Matrix-Game 2.0 (0.79) only the quality mode (0.55) separates (p = 0.035). **State evolution is at the floor for every arm (0–0.08).** SDVG at the paper τ (−0.7) remains slower than the teacher in every measurement.

| operating mode | config | quiet steady s/chunk ↓ | quiet fps ↑ | speed-up vs teacher, quiet steady ↑ | speed-up vs teacher, bench-43 under load ↑ | Δ quality vs teacher (bench-43, paired 86) ↑ | Δ drift vs teacher (bench-43, paired 86) ↓ |
|---|---|---|---|---|---|---|---|
| teacher 14B 4-step (reference) | official 4-step schedule (uniform4) | 2.939 | 4.08 | 1.00× | 1.00× | – | – |
| **ours: teacher-parity mode** | chunk-wise adaptive steps 0.7 (learned 14B head) + safe commit, teacher size (cw07_safe; 3.10 steps quiet, 3.01 on bench-43) | 1.764 | 6.80 | 1.67× | 1.74× | **−0.07 ± 0.06** | **−3.3 ± 3.1** |
| **ours: quality mode** | 832, 2 steps, fused + partial commit (ours832) | <u>1.232</u> | <u>9.74</u> | <u>2.39×</u> | <u>2.5×</u> | <u>−0.30 ± 0.08</u> | <u>+2.5 ± 3.0</u> |
| **ours: realtime mode** | 704, 2 steps, fused + partial commit (ours704) | **0.763** | **15.73** | **3.85×** | **4.1×** | −0.62 ± 0.09 | +5.9 ± 3.4 |

Bold = best, underline = second, among the three modes. Timing comes from facts.md §15: jar + sandcastle, generation only (no decode), one RTX PRO 6000 Blackwell per lane, no other generation jobs (VLM judges ran on other GPUs). The teacher-parity mode is timed on the teacher's basis, the median of diff(chunk_start)[5:]. The quality and realtime modes are cloud-only: median cloud_chunk_s[6:], with the device step excluded (see Caveats, "fps basis"). The under-load column is listed fps ÷ 3.9 (§12); raw s/chunk ratios are 1.72× / 2.48× / 3.99×. Quality and drift come from paper/1012/bench43_summary.json (keys cw07_safe, ours832, ours704; §12). The teacher-parity Δs are both within ~1.5 SE of zero: tied with the teacher, not better.

**Realtime mode vs Matrix-Game 2.0** (bench-43, 86 pairs, same task and seed, warm-up clip excluded; §2):
- Same quality: Δ −0.01 ± 0.10 paired; Δ vs teacher −0.62 vs MG2 −0.60.
- Less drift: Δ −13.1 ± 3.6 paired; 56.9 ± 2.7 vs 70.0 ± 3.0.
- MG2 is still faster. Quiet: 19.7 fps with its own decoder vs ours 15.73 cloud-only (§15). Bench-43 under load: 20.0 vs 15.8.

Figures for this slide:
- **f11_modes** (re-rendered 08:00 CDT): the three modes vs the teacher. Its title, mode labels and table lead with the quiet steady ratios 1.67× / 2.39× / 3.85× (table: quiet fps vs the teacher's 4.08, facts §15) and give the bench-43 under-load ratios 1.74× / 2.5× / 4.1× in parentheses; panel a's axis is bench-43 fps under load. The footer gives the 4.9 correction.
- **f13_latency**: per-frame latency (Chenfeng's method) on jar + sandcastle, one GPU, full-VAE decode streamed on the same GPU. It shows the 8 profile rows (draft/student, target/teacher and SDVG τ −0.7 / −2.2 on the original and optimized stacks), our realtime and quality modes (from the timing pass, a separate run, n = 5), and 4 lighter rows: variants we do not ship (704 separate 1484 ms and 832 separate 2229 ms p50, plus the two below). The realtime mode has the lowest p50 of all measured baselines and modes. Two non-mode variants are lower: 704 fused without partial commit (1180 ms, 12 % boundary jumps) and 576 fused + partial (818 ms, lower resolution). The teacher-parity mode has no row (`spec_early` has no streaming decode). The SDVG claim is scoped to jar + sandcastle. On bench-43, τ −2.2 is faster than the teacher: 6.7 vs 3.9 fps on the same steady basis, under load. The table's 6.4 is a whole-clip mean (facts §2).
- **f01_pipelines**: our recipe vs SDVG.
- **f05_pareto_bench43**: quality vs fps for all arms, under load.
- **f12_stevo**: StEvo-38 physical inaccuracy (Wilson 95 % CI, paired McNemar in the footer); full table data/f12_stevo_table.md.

**Latency, Chenfeng's method** (facts.md §13). Jar + sandcastle, one GPU per run. Every committed chunk is VAE-decoded right away on the same GPU, so decode competes with generation. Prepare is excluded.

| config (optimized stack) | output fps ↑ | action→frame p50 (ms) ↓ | p99 (ms) ↓ | miss > 750 ms ↓ | first frame after gen start ↓ | gen s / 10 s clip ↓ |
|---|---|---|---|---|---|---|
| teacher (14B 4-step, target only) | 4.2 | 3237 | 3810 | 100 % | 1714 ms | 37.8 |
| student (1.3B 4-step, draft only) | <u>10.4</u> | <u>1434</u> | <u>1700</u> | **96.7 %** | **439 ms** | <u>14.6</u> |
| SDVG τ −0.7 | 3.0 | 4463 | 4946 | 100 % | 1794 ms | 51.2 |
| SDVG τ −2.2 | 3.7 | 4156 | 4947 | 100 % | 1791 ms | 43.3 |
| ours: quality mode (832) | 6.9 | 1995 | 2310 | 100 % | 852 ms | 22.4 |
| ours: realtime mode (704) | **10.8** | **1296** | **1495** | **96.7 %** | <u>574 ms</u> | **14.3** |

Bold = best, underline = second; ties share the mark. No underline in the miss column, because all other rows tie at 100 %. Output fps is a whole-clip mean, (frames − 1) / (last − first ready), so it is not steady fps. Do not compare it with the quiet fps above.

Clip counts and runs differ:
- Profile rows: 6 clips (jar + castle × 3 seeds) from sdvg_profile/latency/.
- Ours rows: 5 clips from the timing pass (timing_pass/latency/). Castle s1234 is excluded because it paid a cold torch.compile.

Missing and lower rows:
- **Teacher-parity mode:** no row. spec_early has no streaming decode, so it records no per-frame ready times.
- **MG2 and Waypoint:** not profiled with this method.
- **Two ablations we do not ship go lower:** 704 fused without partial (1180 ms; it has boundary jumps) and 576 fused + partial (818 ms; quality loss at 576) (§15).

**StEvo-Bench subset** (facts.md §14). Setup: 38 camera-WM tasks, seed 1234, 249-frame videos (Waypoint 256) of the 253-frame look-away-and-back path, local Qwen3-VL-32B judge (3-vote majority, 32 frames; not comparable to the paper's Gemini leaderboard). Values are fractions of the 38 tasks. The binomial SE is ≈ 0.05–0.08, so differences below ~0.15 are not significant.

| arm | physical inaccuracy ↓ | control success ↑ | state evolution ↑ | task success ↑ |
|---|---|---|---|---|
| teacher 14B 4-step | <u>0.68</u> | <u>0.13</u> | **0.08** | **0.08** |
| student 1.3B 4-step | 1.00 | <u>0.13</u> | 0.00 | 0.00 |
| SDVG + actions τ −0.7 | 0.71 | 0.11 | 0.00 | 0.00 |
| Matrix-Game 2.0 | 0.79 | **0.24** | 0.00 | 0.00 |
| Waypoint-1.5-1B | 0.92 | 0.00 | 0.03 | 0.00 |
| ours: quality mode (832) | **0.55** | 0.08 | <u>0.05</u> | <u>0.05</u> |
| ours: realtime mode (704) | <u>0.68</u> | 0.08 | <u>0.05</u> | 0.00 |

Bold = best, underline = second; ties share the mark.

Reading:
- **Physical inaccuracy is the informative column.**
  - All 14B-based arms (0.55–0.71) are flagged less often than the student (1.00; paired p < 0.001).
  - Against Waypoint (0.92): the teacher, ours 832 and ours 704 separate (paired p ≤ 0.035); SDVG is borderline (p = 0.057).
  - Against MG2 (0.79), only ours 832 (0.55) separates (p = 0.035). The teacher, the realtime mode (0.68) and SDVG (0.71) are lower in point estimate only (p = 0.39 / 0.42 / 0.58).
  - Paired p-values: exact McNemar on the same 38 tasks (data/f12_stevo.json; facts.md §14; f12_stevo footer).
- **State evolution and task success are at the floor for every arm** (0–0.08), as in the 10/7 85-task run (1–5 %). This judge and protocol do not separate the arms there.
- MG2's high control success comes from turning away visibly (occlusion done 0.97) while rarely showing the prompted action (trigger applied 0.24).
- The teacher-parity mode was not run on StEvo.

Source: lingbot-world-v2/specwm_out/stevo38/eval_runs/lingbot_b38_<arm>/summary.json.

## 10/5 → 10/12 at a glance

| topic | what we said on 10/5 | what we know on 10/12 |
|---|---|---|
| SDVG on LingBot | 99.9 s per clip. The original code ran bf16 eager with T5 on CPU (jar/castle × 3 seeds). 1 of the 12 non-forced chunks was accepted. Target-only took 83.3 s in the same accounting (prepare included); the "≈ 73 s" quoted earlier is the same run without the 11.3 s prepare (generation 64.4 + decode 8.6, paper/1005/step2_lingbot_imagereward/README.md). | We re-ran SDVG on the same optimized stack as every arm. On bench-43 (under load), at the paper τ (−0.7) it is slower than the teacher: 3.5 vs 3.9 fps, or 3.1 fps on the teacher's steady basis. At τ −2.2 it reaches 6.4 fps but loses −0.84 ± 0.10 quality and +19.7 ± 3.4 drift. At τ −0.7, on average 1.65 of 13 chunks come from the draft (median 1; 42/86 clips take none; chunk 0 is always forced to the target). In the per-frame latency profile (same stack, jar/castle) both thresholds are slower than target-only on average over 6 clips: 51.2 / 43.3 vs 37.8 s per clip, and p50 4463 / 4156 vs 3237 ms (τ −0.7 slower on 6/6 clips; τ −2.2 faster on 1/6 per stack, only where ≥ 7 of 13 chunks came from the draft). (f07_sdvg_analysis, f13_latency) |
| Low resolution | "doesn't work" (SDVG / latent-critic loop at 576 / 448 / 384 + Real-ESRGAN; collapse at 384) | In the 10/9–10/11 cloud + refiner path, the 1.3B refiner cost −0.50 ± 0.11 quality at 704 and −0.25 ± 0.12 at 576 (paired, 36 clips), so the device now only decodes. The 14B alone still loses quality as resolution drops (7.53 / 7.22 / 6.83 at 832 / 704 / 576): quality falls steeply below 576 (6.17 at 448, 5.06 at 384) and drift jumps at 384 (66.9), consistent with the 10/5 384 failure coming from the low-res 14B itself. (f08_lowres_curve) |
| Teacher-quality speed-up | none: SDVG was slower than the target alone | **Ours: teacher-parity mode** (chunk-wise adaptive steps 0.7 + safe commit): **6.80 vs 4.08 fps quiet steady (1.67×, same basis as the teacher)**; 1.74× under load on bench-43. Δ quality −0.07 ± 0.06, Δ drift −3.3 ± 3.1 (paired, 86). (f11_modes) |
| Realtime path | none | **Ours: realtime mode** = 704 cloud, 2-step, fused + partial commit; the device does decode + bicubic upsample only. **15.73 fps quiet** (0.763 s/chunk cloud-only, 3.85×) and 15.8 fps under load on bench-43. On one GPU with the full-VAE decode streamed alongside: 10.8 fps output, action→frame p50 1296 ms. (f11_modes, f01_pipelines, f04_speed_bridge, f13_latency) |
| Latency | seconds per 10 s clip only | Chenfeng's per-frame method (output fps, action→frame p50/p90/p99, miss > 750 ms, first frame) for the teacher, student, SDVG (original and optimized stacks) and our quality and realtime modes. No configuration meets the 750 ms per-frame budget. (f13_latency) |
| Evaluation | jar/castle, seconds per 10 s clip | Bench-43: 38 StEvo camera-WM tasks + 5 easy scenes, × 2 seeds = 86 clips per arm. Metrics: VLM drift + quality, paired Δ vs the teacher, and a chunk-boundary jump metric. **StEvo-38** (253-frame protocol, 7 arms) done: physical inaccuracy separates the 14B-based arms from the student and from Waypoint (except SDVG, borderline); against MG2 only the quality mode (0.55 vs 0.79) clears the noise. State evolution is at the floor (0–0.08). Local Qwen3-VL-32B judge, 1 seed, n = 38. (f12_stevo) |
| Baselines | Waypoint planned | Matrix-Game 2.0 and Waypoint-1.5-1B ran on bench-43 and StEvo-38 with the same start images and camera paths. They also ran in the quiet timing pass (MG2 19.7 fps, Waypoint 31.2 fps, each with its own decoder). |

## The recipe that works (cloud does the model, device only decodes)
- **Target acceleration (cloud 14B):**
  - Every LingBot arm uses the same shared stack: FP8 W8A8 linears (14B only) + SageAttention 2 + torch.compile + ring KV.
  - **2 steps** instead of 4 (indices 0/500, σ 1.0 → 0.833).
  - **Fused commit:** reuse the K/V from the last denoising forward, which saves one full forward per chunk.
  - **Partial commit:** a clean t = 0 pass over only the newest latent frame (≈ 1/3 forward). It removes the chunk-boundary jumps that fused commit causes. In the quiet pass it costs ≈ 0.12 s/chunk at 704 on the same task: jar 0.641 → 0.764, sandcastle 0.620 → 0.737 s/chunk (per-task medians, timing_pass/<task>/{c704_fused,ours704}_s*/loop.json cloud_chunk_s[6:]). The §15 table values 0.630 vs 0.763 are medians over different clip sets (6 clips vs 5, castle s1234 excluded for ours704), so their 0.13 gap slightly overstates the cost. (f02_partial_commit, f02b_jump_frames)
  - **Chunk-wise adaptive steps** (teacher-parity mode only): run the official 4-step schedule. After each step, a learned head on 14B features (head14b_v3) predicts which tokens have converged. The whole chunk stops once ≥ 70 % are predicted converged (mean 3.01 steps on bench-43, 3.10 in the quiet pass).
  - **Safe commit** (teacher-parity mode): partial commit if the chunk stopped at step ≥ 3, a full clean pass if it stopped at step 1–2.
- **Low-res cloud:**
  - The 14B runs at max_area 270336 (≈ 704×384; the aspect follows the start image).
  - The device runs the VAE decode and a bicubic upsample. **No 1.3B model on the device.**
- **Three operating modes.** Format: quiet steady fps and speed-up vs the teacher's 4.08 fps (§15) [bench-43 under load, listed ratio]; Δ vs the teacher on bench-43 (paired, 86); f11_modes.
  - **Ours: teacher-parity mode** = chunk-wise adaptive 0.7 + safe commit at the teacher's size. 6.80 fps (1.764 s/chunk), 1.67× [6.8 fps, 1.74×]. Δ quality −0.07 ± 0.06, Δ drift −3.3 ± 3.1.
  - **Ours: quality mode** = 832, 2-step, fused + partial. 9.74 fps (1.232 s/chunk, cloud-only), 2.39× [9.8 fps, 2.5×]. Δ quality −0.30 ± 0.08, Δ drift +2.5 ± 3.0.
  - **Ours: realtime mode** = 704, 2-step, fused + partial. 15.73 fps (0.763 s/chunk, cloud-only), 3.85× [15.8 fps, 4.1×]. Δ quality −0.62 ± 0.09, Δ drift +5.9 ± 3.4.
- **Commit type and resolution both trade speed for quality.** Timing is quiet steady fps (§15); quality is from the 12-prompt × 3-seed matrix (§3):
  - 832 2-step: separate commit 7.98 fps / quality 7.53. Fused + partial (quality mode) 9.74 fps. Fused without partial scored 7.28 on that matrix.
  - 704: separate 12.50 fps / 7.22. Fused + partial 1: 15.73 fps / 6.86. Fused alone: 19.04 fps / 7.08, but with 12 % boundary jumps. On this set the realtime commit scores 6.86 vs 7.22 for 704 separate (unpaired, ~1.6 SE), a drop about as large as 832 → 704 (7.53 → 7.22).
  - Quality steps down with resolution: 7.53 at 832 → 7.22 at 704 → 6.83 at 576 → 6.17 at 448 → 5.06 at 384 (14B alone, native output). It falls slowly down to 576 and steeply below it; drift jumps at 384 (66.9).

## Closed since 10/5 (one line each)
- **SDVG + ImageReward verifier:**
  - At the paper τ (−0.7) it matches teacher quality (Δ +0.00 ± 0.07) but runs at 3.5 fps under load, below the teacher (3.9).
  - At τ −2.2 it reaches 6.4 fps but loses −0.84 ± 0.10 quality and +19.7 ± 3.4 drift.
  - Even if every draft chunk were accepted, the ceiling is ≈ 7.6 fps (≈ 20 s per clip). That is still 2.5× slower than the student alone (8.2 s per clip), at student quality (facts.md §2).
  - In the per-frame latency profile, both thresholds are slower than target-only on the same stack, on average over 6 clips: output fps 3.0 / 3.7 vs 4.2 (§13). τ −0.7 is slower on 6/6 clips; τ −2.2 is faster on 1/6 per stack, only where ≥ 7 of 13 chunks came from the draft.
- **Why the draft fails as a proposal:** 1.3B draft chunks are 0.63 σ from the 14B output, with 7 % of tokens within ε. The 14B's own step 1 is 0.35 σ away, with 36 % within ε.
- **Physics verifier (go/no-go → no-go):**
  - The verify step (decode + physics judge; judge-only time not logged) took 32–37 s per clip at τ = 5 (72–100 s at τ = 4, suspect). With it, no clip was faster than target-only generation (62.5–64.4 s, original stack; facts §7).
  - Off-the-shelf physics scorers saturated and could not separate the arms.
- **Draft-then-Finish:** quality 6.2, i.e. draft level (teacher 7.6). The finish step inherits the draft's world.
- **Verify + redraw with the 14B drift head** (AUROC 0.88 on unseen scenes): redrawn chunks are flagged again. No gain, and time ×2.
- **Adaptive 576 → 832 switch:** it degenerates into scene routing.
- **Multi-GPU splits:**
  - 2-GPU sequence parallel over PCIe gives ≤ 1.1×.
  - Commit offload hallucinated scenes in 6/6 clips.
- **Learned 1.3B refiner:** −0.50 quality at 704, so it was removed. Real-ESRGAN scored worse than bicubic.
- **Fixing low-res drift by training:**
  - Resolution distillation collapsed.
  - RFT/DPO had no effect.
  - More steps at 576 do not reduce drift.

## New evaluation
- **Bench-43:**
  - 38 StEvo camera-WM tasks + 5 easy scenes, × seeds 1234/4321 = 86 clips per arm. 161 frames requested, 153 generated (13 chunks, ≈ 9.6 s at 16 fps).
  - Judge: Qwen3-VL-32B on 8 frames per clip at 448 px.
  - The drift prompt (0–100, ↓) is validated against human severe-drift labels: AUROC 0.82 on 32 unseen scenes.
  - The quality prompt (1–10, ↑) is not validated against humans.
  - Paired Δ vs the teacher is computed on all 86 clips. (f06_paired_deltas)
- **Boundary-jump metric** (new 10/12):
  - Share of chunk boundaries where the frame-to-frame change is > 1.5× its neighbours.
  - Mid-chunk control positions give 0.7–3.7 % across the nine 12-prompt arms (facts.md §5).
  - Fused commit gives 12 % at 704 vs 3 % for 704 separate commit (36 clips each; teacher 832, seed 1234 only: 2 %). Partial commit 1 brings it to 4 %.
- **StEvo-Bench subset** (done 10/12; facts.md §14; 38 tasks, seed 1234, 7 arms, local Qwen3-VL-32B judge, not comparable to the paper's Gemini leaderboard; table above, f12_stevo):
  - Physical inaccuracy: ours quality 0.55, teacher 0.68, ours realtime 0.68, SDVG 0.71, MG2 0.79, Waypoint 0.92, student 1.00.
  - State evolution is 0–0.08 and task success 0–0.08 for every arm, i.e. at the floor.
  - SDVG τ −2.2 and the teacher-parity mode were not run.
- **Quiet timing pass** (done 10/12; facts.md §15): steady s/chunk, generation only, jar + sandcastle, no other generation jobs. It gives the speed-ups in the headline. The teacher is 2.939 s/chunk = 4.08 fps quiet, vs 3.9 under load on bench-43 (3 % apart).
- **Chenfeng-method latency profile** (done 10/12; facts.md §13): per-frame action→frame latency with streaming decode on one GPU. Covers teacher, student and SDVG (both stacks) plus our quality and realtime modes (table above).
- **Chunk-wise adaptive steps + commit policy on bench-43** (done 10/12; facts.md §12; 86 clips per arm). Format: bench-43 under-load fps with listed ratio [quiet steady ratio, §15], and better/equal/worse vs the teacher in parentheses.
  - **Separate commit (cw07_sep):** 2.91 steps, 5.9 fps (1.51× [1.45×]), Δ quality +0.03 ± 0.05 (12/65/9), Δ drift −0.6 ± 3.2. Chunk-wise adaptive keeps teacher quality on 43 prompts.
  - **Fused + partial (cw07_pc1):** 3.01 steps, 6.9 fps (1.77× [1.66×]), −0.09 ± 0.06 (10/59/17), −0.5 ± 3.1.
  - **Safe commit (cw07_safe) = ours: teacher-parity mode:** 3.01 steps, 6.8 fps (1.74× [1.67×]), −0.07 ± 0.06 (10/60/16), −3.3 ± 3.1, boundary jumps 1 %.
  - **Looser stop at 50 % (cw05_pc1):** 2.61 steps, 7.0 fps (1.79× [1.97×]), −0.17 ± 0.07 (9/57/20). Under load on bench-43 it barely saved wall time vs cw07_pc1 (1.722 vs 1.748 s/chunk; cause not diagnosed). In the quiet pass it did save time (1.495 vs 1.771 s/chunk). Either way it costs more quality, so it is not the teacher-parity choice.
  - **Commit policy:** fused/partial commit raises the speed-up from ≈ 1.5× to ≈ 1.7× (quiet 1.45× → 1.66× / 1.67×). Vs the teacher, the quality differences (−0.07 / −0.09) are within ~1.5 SE. Paired against cw07_sep, the commit policy costs −0.10 ± 0.06 / −0.13 ± 0.07 (borderline, < 2 SE; §12).
- **14B-alone 448 / 384** (36 clips each, 12 prompts × 3 seeds, 2-step, separate commit, native; quality / drift, mean ± SE): 448 (448×256) 6.17 ± 0.18 / 58.2 ± 4.8; 384 (384×224) 5.06 ± 0.25 / 66.9 ± 4.2.

## Headline table — bench-43 (38 StEvo tasks + 5 easy scenes × 2 seeds = 86 clips per arm; fps on one RTX PRO 6000 Blackwell)

| arm | fps, bench-43 (under load)¹ ↑ | fps, quiet steady (jar + castle; §15) ↑ | VLM quality ↑ | VLM drift ↓ | Δ quality vs teacher (paired, 86) ↑ | Δ drift vs teacher (paired, 86) ↓ | boundary jumps > 1.5× ↓ |
|---|---|---|---|---|---|---|---|
| teacher 14B 4-step | 3.9 | 4.08 | **7.51 ± 0.08** | <u>51.0 ± 3.0</u> | – | – | **1 %** |
| **ours: teacher-parity mode** (chunk-wise 0.7 + safe commit) | 6.8 | 6.80 | <u>7.44 ± 0.08</u> | **47.7 ± 3.2** | <u>−0.07 ± 0.06</u> | **−3.3 ± 3.1** | **1 %** |
| **ours: quality mode** (832 2-step, fused + partial) | 9.8 | 9.74 | 7.21 ± 0.08 | 53.5 ± 3.0 | −0.30 ± 0.08 | +2.5 ± 3.0 | <u>2 %</u> |
| **ours: realtime mode** (704 2-step, fused + partial) | 15.8 | 15.73 | 6.90 ± 0.08 | 56.9 ± 2.7 | −0.62 ± 0.09 | +5.9 ± 3.4 | <u>2 %</u> |
| SDVG + actions τ −0.7 (paper τ) | 3.5 | – | **7.51 ± 0.09** | 52.9 ± 3.0 | **+0.00 ± 0.07** | <u>+1.9 ± 3.0</u> | **1 %** |
| SDVG + actions τ −2.2 | 6.4 | – | 6.67 ± 0.09 | 70.6 ± 2.5 | −0.84 ± 0.10 | +19.7 ± 3.4 | 3 % |
| student 1.3B 4-step | 17.2 | – | 5.83 ± 0.16 | 75.1 ± 2.2 | −1.69 ± 0.17 | +24.1 ± 3.0 | <u>2 %</u> |
| Matrix-Game 2.0 (640×352) | <u>20.0</u> | <u>19.7</u> ² | 6.91 ± 0.07 | 70.0 ± 3.0 | −0.60 ± 0.09 | +19.0 ± 3.8 | n/a |
| Waypoint-1.5-1B (1024×512) | **25.4** | **31.2** ² | 4.03 ± 0.15 | 75.9 ± 1.6 | −3.48 ± 0.15 | +24.9 ± 3.0 | n/a |

Bold = best, underline = second; ties share the mark. The teacher-parity mode's drift and Δ drift marks are within ~1 SE of the teacher (47.7 vs 51.0; −3.3 ± 3.1), so they do not mean lower drift than the teacher. Sources: paper/1012/bench43_summary.json; quiet column facts.md §15 ("–" = not in the quiet pass).

¹ fps = 12 / seconds per chunk, but the basis differs by arm (scripts/bench43_report.py):
- SDVG: 12 / (gen_s / 13), a whole-clip mean that includes the forced target chunk 0. On the teacher's steady basis (bench-43 loop.json chunk_t, chunks ≥ 5; facts.md §2), SDVG is 3.1 fps (τ −0.7) and 6.7 fps (τ −2.2).
- Teacher, student and teacher-parity mode (chunk-wise arms): steady median, chunks ≥ 5.
- Quality and realtime modes: cloud-only median, chunks ≥ 6, no decode (1.225 / 0.761 s/chunk).
- MG2 and Waypoint: their own whole-clip fps, including their decoders.
- Basis-free alternative: generation seconds per clip, teacher 35.8 s vs SDVG τ −0.7 44.6 s.

² MG2 / Waypoint quiet = their own pipeline including their decoder (timing.json), not a steady per-chunk rate.

- **vs teacher:**
  - Teacher-parity mode: Δ quality −0.07 ± 0.06 (10 better / 60 equal / 16 worse) and Δ drift −3.3 ± 3.1, both within ~1.5 SE, at 1.67× quiet steady (1.74× under load; same fps basis as the teacher).
  - Quality mode (ours 832) loses 0.30 ± 0.08 quality, a clear loss (not within noise). Only its drift Δ (+2.5 ± 3.0) is within noise.
  - Realtime mode (ours 704) loses 0.62 ± 0.09 quality, with drift +5.9 ± 3.4.
- **vs Matrix-Game 2.0** (similar resolution, 640×352):
  - The realtime mode (ours 704) has the same VLM quality (6.90 ± 0.08 vs 6.91 ± 0.07) and lower VLM drift (56.9 ± 2.7 vs 70.0 ± 3.0). It runs at 15.8 vs 20.0 fps under load and 15.73 vs 19.7 quiet.
  - Paired ours 704 − MG2 on bench-43 (86 pairs, warm-up clip excluded; facts.md §2): Δ drift −13.1 ± 3.6, Δ quality −0.01 ± 0.10.
  - Older, weaker evidence from the 12-prompt × 3-seed set: ours 704 fused + partial 1 has drift 53.5 vs MG2 57.9, and quality 6.86 vs 7.06 (MG2 57.9 ± 4.8 / 7.06 ± 0.17, 36 clips, warm-up excluded; facts.md §3).
  - StEvo-38 physical inaccuracy: realtime mode 0.68 vs MG2 0.79. The 0.11 gap is within noise (< ~0.15; paired p = 0.42).
- **By group (quality / drift):**
  - Easy scenes (n = 10): ours 704 7.60/36, teacher 8.00/22, MG2 6.90/48.
  - Tabletop & physics (n = 76): ours 704 6.80/60, teacher 7.45/55, MG2 6.91/73. MG2 is slightly ahead on quality here; we are ahead on drift.

## Claims we can / cannot make yet

| ✅ can claim now (evidence) | ⛔ cannot claim yet (why) |
|---|---|
| **Three operating modes, quiet steady speed-ups vs the teacher's 4.08 fps** (facts.md §15; quality from bench-43, 86 paired clips; f11_modes): **teacher-parity 1.67× at Δ quality −0.07 ± 0.06 (teacher level); quality 2.39× at −0.30 ± 0.08; realtime 3.85× at −0.62 ± 0.09.** The bench-43 under-load ratios (1.74× / 2.5× / 4.1×) agree within ~5 %. | **Like-for-like speed-ups for the quality and realtime modes.** Their s/chunk is cloud-only (device step and decode excluded), while the teacher's is chunk-to-chunk. With the device step counted serially (bench-43, under load) the ratios are 2.25× / 3.44× (§12). The quiet timings cover only 2 prompts (jar + castle), and quality comes from bench-43. |
| **Teacher-parity mode matches the teacher within noise.** Δ quality −0.07 ± 0.06 (10/60/16 better/equal/worse), Δ drift −3.3 ± 3.1 (paired, 86), boundary jumps 1 % (teacher 1 %). It runs at 1.67× on the teacher's own fps basis. With separate commit, chunk-wise adaptive gives +0.03 ± 0.05 at 1.45× quiet. | **"Lower drift than the teacher"** for the teacher-parity mode: −3.3 ± 3.1 is within ~1 SE. |
| **Realtime mode ≈ Matrix-Game 2.0 quality with 13 points less drift.** On bench-43 (86 pairs; facts.md §2): paired Δ quality −0.01 ± 0.10, Δ drift −13.1 ± 3.6 (56.9 ± 2.7 vs 70.0 ± 3.0). | **Faster than MG2:** 15.73 vs 19.7 fps quiet, 15.8 vs 20.0 under load, and MG2 ran without torch.compile (official claim 25 fps on H100). **A drift advantage free of the action-conversion confound or confirmed by humans:** MG2 runs on camera paths we converted to keyboard/mouse inputs, and the drift judge is a VLM (AUROC 0.82 vs human labels). |
| **Lowest per-frame latency among the teacher, student, SDVG and our modes: the realtime mode** (Chenfeng's method, one GPU, decode streamed on the same GPU; facts.md §13). Action→frame p50 1296 ms and p99 1495 ms, vs student 1434 / 1700, teacher 3237 / 3810, SDVG 4156–4463 ms. Output fps 10.8 (student 10.4, teacher 4.2), gen 14.3 s per clip. The quality mode is also below the teacher: p50 1995 ms, 6.9 fps. | **Meeting a real-time per-frame budget:** no configuration meets 750 ms. The realtime mode misses 96.7 % of frames (p50 1.30 s). **Its 15.73 fps with decode on a separate device:** not measured end to end; on one GPU the decode competes and output is 10.8 fps. **Teacher-parity latency:** spec_early has no streaming decode. **Lowest latency vs MG2 / Waypoint:** they were not profiled with this method. Two unshipped ablations go lower (704 fused 1180 ms, 576 fused + partial 818 ms; §15). |
| **StEvo-38 physics** (facts.md §14, 38 tasks, seed 1234, local Qwen3-VL-32B judge; f12_stevo). All 14B-based arms (ours 0.55 / 0.68, teacher 0.68, SDVG 0.71) are flagged less often than the student (1.00; paired p < 0.001). Against Waypoint (0.92): teacher, ours 832 and ours 704 separate (p ≤ 0.035), SDVG is borderline (p = 0.057). Against MG2 (0.79) only ours 832 does (p = 0.035). | **StEvo state evolution / task success:** at the floor for every arm (0–0.08), so this judge and protocol do not separate the arms. **"Better physics than MG2" for the realtime mode or the teacher:** 0.68 vs 0.79 is within noise. **Teacher-parity mode on StEvo:** not run. **Leaderboard comparison:** local Qwen3-VL-32B judge, not Gemini. |
| **Speculative decoding with the 1.3B draft does not accelerate LingBot at teacher quality.** τ −0.7 runs at 3.5 fps under load (teacher 3.9) and has 3.0 vs 4.2 output fps in the latency profile. τ −2.2 reaches 6.4 fps but loses −0.84 ± 0.10 quality. Even accepting every draft chunk would cap at ≈ 7.6 fps. | **"Teacher parity at realtime":** the realtime mode is −0.62 ± 0.09 quality, and the teacher-parity mode runs at 6.80 fps, below the 16 fps playback rate. On the 12-prompt set no configuration reaches ≥ 20 fps and ≥ 7.2 quality. |
| **The device 1.3B refiner cost quality in the 10/9–10/11 cloud + refiner path** (−0.50 ± 0.11 at 704, −0.25 ± 0.12 at 576, paired, 36 clips), so the device now only decodes. The 14B alone still loses quality as resolution drops (7.53 / 7.22 / 6.83 / 6.17 / 5.06 at 832 / 704 / 576 / 448 / 384): it falls steeply below 576, and drift jumps at 384 (66.9). | **Edge-device fps:** the "device" is the same server-class GPU. No network link and no real edge hardware were in any measurement. |
| **Fused commit causes chunk-boundary jumps, and partial commit fixes them.** 704: 12 % vs 3 % for separate commit (36 clips each; teacher 832, seed 1234: 2 %). Partial commit 1 brings it to 4 % (2 % on bench-43), at ≈ 0.12 s/chunk in the quiet pass on the same task (jar 0.641 → 0.764, sandcastle 0.620 → 0.737 s/chunk). | **Quality claims beyond the VLM:** the quality prompt is not validated against humans. |

### Caveats
- **Teacher fps (IMPORTANT CORRECTION, facts.md §15):** the often-quoted "4.9 fps quiet" for the teacher is a whole-clip average: gen 31.2 s / 13 chunks, which includes the cheap early chunks before the KV window fills (10-prompt run, README §N). It is **not** the quiet version of 3.9.
  - On the steady definition used everywhere else, the teacher runs at **4.08 fps quiet** (2.939 s/chunk, jar + castle) and **3.9 fps under load** (3.038 s/chunk, bench-43), only 3 % apart.
  - Do not divide anything by 4.9. The speed-ups in this deck use 4.08 (quiet) or 3.9 (under load).
- **Two timing sources, two prompt sets:**
  - The quiet steady speed-ups (1.67× / 2.39× / 3.85×) come from jar + sandcastle (seed 1 = per-task warm-up excluded; for ours704 / ours832 castle s1234 excluded because of a cold compile).
  - The under-load ratios (1.74× / 2.5× / 4.1×) come from bench-43 with 7 GPU jobs running at once.
  - They agree within ~5 %. Quality and drift Δs are always from bench-43.
- **fps basis by mode:** the teacher-parity mode is timed like the teacher (median of diff(chunk_start)[5:]), so its 1.67× is like-for-like. The quality and realtime modes use cloud-only time per chunk (median cloud_chunk_s[6:], no decode, device step excluded), so 2.39× / 3.85× are not on the same basis.
  - With the in-loop device step counted serially (bench-43, under load): 2.25× / 3.44×.
  - The fully like-for-like comparison is the latency table (same protocol, decode inside): output fps 6.9 / 10.8 vs teacher 4.2.
- **Which GPU sets the fps (realtime mode):** 15.73 fps (quiet) = 12 frames / 0.763 s cloud time per chunk. It assumes the device decodes on its own GPU of the same class:
  - Full VAE at 704 costs 426 ms/chunk, which is faster than the cloud chunk.
  - On one shared GPU with the full VAE streamed alongside, the measured output is 10.8 fps (latency table). The older composed estimates are 10.1 fps with the full VAE and 15.5 fps with TAEHV (TAEHV costs −0.56 ± 0.13 quality; §6).
  - No network link and no real edge hardware were in the measurement.
- **Latency protocol:**
  - The ours rows (timing pass, n = 5 clips) and the profile rows (sdvg_profile, n = 6) are separate runs of the same protocol.
  - A vLLM judge server ran on another GPU throughout. Prepare is excluded: deployable prepare (T5 on GPU) is ≈ 2.0 s (README §Q), but the profile's own prepare took 23–27 s with T5 on CPU.
  - Output fps is a whole-clip mean, not steady fps (on the profile rows it reads 5–33 % above steady; §13).
- **StEvo-38:**
  - One seed, 38 tasks: differences below ~0.15 are not significant.
  - LingBot arms and MG2: 249-frame videos (MG2 needs latent frames divisible by 3); Waypoint: 256 (4 frames per call); all cover the 253-frame look-away-and-back camera path. The protocol applies the camera-controlled occlusion (requested occlusion = camera pan) to every arm, MG2 and Waypoint included.
- **Teacher-parity mode steps:** mean 3.01 steps on bench-43, slightly more than the 2.9 of the 10/8 runs, because exits are counted per chunk on 43 different prompts (3.10 in the quiet pass). The looser 50 % stop (2.61 steps) barely saved wall time under load (cause not diagnosed). In the quiet pass it did save time (1.97×), but it costs −0.17 ± 0.07 quality.
- **Hardware:** all timings are on an NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB), not an H100.
- **Baselines:** MG2 and Waypoint are keyboard/mouse game models driven by camera paths we converted into their inputs. Waypoint has no text input and under-rotates on long camera sweeps. Part of their drift may come from the action conversion.
- **Judge:**
  - 8 frames per clip cannot see chunk-boundary jumps; that is why the jump metric was added.
  - VLM scores are on full-VAE videos.
  - Our judge is local Qwen3-VL-32B, so the numbers are not comparable with the StEvo paper's Gemini leaderboard.
- **Prompt mix:** bench-43 is 38/43 tabletop & physics prompts. The easy-scene group has only n = 10 clips.
- **Resolutions differ:**
  - Teacher / teacher-parity mode / quality mode (ours 832) / SDVG / student: max_area 399360 (≈ 480×832). The size follows the start image, e.g. 848×464 for 31/43 tasks, 832×464 for 7, 864×448 for 4, 768×512 for 1.
  - Realtime mode (ours 704): ≈ 704×384 (e.g. 688×384 for easy_lake).
  - MG2: 640×352.
  - Waypoint: 1024×512.
- **Frame counts:** LingBot arms and MG2 output 153 frames per bench-43 clip, Waypoint 156.

### 발표 메모 (Korean)
- 10/5에는 "SDVG 99.9초, 저해상도는 안 된다"였는데, 지금은 원인이 정리됐고 실제로 동작하는 레시피가 있다는 흐름으로 시작.
- 핵심 메시지는 세 가지 operating mode 한 장으로. 배율은 quiet steady 기준(teacher 4.08 fps = 2.939 s/chunk, jar + sandcastle)을 씀. 품질은 bench-43 86 clip paired.
  - teacher-parity mode (chunk-wise adaptive 0.7 + safe commit)
    - 6.80 fps, **1.67×**, quality −0.07 ± 0.06, drift −3.3 ± 3.1 → teacher와 통계적으로 구분 안 됨.
    - fps 기준도 teacher와 같음.
  - quality mode (832 2-step fused + partial)
    - 9.74 fps, **2.39×**, quality −0.30 ± 0.08 → teacher보다 분명히 낮음.
    - drift +2.5 ± 3.0만 노이즈 안.
  - realtime mode (704 2-step fused + partial)
    - 15.73 fps, **3.85×**, quality −0.62 ± 0.09.
    - MG2와 품질은 같고(paired −0.01 ± 0.10) drift는 13점 낮음(paired −13.1 ± 3.6).
  - bench-43 부하 상태 배율(1.74× / 2.5× / 4.1×)과 ~5 % 안에서 일치한다고 한 줄로 언급.
- **teacher 4.9 fps는 쓰지 말 것.** 4.9는 클립 전체 평균(31.2초/13 chunk, KV가 차기 전 싼 초반 chunk 포함)이라 steady 기준이 아님. steady 기준 teacher는 quiet 4.08, 부하 3.9로 3 %밖에 차이 안 남. 슬라이드에 4.9가 보이면 4.08로 고치거나 말로 정정. f11 제목·라벨·표는 quiet 배율(1.67×/2.39×/3.85×)을 먼저 쓰고, 부하 배율(1.74×/2.5×/4.1×)은 괄호에 있음.
- quality/realtime mode의 s/chunk는 cloud-only(디코드·디바이스 단계 제외)라 teacher와 완전히 같은 기준은 아님.
  - 질문 나오면: 디바이스 단계를 직렬로 넣으면 2.25× / 3.44×(bench-43 부하).
  - 디코드까지 같은 GPU에서 다 넣은 latency 측정에서는 output fps 10.8 / 6.9 vs teacher 4.2.
- latency(Chenfeng 방식, GPU 1장, 같은 GPU에서 streaming decode):
  - realtime mode가 teacher, student, SDVG, ours mode 중 action→frame p50이 가장 낮음: 1296 ms. student 1434, teacher 3237, SDVG 4156–4463 ms.
  - 단, 출시하지 않는 ablation 2개(704 fused 1180 ms, 576 fused + partial 818 ms)는 더 낮음.
  - 750 ms 예산은 아무 config도 못 맞춤(realtime도 96.7 % miss).
  - teacher-parity mode는 streaming decode가 없어 per-frame 행이 없음.
  - MG2/Waypoint는 이 방식으로 측정 안 함.
- StEvo-38(38 tasks, seed 1 개, 로컬 Qwen3-VL 판정):
  - physical inaccuracy에서 14B 기반 arm(ours 0.55/0.68, teacher 0.68, SDVG 0.71)이 모두 student 1.00보다 낮음(paired p < 0.001).
  - Waypoint 0.92 대비로는 teacher, ours 832, ours 704는 구분됨(p ≤ 0.035), SDVG는 경계선(p = 0.057). f12 footer의 paired McNemar와 말이 맞도록 "모든 14B arm이 Waypoint보다 확실히 낮다"고 하지 말 것.
  - MG2 0.79 대비로는 quality mode(0.55)만 노이즈(~0.15) 밖. realtime 0.68 vs 0.79는 노이즈 안이라 "MG2보다 물리 낫다"고 단정하지 말 것.
  - state evolution / task success는 모든 arm이 바닥(0–0.08)이라 이 판정기로는 구분 불가하다고 먼저 말하기.
- SDVG
  - 논문 τ(−0.7)에서 teacher보다 느림(bench-43 3.5 vs 3.9 fps; latency profile에서도 3.0 vs 4.2 fps).
  - τ −2.2는 6.4 fps지만 quality −0.84, drift +19.7.
  - draft를 전부 받아도 상한 ≈ 7.6 fps라서 이 방향은 닫았다고 설명.
- partial commit 비용:
  - quiet pass에서 같은 task끼리 비교하면 704 fused → fused + partial이 jar 0.641 → 0.764, sandcastle 0.620 → 0.737 s/chunk(≈ 0.12 s). §15 표의 0.630 → 0.763은 clip 구성이 달라서(6 vs 5 clip) 차이가 약간 커 보임.
  - 예전 18.7(quiet) vs 15.8(부하) 비교는 측정 조건이 달라서 쓰지 말 것.
- 10/9–10/11 cloud+refiner 경로에서 디바이스 1.3B refiner가 품질을 깎아서(704 −0.50, 576 −0.25) 디바이스는 디코드만 함. 14B 자체도 해상도가 내려가면 품질이 떨어지고(576 아래에서 급하게), 384에서 drift 급증(66.9).
- 아직 말할 수 없는 것도 같이 언급:
  - MG2보다 빠르지는 않음(quiet 15.73 vs 19.7).
  - 실제 edge 기기/네트워크 측정은 없음.
  - StEvo task success로 arm을 구분할 수 없음.
  - teacher-parity mode의 latency와 StEvo는 없음.
