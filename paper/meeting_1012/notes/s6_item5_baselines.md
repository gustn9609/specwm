# Action item 5: on-device SOTA baselines (Waypoint, Matrix-Game 2.0) and the full baseline set

**Headline:** On bench-43 (43 prompts × 2 seeds = 86 clips per arm, paired with the teacher), ours 704 (realtime mode) adds **+5.9 ± 3.4** VLM drift over the 14B teacher, Matrix-Game 2.0 adds **+19.0 ± 3.8**, and Waypoint-1.5-1B adds **+24.9 ± 3.0**. MG2's quality cost is the same as ours (Δ quality −0.60 vs −0.62, within SE; paired ours − MG2 −0.01 ± 0.10), so drift is our only advantage over MG2 (paired ours − MG2 −13.1 ± 3.6). Waypoint's quality cost is far larger (−3.48). Speed, quiet timing pass (jar + sandcastle, one RTX PRO 6000): ours 704 **15.73 fps** as a cloud-chunk rate (**3.85×** the teacher's steady 4.08 fps; decode assumed pipelined on a second GPU), MG2 **19.7 fps** and Waypoint **31.2 fps** with their own decoders on one GPU. When its full-VAE decode streams on the same GPU, ours 704 gives **10.8 fps as a whole-clip output rate** (facts §13; n = 5, jar + castle). Compare that 10.8 with MG2 19.7 / Waypoint 31.2 (also whole-clip or own-decoder rates), not with the steady 4.08. The fps numbers are measured in different ways; see the conventions below. On the StEvo-Bench subset (38 tasks, 249 frames, one seed; local Qwen3-VL-32B judge, not comparable to the StEvo Gemini leaderboard; per-task sign tests derived), task success is at the floor for every arm (0–0.08). Only physical inaccuracy separates arms. The teacher and both ours modes are flagged less often than the student and Waypoint, and only ours 832 is flagged clearly less often than MG2.

Figures for this section: **f05_pareto_bench43** (quality vs fps: teacher, student, SDVG ×2, MG2, Waypoint and the three ours modes; the other chunk-wise arms are in facts §12), **f06_paired_deltas** (paired Δ vs teacher), **f12_stevo** (StEvo-Bench subset, slide 6), **f10_paths_<scene>** + **f10_paths_summary** (rollout paths across all baselines; numbers in facts §11, slide 7).

Conventions: ± = standard error. **bold** = best and <u>underline</u> = second within each column (ties share the mark). All bench-43 fps were measured on one RTX PRO 6000 Blackwell (96 GB) **under load** (7 GPU jobs running at once). The quiet timing pass (facts §15) uses the same per-arm definitions on other prompts: jar + sandcastle, seeds 1234 / 4321 / 9999. The per-task warm-up seed is excluded. For ours 832 / 704, castle s1234 is excluded instead because it paid a cold compile (n = 5). The pass ran 2 lanes on 2 GPUs with no other generation jobs, but the VLM judge servers were running on other GPUs. Teacher: 3.9 fps under load vs **4.08 fps quiet** on the same steady definition. The older "4.9 fps quiet" is a whole-clip average (31.2 s generation for 13 chunks, including the cheap early chunks before the KV window fills), so do not compare it with any steady number. **fps is not measured the same way for every arm** (scripts/bench43_report.py chunk_s(); fps = 12 / median over clips of seconds per chunk):
- LingBot teacher / student / chunk-wise arms: 12 / median chunk interval from chunk 5 on (steady chunks, decode excluded).
- Ours 832 / 704: 12 / median cloud_chunk_s from chunk 6 on (cloud only; decode and device time excluded; decode_s = 0.0 in all 86 pc1_704 loop.json (derived; bench43/*/pc1_704_s*/loop.json)).
- SDVG: 12 / (gen_s / 13), a whole-clip average that includes the forced target chunk 0; the final decode is excluded.
- Matrix-Game 2.0: gen_fps = 153 frames / gen_s (whole clip, first chunk and its own decoder included).
- Waypoint: output_fps (steady calls, first call excluded, its own decoder included; per-clip range 16.4–33.0 under load, 27.5–33.0 quiet (derived; output_fps in bench43/*/wp_s*/timing.json and timing_pass/*/wp_s*/timing.json)).
- Do not rank fps across these groups without this note.

Judge = Qwen3-VL-32B, 8 frames per clip at 448 px wide. Drift is 0–100 ↓ (clip-level AUROC 0.82 vs human labels). Quality is 1–10 ↑ and has not been validated against humans. Paired ours − MG2 is in facts §2. **Numbers not in facts.md are marked (derived) inline**, with their source; keep the "derived" label on a slide. The largest sets are the held-out (34-prompt) numbers (slide 1; from paper/1012/vlm_scores_bench43.json), the easy-scene paired deltas (slide 4; same file) and the StEvo per-task paired counts (slide 6; from lingbot-world-v2/specwm_out/stevo38/eval_runs/lingbot_b38_<arm>/per_task/<task>/physics_report__openai__qwen3-vl-32b.json). Short raw paths that start with bench43/, timing_pass/ or fused/ are under lingbot-world-v2/specwm_out/.

---

## Slide 1: The baseline set (same start image + same camera path for every arm)

| arm | role | inputs | resolution | sampler / settings | caveat |
|---|---|---|---|---|---|
| teacher: LingBot-World v2 14B causal-fast | quality reference (cloud) | image + text + camera path | ≈ 832×480 area (max_area 399360; mostly 848×464) | 4 steps (σ 1.0/0.937/0.833/0.625); FP8 W8A8, SageAttention 2, torch.compile, ring KV | steady 3.9 fps under load, 4.08 fps quiet (the old 4.9 is a whole-clip average) |
| student: LingBot 1.3B causal-fast | edge-only, same family | image + text + camera path | ≈ 832×480 area (max_area 399360; mostly 848×464) | 4 steps, same stack (FP8 is used on the 14B only) | – |
| SDVG + actions, τ −0.7 (paper τ) | speculative baseline (Hu & Zhang, arXiv 2604.17397) | image + text + camera path | ≈ 832×480 area (max_area 399360; mostly 848×464) | 1.3B drafts (4-step) → ImageReward worst-frame check on the decoded chunk → 14B (4-step) regenerates rejected chunks from the same noise. Chunk 0 is forced to the target, and both KV caches take every committed chunk | ImageReward cannot see physics |
| SDVG + actions, τ −2.2 | same, permissive threshold | same | ≈ 832×480 area (max_area 399360; mostly 848×464) | same, τ −2.2 | not run on the StEvo subset (disk) |
| Matrix-Game 2.0 (Skywork) | edge SOTA, game-trained | image + keyboard/mouse (**no text**) | 640×352 | 3 steps; flash-attn → SDPA shim, **no torch.compile** (the official repo compiles and claims 25 fps on H100) | camera → mouse 0.067 × deg/frame clipped to ±0.1, plus WASD |
| Waypoint-1.5-1B (Overworld) | edge SOTA, game-trained | image + keyboard/mouse (**no text**) | 1024×512 | torch.compile max-autotune; 4 frames per call; mouse ≈ 118 units per degree per call | **stretches the start image to 2:1**; under-rotates on long camera sweeps |
| ours 832 (quality mode) | cloud 14B + device decode | image + text + camera path | ≈ 832×480 area (max_area 399360; mostly 848×464) | 14B 2 steps (σ 1.0 → 0.833), fused + partial commit; same recipe as ours 704, run at 832 | bench-43 runs carry a leftover 1.3B device KV commit (see ours 704) |
| ours 704 (realtime mode) | cloud 14B low-res + device decode/upsample | image + text + camera path | ≈ 704×384 (max_area 270336; size follows the start image's aspect) | same at 704; the method's device = VAE decode + bicubic upsample, **no 1.3B** | the bench-43 runs still did a 1.3B device KV commit per chunk (≈ 0.12 s/chunk, left over from the run script; outside the fps; slide 5) |
| ~~Matrix-Game 3.5~~ (excluded) | – | image + camera path + text (same interface as ours) | 1280×704 | 5B, distilled | **0.9 fps end-to-end**, which is cloud-class. Excluded from the efficient set |

- **Prompt set (bench-43):** the 38 camera-WM tasks of StEvo-Bench (arXiv 2603.13215, 225 tasks / 6 categories) plus 5 easy scenes. Seeds 1234 / 4321. 13 chunks = 153 frames (≈ 9.6 s at 16 fps; 161 requested). Waypoint outputs 156 frames (derived; "frames" in bench43/*/wp_s*/timing.json).
- **Bench-43 is not the StEvo protocol.** StEvo camera tasks use 253 frames (21 chunks); bench-43 stops at 153. The StEvo-protocol run (38 tasks, 249 frames, seed 1234, 7 arms) is on slide 6.
- **Bench-43 is not fully held out.** 9 of its 43 prompts (derived; prompt folders of fused/ vs bench43/) come from the 12-prompt development matrix on which 704, fused commit and partial commit K = 1 were chosen (README V3, W): all 5 easy scenes plus billiard_break, domino_falling, newtons_cradle and pouring_water_into_cup. The seeds 1234 / 4321 are also shared.
  - On the 34 held-out prompts (68 clips, paired vs the teacher; derived), the order is the same. Δ drift: ours 832 +1.5 ± 3.4, ours 704 +4.5 ± 3.8, MG2 +19.9 ± 4.3, Waypoint +24.1 ± 3.2. Δ quality: −0.32 ± 0.08, −0.62 ± 0.09, −0.60 ± 0.09, −3.47 ± 0.18.
- Our judge is a local Qwen3-VL-32B. StEvo's default judge is Gemini, so our scores are **not comparable to the StEvo leaderboard**.

## Slide 2: Bench-43 results (86 clips per arm) + quiet fps → f05_pareto_bench43

| arm | fps under load ↑ (bench-43) | fps quiet ↑ (jar + castle; facts §15) | fps measured as | VLM drift ↓ | VLM quality ↑ | Δ quality vs teacher ↑ (paired, 86) | Δ drift vs teacher ↓ (paired, 86) |
|---|---|---|---|---|---|---|---|
| teacher 14B 4-step | 3.9 | 4.08 (1.00×) | steady chunks, decode excluded | <u>51.0 ± 3.0</u> | **7.51 ± 0.08** | – | – |
| student 1.3B 4-step | 17.2 | – (not in the quiet pass) | steady chunks, decode excluded | 75.1 ± 2.2 | 5.83 ± 0.16 | −1.69 ± 0.17 | +24.1 ± 3.0 |
| SDVG + actions τ −0.7 | 3.5 | – (not in the quiet pass) | whole-clip average incl. forced chunk 0, final decode excluded | 52.9 ± 3.0 | **7.51 ± 0.09** | **+0.00 ± 0.07** | <u>+1.9 ± 3.0</u> |
| SDVG + actions τ −2.2 | 6.4 | – (not in the quiet pass) | whole-clip average incl. forced chunk 0, final decode excluded | 70.6 ± 2.5 | 6.67 ± 0.09 | −0.84 ± 0.10 | +19.7 ± 3.4 |
| Matrix-Game 2.0 (640×352) | <u>20.0</u> | <u>19.7</u> | whole clip incl. first chunk + own decoder | 70.0 ± 3.0 | 6.91 ± 0.07 | −0.60 ± 0.09 | +19.0 ± 3.8 |
| Waypoint-1.5-1B (1024×512) | **25.4** | **31.2** | steady calls incl. own decoder | 75.9 ± 1.6 | 4.03 ± 0.15 | −3.48 ± 0.15 | +24.9 ± 3.0 |
| ours teacher-parity (chunk-wise 0.7 + safe commit) | 6.8 | 6.80 (1.67×) | steady chunks (chunk_start, chunks ≥ 5), decode excluded | **47.7 ± 3.2** | <u>7.44 ± 0.08</u> | <u>−0.07 ± 0.06</u> | **−3.3 ± 3.1** |
| ours 832 fused + partial (quality mode) | 9.8 | 9.74 (2.39×) | steady cloud chunks only, decode excluded | 53.5 ± 3.0 | 7.21 ± 0.08 | −0.30 ± 0.08 | +2.5 ± 3.0 |
| ours 704 fused + partial (realtime mode) | 15.8 | 15.73 (3.85×) | steady cloud chunks only, decode excluded | 56.9 ± 2.7 | 6.90 ± 0.08 | −0.62 ± 0.09 | +5.9 ± 3.4 |

The fps bases differ (column 4), so the fps bold/underline is a ranking across different measurements. The quiet column uses the same per-arm basis as the under-load column, but other prompts (jar + sandcastle). Same-basis check for SDVG vs the teacher (bench-43, under load): gen s per clip 44.6 vs 35.8, so SDVG is ≈ 1.25× slower (derived ratio). A same-stack check without other generation jobs (facts §13; jar + castle, streamed decode, n = 6) gives 51.2 vs 37.8 s per clip, also slower. On the teacher's steady-chunk basis SDVG is 3.1 fps (τ −0.7) and 6.7 fps (τ −2.2) (facts §2). On SDVG's whole-clip basis the teacher is 4.4 fps (derived; = 12 / (35.8 s / 13)).

- **Quiet vs under load:** the LingBot arms agree within ~5 % (teacher 4.08 vs 3.9, teacher-parity 6.80 vs 6.8, ours 832 9.74 vs 9.8, ours 704 15.73 vs 15.8), so the speed-ups hold. Prefer the quiet ratios on slides: teacher-parity 1.67×, quality 2.39×, realtime 3.85× (bench-43 under load: 1.74× / 2.5× / 4.1×). MG2 is not slower under load (19.7 quiet vs 20.0). Waypoint is ≈ 23 % faster quiet (31.2 vs 25.4; derived ratio).
- **Ours 704 vs MG2 (15.73 vs 19.7 fps quiet; 15.8 vs 20.0 under load; ours = cloud chunks only, MG2 = whole clip with its own decoder):**
  - With its full-VAE decode streamed on the same GPU, ours 704 outputs 10.8 fps (facts §13; jar + castle, n = 5, whole-clip output rate). Its action→frame p50 is 1296 ms, vs teacher 3237 ms and student 1434 ms (facts §13: jar + castle, one GPU, full-VAE decode streamed on the same GPU, prepare excluded; teacher / student n = 6 clips from the SDVG profile, ours n = 5 from the timing pass).
  - Quality cost is about the same: −0.62 ± 0.09 vs −0.60 ± 0.09 (within SE). Paired ours − MG2 quality: −0.01 ± 0.10 (22 better / 44 equal / 20 worse; counts derived from paper/1012/vlm_scores_bench43.json). The quality score is an integer at temperature 0, so half the pairs tie.
  - Drift is our only advantage: Δ vs teacher **+5.9 ± 3.4 vs +19.0 ± 3.8**; paired ours − MG2 drift **−13.1 ± 3.6**.
  - Absolute VLM quality is the same (6.90 vs 6.91). Absolute drift is 56.9 vs 70.0.
- **Waypoint is the fastest arm (31.2 fps quiet, 25.4 under load, own decoder included) and the worst on quality:** 4.03, Δ −3.48 ± 0.15. Its drift (75.9 ± 1.6) is about the same as our own 1.3B student's (75.1 ± 2.2, within SE).
- **Among arms at ≥ 2× the teacher's speed, ours 832 is closest to the teacher:** Δ quality −0.30 ± 0.08, Δ drift +2.5 ± 3.0, at 9.74 vs 4.08 fps quiet (2.39×; 9.8 vs 3.9 under load). The teacher-parity mode is closer still (−0.07 / −3.3) at 1.67× (slide 3).
- **SDVG τ −0.7 is teacher-equal but ≈ 1.25× slower than the teacher** on the same basis (bench-43, under load: gen s per clip 44.6 vs 35.8; derived ratio). The 3.5 vs 3.9 fps pair mixes a whole-clip and a steady-chunk measurement and understates the slowdown. See slide 5.
- **Chunk-boundary jumps (share of boundaries > 1.5×):** teacher 1 %, student 2 %, SDVG 1 % / 3 %, ours teacher-parity 1 %, ours 832 2 %, ours 704 2 % (facts §2, §12). The metric does not apply to MG2 or Waypoint (n/a).

## Slide 3: Paired deltas vs the teacher → f06_paired_deltas
- Every arm is paired with the teacher clip for the same prompt and seed (n = 86), so prompt difficulty cancels out.
- Two clusters by drift:
  - **Near-teacher (Δ drift −3.3 to +5.9):** ours: teacher-parity mode (−3.3), SDVG τ −0.7 (+1.9), ours: quality mode (+2.5), ours: realtime mode (+5.9). All four are within ≈ 2 SE of zero (95 % CI includes 0).
  - **Lost-scene (Δ drift +19.0 to +24.9):** MG2 (+19.0), SDVG τ −2.2 (+19.7), student (+24.1), Waypoint (+24.9).
- **Of the arms shown, only ours: teacher-parity mode is faster than the teacher and within noise on both axes:** Δ quality −0.07 ± 0.06, Δ drift −3.3 ± 3.1 (paired, 86), at 6.80 vs 4.08 fps quiet (**1.67×**; 6.8 vs 3.9 = 1.74× under load on bench-43). SDVG τ −0.7 is also within noise on both axes (+0.00 / +1.9) but is not faster (44.6 s vs 35.8 s per clip). Two chunk-wise arms not in f06 also pass both tests: cw07_pc1 (6.78 fps quiet / 6.9 under load; −0.09 ± 0.06 / −0.5 ± 3.1) and cw07_sep (5.94 quiet / 5.9 under load; +0.03 ± 0.05 / −0.6 ± 3.2) (facts §12, §15).
- **Every fast arm (≥ 15 fps) except ours: realtime mode is in the lost-scene cluster:** student 17.2, MG2 20.0 (19.7 quiet), Waypoint 25.4 (31.2 quiet) fps (the fps bases differ; slide 2).

## Slide 4: Easy scenes vs tabletop & physics (bench-43 split)

| arm | easy: quality ↑ (5 prompts, n=10) | easy: drift ↓ | tabletop & physics: quality ↑ (38 prompts, n=76) | tabletop & physics: drift ↓ |
|---|---|---|---|---|
| teacher 14B | <u>8.00</u> | **22** | **7.45** | **55** |
| student 1.3B | 7.60 | <u>24</u> | 5.59 | 82 |
| SDVG τ −0.7 | **8.20** | **22** | <u>7.42</u> | <u>57</u> |
| SDVG τ −2.2 | 7.20 | 32 | 6.61 | 76 |
| Matrix-Game 2.0 | 6.90 | 48 | 6.91 | 73 |
| Waypoint-1.5-1B | 4.70 | 55 | 3.95 | 79 |
| ours 832 | 7.90 | 25 | 7.12 | <u>57</u> |
| ours 704 | 7.60 | 36 | 6.80 | 60 |

- **Easy scenes:**
  - Our own 1.3B student is already close to the teacher (7.60 / 24 vs 8.00 / 22).
  - Waypoint is clearly worse than the student (paired Δ quality −2.90 ± 0.55, Δ drift +31.5 ± 8.9; n = 10, derived). MG2 trends worse (−0.70 ± 0.45, +25.0 ± 13.2; < 2 SE, derived). Absolute: MG2 6.90 / 48, Waypoint 4.70 / 55 vs student 7.60 / 24. Paired deltas are from paper/1012/vlm_scores_bench43.json (same task and seed).
- **Tabletop & physics:**
  - The 1.3B student collapses (5.59 / 82).
  - MG2 keeps its quality (6.91, flat across both groups) and drifts less than the student (73 vs 82).
  - Ours stays near the teacher: ours 832 7.12 / 57, ours 704 6.80 / 60.
- **Honest points:**
  - On the 10 easy clips, ours 704 drifts more than the student (36 vs 24; paired +13.0 ± 11.0, within noise; derived). The advantage of ours comes from the 38 hard prompts.
  - All 5 easy scenes (and 4 of the 38 hard prompts) are in the development matrix (slide 1).

## Slide 5: Generation time per clip, and where SDVG's time goes (bench-43, median per 153-frame clip ≈ 9.6 s, under load) → f07_sdvg_analysis

| arm | gen s per clip ↓ | breakdown / what is excluded (s) | chunks from the draft (of 13) |
|---|---|---|---|
| teacher 14B 4-step | 35.8 | excludes decode 8.6 (full VAE) and prepare 6.3 | – |
| student 1.3B 4-step | 8.2 | excludes decode 8.7 | – |
| SDVG τ −0.7 | 44.6 | draft 4.6, verify 9.0, target 21.3, commit 9.1; excludes final decode 8.6 | 1.65 (median 1; 42/86 clips take none) |
| SDVG τ −2.2 | 24.4 | draft 4.6, verify 9.0, target 3.0, commit 7.0 | 10.1 (median 11) |
| Matrix-Game 2.0 | <u>7.6</u> | own decoder **included** | – |
| Waypoint-1.5-1B | **6.6** | own decoder **included** | – |
| ours 832 | 16.1 | = cloud 14.6 + ≈ 1.46 s of a 1.3B device KV commit that the method does not need (left over from the run script; not in fps) | – |
| ours 704 | 10.7 | = cloud 9.2 + ≈ 1.45 s of the same leftover 1.3B device KV commit; full-VAE decode at 704 = 5.5 s/clip (runs on the device; not timed in these runs, decode_s = 0, derived) | – |

The rows do not all cover the same scope (decode is included only for MG2 and Waypoint). Treat the column as a ranking, not as additive numbers.

- **τ −0.7:** almost every chunk still goes to the 14B, so SDVG pays draft + ImageReward on top of the teacher. Result: 44.6 vs 35.8 s per clip (bench-43, under load), ≈ 1.25× slower than the teacher (derived ratio).
- **τ −2.2:** the draft is accepted for 10.1 / 13 chunks, giving 6.4 fps (whole-clip average). Drift then reaches MG2 level (+19.7 vs +19.0) while running at 6.4 vs 20.0 fps (both whole-clip averages; MG2's includes its decoder, SDVG's excludes the final decode).
- **Accept-all ceiling:** draft 0.35 + verify 0.69 + commit to both caches 0.54 ≈ 1.6 s/chunk ≈ 7.6 fps. That is still 2.5× slower than the student alone, at student quality.

## Slide 6: StEvo-Bench subset (StEvo look-away-and-back protocol) → f12_stevo
- **Setup (facts §14):** the 38 StEvo camera-WM tasks of bench-43, seed 1234 only, 249-frame videos covering the 253-frame look-away-and-back camera path (≈ 21 chunks vs 13 in bench-43). Judge: local Qwen3-VL-32B, 3-vote majority, 32 frames; criteria control / physics / state (scripts/stevo_eval.sh). Every arm, MG2 and Waypoint included, is scored under the camera-controlled protocol (requested occlusion = camera pan). Not comparable to the paper's Gemini leaderboard.
- **Arms:** 7. SDVG τ −2.2 was not run (disk). Ours teacher-parity mode (chunk-wise) was not run (time). MG2 ran at 249 frames (its latent frame count must be divisible by 3).

| arm | occlusion done ↑ | trigger applied ↑ | physical inaccuracy ↓ (tasks flagged / 38) | control success ↑ | state evolution ↑ | task success ↑ |
|---|---|---|---|---|---|---|
| teacher 14B 4-step | 0.34 | <u>0.45</u> | <u>0.68 (26)</u> | <u>0.13</u> | **0.08** | **0.08** |
| student 1.3B 4-step | <u>0.42</u> | 0.39 | 1.00 (38) | <u>0.13</u> | 0.00 | 0.00 |
| SDVG + actions τ −0.7 | 0.18 | **0.61** | 0.71 (27) | 0.11 | 0.00 | 0.00 |
| Matrix-Game 2.0 | **0.97** | 0.24 | 0.79 (30) | **0.24** | 0.00 | 0.00 |
| Waypoint-1.5-1B | 0.05 | 0.26 | 0.92 (35) | 0.00 | 0.03 | 0.00 |
| ours quality mode (832) | 0.32 | 0.39 | **0.55 (21)** | 0.08 | <u>0.05</u> | <u>0.05</u> |
| ours realtime mode (704) | 0.24 | 0.37 | <u>0.68 (26)</u> | 0.08 | <u>0.05</u> | 0.00 |

Fractions of 38 tasks; binomial s.e. ≈ 0.05–0.08, so differences < ~0.15 are not significant. Metrics (stevo_bench_code/eval/summarize_results.py): occlusion done = the look-away happened; trigger applied = the prompted action happened; control success = both; state evolution = the expected state change is visible after looking back; task success = state evolution AND no physical inaccuracy.

- **Honest reading:**
  - **The success columns do not separate the arms.** State evolution and task success are at the floor for every arm, as in the 10/7 85-task run (1–5 %). State evolution: teacher 3 of 38 tasks, ours 832 2, ours 704 2, Waypoint 1, the others 0. Task success: teacher 3, ours 832 2, the others 0. Do not read the teacher's 0.08 vs ours 0.05 / 0.00 as a gap.
  - **Physical inaccuracy is the only informative column.** Paired per task (derived; "A k vs l" = tasks where only A is clean vs only the other arm is clean; sign test):
    - vs the student (flagged on all 38): every 14B arm is cleaner. Teacher 12 vs 0, ours 704 12 vs 0, ours 832 17 vs 0, SDVG τ −0.7 11 vs 0.
    - vs Waypoint (35/38 flagged): teacher 12 vs 3 (p 0.035), ours 704 12 vs 3 (p 0.035), ours 832 17 vs 3 (p 0.003); SDVG τ −0.7 11 vs 3 (p 0.057, borderline).
    - vs MG2 (30/38 flagged): **only ours 832 separates** (12 vs 3, p 0.035). Teacher 8 vs 4 (p 0.39), ours 704 9 vs 5 (p 0.42) and SDVG τ −0.7 8 vs 5 (p 0.58) do not. So "14B arms beat MG2 on physics" is a trend, not a result.
    - vs the teacher: ours 704 ties (5 vs 5). Ours 832's 0.55 vs 0.68 is 8 vs 3 (p 0.23), not significant. **Do not claim ours 832 is more physical than the teacher.** SDVG τ −0.7 vs the teacher: 3 vs 4.
  - **Control reflects how each model moves the camera, not the scene:** MG2 almost always looks away (0.97; game-style mouse look) but rarely shows the prompted action (0.24). Waypoint almost never looks away (0.05), consistent with its under-rotation on long camera sweeps (slide 1). The judge sees the look-away in only 0.18–0.42 of the LingBot videos (cause not diagnosed).
- **What it adds to bench-43:** the same direction as the VLM drift on slide 3 (14B-based arms ahead of the student and Waypoint; ours ≈ teacher). It cannot answer "does the model complete the prompted state change", because no arm does.

## Slide 7: Rollout paths across all baselines → f10_paths_<scene>, f10_paths_summary
- **Scenes:** easy_lake, easy_stonehenge_sunset, easy_urban_rain, pouring_water_into_cup, sandcastle_wave_impact, sand_filling_up_jar. 10 seeds per arm, except the teacher on jar / castle (n = 3), so the distances on those two scenes use 3 seeds per arm.
- **Same latent space for every model:**
  - Each arm's decoded video goes through the same steps: undo Waypoint's 2:1 stretch, then centre-crop every arm (MG2 included) to the teacher's aspect, resize to half the teacher grid, and re-encode with the Wan2.1 VAE.
  - Left panel: one shared PCA over all arms and seeds.
  - Right panel: relative latent distance to the same-seed teacher frame per chunk (↓). The grey dashed line is teacher vs teacher with a different seed (the reference's own spread).
- **Numbers (facts §11):** ratio = time-mean distance to the teacher ÷ teacher-vs-teacher distance (different seed); 1.0 = as far as another teacher sample.

| arm | ratio to teacher spread, mean of 6 scenes ± s.e. ↓ | range over scenes |
|---|---|---|
| student 1.3B | 1.17 ± 0.05 | 1.03–1.32 |
| SDVG τ −0.7 | **1.06 ± 0.03** | 0.99–1.15 |
| SDVG τ −2.2 | 1.16 ± 0.05 | 0.98–1.33 |
| Matrix-Game 2.0 | 1.45 ± 0.07 | 1.22–1.72 |
| Waypoint-1.5-1B | 1.24 ± 0.09 | 0.97–1.52 |
| ours 832 (shares the teacher's noise; not ranked) | 1.00 ± 0.02 | 0.94–1.07 |
| ours 704 | <u>1.15 ± 0.05</u> | 1.03–1.33 |

- **Reading:**
  - MG2 is clearly farther from the teacher than ours 704 (+0.30 ± 0.07) and the student (+0.28 ± 0.08).
  - Ours 704 is as far from the teacher as the student (−0.02 ± 0.04, within seed noise). The path metric does not show the drift advantage that the VLM judge shows.
  - Waypoint − student +0.07 ± 0.07: varies by scene, not clearly farther.
  - Ours 832 starts each chunk from the teacher's noise, and still ends as far from the teacher as an independent teacher sample (1.00). The other arms use other noise streams, so compare them against the grey band, not against ours 832.
  - This is distance from the teacher's samples, not quality: Waypoint is 0.97 on jar but has bench-43 quality 4.03.

## Slide 8: Answer to action item 5 ("have the on-device SOTA at hand; build on it only if our draft is far worse")
- **Waypoint-1.5-1B:**
  - Lower quality than our own 1.3B (4.03 vs 5.83) at about the same drift (75.9 vs 75.1, within SE).
  - StEvo subset: physical inaccuracy 0.92 (vs student 1.00), and it looks away in only 0.05 of tasks.
  - Waypoint is not a better base than our draft.
- **Matrix-Game 2.0:**
  - Better than our 1.3B on tabletop & physics (6.91 / 73 vs 5.59 / 82). Also less often flagged for physical inaccuracy on the StEvo subset (0.79 vs 1.00; 8 vs 0 tasks, derived), but it shows the prompted action in only 0.24 of tasks.
  - Trends worse on easy photo scenes (6.90 / 48 vs 7.60 / 24; n = 10, not significant: paired −0.70 ± 0.45 / +25.0 ± 13.2, derived).
  - Takes no text.
  - We keep it as a baseline (it is already our second base model in the scalability check, README M), not as the device model.
- **The question matters less now:** the realtime method (ours 704) needs no draft model on the device, only VAE decode + upsample. (The bench-43 runs still carried a leftover 1.3B device KV commit, ≈ 1.45–1.46 s per clip outside the fps; slide 5.)
- **Matrix-Game 3.5:**
  - It is the only open model with our exact interface (image + camera path + text).
  - It runs at 0.9 fps end-to-end at 1280×704, so it belongs to the cloud class. Excluded from the efficient set.
- **Done since the last revision:**
  - StEvo-Bench subset (38 tasks, 249 frames, seed 1234, 7 arms): slide 6, facts §14.
  - Chunk-wise adaptive + commit policy on bench-43 (teacher-parity mode): slide 3, facts §12.
  - Quiet timing pass: slide 2, facts §15.

## Backup: the 12-prompt × 3-seed matrix agrees on drift, not clearly on quality (36 clips per arm; facts.md §3)

This is the development set on which 704, fused commit and partial commit K = 1 were chosen. It shares 9 prompts and seeds 1234 / 4321 with bench-43.

| config | fps quiet ↑ (facts §15, jar + castle) | fps basis; older values (text only) | VLM drift ↓ | VLM quality ↑ |
|---|---|---|---|---|
| teacher 14B 4-step | 4.08 | steady chunks, decode excluded. The 10/10 run (README §N: 10 prompts, seed 1234) gave a **whole-clip** 153 frames / 31.2 s = 4.9; do not divide another arm's fps by that | **45.4 ± 5.2** | **7.64 ± 0.16** |
| 704 fused + partial 1 (= ours 704) | 15.73 | steady cloud chunks only, decode excluded; 15.8 under load (bench-43) | <u>53.5 ± 4.6</u> | 6.86 ± 0.17 |
| Matrix-Game 2.0 | <u>19.7</u> | whole clip incl. first chunk + own decoder; 20.5 in the 10/10 run (GPU sharing not verified) | 57.9 ± 4.8 | <u>7.06 ± 0.17</u> |
| Waypoint-1.5-1B | **31.2** | steady calls incl. own decoder; 28.3 in the 10/10 run (GPU sharing not verified) | 58 | 5.1 |

- The MG2 row uses its 36 real clips (the duplicated warm-up clip mg2/seeds/easy_lake_s1234_warm is excluded), as facts §3 does.
- The fps column is the quiet timing pass (jar + castle), not this 36-clip matrix. The measurement method still differs per arm (see the "fps basis" column). The teacher's older 4.9 is a whole-clip average and is kept as text only.
- **On this set, MG2's quality is above ours 704 (7.06 vs 6.86).** The SEs overlap. The drift order (teacher < ours 704 < MG2) is the same as on bench-43, but the SEs overlap here too.
- Use bench-43 (paired, 86 clips) as the main claim and this table as backup.

### Caveats
- **Bench-43 fps is measured under load** (7 jobs at once). The quiet timing pass (facts §15; jar + sandcastle, same per-arm definitions; 2 lanes on 2 GPUs with no other generation jobs, VLM judge servers on other GPUs; warm-up seed excluded, and castle s1234 also excluded for ours 832 / 704) agrees for the LingBot arms within ~5 % (teacher 4.08 vs 3.9, teacher-parity 6.80 vs 6.8, ours 832 9.74 vs 9.8, ours 704 15.73 vs 15.8). The load effect is not uniform across models: MG2 19.7 quiet vs 20.0 under load, Waypoint 31.2 quiet vs 25.4 under load. The prompts also differ (2 tasks vs 43). MG2 quiet 19.7 is the median of its 6 runs without the warm-up run (facts §15: gen_fps 18.9, 19.1, 19.5, 19.9, 19.9, 20.1); the 19.5 quoted before 10/12 also counted the warm-up run (19.0).
- **The teacher's "4.9 fps quiet" is a whole-clip average** (31.2 s for 13 chunks, including the cheap early chunks). On the steady definition used for every speed-up, the teacher is 4.08 quiet / 3.9 under load. Quote the quiet steady ratios: 1.67× / 2.39× / 3.85×.
- **fps is not measured the same way for every arm** (conventions above):
  - LingBot teacher / student / ours: steady chunks, decode excluded (ours: cloud only).
  - SDVG: whole-clip average, final decode excluded.
  - MG2: whole clip, first chunk + own decoder included.
  - Waypoint: steady calls, own decoder included.
- **MG2 ran without torch.compile, through an SDPA shim for flash-attn.** The official repo compiles and claims 25 fps on H100. Our MG2 fps is therefore a lower bound. Waypoint ran with compile max-autotune.
- **Waypoint stretches the start image to 2:1 (1024×512).**
  - The judge sees the stretched video, which may lower its VLM quality. This was not tested.
  - The path figure un-stretches before comparing.
- **The game models take no text and no camera trajectory.**
  - Camera paths become keyboard/mouse approximations. MG2's mouse is clipped at ±0.1, and Waypoint under-rotates on long sweeps, so neither follows the full commanded path.
  - The action described in the text of the physics prompts is invisible to them, so on the StEvo subset they cannot be told the prompted action (trigger applied: MG2 0.24, Waypoint 0.26).
  - On photos they are out of their training domain, so these scores are not their game-domain quality.
- **Decoder accounting differs.**
  - MG2 and Waypoint include their own decoder on the same GPU.
  - Ours 704 at 15.73 fps (quiet; 15.8 under load) is the cloud chunk time, with the full VAE (426 ms/chunk < 761 ms cloud) assumed pipelined on a separate device GPU of the same class.
  - On one shared GPU: measured streaming output 10.8 fps for ours 704 and 6.9 fps for ours 832 (facts §13, jar + castle, n = 5, whole-clip output rate; do not compare it with any steady fps). Composed from per-chunk times (facts §6): ours 704 = 10.1 fps (full VAE) or 15.5 fps (TAEHV). TAEHV costs −0.56 ± 0.13 quality and +7.4 drift (judge 448, 36 clips).
  - The bench-43 ours runs also did a 1.3B device KV commit (≈ 1.45–1.46 s per clip) that the method does not need. It is outside the fps.
- **The compute class differs.**
  - Ours and the teacher run a 14B on a 96 GB server GPU.
  - MG2 and Waypoint are 1–2B edge-class models, but they were also timed on that server GPU, not on a device.
  - All timings are on an RTX PRO 6000 Blackwell, not an H100.
- **Resolutions are not matched** (≈ 832×480 area, mostly 848×464; ≈ 704×384; 640×352; 1024×512). LingBot output size follows the start image's aspect at a fixed max_area (832-class: 848×464 on 62/86 clips, 832×464 on 14, 864×448 on 8, 768×512 on 2; derived from the bench-43 video headers). The judge's 448 px input hides part of the difference. An edge-SOTA run at our exact resolution (asked for on 10/10) has not been done.
- **The evaluation is thin.**
  - Bench-43: 2 seeds per prompt; the easy group has only n = 10 clips. 9 of 43 prompts and both seeds are shared with the development matrix; the order of the arms holds on the 34 held-out prompts (slide 1). Clips are 153 frames, not StEvo's 253.
  - StEvo subset: one seed, 38 tasks, binomial s.e. ≈ 0.05–0.08; its success columns are at the floor, and only ours 832 vs MG2 separates on physics among the 14B-vs-MG2 pairs (slide 6). The per-task paired counts are derived here, not in facts.md.
  - VLM quality is not human-validated.
  - The scores are not comparable with the StEvo leaderboard.
- **Path figure:** the teacher has n = 3 seeds on jar / castle vs 10 on the other scenes, so every arm's distance on those two scenes uses 3 seeds. The leave-one-teacher-seed-out s.e. of the reference is 15 % on jar (2–8 % elsewhere) and shifts all arms of a scene together (facts §11).

### 발표 메모 (Korean)
- 10/5 액션 아이템 5: Waypoint와 Matrix-Game 2.0을 bench-43(86클립)에서 같은 시작 이미지와 같은 카메라 경로로 돌렸고, StEvo 프로토콜(38 task, 249프레임, seed 1234)로도 채점했습니다.
- 핵심 숫자: teacher 대비 drift 증가가 ours: realtime mode(704)는 +5.9, MG2는 +19.0, Waypoint는 +24.9입니다. 품질 손실은 realtime mode와 MG2가 같습니다(−0.62 vs −0.60, SE 이내; ours − MG2 직접 비교 −0.01 ± 0.10). MG2 대비 우리 이점은 drift뿐입니다(직접 비교 −13.1 ± 3.6).
- 속도는 quiet timing pass 기준으로 말하세요: teacher 4.08 fps, ours realtime 15.73 fps(3.85×), quality 9.74 fps(2.39×), teacher-parity 6.80 fps(1.67×). MG2 19.7, Waypoint 31.2 fps는 자체 디코더까지 포함한 숫자이고, ours는 cloud chunk만(디코드 제외)입니다. 같은 GPU에서 full VAE 디코드까지 스트리밍하면 ours 704는 10.8 fps입니다(클립 전체 평균, n = 5; steady 숫자와 비교 금지).
- teacher "4.9 fps quiet"는 클립 전체 평균(초반 싼 청크 포함)이라 steady 숫자와 비교하면 안 됩니다. 이전 슬라이드에 4.9가 남아 있으면 4.08로 바꾸세요.
- StEvo 결과는 솔직하게(로컬 Qwen3-VL-32B judge, 38 task·seed 1개, StEvo 리더보드(Gemini)와 비교 불가; task별 sign test는 derived): state evolution / task success는 모든 arm이 바닥(38개 중 0–3개)이라 구분이 안 됩니다. 의미 있는 건 physical inaccuracy뿐이고, teacher와 ours 두 모드는 student(38/38 flagged)와 Waypoint(35/38)보다 확실히 덜 flagged됩니다(SDVG는 Waypoint 대비 경계선, p 0.057). MG2(30/38) 대비로는 ours 832만 유의합니다(12 vs 3). ours 832 0.55 vs teacher 0.68은 유의하지 않으니 "teacher보다 물리적으로 낫다"고 말하지 마세요.
- 쉬운 장면에서는 우리 1.3B도 teacher에 가깝습니다. 차이는 탁상·물리 프롬프트 38개에서 나옵니다(1.3B 5.59/82, ours 704 6.80/60).
- SDVG는 논문 τ(−0.7)에서 teacher와 품질이 같지만, 같은 기준(클립당 생성 시간)으로 44.6 s vs 35.8 s, 약 1.25배 느립니다(bench-43, under load). 13청크 중 평균 1.65청크만 draft를 씁니다.
- 경로(rollout path) 거리: MG2가 확실히 가장 멀고(ours 704 대비 +0.30 ± 0.07), ours 704는 student와 비슷합니다(−0.02 ± 0.04). 거리는 품질이 아니라는 점을 같이 말하세요.
- 공정성 질문이 나오면 먼저 말하세요: arm마다 fps 측정 방식이 다르고, MG2는 compile 없이 돌렸고, Waypoint는 이미지를 2:1로 늘리며, 게임 모델은 텍스트를 받지 않습니다. bench-43의 43개 프롬프트 중 9개는 개발용 12-프롬프트 세트와 겹치지만, 나머지 34개만 봐도 순서는 같습니다.
- 결론: Waypoint와 MG2 모두 base로 갈아탈 이유는 없습니다. MG3.5는 0.9 fps라 효율 비교에서 제외했습니다.
