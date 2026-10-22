# Next steps and asks

**Headline:** Five items are done since the last version: chunk-wise adaptive + commit policy (now ours: teacher-parity mode), the quiet timing pass, the Chenfeng-method latency profile (now with rows for ours), the StEvo-38 subset and the 448 / 384 low-res points. On the quiet steady basis (jar + castle, generation only; facts §15) our three modes run at:
- teacher-parity: 6.80 fps, **1.67×** the teacher's 4.08 fps;
- quality: 9.74 fps, **2.39×**;
- realtime: 15.73 fps, **3.85×**.

The bench-43 under-load ratios (1.74× / 2.5× / 4.1×) agree with these within ~5 %. Δ quality vs the teacher on bench-43 (86 paired clips) is −0.07 ± 0.06 / −0.30 ± 0.08 / −0.62 ± 0.09.

Correction: the teacher's often-quoted "4.9 fps quiet" is a whole-clip average. On the steady basis the teacher runs at 4.08 fps quiet and 3.9 fps under load.

No mode reaches ≥ 20 fps at quality ≥ 7.2. So the first next step is now a DMD fine-tune of the 14B, starting at 704. The cheapest next result is a uniform 3-step control. Two items need outside help: the human blind study still has no raters, and running StEvo with the paper's judge (Gemini) needs an API key.

Figures for this slide:
- **f11_modes:** the three operating modes. Re-rendered 08:00 CDT: its title, mode labels and table lead with the quiet steady ratios 1.67× / 2.39× / 3.85× (teacher 4.08 fps quiet) and give the bench-43 under-load ratios 1.74× / 2.5× / 4.1× in parentheses (teacher 3.9 fps); panel a's axis is bench-43 fps under load. Put the quiet steady ratios in the slide text.
- **f05_pareto_bench43:** the "≥ 20 fps and ≥ 7.2" corner is empty (bench-43, under load).
- **f08_lowres_curve:** why resolution needs training. It now includes 448 and 384.
- **f13_latency:** the latency profile (facts §13).
- **f04_speed_bridge:** where the cloud time goes, and Δ quality for each lever.
- **f03_splits_map:** which cloud/edge splits survived.

## Done since the last version (one line each)

| item | result (one line) | source |
|---|---|---|
| Chunk-wise adaptive + commit policy (old step 1) | Became **ours: teacher-parity mode** (`cw07_safe`). Bench-43: Δ quality −0.07 ± 0.06, Δ drift −3.3 ± 3.1, 1 % boundary jumps. Speed: 1.74× under load, 1.67× quiet steady. | facts §12, §15; full table in s5 notes |
| Quiet timing pass | Teacher 2.939 s/chunk = 4.08 fps. Modes 1.764 / 1.232 / 0.763 s/chunk = 1.67× / 2.39× / 3.85×. Realtime is 0.763 s/chunk quiet vs 0.761 under load, so load does not explain its gap to 704 fused (0.630 s/chunk in the same quiet pass). | facts §15 |
| Chenfeng-method latency profile (incl. ours) | Same stack and protocol: SDVG is slower than target only on average over the 6 clips (optimized: 51.2 / 43.3 vs 37.8 s per clip); τ −2.2 is faster on 1 of 6 clips per stack, only where ≥ 7 of 13 chunks came from the draft. Ours realtime has a lower action→frame p50 than every profiled arm: 1296 ms vs teacher 3237 ms and student 1434 ms (one GPU, decode competing), at 10.8 fps output. No mode or profiled arm meets the 750 ms per-frame budget. | facts §13 |
| StEvo-38 subset (7 arms, local Qwen3-VL-32B judge) | State evolution and task success sit at the floor (0–0.08) for every arm. On physical inaccuracy ↓ (38 tasks, seed 1234, local Qwen3-VL-32B judge), the student (1.00) is clearly worse than every 14B-based arm (0.55–0.71; paired p < 0.001). Waypoint (0.92) is worse by 0.21–0.37: paired p 0.003–0.035 for the teacher and both our modes, borderline for SDVG (p = 0.057; uncorrected). MG2 (0.79) is within ~0.15 of the teacher (0.68); only ours quality mode separates from MG2 (p = 0.035). | facts §14; paired tests data/f12_stevo_table.md |
| 448 / 384 low-res points (12-prompt × 3-seed, 2-step) | Quality 6.17 ± 0.18 (448) and 5.06 ± 0.25 (384) vs 6.83 ± 0.14 at 576. Drift 58.2 / 66.9. Quality falls steeply below 576. | facts §4; f08 |

### Where the three modes stand

| arm | quiet steady fps ↑ | speed-up, quiet steady ↑ | speed-up, bench-43 under load ↑ | Δ quality vs teacher, bench-43 ↑ | Δ drift vs teacher, bench-43 ↓ | boundary jumps > 1.5× ↓ | action→frame p50, one GPU ↓ |
|---|---|---|---|---|---|---|---|
| *reference: teacher 14B 4-step* | *4.08* | *1.00×* | *1.00×* | – | – | *1 %* | *3237 ms* |
| cw07_sep: chunk-wise 0.7, separate commit (strict-parity alternative, Q8) | 5.94 | 1.45× | 1.51× | **+0.03 ± 0.05** | <u>−0.6 ± 3.2</u> | <u>2 %</u> | n/a |
| ours: teacher-parity mode (cw07_safe: chunk-wise 0.7, safe commit) | 6.80 | 1.67× | 1.74× | <u>−0.07 ± 0.06</u> | **−3.3 ± 3.1** | **1 %** | n/a |
| ours: quality mode (ours832: 832 2-step fused + partial) | <u>9.74</u> | <u>2.39×</u> | <u>2.5×</u> | −0.30 ± 0.08 | +2.5 ± 3.0 | <u>2 %</u> | <u>1995 ms</u> |
| ours: realtime mode (ours704: 704 2-step fused + partial) | **15.73** | **3.85×** | **4.1×** | −0.62 ± 0.09 | +5.9 ± 3.4 | <u>2 %</u> | **1296 ms** |

Bold = best and underline = second among the ranked rows; ties share the mark. The teacher row is a reference and is not ranked.

Sources:
- Quiet steady: facts §15. This is jar + castle, generation only: decode excluded from the timed interval (the quality/realtime runs had streaming full-VAE decode on the same GPU; the teacher and chunk-wise runs had no decode). c704_fused gives 0.630 s/chunk here vs 0.640 on an exclusive GPU (README V3), so the contention looks negligible. The chunk-wise rows use chunk-to-chunk time. The quality and realtime rows use the cloud-only `cloud_chunk_s`, as on bench-43, so they are not strictly like-for-like with the teacher.
- Under-load ratios: facts §12. These are listed fps ÷ 3.9. Computed from the raw s/chunk instead, they are 1.49× / 1.72× / 2.48× / 3.99×.
- Δs and boundary jumps: bench-43, 86 paired clips (facts §12).
- Action→frame p50: facts §13. One GPU, with streaming full-VAE decode on the same GPU. Ours has n = 5 clips (castle s1234 excluded because of a recompile); the teacher has n = 6 (sdvg_profile run). The chunk-wise arms run through `spec_early`, which has no streaming decode, so they have no per-frame latency (n/a).

The Δ marks rank point estimates. The cw07_sep and teacher-parity Δ qualities are each within ~1.5 SE of zero. Paired against each other, the safe commit costs −0.10 ± 0.06 (1.7 SE; facts §12).

## Ranked plan (open items)

| # | next step | status | cost | expected value | evidence |
|---|---|---|---|---|---|
| 1 | **Main proposal: DMD fine-tune of the 14B at 704** (1 step). Fallback: 576–672 at 2 steps. | not started | **high**: 14B training (generator + real score + fake score) | the only route we see to ≥ 20 fps at ≥ 7.2 | Corner empty (f05). Realtime: 0.763 s/chunk quiet vs the 0.6 s budget. 576 fused + partial meets the budget (0.468 s/chunk quiet) but its quality was never scored; the other 576 arms score 6.58 (fused) to 6.83 (separate). Below 576 quality falls steeply (448: 6.17, 384: 5.06). |
| 2 | Uniform 3-step control (bench-43 and 12-prompt × 3-seed), separate and fused + partial commit | not started | low: queue + VLM scoring + quiet timing, no training | shows whether the learned head adds anything over 3 fixed steps; the professors will ask | The cw07 arms run 2.91–3.01 steps on bench-43 and 2.94–3.15 in the quiet pass. No uniform 3-step arm exists in either run. |
| 3 | Recipe on Matrix-Game 2.0 (scalability) | speed levers done 10/10; commit levers and chunk-wise head not ported | low–medium: MG2 runner exists | answers Guandao's 10/10 key criterion | steps + resolution: −40 % time → 34.3 fps; drift head AUROC 0.77 |
| 4 | Human blind study | page ready (90 trials) but **still no raters**; candidate set predates all three modes | low compute; **needs raters**; candidate set must be rebuilt | validates the VLM quality prompt, on which the teacher-parity claim rests | README §O(iii) |
| 5 | StEvo with the paper's judge (Gemini) | new; **needs a Gemini API key** (none in our environment) | low compute for re-judging the 7 arms' existing videos (+ API cost). Adding the teacher-parity mode needs generation and disk. | numbers comparable to the StEvo leaderboard; shows whether the floor in state evolution comes from our local judge | Local judge: state evolution / task success 0–0.08 for every arm (facts §14). Default judge of the eval code is Gemini (facts §9). |
| 6 | Idle-action trick | idea only | low–medium: scheduling only, no training | lower action→frame latency during idle periods; no throughput gain | measured target: realtime p50 1296 ms (one GPU); not measured with the trick |
| 7 | Distribution-matched distillation of the 1.3B | per-scene MSE pilot only (blurred) | **high**: new training pipeline | a better edge-only arm; it does **not** rescue SDVG | 10/8 calibration: draft 0.607 σ from the 14B, 7 % of tokens within ε (14B's own step 1: 0.308 σ, 44 %) |

How the order changed:
- The old #1 (chunk-wise + commit policy) and the old #8 (StEvo-38 subset) are done. They moved to the table above.
- DMD moves to #1. The new low-res points (448 / 384) and the quiet timings both point to starting at 704 (§1).
- StEvo with the Gemini judge is new at #5.

## 1. DMD fine-tune of the 14B, starting at 704 (the only route to ≥ 20 fps at ≥ 7.2)
- **Budget.**
  - 20 fps means ≤ 0.6 s of cloud time per 12-frame chunk.
  - The realtime mode needs 0.763 s quiet (0.761 s under load), so it has to lose ≈ 0.16 s/chunk.
  - Two other configs are closer to the budget:
    - 704 fused without partial: 0.630 s quiet (19.04 fps), but 12 % boundary jumps.
    - 576 fused + partial: 0.468 s/chunk quiet (25.66 fps), so it meets the budget. Its VLM quality was never scored. 576 fused scored 6.58 and 576 separate 6.83, both below 7.2.

| config (VLM: 12-prompt × 3-seed) | fps as listed in facts §3/§5 ↑ | quiet steady fps, facts §15 ↑ | VLM quality ↑ | VLM drift ↓ | boundary jumps > 1.5× ↓ | meets ≥ 20 fps and ≥ 7.2? |
|---|---|---|---|---|---|---|
| 704 cloud, separate commit, native | 12.7 | 12.50 | **7.22 ± 0.15** | 54.2 ± 5.0 | **3 %** | no (fps) |
| ours: realtime mode (704 fused + partial 1) | 15.8 (under load) | 15.73 | 6.86 ± 0.17 | <u>53.5 ± 4.6</u> | <u>4 %</u> | no (both) |
| 704 fused, no partial (jumpy) | 18.7 (exclusive GPU) | <u>19.04</u> | <u>7.08 ± 0.16</u> | **49.3 ± 4.7** | 12 % | no (both) |
| 672 fused (jumpy) | <u>22.4</u> | not in the pass | 6.89 ± 0.15 | 55.4 ± 4.6 | 12 % | no (quality) |
| 576 fused (jumpy) | **32.0** (measured, 0.375 s/chunk) | not in the pass | 6.58 ± 0.18 | 60.3 ± 4.6 | 18 % | no (quality) |
| 576 fused + partial 1 | – | **25.66** | not scored | not scored | not measured | fps yes; quality unknown |
| Matrix-Game 2.0 (640×352; 36 clips) | 20.5 | 19.7 (whole-clip, own pipeline incl. its decoder; not steady) | 7.06 ± 0.17 | 57.9 ± 4.8 | n/a | no (quality) |
| *reference: 14B 4-step teacher, 832* | *4.9 (whole-clip, quiet)* | *4.08* | *7.64 ± 0.16* | *45.4 ± 5.2* | *2 % (s1234)* | – |

Bold = best and underline = second; ties share the mark. The teacher row is a reference and is not ranked.

Notes on the table:
- The two fps columns use different bases and runs:
  - listed: facts §3/§5, mixed load conditions;
  - quiet steady: facts §15, jar + castle, generation only, decode excluded from the timed interval (the 704 / 576 arms ran streaming decode on the same GPU; see the Sources note above).
- MG2's quiet value is a whole-clip average (153 frames / gen s) including its own decoder; ours are steady, cloud-only, decode excluded. So MG2 is not ranked in the quiet column.
- The 576 drift/quality SEs come from README §V2. The 576 fps is the median measured cloud time per chunk (facts §5).
- The MG2 row excludes a duplicated warm-up clip, so MG2 (7.06) ranks third on quality, behind 704 fused (7.08).

On bench-43 the corner is empty too (f05, under load):
- MG2: 20.0 fps at quality 6.91.
- Ours, quality mode: 7.21 at 9.8 fps.
- Ours, teacher-parity mode: 7.44 at 6.8 fps.

- **Why 704 first.**
  - Along the native 2-step curve (832 / 704 / 576 / 448 / 384), quality is 7.53 / 7.22 / 6.83 / 6.17 / 5.06. 704 has the smallest gap to recover. 448 and 384 rule out going below 576 (f08).
  - Variant (a), 704 at 1 step, removes one of the two denoising forwards: one 704 forward = 0.31 s (README §U), more than the ≈ 0.16 s the realtime mode must lose.
  - Variant (b), 576–672 at 2 steps, is the fallback. 576 fused + partial already meets the time budget (0.468 s/chunk quiet), so (b) only has to recover quality (≥ 7.2; the scored 576 arms are at 6.58–6.83, and 576 fused + partial itself is unscored). On one GPU with streaming decode it gives 17.1 fps whole-clip output and action→frame p50 818 ms, but 60 % of frames miss 750 ms (facts §15). Guandao on 10/5: about 100 DMD fine-tune steps on the missing resolution usually recover quality.
- **Inference-time levers are used up.** None of these closed the low-res gap:
  - more steps at 576 did not reduce drift (2/3/4 steps: 65–77 % of chunks flagged vs 40–55 % at 832; jar/castle × 3 seeds);
  - anchoring the first chunks at 832;
  - an adaptive 576→832 switch;
  - the 1.3B refiner, or Real-ESRGAN.

  Chunk-wise adaptive steps speed up the 4-step schedule. They do not touch the resolution gap. Post-training on generated data failed too: resolution distillation collapsed, and RFT/DPO had no effect (f08, s4 notes).
- **Why DMD might work where our fine-tunes failed.** Our resolution distillation regressed x0 from σ = 1, which taught the mean (flat grey tabletops). DMD uses a distribution-matching gradient instead (real score vs fake score). That difference is exactly what has to be tested.
- **Known pitfalls:**
  - SageAttention has no backward, so training must use SDPA. All three earlier 14B LoRA runs diverged with Sage.
  - 704 and lower are outside the official sizes (480×832, 720×1280), so even the "real" score network is off-distribution there.
- **Proposed go/no-go** (12-prompt × 3-seed, then bench-43): quality ≥ 7.2 at ≤ 0.6 s/chunk (quiet steady), with boundary jumps ≤ 4 %.

## 2. Uniform 3-step control
- **Why.** The teacher-parity mode averages 3.01 steps on bench-43 (3.10 in the quiet pass). Its 1.67× (quiet) could come almost entirely from two things: running 3 steps instead of 4, plus the commit saving.
  - If uniform 3-step + partial commit matches it on paired Δ quality and Δ drift, the paper should present "3 steps + partial commit" and drop the head, which is simpler.
  - If the head is clearly better, the head is a contribution.
- **Arms:**
  - uniform 3-step with separate commit (control for cw07_sep);
  - uniform 3-step with fused + partial commit. At 3 steps the safe policy always picks partial, so this is also the control for the teacher-parity mode.
  - Run both on bench-43 (86 clips) and on the 12-prompt × 3-seed matrix (36 clips). Time them with the quiet-pass protocol (jar + castle) so the speed comparison is like-for-like.
- **Smaller-set evidence (unchanged).**
  - 10-prompt seed-1234 matrix (README §N): uniform 3-step runs at 6.2 fps, drift 44, quality 7.7. Chunk-wise runs at 6.5 fps, drift 36, quality 7.8 (whole-clip fps, 153 frames / gen s; not comparable with the quiet steady values).
  - README §H1: 14B denoise time 18.3 s (uniform 3-step) vs 17.8 s (chunk-wise, 2.92 steps). Error vs dense 4-step: 0.162 vs 0.146.
- **The quiet pass changes the cw05 picture.**
  - On jar + castle, the looser stop (cw05_pc1) does save time: 2.47 steps at 1.495 s/chunk (8.03 fps, 1.97×), vs cw07_pc1 at 3.15 steps and 1.771 s/chunk.
  - On bench-43 under load it saved almost none (1.722 vs 1.748 s/chunk).
  - The prompts differ, so the cause is still not isolated. Its bench-43 quality cost (−0.17 ± 0.07) stands.
- **Cost:** queue + VLM scoring, no training. It needs free disk for the videos.
- **Proposed decision rule:** paired Δ (chunk-wise − uniform 3-step) over 86 clips. Keep the head only if it wins on Δ quality or Δ drift by > 1 SE at equal or better quiet fps.

## 3. Recipe on Matrix-Game 2.0 (scalability)
Done on 10/10 (10 prompts, seed 1234; README §M, §N):

| MG2 config | fps ↑ | DINO min cos ↑ | VLM drift ↓ | VLM quality ↑ |
|---|---|---|---|---|
| 3-step 640×352 (as released) | 20.5 | **0.149** | **55.0** | **7.1** |
| 2-step 640×352 | 23.6 | <u>0.136</u> | – | – |
| 3-step 448×256 | <u>28.8</u> | 0.100 | – | – |
| 2-step 448×256 (both levers) | **34.3** | 0.114 | <u>71.5</u> | <u>5.7</u> |

Bold = best and underline = second in each column. VLM columns average the 10 prompts and exclude a duplicated warm-up clip (easy_lake_warm). README §N includes it and lists 57 / 7.2 and 72 / 5.8. In the 10/12 quiet pass, MG2 as released ran at 19.7 fps (whole-clip, own pipeline incl. its decoder; facts §15).

- **What transfers:**
  - The speed levers apply unchanged: steps 3→2 save 13 % of the time, resolution (0.51× tokens) saves 30 %, and both together save 40 %.
  - The drift head transfers too: AUROC 0.77 on held-out scenes (block 10, VLM labels).
  - The quality trend also replicates: lower resolution loses the scene earlier, as it does in LingBot.
- **Not yet done:**
  - port fused + partial commit to MG2 (first check whether MG2 has a separate clean K/V pass);
  - chunk-wise adaptive steps on MG2. The convergence head is trained on the LingBot 14B's hidden states, so MG2 would need its own head;
  - bench-43 VLM scores for MG2 with the recipe.
- **Open question for the professors:** does MG2 count as the second base model, or is a camera + text model needed? Matrix-Game 3.5 runs at 0.9 fps end to end.

## 4. Human blind study (still needs raters)
- **Ready:** the page https://claude.ai/artifact/HyFxbLBV5Ab1ouMPnLXXdW has 90 pairwise trials (10 prompts × 9 candidates, each vs the teacher, seed 1234). Raters need Contributor access to save. **No one has rated yet.**
- **Must be rebuilt before recruiting.** The page was built on 10/10, before the current modes existed:
  - 4 of its 9 candidates are now retired: 576 + ESRGAN, 576 + refiner, adaptive 576→832 and 704 + refiner.
  - All three operating modes (teacher-parity, quality, realtime) and SDVG are missing.
  - Rebuilding is cheap (`scripts/blind_build.py`).
- **Why it matters.** The teacher-parity claim (Δ quality −0.07 ± 0.06) rests entirely on the VLM quality prompt, which is **not** validated against humans. The drift prompt is (AUROC 0.82).

## 5. StEvo with the paper's judge (Gemini) (new)
- **Blocker:** a Gemini API key. The StEvo eval code defaults to Gemini (facts §9). We ran it with a local Qwen3-VL-32B judge, and our environment has no Gemini key.
- **What the local-judge run showed** (facts §14, 38 tasks, seed 1234, 7 arms; binomial s.e. ≈ 0.05–0.08):
  - State evolution and task success are at the floor (0–0.08) for every arm, so they do not separate the arms.
  - Physical inaccuracy ↓ is the informative column. The 14B-based arms score 0.55–0.71: teacher 0.68, ours 0.55 (quality) and 0.68 (realtime), SDVG τ −0.7 0.71. The student (1.00) is clearly worse than every 14B-based arm (paired p < 0.001). Waypoint (0.92) is worse by 0.21–0.37: paired p 0.003–0.035 for the teacher and both our modes, borderline for SDVG (p = 0.057; uncorrected exact McNemar, data/f12_stevo_table.md). MG2 (0.79) is within ~0.15 of the teacher.
  - MG2's occlusion-done is 0.97 because it turns away visibly, but it shows the prompted action in only 0.24 of tasks.
- **Plan once a key exists:**
  - re-judge the existing videos of the 7 arms (`lingbot-world-v2/specwm_out/stevo38/`);
  - then add the teacher-parity mode (not run, for lack of time) and SDVG τ −2.2 (not run, for lack of disk). Both need generation and disk.
- **Value:** numbers comparable to the paper's leaderboard, and a check whether the state-evolution floor comes from our judge.

## 6. Idle-action trick (10/5 remark on idle actions)
- **Idea.** Actions do not change every chunk. While the input is unchanged, the next chunk's conditioning is known one chunk (0.75 s) ahead. Starting chunk n+1 early could hide the action→frame latency during idle periods.
- **Measured target (facts §13).** On one GPU with decode competing:
  - ours realtime has action→frame p50 1296 ms, and ours quality 1995 ms;
  - the teacher has 3237 ms;
  - no configuration meets the 750 ms per-frame budget.
- **What it cannot do.** All three modes run at or below the 16 fps playback rate (quiet steady 6.80 / 9.74 / 15.73 fps; realtime 0.763 s/chunk vs 0.75 s). The idle trick can hide latency, but it cannot make a mode that is slower than playback keep up. That needs a start-up buffer, which adds delay.
- **Missing input:** the share of idle chunks in real action streams. Our bench camera paths are scripted and fully known in advance, so they cannot measure this.
- **Cost:** scheduling code only. **Value:** action→frame latency only, not throughput.

## 7. Distribution-matched distillation of the 1.3B
- **Why.** The 1.3B is not a usable proposal for the 14B.
  - In the 10/8 6-scene calibration (README §E), its chunks are 0.607 σ from the 14B output, with 7 % of tokens within ε.
  - The 14B's own step 1 is 0.308 σ away (44 % within ε), and the 14B with a different noise is 0.405 σ away (34 %).
  - On bench-43 the student scores quality 5.83 ± 0.16 and drift 75.1 ± 2.2, at 17.2 fps (under load).
- **Pilot (10/8):** a per-scene paired LoRA with x0/MSE matching on jar + castle (1,152 steps, 90 min):
  - Against the same 10/8 calibration, the gap fell from 0.607 to 0.483 σ, and tokens within ε rose from 7 % to 18 % (12 % on unseen scenes).
  - But textures were smeared outside one scene. This needs a distribution term, not MSE.
- **Value is on the edge-only side** (Guandao 10/10: "bring the draft's quality up to the target's level"). It is **not** a way to make SDVG fast:
  - accepting every draft chunk would still cap SDVG at ≈ 7.6 fps (under load; f07);
  - in the latency profile, SDVG is slower than target only on average over the 6 clips (optimized: 51.2 / 43.3 vs 37.8 s per clip); τ −2.2 is faster on 1 of 6 clips per stack, only where ≥ 7 of 13 chunks came from the draft (facts §13).

## Questions for the professors
**Guandao**
1. DMD: we plan to start at 704 at 1 step. The realtime mode is 0.763 s/chunk quiet, against a 0.6 s budget, and one 704 forward is 0.31 s. The fallback is 576–672 at 2 steps, where 576 fused + partial already runs at 0.468 s/chunk and only quality must be recovered. Do you agree with this order? Are about 100 fine-tune steps on the missing resolution enough, as you suggested on 10/5?
2. DMD real score: at low res the 14B is outside its supported sizes. Should the real score be the 14B at 704, or the 832 teacher with its outputs downsampled? Real video or self-rollouts? (Our fine-tunes on generated data failed.)
3. Is a public DMD training recipe for causal few-step video models (e.g. Self-Forcing / CausVid style) the right starting point, or do you have code to share?
4. Scalability: is Matrix-Game 2.0 enough as the second base model, or do we need a camera + text model?
5. Idle actions: is there an action-stream dataset or usage statistic we can use for the idle share?

**Diana**

6. Is f04_speed_bridge (time removed and Δ quality per lever) the sensitivity analysis you asked for on 10/5? Or do you want a speed-up-per-quality-loss table?
7. Human study: the page is ready but has no raters. Can your group provide raters? How many raters and trials would be enough for the paper?

**Both**

8. Teacher parity: is the safe-commit mode enough to call "teacher parity"? It runs at 1.67× quiet (1.74× under load) with Δ quality −0.07 ± 0.06 and Δ drift −3.3 ± 3.1 over 86 paired clips. Or should the paper use separate commit, the only arm inside 1 SE on both Δs (1.45× quiet, 1.51× under load, Δ quality +0.03 ± 0.05)?
9. Realtime target: is 16 fps playback (0.75 s per chunk) enough, or do we need ≥ 20 fps? The realtime mode runs at 0.763 s/chunk quiet (15.73 fps), and its action→frame p50 is 1.30 s on one GPU with decode.
10. Device: is a server-class GPU acting as the "device" acceptable for now, or do we need real edge hardware (which)?
11. Framing: speculative decoding with the 1.3B is closed. Should the paper be framed as follows?
    - cloud-side acceleration with three operating modes: teacher-parity 1.67×, quality 2.39×, realtime 3.85× (quiet steady);
    - plus a low-res cloud / device-decode split (f03);
    - with SDVG as a baseline.
12. StEvo: can the lab provide a Gemini API key (or budget), so we can report StEvo with the paper's judge?

### Caveats
- **The ranking and the expected values are judgment, not measurements.** Every open item (1–7) is a plan.
- **"Teacher parity" is a VLM-only, within-noise statement:**
  - Δ quality −0.07 ± 0.06 (10 better / 60 equal / 16 worse) is just over 1 SE below zero. The strict 1-SE check is passed only by cw07_sep.
  - Δ drift −3.3 ± 3.1 favours ours but is not significant.
  - The quality prompt is not human-validated. Its 8 frames per clip cannot see chunk-boundary jumps, which are measured separately (1 %).
- **The learned head's own contribution is unproven.** The chunk-wise arms run ≈ 3 steps, and no run has a uniform 3-step arm (step 2).
- **Speed and quality come from different prompt sets:**
  - The preferred speed-ups (1.67× / 2.39× / 3.85×) come from the quiet pass on jar + castle. That is 6 clips per arm; ours quality / realtime have 5, because castle s1234 paid a recompile.
  - Every Δ comes from bench-43. Mean steps differ too (teacher-parity 3.10 quiet vs 3.01 on bench-43).
  - The quality / realtime s/chunk count the cloud GPU only. With the device step run serially on the same GPU, bench-43 gives 8.9 / 13.6 fps (facts §12).
- **Latency numbers are whole-clip output fps and per-frame times on one GPU, with the decode competing.** A VLM judge server ran on another GPU throughout. A separate device GPU was not measured end to end.
- **"Only route to ≥ 20 fps at ≥ 7.2" covers only what we tried:**
  - inference-time levers and post-training on generated data, on this base model;
  - training on real low-res video, or a base model trained at low res, is untested.
  - 576 fused + partial meets the fps budget, but its quality was never scored.
- **The corner table mixes timing bases.** The listed fps come from facts §3/§5 (mixed load conditions). The quiet column is facts §15. MG2's quiet value is whole-clip and includes its decoder, so it is not ranked there.
- **StEvo-38 used a local judge** and is not comparable to the paper's leaderboard. Differences < ~0.15 are not significant, and two arms are missing (teacher-parity, SDVG τ −2.2).
- **The MG2 transfer evidence is thin:** 10 prompts, one seed, DINO + VLM. The commit levers and the chunk-wise head are not tested on MG2.
- **The blind study has no ratings yet,** and its current candidate set includes none of the three modes.
- **No real edge device or network link** was used in any measurement.

### 발표 메모 (Korean)
- 첫 문장: 지난 버전 이후 계획했던 다섯 가지 (chunk-wise + commit policy 포함)가 끝남. chunk-wise + commit policy(이제 ours: teacher-parity mode), quiet timing pass, Chenfeng 방식 latency profile(ours 행 포함), StEvo-38 subset, 448/384 저해상도 점.
- 속도는 quiet steady 기준으로 말하기: teacher-parity 6.80 fps(1.67×), quality 9.74 fps(2.39×), realtime 15.73 fps(3.85×). bench-43 부하 상태 비율(1.74× / 2.5× / 4.1×)과 ~5 % 안에서 일치함. 품질 Δ는 bench-43 기준 −0.07 / −0.30 / −0.62.
- 정정부터 말하기: teacher "4.9 fps quiet"는 clip 전체 평균(초반의 싼 chunk 포함)이었음. steady 기준으로는 quiet 4.08 / 부하 3.9. 그래서 4.9 기준으로 낸 1.4× / 2.0× / 3.2×는 쓰지 않음.
- realtime mode는 quiet에서도 0.763 s/chunk(부하 0.761과 거의 같음). 같은 pass에서 704 fused가 0.630이니, 그 차이는 load가 아니라 partial commit 비용임. 16 fps 재생(0.75 s)보다 아주 조금 느림.
- latency: 한 GPU에서 decode가 같이 돌 때 realtime의 action→frame p50은 1.30 s로 프로파일한 arm(draft / target / SDVG)보다 모두 낮음(teacher 3.24 s, student 1.43 s). 그래도 750 ms 기준을 맞추는 모드나 arm은 없음. 같은 stack에서 SDVG는 6 clip 평균으로 target only보다 느림(τ −2.2는 13 chunk 중 7개 이상을 draft에서 가져온 clip에서만, stack마다 6개 중 1개 clip에서 더 빠름).
- StEvo-38: state evolution / task success는 모든 arm이 바닥(0–0.08)이라 구분이 안 됨. 정보가 있는 건 physical inaccuracy뿐: 14B 계열 0.55–0.71 vs student 1.00(모든 14B 계열 대비 paired p < 0.001, 확실히 나쁨), Waypoint 0.92(0.21–0.37 나쁨; teacher와 우리 두 모드 대비 paired p 0.003–0.035로 유의, SDVG 대비는 p 0.057로 경계선). MG2(0.79)는 teacher(0.68)와 ~0.15 안이라 유의하지 않음. 로컬 Qwen judge라 논문 리더보드와 비교할 수 없고, 논문 judge(Gemini)로 돌리려면 API key가 필요함(Q12).
- 448/384는 6.17 / 5.06으로 576 아래에서 품질이 급락함. 그래서 DMD는 704부터 하고, 576 아래로는 내려가지 않음.
- 다음 단계 1번은 DMD at 704(1-step). realtime은 0.6 s/chunk까지 ≈ 0.16 s를 줄여야 하는데, 704 forward 하나가 0.31 s임. 대안은 576–672 2-step: 576 fused + partial은 이미 quiet 25.66 fps라 품질(≥ 7.2)만 회복하면 됨. 품질 점수는 아직 없음. Guandao 교수님께 순서와 ~100 step이면 되는지 여쭙기(Q1–Q3).
- 가장 싼 다음 결과는 uniform 3-step control(2번): learned head가 고정 3 step보다 나은지 확인. 참고로 quiet pass에서는 cw05가 실제로 시간을 줄였음(1.495 vs 1.771 s/chunk). bench-43 부하 결과와 다르고 프롬프트도 다르니, 원인은 아직 모른다고 말하기.
- 블라인드 스터디: 페이지는 준비됐지만 평가자가 아직 없음. teacher-parity 주장은 사람 검증이 없는 VLM quality prompt 하나에만 기대고 있으니, 세 모드를 넣어 다시 만들고 Diana 교수님께 평가자 지원을 부탁드리기(Q7).
- 마지막으로 결정 요청: parity를 safe(quiet 1.67×, 부하 1.74×)로 부를지 separate(quiet 1.45×, 부하 1.51×)로 부를지(Q8), 목표 fps 16 vs 20(Q9), 디바이스 정의(Q10), 논문 framing(Q11), Gemini key(Q12).
