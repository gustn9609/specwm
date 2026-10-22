# Meeting package 10/12–13 (Guandao, Diana): answers to the five 10/5 action items

Final index, 10/12 08:15 CDT (completeness pass). Every number comes from `facts.md` (the single source of truth) or from a raw file it names; numbers computed in the notes but not listed in `facts.md` are marked **derived** and must keep that label on a slide. Paths are relative to `/home/hk28965/specwm`, except `figs/`, `notes/`, `data/` and `scripts/`, which are relative to this folder. Times are Austin CDT. All timings: one NVIDIA RTX PRO 6000 Blackwell (96 GB), not an H100.

**Speed convention (read first).** The teacher's old "4.9 fps quiet" is a whole-clip average (31.2 s / 13 chunks, including the cheap early chunks), not a steady rate. On the steady basis used everywhere else the teacher runs at **4.08 fps quiet / 3.9 fps under load** (facts §15). Slides quote the **quiet steady speed-ups 1.67× / 2.39× / 3.85×** (teacher-parity / quality / realtime); the bench-43 under-load ratios 1.74× / 2.5× / 4.1× agree within ~5 % and go in parentheses only. Never divide anything by 4.9.

## 1. Headline numbers

Speed: quiet timing pass (facts §15; jar + sandcastle, steady s/chunk, generation only). Quality and drift: bench-43 (38 StEvo camera-WM tasks + 5 easy scenes × 2 seeds = 86 clips per arm, paired with the teacher on the same prompt and seed; facts §12). Latency: Chenfeng method, one GPU, full-VAE decode streamed on the same GPU (facts §13). StEvo-38: facts §14.

| mode | config | quiet steady s/chunk ↓ | quiet fps ↑ | speed-up vs teacher, quiet (under load) ↑ | Δ VLM quality vs teacher ↑ | Δ VLM drift vs teacher ↓ | boundary jumps > 1.5× ↓ | action→frame p50, one GPU ↓ | StEvo-38 physical inaccuracy ↓ |
|---|---|---|---|---|---|---|---|---|---|
| *teacher 14B 4-step (reference)* | *official 4-step schedule* | *2.939* | *4.08* | *1.00×* | – | – | *1 %* | *3237 ms* | *0.68* |
| **ours: teacher-parity mode** | chunk-wise adaptive 0.7 (learned 14B head) + safe commit (`cw07_safe`) | 1.764 | 6.80 | 1.67× (1.74×) | **−0.07 ± 0.06** | **−3.3 ± 3.1** | **1 %** | n/a ᵃ | not run |
| **ours: quality mode** | 832, 2-step, fused + partial commit (`ours832`, run dirs `pc1_832`) | <u>1.232</u> | <u>9.74</u> | <u>2.39× (2.5×)</u> | <u>−0.30 ± 0.08</u> | <u>+2.5 ± 3.0</u> | <u>2 %</u> | <u>1995 ms</u> | **0.55** |
| **ours: realtime mode** | 704, 2-step, fused + partial commit, device = VAE decode + bicubic only (`ours704`, `pc1_704`) | **0.763** | **15.73** | **3.85× (4.1×)** | −0.62 ± 0.09 | +5.9 ± 3.4 | <u>2 %</u> | **1296 ms** | <u>0.68</u> |

Bold = best, underline = second among the three modes (ties share the mark; the teacher row is a reference). Δ = paired mean ± s.e. over 86 clips. ᵃ `spec_early` has no streaming decode, so the teacher-parity mode has no per-frame latency yet.

How to read it:
- **Teacher-parity is a tie, not a win.** Both Δs are within ~1.5 s.e. (95 % CIs include 0). It rests on the VLM quality prompt, which is not validated against humans (the drift prompt is: AUROC 0.82).
- **Timing basis.** Teacher-parity is timed like the teacher (chunk-to-chunk), so 1.67× is like-for-like. Quality and realtime are cloud-GPU-only (`cloud_chunk_s`) and assume the device step runs in parallel on a separate device; counted serially on one GPU they are 8.9 / 13.6 fps (2.25× / 3.44×, bench-43 under load; facts §12). With the full-VAE decode streamed on the same GPU they output 6.9 / 10.8 fps (whole-clip; teacher 4.2; facts §13).
- **Realtime vs Matrix-Game 2.0** (bench-43, 86 pairs; facts §2): same quality (paired −0.01 ± 0.10; 6.90 vs 6.91), 13 points less drift (paired −13.1 ± 3.6). MG2 is faster: 20.0 vs 15.8 fps under load, 19.7 vs 15.73 quiet (MG2 = whole clip incl. its decoder, no torch.compile).
- **Realtime is just below real time:** 0.763 s/chunk vs 0.75 s of playback. No configuration meets the 750 ms per-frame deadline (realtime misses 96.7 % of frames).
- **The student is faster than every mode** (17.2 fps under load, bench-43) but at quality 5.83 / drift 75.1 (realtime 6.90 / 56.9). The pitch is quality at near-student speed.

Other headline facts (facts § in brackets):
- **Speculative decoding (SDVG + actions) is closed.** Paper τ −0.7: 3.5 fps whole-clip (3.1 steady) vs teacher 3.9 under load, 44.6 vs 35.8 s per clip at equal quality [§2]. τ −2.2: 6.4 fps (6.7 steady) at Δ quality −0.84 ± 0.10, Δ drift +19.7 ± 3.4 [§2]. Accept-all ceiling ≈ 7.6 fps at student quality [§2]. Same stack, per frame, jar + sandcastle: slower than target-only at both τ (51.2 / 43.3 vs 37.8 s per clip; p50 4463 / 4156 vs 3237 ms) [§13]. Physics verifier: never faster than target-only (66–128 vs 62.5–64.4 s, original stack); its go criterion was not measured (videos deleted 10/9) [§7].
- **Low resolution (14B alone, 12 prompts × 3 seeds):** quality 7.53 / 7.22 / 6.83 / 6.17 / 5.06 at 832 / 704 / 576 / 448 / 384; drift jumps at 384 (66.9) [§4]. The 1.3B refiner cost −0.50 ± 0.11 at 704, so the device only decodes [§3].
- **Fused-commit jumps:** 704 fused 12 % of boundaries → partial commit 4 % (separate 3 %) [§5], for +0.13 s/chunk quiet (0.630 → 0.763; ≈ +0.12 on the same 5 clips, derived) [§15].
- **StEvo-38:** physical inaccuracy separates the 14B-based arms (0.55–0.71) from the student (1.00, all p < 0.001) and, except SDVG (p = 0.057), from Waypoint (0.92); against MG2 (0.79) only the quality mode separates (p = 0.035). State evolution / task success are at the floor (0–0.08) for every arm [§14].
- **Rollout paths:** ours 704 is as far from the teacher as the student (1.15 vs 1.17; paired −0.02 ± 0.04); MG2 is clearly farther (1.45). Distance, not quality [§11].

## 2. The five 10/5 action items: answer, evidence, figure

| # | 10/5 action item | answer | evidence | figures | note |
|---|---|---|---|---|---|
| 1 | Report latency as TTFF, steady fps, action→frame (Chenfeng's code), keep the 10 s clip time | **Done.** Steady fps for every arm (bench-43 under load; quiet pass). Per-frame (Chenfeng method, one GPU, decode streamed): realtime p50 1296 ms, quality 1995 ms, teacher 3237 ms, student 1434 ms; output 10.8 / 6.9 / 4.2 / 10.4 fps. TTFF ≈ 2.57 s (realtime) vs 3.71 s (teacher) and 10 s clip e2e ≈ 16.6 vs 40.2 s, composed with an assumed 2.0 s prepare (derived, s2 table e2). **Open:** teacher-parity per-frame row. | facts §2, §13, §15 | **f13_latency**, f01_pipelines, f09_decoder | s2 |
| 2 | Physics-consistency verifier in SDVG: do the final videos beat the baselines on physics? (go/no-go) | **NO-GO on cost and proxy evidence.** Physics-verifier SDVG never faster than target-only; ImageReward SDVG slower than the teacher at paper τ and student-like at τ −2.2; StEvo-38 shows no physics benefit (SDVG 0.71 vs teacher 0.68); the draft has almost no physics to accept (StEvo-85: 1.3B 1.2 % vs 14B 32.9–48.2 %). **Not measured:** physics of the physics-verifier's final videos (deleted 10/9). | facts §2, §7, §13, §14 | **f07_sdvg_analysis**, f13_latency, f12_stevo, f01_pipelines | s3 |
| 3 | Isolate the low-res loss (model outside its range vs resolution itself; supported sizes) | **Done.** 14B alone loses quality slowly to 576, steeply below (6.17 at 448, 5.06 at 384); drift jumps at 384. Officially supported sizes are only 480×832 and 720×1280, so every low-res arm is off the list. At 704 the refiner cost about as much as the resolution step (−0.50 ± 0.11 vs −0.31 ± 0.10; difference not significant, derived); refiner removed. Inference levers and post-training on generated data do not fix low-res drift. | facts §0, §3, §4, §8 | **f08_lowres_curve**, f09_decoder | s4 |
| 4 | (Diana) Combine the techniques; sensitivity = speed-up per quality loss for each | **Done.** Levers multiply (each moves ≤ 0.05× across contexts). Quiet bridge: 2 steps 1.95× · fused + partial 1.22× · 832→704 1.61× = 3.85×. Chunk-wise adaptive is free (12×3: +0.00 ± 0.11; bench-43: +0.03 ± 0.05) at 1.45×; the commit policy adds speed (1.66–1.67×) at a borderline cost vs chunk-wise alone (−0.10 ± 0.06 / −0.13 ± 0.07). Combined → the three modes. | facts §3, §5, §12, §15 | **f04_speed_bridge**, **f11_modes**, f02_partial_commit | s5 |
| 5 | Have the on-device SOTA (Waypoint) at hand; build on it only if our draft is far worse | **Done; do not build on either.** Waypoint-1.5-1B: quality 4.03 (below our student 5.83), drift +24.9. Matrix-Game 2.0: same quality cost as realtime (−0.60 vs −0.62) but +19.0 drift; paired ours − MG2 drift −13.1 ± 3.6. Both ran on bench-43, StEvo-38 and the path metric with the same start images and camera paths. | facts §2, §9, §11, §14, §15 | **f05_pareto_bench43**, **f06_paired_deltas**, f12_stevo, f10_paths_summary | s6 |

Related 10/5 remarks, not counted as the five items: "stress the draft" (s8 #7, 1.3B distillation pilot only) and "exploit idle actions" (s8 #6, idea only) are open.

## 3. The 10/10 points (Guandao)

| 10/10 point | status | evidence | figure / note |
|---|---|---|---|
| Scalability: the recipe must transfer to next-gen base models | **Partial.** Speed levers and the drift head transfer to Matrix-Game 2.0 (steps 3→2 −13 %, 0.51× tokens −30 %, both −40 % → 34.3 fps; drift head AUROC 0.77). Commit levers and chunk-wise head not ported; no next-gen base model run. | facts §9; README §M | s7 Part A, s8 §3 |
| Three cloud/edge splits (spec decoding is one of them) | **Mapped.** Low/high-res over time: closed. Device drafts + cloud corrects: closed. Cloud low-res + device decode/upsample: kept (realtime mode, without SR). | facts §7, §8 | **f03_splits_map**, s7 |
| Fair comparison: beat the best hybrid; edge-only at the same resolution | **Done on bench-43** (quality, drift, fps) and per frame for the LingBot arms. Teacher-parity 6.8 vs SDVG τ −0.7 3.1 fps (steady) at teacher quality; realtime beats SDVG τ −2.2 on fps, quality and drift; ≈360p block: realtime vs MG2. **Missing:** Waypoint at a matched resolution, the student at 704, per-frame latency for MG2 / Waypoint. | facts §2, §12, §13 | f05, f06, f13, s7 matched-resolution table |
| Use easier prompts | **Done.** 5 easy scenes in bench-43 (n = 10 clips): teacher drift 22 on easy vs 55 on tabletop & physics; the student is near the teacher on easy scenes. | facts §2 | s6 slide 4, s7 |

## 4. Figures (`figs/`, each .png + .svg; generators in `scripts/`, CPU only)

| id | file | caption (finding) | supports |
|---|---|---|---|
| f01 | `f01_pipelines` | What runs where per chunk: teacher, SDVG + actions, ours realtime. Ours needs 0.761 s cloud time per chunk (15.8 fps under load; 0.763 s quiet). SDVG τ −0.7 is slower than the teacher (3.5 whole-clip vs 3.9). Partial commit +0.13 s quiet. The device side (426 ms full VAE < 761 ms) is proposed, not measured end to end. Footer re-rendered 08:07 CDT. | overview; items 1, 2 |
| f02 | `f02_partial_commit` | (a) What one chunk leaves in the KV cache (separate / fused / partial). (b) Boundaries with a jump > 1.5×: 704 fused 12 % → partial 4 % (separate 3 %); 832 fused 7 % → 0 % (s1234); mid-chunk control 0.7–3.7 %. Title: +0.13 s per chunk quiet; footer gives the same-clip ≈ +0.12. | item 4; finding B1 |
| f02b | `f02b_jump_frames` | Frames 126–131 of easy_urban_rain s1234: 704 fused hard-cuts at 128→129 (r = 4.34); ours 704 is continuous (r = 1.21). | finding B1 |
| f03 | `f03_splits_map` | Cloud/edge splits: only routes that keep the 14B generating survive (ours 832 9.8 fps, ours 704 15.8 fps under load; chunk-wise + safe commit 1.67× quiet). Closed: spec decoding, Draft-then-Finish, verify + redraw, adaptive 576→832, commit offload, sequence parallel. Labels "ours 832 / 704" = quality / realtime mode. Footer re-rendered 08:07 CDT. | 10/10 splits; items 2, 3 |
| f04 | `f04_speed_bridge` | Quiet cloud time per chunk, one lever at a time: 2.94 s (4.08 fps) → 2 steps 1.50 s → fused + partial 1.23 s (quality mode, 9.74 fps) → 704 0.763 s (realtime, 15.73 fps) = 1.95× · 1.22× · 1.61× = 3.85×; 1.7 % over the 0.75 s budget. Teacher-parity mode is not a bridge step. Hatched reference: original 10/5 stack. | item 4; item 1 |
| f05 | `f05_pareto_bench43` | Quality and drift vs fps (bench-43, under load) for 9 arms. Realtime matches MG2 quality (6.90 vs 6.91) with 13.1 less drift, at 15.8 vs 20.0 fps; the ≥ 20 fps / ≥ 7.2 corner is empty. Title ratios quiet (under load in parentheses). Footer re-rendered 08:07 CDT. | item 5; next steps (DMD) |
| f06 | `f06_paired_deltas` | Paired Δ vs teacher (86 clips). Only the teacher-parity mode is faster than the teacher and within noise on both axes (−0.07 ± 0.06 / −3.3 ± 3.1, 1.67× quiet). Drift splits arms into near-teacher (SDVG τ −0.7, three modes) and lost-scene (MG2, SDVG τ −2.2, student, Waypoint: +19.0 to +24.9). | item 5; item 2 |
| f07 | `f07_sdvg_analysis` | Where SDVG's time goes: τ −0.7 44.6 s per clip vs teacher 35.8 s (draft + verify add 13.6 s); τ −2.2 24.4 s at quality 6.67; accept-all ceiling ≈ 20 s per clip (2.5× the student). Top bar: 10/5 original stack (80.0 s generation, 99.9 s total). Footer re-rendered 08:07 CDT. | item 2; item 1 |
| f08 | `f08_lowres_curve` | 14B alone, native output, 12 × 3: quality 7.53 / 7.22 / 6.83 / 6.17 / 5.06 at 832 → 384; drift jumps only at 384 (66.9); refiner −0.50 ± 0.11 at 704, −0.25 ± 0.12 at 576. No fps on this figure. | item 3 |
| f09 | `f09_decoder` | Full VAE 426 ms/chunk at 704 hides under the 761 ms cloud chunk on a separate device GPU of this class; one shared GPU: 10.1 fps (full VAE) / 15.5 fps (TAEHV); TAEHV costs −0.56 ± 0.13 (judge 448) / −1.06 ± 0.14 (judge 832). | items 1, 3 |
| f10 | `f10_paths_summary` + `f10_paths_<scene>` (lake, Stonehenge, urban rain, pouring, sandcastle, jar) | Rollout-path distance to the teacher, ratio to the teacher's own seed spread: ours 704 1.15 vs student 1.17 (paired −0.02 ± 0.04), MG2 1.45, Waypoint 1.24, SDVG 1.06 / 1.16; ours 832 (1.00) shares the teacher's noise, not comparable. Jar / castle n = 3 and empty text prompt. Distance, not quality. | item 5; 10/10 fair comparison |
| f11 | `f11_modes` | **Headline.** Teacher-level VLM quality at 1.67× quiet (1.74× under load); −0.30 at 2.39× (2.5×); −0.62 at 3.85× (4.1×). (a) quality vs bench-43 fps under load, chunk-wise inset; (b) paired Δ quality ± 1.96 s.e. with a table of Δ quality, Δ drift, mean steps and quiet fps (× teacher), incl. cw07_sep 1.45×, cw07_pc1 1.66×, cw05_pc1 1.97×. | overview; item 4; direction 1 |
| f12 | `f12_stevo` | StEvo-38 (38 tasks, seed 1234, local Qwen3-VL-32B judge): physical inaccuracy 0.55 (quality) / 0.68 (teacher, realtime) / 0.71 (SDVG) vs MG2 0.79, Waypoint 0.92, student 1.00, Wilson 95 % CIs; paired McNemar in the subtitle; other criteria in panel b (state evolution / task success at the floor). Title re-rendered 08:07 CDT so it no longer implies every 14B arm beats MG2. | item 5; item 2 |
| f13 | `f13_latency` | Per-frame action→frame latency (Chenfeng method, jar + sandcastle, one GPU, full-VAE decode streamed on the same GPU): realtime p50 1296 ms / 10.8 fps output, lowest of all baselines and modes (teacher 3237 ms / 4.2 fps, student 1434 ms / 10.4 fps); SDVG slower than target-only at both τ on both stacks; no config meets 750 ms. Lighter rows are non-mode variants (704 fused 1180 ms, 576 fused + partial 818 ms are lower). Teacher-parity has no row. | item 1; item 2 |

Data behind the figures: `data/f02_boundary_counts.json` (k/N for facts §5), `data/f10_paths_<scene>.json` + `f10_paths_summary.json` (facts §11), `data/f12_stevo.json` + `f12_stevo_table.md` (facts §14, incl. paired tests), `data/f13_latency_table.md` (facts §13), `data/e_steady_fps.json` (steady output fps, facts §13). `f11_modes` reads `paper/1012/bench43_summary.json` and the bench-43 `loop.json` files directly.

## 5. Notes, in presenting order (`notes/`; English body, Korean 발표 메모 at the end of each)

| # | file | covers | one-line headline |
|---|---|---|---|
| 1 | `s1_overview.md` | overview, 10/5 → 10/12 table, claims we can / cannot make | Three modes of the 14B: 1.67× / 2.39× / 3.85× quiet at Δ quality −0.07 / −0.30 / −0.62; realtime = MG2 quality with 13 points less drift. |
| 2 | `s2_item1_latency.md` | item 1 | All four latency metrics for teacher, quality and realtime (one protocol); teacher-parity per-frame n/a; no config meets 750 ms. |
| 3 | `s3_item2_specdec.md` | item 2 | NO-GO: slower than target-only per frame at both τ, no physics benefit on StEvo-38, accept-all ceiling ≈ 7.6 fps; go criterion of the physics verifier not measured. |
| 4 | `s4_item3_lowres.md` | item 3 | Quality falls slowly to 576 and steeply below; 704 is the lowest size where the 14B stays clearly above the student (+0.67 ± 0.25, derived). |
| 5 | `s5_item4_sensitivity.md` | item 4 | Levers multiply; chunk-wise is free; commit policy borderline cost; jump-free commit ≈ one resolution step (−0.36 vs −0.31). |
| 6 | `s6_item5_baselines.md` | item 5 | Drift added over the teacher: realtime +5.9, MG2 +19.0, Waypoint +24.9; MG2 quality cost equals ours; StEvo-38 and path distance agree in direction. |
| 7 | `s7_findings.md` | 10/10 points, new findings B1–B9 | 10/10 status; jump fix; refiner vs resolution; decoder; student 17.2 fps; empty text prompts; teacher-parity mode; 4.9 correction (B7); latency (B8); StEvo (B9). |
| 8 | `s8_next.md` | next steps, asks | DMD at 704 is the main proposal; uniform 3-step control is the cheapest next result; 12 questions for the professors. |

## 6. Still open (say these out loud; none is resolved by `facts.md`)

1. **Teacher-parity per-frame latency** (first frame, action→frame, miss): `spec_early` has no streaming decode (facts §13).
2. **Uniform 3-step control:** the learned head's own contribution over ≈ 3 fixed steps is unproven on bench-43 and in the quiet pass (only the 10-prompt seed-1234 run: 6.2 vs 6.5 fps whole-clip, facts §3).
3. **Physics-verifier go criterion:** physics of its final videos never measured (videos deleted 10/9); re-run with videos kept is proposed, not done.
4. **Human validation:** the VLM quality prompt (on which teacher parity rests) is not human-validated; the blind-study page has no raters and its candidate set predates the three modes.
5. **Quiet timings missing:** the student, SDVG, 576 separate, 448 / 384 (only diag-run cloud times 0.367 / 0.250 s/chunk, load not recorded, derived in s4), MG2 / Waypoint (only their own whole-clip pipelines, 19.7 / 31.2 fps).
6. **StEvo:** teacher-parity mode and SDVG τ −2.2 not run; state evolution / task success at the floor; paper's Gemini judge needs an API key.
7. **No real edge device or network** in any measurement; the separate-device pipeline (decode off the cloud GPU) is composed, not measured end to end.
8. **576 fused + partial:** 25.66 fps quiet (only config ≥ 16 fps with streaming decode on one GPU: 17.1 whole-clip / 16.07 steady) but no VLM score and no jump count.
9. **Scalability:** MG2 only; commit levers and chunk-wise head not ported; no next-gen base model.
10. **Matched-resolution gaps:** Waypoint at ≈ 360p, student at 704.
11. **cw05_pc1 timing:** its median speed is not a stable number (the median picks the modal stop step; on a mean basis it is faster than cw07_pc1 in both passes; derived in s5). Not a mode either way (Δ quality −0.17 ± 0.07).
12. **Empty text prompts** on jar / castle / hourglass: the quiet timing pass, the latency profile and all 10/5 jar/castle results ran without text; effect not quantified (bench-43 unaffected).
13. **Bench-43 is not fully held out:** 9 of 43 prompts and both seeds overlap the development matrix (arm order holds on the 34 held-out prompts, derived).
14. **Ideas only:** DMD at 704 (main proposal), 1.3B distribution-matched distillation ("stress the draft"), idle-action trick.

## 7. Fixed in the completeness pass (10/12 08:15 CDT)

- MG2 quiet fps 19.5 → **19.7** everywhere in the notes (s1, s6, s7, s8): facts §15 corrected it (19.5 included the warm-up run).
- Partial-commit cost: s5 and s7 quoted +0.133 s/chunk without noting that 0.630 vs 0.763 are medians over 6 vs 5 clips; added the same-clip ≈ +0.12 (0.640 → 0.763; derived, matches s1, s2 and the f02 footer).
- s7 B2 no longer says the refiner caused "most" of the 704 loss: about the same size as the resolution step (−0.50 ± 0.11 vs −0.31 ± 0.10, difference not significant) — README §5 #7 of the earlier index; Korean memo too.
- s7: MG2 token saving 29 % → 30 % (README §M); f11 description now says quiet ratios lead; "no steady value exists for the ours rows" replaced by the facts §13 steady values (9.97 / 6.39).
- s8: stale f11 description (07:44 render) replaced; Waypoint StEvo wording harmonized with f12 / s1 (p 0.003–0.035 for teacher and both modes, SDVG borderline p = 0.057).
- s5: "no fps for 448/384" now says no quiet fps, diag-run cloud times only (consistent with s4).
- facts.md: stale PENDING lines in §1, §2, §10 and §12 now point to §13–§15 (no numbers changed); §10 lists what is still open.
- Figures re-rendered on CPU (no data change): f01, f03, f05, f07 footers now describe bench-43 as 38 StEvo camera-WM tasks + 5 easy scenes (was "43 StEvo prompts"); f12 title no longer implies every 14B arm beats MG2 (only the quality mode separates). Backups of the previous renders are in the session scratchpad, not in this folder.

Derived numbers in the notes (keep the "derived" label on slides): held-out 34-prompt deltas, easy-scene paired deltas (s6); 12 × 3 single-lever paired SEs and resolution steps (s4, s5); TTFF / e2e with the assumed 2.0 s prepare and the steady-output ratios 1.68× / 2.62× (s2); same-clip partial cost ≈ 0.12 s; 448 / 384 diag-run fps; cw05 mean-basis timings; SDVG 1.25× slower in gen s and teacher 4.4 fps on SDVG's basis (s6).

## 8. Folder map
- `facts.md`: canonical numbers (§0 setup … §11 rollout paths, §12 direction 1, §13 latency, §14 StEvo-38, §15 quiet timing pass; §10 open items).
- `figs/`: f01–f13 (20 png + 20 svg; f02 has a companion f02b; f10 has a summary plus six scenes).
- `notes/`: 8 speaker notes (s1–s8).
- `data/`: recomputed numbers for f02, f10, f12, f13 and the steady output fps.
- `scripts/`: one generator per figure plus `e_steady_fps.py` (CPU only).
- `README_ko.md`: Korean slide plan (14 slides + backups), one-line messages, caveats to say out loud, what changed since 10/5.
