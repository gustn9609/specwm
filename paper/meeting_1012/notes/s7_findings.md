# New findings worth telling: the 10/10 points, and what changed our numbers

**Headline:** With the jump fix, **ours: realtime mode** (704 2-step fused + partial) runs at **15.73 fps in the quiet timing pass** (0.763 s/chunk, generation only; 15.8 fps under load on bench-43), **3.85× the teacher** in the quiet timing pass (jar + castle; steady s/chunk, ours = cloud-only `cloud_chunk_s`, teacher = chunk-to-chunk interval; facts §12, §15). Fused-only runs at 19.04 fps in the same pass (0.630 s/chunk), so the fix costs ≈ 0.13 s/chunk (≈ 0.12 on the same 5 clips: 0.640 → 0.763; the 0.630 median also includes castle s1234); in return, boundary jumps fall from 12 % to 4 % (separate commit 3 %; > 1.5× their neighbours, 12 prompts × 3 seeds). Note: 0.763 s/chunk is slightly longer than the 0.75 s a 12-frame chunk plays at 16 fps, so the mode is just below real time. **Methodology correction (B7):** the teacher's often-quoted "4.9 fps quiet" is a whole-clip average. On the steady basis it runs at 4.08 fps quiet (3.9 under load), so the preferred speed-ups of our three modes are **1.67× / 2.39× / 3.85×** (teacher-parity / quality / realtime; quiet steady, facts §15).

Figures for this section: **f03_splits_map** (where each of the three 10/10 splits stands, plus our other routes), **f02_partial_commit** and **f02b_jump_frames** (jumps and the fix), **f08_lowres_curve** (refiner vs resolution diagnosis), **f09_decoder** (decoder cost), **f10_paths_summary** (rollout-path summary, 6 scenes (3 easy + pouring + jar + castle), 10 seeds per arm (jar and castle compared on n = 3)), **f11_modes** (our three operating modes; quiet steady ratios, bench-43 under-load ratios in parentheses), **f13_latency** (per-frame action→frame latency, B8). StEvo-38 (B9): **f12_stevo** (physical inaccuracy with Wilson 95 % CI, other criteria, paired McNemar tests).

## Part A: status of the 10/10 meeting points (Guandao)

| 10/10 point | status on 10/12 | key evidence (prompt set) | figure |
|---|---|---|---|
| **Scalability:** the recipe must transfer to next-gen base models | **Partial** | On Matrix-Game 2.0 (1.8B, keyboard/mouse), both speed levers transfer: 3 → 2 steps saves 13 %, 0.51× tokens saves 30 %, and both together save 40 % (34.3 fps). The quality pattern also repeats: lower resolution loses the scene earlier (DINO min 0.149 → 0.100). With our recipe (2-step 448), VLM drift goes 55.0 → 71.5 and quality 7.1 → 5.7 vs MG2 as released (warm-up clip excluded; facts §9). The drift head transfers too, with AUROC 0.77 on held-out scenes vs VLM labels (LingBot: 0.88 vs human labels). (10 prompts, seed 1234; head data 36 scenes × 2 seeds + 10 prompts; source README M, M2, N) | – |
| **The three cloud/edge splits** (spec decoding is only one of them) | **Mapped.** Two closed, one kept in reduced form | See the split table below. | **f03_splits_map** |
| **Fair comparison:** beat the best hybrid, and compare against edge-only models at the same resolution | **Done on bench-43** for quality, fps and drift. **Per-frame latency done on one GPU** (jar + castle, facts §13). | Ours: realtime mode (704 2-step fused + partial) vs Matrix-Game 2.0 (≈704×384 vs 640×352): quality 6.90 vs 6.91, drift 56.9 vs 70.0, 15.8 vs 20.0 fps (fps not like-for-like on decode and GPUs; see the footer of the matched-resolution table). The realtime mode also beats SDVG τ −2.2 on all three axes. At quality within noise of the teacher, ours: teacher-parity mode runs at 6.8 fps vs SDVG τ −0.7 at 3.1 (both steady, under load; SDVG's whole-clip value is 3.5, B7). On one GPU with the decode streamed on the same GPU, the realtime mode has the lowest action→frame latency among the operating modes and baselines measured: p50 1296 ms vs student 1434, teacher 3237, SDVG 4156–4463 (B8; two non-mode variants are lower). MG2 and Waypoint were not measured per frame. | f05_pareto_bench43, f06_paired_deltas, f13_latency |
| **Easier prompts:** failures may come from very hard prompts or actions | **Done.** Bench-43 now has 5 easy scenes (n = 10 clips) next to 38 tabletop & physics tasks (n = 76). | Drift is mostly a hard-prompt effect: the teacher drifts 21.5 on easy scenes vs 55 on tabletop & physics. | f10_paths_summary (rollout-path summary, 6 scenes (3 easy + pouring + jar + castle), 10 seeds per arm (jar and castle compared on n = 3)) |

### The three splits (f03_splits_map). There are no metric marks here because the rows are not comparable arms.

| split | what we ran | best result (prompt set) | status |
|---|---|---|---|
| *The three splits on the 10/10 list* | | | |
| **Low-res / high-res decomposition over time** | Adaptive 576 → 832 switch; first K chunks at 832, then 576 | Quality moves in proportion to the share of the clip generated at 832. The switch policy degenerates into scene routing (12 prompts, seed 1234). | **Closed.** |
| **Device drafts, cloud corrects** (speculative decoding) | SDVG with actions (1.3B draft, 14B target, ImageReward verifier); Draft-then-Finish; verify + redraw with the 14B drift head | **SDVG:** at τ −0.7 it runs at 3.5 fps (whole-clip; 3.1 on the teacher's steady basis, B7) vs the teacher's 3.9 (bench-43, under load). Even accepting every draft chunk caps it at ≈ 7.6 fps. **Draft-then-Finish:** quality 6.2 vs teacher 7.6 (12 prompts, seed 1234). | **Closed.** The 1.3B draft is 0.63 σ from the 14B output, with 7 % of tokens within ε. |
| **Cloud low-res, device super-resolution** | 14B at 704 / 576 in the cloud; the device runs a learned 1.3B refiner, Real-ESRGAN, or a bicubic upsample | The refiner costs −0.50 ± 0.11 quality at 704 (paired, 36 clips), so bicubic wins. **Ours: realtime mode (704 2-step fused + partial): 15.8 fps (under load), quality 6.90** (bench-43). | **Kept.** The device job is now only VAE decode + bicubic upsample. |
| *Extra split we also tried (our own, 10/8; not on the 10/10 list)* | | | |
| **Cloud anchor, device finishes** (anchor split; README N, method: H2, J1) | 14B step 1 in the cloud, 1.3B finishes on the device | 8.5 fps, quality 7.0, drift 43 on 1+1 GPUs. The 14B 2-step 832 on one GPU gives 8.3 fps, quality 7.5, drift 42 (whole-clip fps, README N: 153 frames / 18.0 s and / 18.5 s; 10 prompts, seed 1234). On the steady basis, 14B 2-step 832 separate is 7.98 fps quiet (facts §15). | **Closed.** No gain over cloud-only. |

### Fair comparison at matched resolution (bench-43: 43 prompts × 2 seeds = 86 clips per arm; fps under load on RTX PRO 6000 Blackwell GPUs, see the footer for GPUs and decode)
Bold = best and underline = second **within each resolution block**; ties share the mark.

| arm | output resolution | fps (under load) ↑ | VLM quality ↑ | VLM drift ↓ | Δ quality vs teacher (paired, 86) ↑ |
|---|---|---|---|---|---|
| *480×832 block* | | | | | |
| teacher 14B 4-step | 480×832 | 3.9 | **7.51 ± 0.08** | <u>51.0 ± 3.0</u> | – |
| ours: teacher-parity mode (chunk-wise adaptive 0.7 + safe commit) | 480×832 | 6.8 | <u>7.44 ± 0.08</u> | **47.7 ± 3.2** | <u>−0.07 ± 0.06</u> |
| ours: quality mode (832 2-step fused + partial) | 480×832 | <u>9.8</u> | 7.21 ± 0.08 | 53.5 ± 3.0 | −0.30 ± 0.08 |
| SDVG + actions τ −0.7 (best hybrid on quality) | 480×832 | 3.5 (steady 3.1) | **7.51 ± 0.09** | 52.9 ± 3.0 | **+0.00 ± 0.07** |
| SDVG + actions τ −2.2 | 480×832 | 6.4 (steady 6.7) | 6.67 ± 0.09 | 70.6 ± 2.5 | −0.84 ± 0.10 |
| student 1.3B 4-step (edge-only, same model family) | 480×832 | **17.2** | 5.83 ± 0.16 | 75.1 ± 2.2 | −1.69 ± 0.17 |
| *≈360p block* | | | | | |
| ours: realtime mode (704 2-step fused + partial) | ≈704×384 | <u>15.8</u> | <u>6.90 ± 0.08</u> | **56.9 ± 2.7** | <u>−0.62 ± 0.09</u> |
| Matrix-Game 2.0 (edge-only SOTA) | 640×352 | **20.0** | **6.91 ± 0.07** | <u>70.0 ± 3.0</u> | **−0.60 ± 0.09** |
| *no matched run* | | | | | |
| Waypoint-1.5-1B (edge-only SOTA) | 1024×512 | 25.4 | 4.03 ± 0.15 | 75.9 ± 1.6 | −3.48 ± 0.15 |

Source: `paper/1012/bench43_summary.json` (teacher-parity mode: key cw07_safe, facts §12). The teacher-parity mode's bold drift is within noise of the teacher (paired Δ drift −3.3 ± 3.1), so it does not mean lower drift than the teacher; its Δ quality −0.07 ± 0.06 is likewise within noise.
Footer (fps is not like-for-like on decoding): LingBot fps (teacher, student, SDVG, and ours in all three modes, including the chunk-wise teacher-parity mode) = generation only, with the VAE decode excluded (teacher +8.6 s and student +8.7 s per clip for the full VAE). The quality and realtime fps are cloud-GPU time only (`cloud_chunk_s`) and assume the device step and the full-VAE decode (5.5 s per clip at 704) run in parallel on a separate device GPU (not measured end to end). Counting the measured device step serially gives 13.6 / 8.9 fps (3.44× / 2.25× under load; facts §12). Matrix-Game 2.0 and Waypoint fps include their own decoder on one GPU. The realtime mode on one shared GPU with the full VAE: 10.1 fps (sum of component timings, not measured end-to-end; facts §6); measured end to end with streaming decode on jar + castle it gives 10.8 fps output (whole-clip mean; facts §13, B8). SDVG's 3.5 / 6.4 are whole-clip means (12 / (gen s / 13), including the forced target chunk 0); on the steady basis of the other LingBot rows they are 3.1 / 6.7 (facts §2; B7).
- **≈360p:** ours: realtime mode ties Matrix-Game 2.0 on VLM quality (6.90 vs 6.91) and has lower drift (56.9 vs 70.0). It is slower: 15.8 vs 20.0 fps, and the 15.8 excludes the decode that MG2's 20.0 includes (see the footer).
- **Best hybrid:** SDVG at the paper τ matches the teacher (+0.00 ± 0.07) but runs at 3.5 fps (3.1 steady), below the teacher's 3.9. At quality within noise of the teacher, ours: teacher-parity mode runs at 6.8 fps vs SDVG τ −0.7 at 3.1 (both steady, under load; SDVG's whole-clip value is 3.5). On speed it is level with SDVG τ −2.2 (6.8 vs 6.7 fps, both steady; SDVG 6.4 whole-clip), with much better quality (7.44 vs 6.67) and drift (47.7 vs 70.6). At a usable speed (τ −2.2, 6.7 fps steady / 6.4 whole-clip), the realtime mode is better on all three axes: higher fps (15.8 vs 6.7, both steady under load), higher quality (6.90 vs 6.67) and lower drift (56.9 vs 70.6).
- **Edge-only at 480×832:** the 1.3B student is the fastest LingBot arm (17.2 fps), at −1.69 ± 0.17 quality vs the teacher. The quality mode is at −0.30 ± 0.08 and the teacher-parity mode at −0.07 ± 0.06.
- **Missing:** Waypoint at a matched ≈360p, and the student at 704. Neither is in the canonical numbers.

### Easy vs hard prompts (bench-43 groups; quality ↑ / drift ↓)
Bold = best and underline = second, per column. Ties share the mark. Easy-scene drift is shown with one decimal (raw means of 10 clips, `paper/1012/vlm_scores_bench43.json`).

| arm | easy scenes: quality ↑ (n = 10) | easy scenes: drift ↓ | tabletop & physics: quality ↑ (n = 76) | tabletop & physics: drift ↓ |
|---|---|---|---|---|
| teacher 14B 4-step | <u>8.00</u> | **21.5** | **7.45** | **55** |
| ours: quality mode (832 2-step fused + partial) | 7.90 | 25.0 | 7.12 | <u>57</u> |
| ours: realtime mode (704 2-step fused + partial) | 7.60 | 36.5 | 6.80 | 60 |
| SDVG τ −0.7 | **8.20** | **21.5** | <u>7.42</u> | <u>57</u> |
| SDVG τ −2.2 | 7.20 | 32.0 | 6.61 | 76 |
| student 1.3B 4-step | 7.60 | <u>23.5</u> | 5.59 | 82 |
| Matrix-Game 2.0 | 6.90 | 48.5 | 6.91 | 73 |
| Waypoint-1.5-1B | 4.70 | 55.0 | 3.95 | 79 |

- On easy scenes, the student comes close to the teacher (7.60 / 23.5 vs 8.00 / 21.5). The cloud model earns its cost on tabletop & physics, where the student falls to 5.59 / 82.
- On easy scenes, the realtime mode matches the student on quality (7.60) but drifts more (36.5 vs 23.5). The quality mode (7.90 / 25.0) is close to the teacher there. (facts §2 has no easy/hard split for the teacher-parity mode, so it is not in this table.)
- Matrix-Game 2.0 is slightly ahead of the realtime mode on tabletop & physics quality (6.91 vs 6.80) and well behind it on drift (73 vs 60).

## Part B: new findings since 10/10

### B1. Fused commit → chunk-boundary jumps → partial commit (f02_partial_commit, f02b_jump_frames)
- **Fused commit** reuses the K/V that the last denoising forward wrote. Its input is noisy (the σ 0.833 node). This saves one full forward per chunk.
- **Side effect:** the next chunk starts from a slightly different state, and occasionally from a different scene. Example: `easy_urban_rain`, 704 fused, seed 1234, hard-cuts to another corridor at frame 128 → 129, which is a chunk boundary.
- **Why we missed it:** the VLM judge sees 8 frames per clip, and the 3-seed averages did not move. The user spotted the jumps on the comparison site.
- **Partial commit:** after the fused step, run a clean t = 0 pass over only the newest latent frame's tokens (≈ 1/3 forward).

| arm (12 prompts × 3 seeds unless marked s1234) | cloud s/chunk ↓ | fps ↑ | timing condition | quiet steady s/chunk ↓ (fps ↑), same condition for every row (facts §15, jar + castle) | boundaries > 1.5× ↓ | VLM drift ↓ | VLM quality ↑ |
|---|---|---|---|---|---|---|---|
| teacher 14B 4-step 832 (jumps s1234, 12 clips; VLM 3 seeds, 36 clips) | 2.45 | 4.9 | quiet, but a **whole-clip average** (153 frames / 31.2 s; B7) | 2.939 (4.08) | <u>2 %</u> | **45.4** | **7.64** |
| 704 separate commit | 0.95 | 12.7 | README U, s1234 run (704 + refiner, separate); load not stated | 0.960 (12.50) | 3 % | 54.2 | 7.22 |
| 704 fused | **0.640** | **18.7** | exclusive GPU (README V3) | **0.630 (19.04)** | 12 % | <u>49.3</u> | 7.08 |
| 704 fused + partial 1 (= ours: realtime mode) | <u>0.761</u> | <u>15.8</u> | under load | <u>0.763 (15.73)</u> | 4 % | 53.5 | 6.86 |
| 704 fused + partial 2 | 0.874 | 13.7 | not recorded (facts §5: n/a) | – (not in the quiet pass) | 4 % | 52.9 | 7.08 |
| 832 fused (s1234) | 1.02 | 11.8 | README U, s1234; load not stated | – (not in the quiet pass) | 7 % | 54 | 7.4 |
| 832 fused + partial 1 (s1234; = ours: quality mode) | 1.24 | 9.7 | not recorded (facts §5: n/a) | 1.232 (9.74) | **0 %** | 52.5 | <u>7.5</u> |

Sources: `scripts/boundary_jump.py`, README sections U, V3 and W. fps = 12 / cloud s per chunk on one GPU, decode excluded; the load condition differs by row (column "timing condition"), so the fps column is not a like-for-like comparison. The quiet column puts every row in one condition: one timing pass, no other generation jobs, generation only, no decode, steady chunks (facts §15: median of `cloud_chunk_s[6:]` for the 704/832 arms, of `diff(chunk_start)[5:]` for the teacher). It runs on jar + sandcastle, not on the 12 prompts of the other columns; 704 separate quiet 0.960 s/chunk matches the 0.959 of the 3-seed dg704 runs below. For the 704 separate row, the timing comes from the single-seed README U run (12 / 0.95 gives 12.6 from the rounded time; 12.7 is the canonical value). The 3-seed dg704 runs behind this row's VLM and jump numbers have a median steady cloud time of 0.959 s/chunk (12.5 fps; `lingbot-world-v2/specwm_out/fused/*/dg704_s*/loop.json`).
- **Cost of the fix at 704 (same condition, quiet pass, facts §15):** +0.133 s/chunk (0.630 → 0.763 s/chunk, 19.04 → 15.73 fps). These two medians cover 6 vs 5 clips; on the same 5 clips (castle s1234 excluded, as for ours 704) it is ≈ +0.12 s/chunk (0.640 → 0.763; jar 0.641 → 0.764, sandcastle 0.620 → 0.737; derived from the §15 loop.json files, as in the f02 footer). The earlier +0.12 s (0.640 s quiet vs 0.761 s under load) was therefore not a load artefact: the cost is real. It is ≈ 40 % of a full clean commit (704 separate − fused = 0.960 − 0.630 = 0.330 s/chunk), while the partial pass covers ≈ 1/3 of the tokens by construction. Fused + partial stays faster than separate commit (15.73 vs 12.50 fps quiet).
- **Control:** mid-chunk control positions give 0.7–2.5 % in the arms of this table (0.7–3.7 % across all nine arms, facts §5), so the 12 % is specific to chunk boundaries.
- **Bench-43:** the realtime and quality modes have 2 % boundary jumps vs 1 % for the teacher; the teacher-parity mode has 1 % (facts §12).
- **Retraction:** the 10/11 claim "704 fused at 18.7 fps ties Matrix-Game 2.0" held only for the jumpy variant.

### B2. Where the low-res quality loss comes from: at 704 the device refiner costs about as much as the lower resolution; at 576 the lower resolution dominates (f08_lowres_curve)

At 704 the device refiner cost about as much as the lower resolution (−0.50 ± 0.11 vs −0.31 ± 0.10, both paired; per-clip difference −0.19 ± 0.18, not significant; s4 §3); at 576 the low-res 14B itself loses more (−0.70) than the refiner (−0.25). (Resolution loss = difference of 36-clip mean quality vs the 14B 2-step 832 at 7.53; refiner loss = paired, see below.)

| output (12 prompts × 3 seeds, judge at 448 px) | VLM quality ↑ | VLM drift ↓ |
|---|---|---|
| 704 cloud, native (no refiner) | **7.22 ± 0.15** | **54.2 ± 5.0** |
| 704 cloud + 1.3B refiner → 832 | 6.72 ± 0.14 | 57.1 ± 4.8 |
| 576 cloud, native | <u>6.83 ± 0.14</u> | <u>56.9 ± 4.7</u> |
| 576 cloud + 1.3B refiner → 832 | 6.58 ± 0.18 | 60.3 ± 4.7 |

- **Refiner cost, paired by clip (36 clips):**
  - 704: −0.50 ± 0.11 quality. Per clip: 1 better, 18 equal, 17 worse.
  - 576: −0.25 ± 0.12.
- **With the judge at 832 px,** where upsampling blur is visible:

  | output | VLM quality ↑ |
  |---|---|
  | teacher | **7.78** |
  | 14B 2-step 832 | <u>7.22</u> |
  | 704 cloud, bicubic up | 7.11 |
  | 704 + refiner | 6.58 |
- **14B-alone resolution curve** (2-step, native output, quality / drift):

  | resolution | VLM quality ↑ | VLM drift ↓ |
  |---|---|---|
  | 832 | **7.53** | **50.8** |
  | 704 | <u>7.22</u> | <u>54.2</u> |
  | 576 | 6.83 | 56.9 |
  | 448 | 6.17 | 58.2 |
  | 384 | 5.06 | 66.9 |

  Quality falls slowly down to 576 and steeply below it. Relative to 832: −0.31 at 704, −0.70 at 576.
- **Consequence:** the device no longer runs a model. Real-ESRGAN also scored below plain bicubic.

### B3. Decoder choice matters: TAEHV costs quality (f09_decoder)
Bold = better of the two decoders; ties share the mark.

| decoder on the device (realtime mode = 704 fused + partial 1, 12 prompts × 3 seeds) | ms/chunk at 704 ↓ | fps, separate device GPU (under load, assumed) ↑ | fps, one shared GPU (under load, computed) ↑ | quality, judge 448 ↑ | quality, judge 832 ↑ | drift, judge 448 ↓ | drift, judge 832 ↓ |
|---|---|---|---|---|---|---|---|
| full Wan2.1 VAE (bf16) | 426 | **15.8** | 10.1 | **6.86** | **6.81** | **53.5** | **51.1** |
| TAEHV taew2_1 (fp16) | **12.8** | **15.8** | **15.5** | 6.31 | 5.75 | 60.8 | 62.2 |

Both fps columns use the cloud time measured under load (0.761 s/chunk). The shared-GPU column is a serial sum of component timings (cloud + decoder), not measured end-to-end; the separate-device column assumes a second GPU that was not measured either (facts §12). Source: README W2, facts §6.

- **Paired TAEHV − full VAE (36 clips):**

  | judge input | Δ quality | Δ drift |
  |---|---|---|
  | 448 px | −0.56 ± 0.13 | +7.4 ± 3.0 |
  | 832 px | −1.06 ± 0.14 | +11.1 ± 3.2 |
- **Hidden mismatch in the earlier slides:** all VLM scores of the LingBot arms (ours, teacher, student, SDVG) were computed on full-VAE videos, while the 10/11 end-to-end fps assumed TAEHV. MG2 and Waypoint use their own decoders.
- **Separate device GPU (assumed, not measured end to end):** the full VAE (426 ms) finishes inside the cloud chunk (761 ms, under load), so on paper it costs no fps (facts §6). TAEHV only matters on a weaker device, or when the cloud and the device share one GPU.
- **At 832:** full VAE 631 ms/chunk vs TAEHV 19 ms/chunk.

### B4. The student runs at 17 fps on the same stack (correction of the earlier ≈ 10 fps)
- **The old number is superseded:** "edge: LingBot 1.3B 4-step 832 ≈ 10 fps" (10.4 fps, 14.7 s per clip in the 10/10 baseline matrix).
  - facts.md labels it "old run, without compile". However, the raw `cmp10/*/draft_1p3b_s1234/loop.json` records compile = True, fp8 = True, stream_decode = True and decode_s = 0.0.
  - That points to the streaming full-VAE decode running inside the timed generation window (`gen_fast.py` pushes each chunk to the stream decoder inside the generation loop).
  - Consistent with facts §13: the same stack with streaming decode on the same GPU gives 14.6 s / 10.4 fps (jar + castle) vs the old 14.7 s / 10.4 fps (10 prompts). Not a direct re-run of the old prompts, so this is the likely cause, not a confirmed one.
- **Same stack:** with SageAttention 2, torch.compile, ring KV and the exact patches (FP8 is on the 14B only), the student runs at **17.2 fps under load** on bench-43. Generation and decode are timed separately: 8.2 s per 10 s clip, plus 8.7 s for the full-VAE decode.
- **The student is faster than ours: realtime mode** (17.2 vs 15.8 fps, both under load on bench-43). Neither number includes the decode: the student's full-VAE decode (+8.7 s per clip at 832) is excluded, and the realtime mode's 15.8 is cloud-GPU time only, assuming the device step and the full-VAE decode (5.5 s per clip at 704) run in parallel on a separate device GPU (not measured end to end; with the measured device step counted serially it is 13.6 fps, facts §12). Our pitch is quality at near-student speed, not raw speed:
  - Quality: 6.90 vs 5.83.
  - Drift: 56.9 vs 75.1.
  - Paired Δ quality vs the teacher: −0.62 ± 0.09 vs −1.69 ± 0.17.
  - With the decode on the same GPU (facts §13, jar + castle), the realtime mode is ahead of the student on latency: action→frame p50 1296 vs 1434 ms, output 10.8 vs 10.4 fps (whole-clip means; B8). The student still shows its first frame sooner (439 vs 574 ms).
- **This hurts SDVG further:** even its accept-all ceiling (≈ 7.6 fps, ≈ 20 s per 10 s clip) is still 2.5× slower than the student alone (8.2 s per clip), at student quality (facts §2).
- **Superseded:** every earlier table row with the 1.3B at ≈ 10 fps (README M, N, R, T; facts §3 row). The student's quiet generation-only fps is still **PENDING**: the quiet timing pass (facts §15) has no student row. With streaming decode on one GPU (jar + castle, decode inside) the student gives 9.86 fps steady / 10.4 fps whole-clip (facts §13).

### B5. Hourglass, jar and castle ran with an empty text prompt: a caveat for every earlier jar/castle result
- **Cause:** in `data/stevo/tasks_camera.json`, `camera_WM` is `''` for `hourglass_accumulation`, `sand_filling_up_jar` and `sandcastle_wave_impact` (47 of the 90 entries are empty). Every LingBot generator passes `T.get("camera_WM", "")` as the text prompt (`gen_fast.py`, `res_split_infer.py`, `spec_early.py`, `spec_v0.py`, …). So for these scenes the model saw only the image and the camera path, never any text.
- **10/5 SDVG on jar/castle:** the ImageReward verifier scored chunks against the full T2V caption, while LingBot generated from `''`. The 10/5 inputs file records this (`paper/1005/meeting/01_sdvg_to_action/inputs/jar_prompts.txt`), but it was never flagged as a caveat.

| result | prompt set | affected? |
|---|---|---|
| Bench-43 headline numbers | 43 tasks, all with a non-empty `camera_WM` | **No** |
| StEvo-38 subset (facts §14, B9) | the same 38 StEvo tasks | **No** |
| 12-prompt × 3-seed matrix (jump, refiner, decoder and resolution tables above) | 3 of 12 prompts = 9 of 36 clips per arm | **Partly** (the same empty prompt for every LingBot arm) |
| jar/castle-only results: 10/5 SDVG breakdown (99.9 s), physics-verifier attempt, jar/castle path figures (f10_paths_sand_filling_up_jar, f10_paths_sandcastle_wave_impact), the earlier sandcastle + sand-jar test set | jar + castle | **Yes, all** |
| Chenfeng-method latency profile and ours latency rows (facts §13, B8) | jar + castle × 3 seeds | Yes (timing run, same empty prompt) |
| Quiet timing pass and the quiet speed-ups 1.67× / 2.39× / 3.85× (facts §15, B7) | jar + castle | Yes (timing run, same empty prompt for every arm) |

- **Comparisons stay fair:** within one prompt, every LingBot arm got the same empty prompt, so arm-vs-arm comparisons are like-for-like. Absolute scores on these scenes describe image + camera conditioning only.
- **No re-run yet:** no re-run with a text prompt is in the canonical numbers. Whether part of the jar/castle "hardness" came from the missing text is **open**.

### B6. Ours: teacher-parity mode (chunk-wise adaptive 0.7 + safe commit) on bench-43: quality within noise of the teacher at 1.67× the teacher's speed (quiet steady, 6.80 vs 4.08 fps; 1.74× under load on bench-43, 6.8 vs 3.9) (facts §12, §15; f11_modes leads with the quiet ratios and gives the under-load ratios in parentheses)
- **New since 10/10:** the bench-43 evaluation and the commit policy. The chunk-wise head itself dates from 10/8 (facts §3).
- **Ours: teacher-parity mode** (`cw07_safe`, chunk-wise adaptive 0.7 + safe commit) runs at **6.8 fps under load, 1.74× the teacher** (3.9 fps under load); in the quiet timing pass it is 6.80 vs 4.08 fps = **1.67×** (facts §15; the old "4.9 quiet" is a whole-clip average, B7). It keeps VLM quality within noise of the teacher (Δ −0.07 ± 0.06; 10 better / 60 equal / 16 worse; the quality prompt is not human-validated) on 43 prompts × 2 seeds. After each 14B step, a learned head (head14b_v3) predicts per-token convergence, and the whole chunk stops once ≥ 70 % of its tokens are predicted converged (3.01 mean steps). The chunk is then committed with a partial pass if it stopped at step ≥ 3, or with a full clean pass if it stopped at step 1–2. Paired vs the teacher (86 clips), Δ quality is −0.07 ± 0.06 (10 better / 60 equal / 16 worse) and Δ drift is −3.3 ± 3.1. Δ drift −3.3 ± 3.1 is within noise (not evidence of less drift). Boundary jumps are 1 %, the same as the teacher. The fps include the head's cost because there is no reference pass.
  - **The head alone** (`cw07_sep`, separate commit) keeps teacher quality (Δ +0.03 ± 0.05) at 1.51× under load (5.9 fps; 1.45× quiet). The commit policy raises the speed-up to ≈ 1.75× under load (`cw07_pc1` 1.77×, Δ −0.09 ± 0.06; 1.66× quiet), with quality differences vs the teacher within ~1.5 s.e. Paired against `cw07_sep`, the commit policy costs −0.10 ± 0.06 (safe; 10 better / 57 equal / 19 worse) / −0.13 ± 0.07 (pc1; 10 / 56 / 20) quality: borderline (< 2 s.e.), a small cost of the extra speed (facts §12).
  - **The looser stop (`cw05_pc1`, ≥ 50 % converged) is not one of our modes.** Under load on bench-43 it ran fewer steps than `cw07_pc1` (2.61 vs 3.01) but saved almost no time per chunk (1.722 vs 1.748 s/chunk, 7.0 vs 6.9 fps). In the quiet timing pass (jar + castle) the saving does appear: 2.47 vs 3.15 steps, 1.495 vs 1.771 s/chunk, 8.03 vs 6.78 fps (1.97× vs 1.66×; facts §15). So the bench-43 non-saving does not reproduce in the quiet pass (different prompts; the cause is still not diagnosed). It costs more quality on bench-43 (Δ −0.17 ± 0.07, Δ drift +2.2 ± 3.1), so it would sit between the teacher-parity mode and the quality mode (2.39× quiet, Δ −0.30 ± 0.08) rather than replace either.
  - **Where it sits among our modes:** **ours: quality mode** (`ours832`, 832 2-step fused + partial) runs at 9.74 fps quiet (**2.39×**; 9.8 fps, 2.5× under load on bench-43) with Δ quality −0.30 ± 0.08, and **ours: realtime mode** (`ours704`, 704 2-step fused + partial) runs at 15.73 fps quiet (**3.85×**; 15.8 fps, 4.1× under load) with −0.62 ± 0.09. The quiet steady ratios (facts §15) are the ones to quote; the bench-43 under-load ratios agree within ~5 %. The best hybrid at teacher quality, SDVG τ −0.7 (Δ +0.00 ± 0.07), runs at 3.5 fps (3.1 steady), below the teacher. Source: `paper/1012/bench43_summary.json` (keys cw07_sep, cw07_pc1, cw07_safe, cw05_pc1); quiet timings `lingbot-world-v2/specwm_out/timing_pass/` (facts §15).

### B7. Methodology note: the teacher's "4.9 fps quiet" was a whole-clip average, not steady fps (facts §15, §2)
- **What 4.9 is:** 153 frames / 31.2 s of generation for the whole clip (README §N: 10 prompts, 832×480, seed 1234). It averages over all 13 chunks, including the cheap early chunks before the KV window fills.
- **What every fps in the speed-up tables (facts §12, §15) is:** steady fps = 12 / the median chunk-to-chunk time over the later chunks (`diff(chunk_start)[5:]`, or `cloud_chunk_s[6:]` for the 704/832 arms). Not every fps in this package is steady: see "Other whole-clip numbers" below. MG2 and Waypoint fps come from their own pipelines and have no steady s/chunk (facts §15).
- **On that basis the teacher runs at 4.08 fps quiet (2.939 s/chunk) and 3.9 fps under load (3.038 s/chunk on bench-43)**, 3 % apart. So "3.9 under load vs 4.9 quiet" was never a load effect alone: the two numbers differ in basis and prompt set.

The same teacher measured three ways. No best/second marks: these rows are one model on three bases, not competing arms.

| teacher 14B 4-step measurement | basis | prompt set | load | s/chunk ↓ | fps ↑ |
|---|---|---|---|---|---|
| old "4.9 fps quiet" (README §N; facts §3, §5) | whole clip: 153 frames / generation s | 10 prompts, seed 1234 | quiet | 2.45 (= 12 / 4.9) | 4.9 |
| quiet timing pass (facts §15) | steady, chunks ≥ 5 | jar + sandcastle | quiet | 2.939 | 4.08 |
| bench-43 (facts §2, §12) | steady, chunks ≥ 5 | 43 prompts × 2 seeds | under load | 3.038 | 3.9 |

Speed-ups of our three modes on the steady basis. Bold = best, underline = second per column (teacher excluded).

| arm | quiet steady s/chunk ↓ | quiet fps ↑ | speed-up, quiet steady (quote this) ↑ | bench-43 fps, under load ↑ | speed-up, under load ↑ |
|---|---|---|---|---|---|
| teacher 14B 4-step | 2.939 | 4.08 | 1.00× | 3.9 | 1.00× |
| ours: teacher-parity mode (chunk-wise 0.7 + safe commit) | 1.764 | 6.80 | 1.67× | 6.8 | 1.74× |
| ours: quality mode (832 2-step fused + partial) | <u>1.232</u> | <u>9.74</u> | <u>2.39×</u> | <u>9.8</u> | <u>2.5×</u> |
| ours: realtime mode (704 2-step fused + partial) | **0.763** | **15.73** | **3.85×** | **15.8** | **4.1×** |

Sources: facts §15 (quiet: jar + sandcastle, generation only, no decode; castle s1234 excluded for the 704/832 arms after a cold compile) and facts §12 (under load: bench-43, 86 clips, 7 concurrent jobs). The quality and realtime rows use cloud-only `cloud_chunk_s`; the teacher and teacher-parity rows use `chunk_start` intervals (facts §12 notes this basis difference).
- **Rule for slides:** quote 1.67× / 2.39× / 3.85×. Never divide one of our steady fps by 4.9: that mixes bases and understates every speed-up by ≈ 17 % (4.08 / 4.9).
- **Other whole-clip numbers in this package:** SDVG's bench-43 3.5 / 6.4 fps (12 / (gen s / 13), including the forced target chunk 0) are 3.1 / 6.7 on the steady basis (facts §2). README §N values (153 frames / gen s, 10 prompts, seed 1234): 14B 2-step 832 8.3, anchor split 8.5 (Part A split table), old student 10.4 (B4), chunk-wise adaptive 6.5 (facts §3); the quiet steady 14B 2-step 832 separate is 7.98 fps (facts §15). The output fps of the per-frame latency table (B8) are whole-clip means too: +5 to +13 % above steady for the optimized profile rows in B8 (+5 to +33 % across both stacks, facts §13; `data/e_steady_fps.json`); for the ours rows the steady values are 9.97 (realtime) and 6.39 (quality) vs 10.8 / 6.9 whole-clip (facts §13). Compare them only with each other.
- **Load:** for our three modes, the bench-43 under-load and the quiet steady s/chunk are within 1 % (1.764 vs 1.764, 1.225 vs 1.232, 0.761 vs 0.763), and the teacher's are 3 % apart. The prompt sets differ, so this is an observation, not a controlled load test; facts §0/§2 keep the "≈ 10–20 % slower under load" allowance as the rule. Either way the ratios agree within ~5 %.

### B8. Ours: realtime mode has the lowest action→frame latency among the operating modes and baselines measured on one GPU (facts §13; f13_latency)
Protocol (Chenfeng's sdv2_latency definitions, our port `scripts/wm_latency.py`): jar + sandcastle × seeds 1234/4321/9999, empty text prompt, one RTX PRO 6000 GPU, prepare excluded, every committed chunk VAE-decoded right away on a side CUDA stream of the **same** GPU (decode competes with generation). Ours rows = `res_split_infer --stream-decode 1 --no-device` (cloud 14B + streaming decode at the cloud resolution, no device stage or upsample: realtime 368–384×704, quality 448–464×848–864; `data/f13_latency_table.md`); the student and teacher decode at ≈ 832×480 (full VAE 631 vs 426 ms/chunk at 704, facts §6), so the decode load is not symmetric. Action→frame latency (`server_ms`) = ready(frame) − the moment its chunk consumed its action. Profile arms: 6 clips (10/12 profile run). Ours: 5 clips (timing pass; castle s1234 excluded after a cold compile). Bold = best, underline = second per column.

| arm (optimized stack, decode on the same GPU) | output fps, whole clip ↑ | action→frame p50 (ms) ↓ | p90 (ms) ↓ | p99 (ms) ↓ | miss > 750 ms ↓ | first frame after gen start (ms) ↓ | gen s per 10 s clip ↓ |
|---|---|---|---|---|---|---|---|
| teacher 14B 4-step (target only) | 4.2 | 3237 | 3728 | 3810 | 100 % | 1714 | 37.8 |
| student 1.3B 4-step (draft only) | <u>10.4</u> | <u>1434</u> | <u>1680</u> | <u>1700</u> | **96.7 %** | **439** | <u>14.6</u> |
| SDVG + actions τ −0.7 | 3.0 | 4463 | 4897 | 4946 | 100 % | 1794 | 51.2 |
| SDVG + actions τ −2.2 | 3.7 | 4156 | 4851 | 4947 | 100 % | 1791 | 43.3 |
| ours: quality mode (832 2-step fused + partial) | 6.9 | 1995 | 2298 | 2310 | 100 % | 852 | 22.4 |
| ours: realtime mode (704 2-step fused + partial) | **10.8** | **1296** | **1492** | **1495** | **96.7 %** | <u>574</u> | **14.3** |
| ours: teacher-parity mode | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

Miss column: the two 96.7 % share bold; every other row is 100 %, so it has no underline. Raw: `lingbot-world-v2/specwm_out/sdvg_profile/latency/<label>/summary.json` (profile arms) and `lingbot-world-v2/specwm_out/timing_pass/latency/{ours704,ours832}/summary.json`. The teacher-parity mode runs through `spec_early`, which has no streaming decode, so it has no per-frame row. f13_latency shows the profile rows (n = 6), the realtime and quality modes (n = 5, castle s1234 excluded) and four non-mode variants of ours (lighter rows).
- **The lowest action→frame latency among the operating modes and baselines measured** (two non-mode variants are lower; see "Only among operating modes and baselines" below): the realtime mode leads p50, p90 and p99 (p50 1296 ms vs student 1434, quality mode 1995, teacher 3237, SDVG 4156–4463). That is ≈ 2.5× lower than the teacher's p50. Output fps and generation time are level with the student (10.8 vs 10.4 fps, 14.3 vs 14.6 s per clip; small gaps across two runs, see caveats).
- **The student still shows its first frame sooner** (439 vs 574 ms). Generation alone, the student is also faster (17.2 vs 15.8 fps under load, B4). A likely reason for the flip with decode on the same GPU is that the student decodes 832 frames (full VAE 631 vs 426 ms/chunk at 704, facts §6). This run did not time the decode separately, so that is not a measured cause.
- **Only among operating modes and baselines.** Two of our variants that are not modes have lower p50: 704 fused without partial, 1180 ms (12.0 fps; 12 % boundary jumps), and 576 fused + partial, 818 ms (17.1 fps; 576 loses quality, B2). The partial commit also beats separate commit on latency: 704 separate commit has p50 1484 ms (9.3 fps) (facts §15).
- **No configuration meets the 750 ms per-frame budget:** the realtime mode misses it on 96.7 % of frames, like the student. With the decode on a separate device, the output rate would be bounded by the cloud (0.761 s/chunk → 15.8 fps), but that was not measured end to end.
- **Not measured with this protocol:** Matrix-Game 2.0 and Waypoint (we have only their throughput from their own pipelines, decoder included: 19.7 / 31.2 fps quiet, facts §15), and the teacher-parity mode (n/a, above).

### B9. StEvo-38: the 14B-based arms, including both measured modes of ours, keep the teacher's physics; the student and Waypoint do not (facts §14; f12_stevo)
Setup: 38 camera-WM StEvo-Bench tasks, seed 1234, 249-frame videos covering the 253-frame look-away-and-back camera path. Judge: local Qwen3-VL-32B, 3-vote majority, 32 frames, StEvo criteria (control / physics / state). NOT comparable with the paper's Gemini leaderboard. The teacher-parity mode was not run (time), and neither was SDVG τ −2.2 (disk). Bold = best, underline = second per column; ties share the mark.

| arm | physical inaccuracy ↓ (tasks of 38) | trigger applied ↑ | occlusion done ↑ | control success ↑ | state evolution ↑ | task success ↑ |
|---|---|---|---|---|---|---|
| teacher 14B 4-step | <u>0.68</u> (26) | <u>0.45</u> | 0.34 | <u>0.13</u> | **0.08** | **0.08** |
| ours: quality mode (832) | **0.55** (21) | 0.39 | 0.32 | 0.08 | <u>0.05</u> | <u>0.05</u> |
| ours: realtime mode (704) | <u>0.68</u> (26) | 0.37 | 0.24 | 0.08 | <u>0.05</u> | 0.00 |
| SDVG + actions τ −0.7 | 0.71 (27) | **0.61** | 0.18 | 0.11 | 0.00 | 0.00 |
| student 1.3B 4-step | 1.00 (38) | 0.39 | <u>0.42</u> | <u>0.13</u> | 0.00 | 0.00 |
| Matrix-Game 2.0 | 0.79 (30) | 0.24 | **0.97** | **0.24** | 0.00 | 0.00 |
| Waypoint-1.5-1B | 0.92 (35) | 0.26 | 0.05 | 0.00 | 0.03 | 0.00 |

Fractions of 38 tasks; binomial s.e. ≈ 0.05–0.08, so differences < ~0.15 are not significant (facts §14). Task counts from `lingbot-world-v2/specwm_out/stevo38/eval_runs/lingbot_b38_<arm>/summary.json` (avg_physical_inaccuracy × 38).
- **Physical inaccuracy is the informative column.** All four 14B-based arms sit at 0.55–0.71. The student is flagged on every task (1.00) and Waypoint on 35 of 38 (0.92). The student gap clears noise for every 14B arm (paired p < 0.001). Waypoint's gap clears it for the teacher and both of our modes (p ≤ 0.035), but SDVG vs Waypoint is borderline (3 vs 11 discordant tasks, p = 0.057). (Exact two-sided McNemar on the same 38 tasks, uncorrected; `data/f12_stevo_table.md`.)
- **Our cheaper modes keep the teacher's physics:** the realtime mode equals the teacher (26 of 38 each). The quality mode's 0.55 is within noise of the teacher's 0.68 (gap 0.13), so it is not evidence of better physics than the teacher.
- **Against Matrix-Game 2.0 (0.79)** the direction favours the 14B arms, but only the quality mode's gap (0.24) clears ~0.15. The teacher, realtime mode and SDVG (gaps 0.08–0.11) are within noise of it.
- **Message for the slide:** the cloud 14B is what buys physical plausibility. The same-family 1.3B student fails it everywhere, which matches the bench-43 tabletop & physics group (student quality 5.59 vs teacher 7.45).
- **State evolution and task success are at the floor (0–0.08) for every arm,** as in the 10/7 run (85 tasks: 1–5 %). This judge and protocol do not separate the arms on the benchmark's main metric.
- **Matrix-Game 2.0's 0.97 "occlusion done"** reflects that it turns away visibly (game-style mouse look) while rarely showing the prompted action (trigger applied 0.24).

### Caveats
- **Load conditions:**
  - Bench-43 fps were measured under load (7 GPU jobs at once; facts §0 allows ≈ 10–20 %). The teacher's 3.9 under load vs 4.08 quiet are 3 % apart on the same steady basis; the old "4.9 quiet" is a whole-clip average on another prompt set, not a load comparison (B7).
  - The 12-prompt fps are 12 / cloud s per chunk on one GPU, decode excluded, under mixed load: 704 fused 0.640 s (18.7 fps) is exclusive GPU (README V3); 704 fused + partial 1 (0.761 s → 15.8 fps) is under load; 704 fused + partial 2 (0.874 s → 13.7 fps) and 832 fused + partial 1 (1.24 s → 9.7 fps) have no recorded load condition. (The bench-43 quality-mode 9.8 fps is under load, facts §2.) Same-condition (quiet) timings are now in facts §15 and the quiet column of the B1 table; 704 fused + partial 2 and 832 fused have none.
  - LingBot fps exclude the VAE decode; the quality and realtime fps are cloud-GPU time only and assume the device step and the decode run on a separate device GPU (not measured end to end; serial device step: 13.6 / 8.9 fps, facts §12). MG2 and Waypoint fps include their own decoder on one GPU.
  - The quiet timing pass covers jar + sandcastle only; no quiet bench-43 fps exist. Per-frame latency (B8) exists for the realtime and quality modes but not for the teacher-parity mode, Matrix-Game 2.0 or Waypoint.
  - The latency table mixes two runs with the same protocol and GPU type: the profile arms (6 clips; our vLLM judge was running on another GPU) and ours (5 clips, timing pass; castle s1234 dropped after a cold compile). Treat gaps of a few percent as ties (e.g. 10.8 vs 10.4 fps, 14.3 vs 14.6 s per clip); the realtime mode's p50 lead over the student (1296 vs 1434 ms, ≈ 10 %) is larger, but it also compares two runs.
- **Partial seeds:** in the jump table, the 832 fused and 832 fused + partial 1 rows are seed 1234 only (12 clips). The teacher row's jump share is seed 1234 only, but its VLM values (45.4 / 7.64) are 3 seeds (36 clips); on seed 1234 alone the teacher scores 46.7 drift / 7.58 quality (`paper/1012/vlm_scores_3seed.json`), which is the matching reference for the two 832 s1234 rows. The 1.5× threshold of the jump metric is our own choice; mid-chunk control positions give 0.7–2.5 % in the arms of the jump table (0.7–3.7 % across all nine arms, facts §5).
- **Scalability is shown on one second model only:**
  - Matrix-Game 2.0, 10 prompts, seed 1234, DINO + VLM.
  - Fused / partial commit has not been tried on it.
  - The drift-head transfer is measured against VLM pseudo-labels, which agree with humans at AUROC 0.82.
  - No next-gen base model (e.g. Wan2.2-5B) has been run.
- **Resolutions are only roughly matched:** ≈704×384 vs 640×352 is close, not equal. MG2 ran without torch.compile (official claim 25 fps on H100). Waypoint has no matched run.
- **Small groups:** the easy group is n = 10 clips (5 prompts × 2 seeds).
- **Judge limits:** the VLM quality prompt is not validated against humans, and 8 frames per clip cannot see chunk-boundary jumps.
- **Empty prompts not quantified:** we have not checked how much they changed the scores. Nor have we audited which other experiments (e.g. drift-head training scenes) drew from the 47 empty-prompt tasks.
- **StEvo-38 (B9):** one seed, 38 tasks, binomial s.e. ≈ 0.05–0.08, so only gaps > ~0.15 count. The judge is our local Qwen3-VL-32B, not the paper's Gemini, and facts §14 records no human check of its physics calls. The teacher-parity mode was not run.
- **Rollout-path figures (f10_paths_*)** measure distance from the teacher's samples, not quality (facts §11).

### 발표 메모 (Korean)
- 10/10 미팅 포인트 네 개 상태부터 짧게 정리: scalability는 MG2에서 speed lever, 저해상도 drift 패턴, drift head(AUROC 0.77)까지 전이 확인; fused/partial commit과 차세대 base model은 미확인 → 부분 성공. 10/10에 나온 세 가지 split은 두 개 닫고 하나(클라우드 저해상도 + 디바이스 디코드)만 남김(anchor split은 10/8에 우리가 따로 해본 것, 역시 닫음). matched-resolution 비교는 bench-43으로 완료, 쉬운 프롬프트도 추가함.
- 가장 먼저 정정할 것: 점프 수정 후 ours: realtime mode(704 2-step fused + partial)는 클라우드 기준 15.8 fps(부하 하 측정, 0.761 s/chunk, 클라우드 GPU 시간만. 디바이스 단계와 full-VAE 디코드는 별도 디바이스 GPU에서 병렬로 돈다고 가정했고 end-to-end로는 안 쟀음. 측정된 디바이스 단계를 직렬로 넣으면 13.6 fps). 이전 18.7 fps는 fused만 쓴 config를 단독 GPU에서 잰 값이라 같은 조건 비교가 아니었음. 이제 quiet timing pass(같은 조건, jar + castle)에서 fused만 0.630 s/chunk(19.04 fps), fused + partial 0.763 s/chunk(15.73 fps) → partial commit 비용 ≈ 0.13 s/chunk(같은 5 clip끼리는 0.640 → 0.763, ≈ 0.12 s)는 부하 탓이 아니라 실제 비용(full clean commit 0.330 s의 ≈ 40 %). 그래도 separate commit(12.50 fps)보다 빠름. fused commit이 청크 경계 점프(12 %)를 만들었고 partial commit으로 4 %까지 낮춤(separate commit 3 %).
- 방법론 정정(B7): teacher의 "4.9 fps quiet"는 whole-clip 평균(초반 KV window가 차기 전 싼 chunk까지 포함)이었음. speed-up 표(facts §12, §15)의 fps와 같은 steady 기준으로는 quiet 4.08 / under load 3.9(3 % 차이) → "3.9 vs 4.9"는 부하 차이가 아니라 기준 차이. 속도 향상은 quiet steady 기준 1.67× / 2.39× / 3.85×(teacher-parity / quality / realtime)로 말할 것. bench-43 under-load 1.74× / 2.5× / 4.1×와 5 % 이내로 일치. 4.9로 나누면 ≈ 17 % 과소평가됨. SDVG의 bench-43 3.5 / 6.4 fps도 whole-clip이라 steady로는 3.1 / 6.7(τ −2.2는 teacher-parity mode 6.8과 속도가 거의 같음, 품질은 훨씬 나쁨).
- latency(B8): 같은 GPU에서 decode까지 streaming으로 돌릴 때 realtime mode가 측정한 operating mode와 baseline 중 action→frame latency 최저(p50 1296 ms vs student 1434, teacher 3237, SDVG 4156–4463 ms; teacher 대비 ≈ 2.5배 낮음). 단 750 ms 예산은 못 맞춤(miss 96.7 %, student와 같음), 첫 프레임은 student가 더 빠름(439 vs 574 ms). 우리 변형 중 점프가 있는 704 fused(1180 ms)와 576 fused + partial(818 ms)은 더 낮지만 operating mode가 아님. MG2 / Waypoint와 teacher-parity mode는 이 프로토콜로 안 쟀다는 점도 같이 말할 것.
- StEvo-38(B9): 의미 있는 칸은 physical inaccuracy뿐. 14B 기반 arm(teacher 0.68, quality 0.55, realtime 0.68, SDVG 0.71)은 student(1.00, 38개 전부 flag)보다 확실히 나음(paired p < 0.001). Waypoint(0.92)보다는 teacher와 우리 두 mode가 나음(p ≤ 0.035). 다만 SDVG vs Waypoint는 경계선(3 vs 11 tasks, p = 0.057)이라 "확실히 낫다"고 하면 안 됨. realtime은 teacher와 같음(26/38) → 해상도와 step을 줄여도 물리 타당성은 유지. quality mode 0.55는 teacher와 noise 이내라 "teacher보다 낫다"고 하면 안 됨. MG2(0.79)와는 quality mode만 noise(~0.15)를 넘음. state evolution / task success는 모든 arm이 바닥(0–0.08)이라 구분 불가. 로컬 Qwen3-VL-32B 판정, seed 1개라 논문 Gemini leaderboard와 비교 불가.
- 저해상도 품질 손실: 704에서는 디바이스 1.3B refiner 손실(−0.50)과 해상도 손실(−0.31)이 비슷한 크기(차이 −0.19 ± 0.18, 유의하지 않음), 576에서는 저해상도 14B 자체 손실(−0.70)이 refiner(−0.25)보다 큼. 디코더도 중요해서 TAEHV는 quality −0.56 ~ −1.06을 잃음. 이전 fps는 TAEHV 기준, LingBot arm의 VLM 점수는 full VAE 기준이었다는 불일치도 솔직히 언급(MG2/Waypoint는 자체 디코더).
- student는 같은 스택에서 17.2 fps. 이전 ≈10 fps는 옛 측정으로, 스트리밍 VAE 디코드가 생성 시간에 포함된 것이 유력한 원인(facts §13: 같은 스택 + 같은 GPU 스트리밍 디코드로 14.6 s / 10.4 fps, 옛 14.7 s / 10.4 fps와 거의 같음; 옛 프롬프트를 직접 재실행한 건 아니라 '확인'이 아니라 '유력'으로 말할 것). student가 realtime mode보다 빠르니(17.2 vs 15.8, 둘 다 부하 하, 디코드 제외), 우리 메시지는 "속도"가 아니라 "student급 속도에서 훨씬 높은 품질"(6.90 vs 5.83, drift 56.9 vs 75.1). MG2와의 fps 비교(15.8 vs 20.0)도 MG2는 디코드 포함 1 GPU, 우리는 클라우드 GPU 시간만(디바이스 단계·디코드는 별도 GPU 가정, 미측정)이라는 점 언급.
- hourglass/jar/castle은 지금까지 텍스트 프롬프트가 비어 있었음. bench-43은 영향이 없지만 이전 jar/castle 결과(10/5 SDVG 포함)는 전부 이 전제를 달고 봐야 한다고 먼저 밝히기.
