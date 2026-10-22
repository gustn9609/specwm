# Night 3 (2026-09-24, from 1:54 AM CDT) — autonomous loop, goal: Base-like video at 200–400 s

## Done before the loop (Sep 23 9:30 PM – Sep 24 1:50 AM CDT)
- Exp 1 LoRA 4-step (lightx2v 1022) correction: 245–399 s but castle 4321 creatures, castle 9999 road, jar duplicates -> rejected.
- Exp 2 KV-reuse RePaint: clean t=0 prefix breaks high expert (rel err 1.1-1.9); fixed with noisy prefix cache; still drifts (white liquid) and 81 s/correction -> stopped.
- Exp 4 handoff (Base high-expert only -> Fast finishes): hand1 1393/931 s, hand2 465/905 s (castle4321/jar9999); no better than 16-step.
- Timing caveat: GPU 6 shared with lx2995 since 2:52 UTC; LoRA runs' Fast part contended.

## Loop log
### 2:43 AM CDT — Base probe round 1 (castle 4321, jar 1234; chunks 6, 10; 16-step CFG RePaint on the Base 70-step prefix)
A (DINO dist to the full-prefix result; seed-1 rerun = sampling-noise floor):
| case | seed floor | blur2 old | blur4 old | drop old |
| castle c6 | 0.27 | 0.04 | 0.07 | 0.22 |
| castle c10 | 0.32 | 0.27 | 0.30 | 0.33 |
| jar c6 | 0.94 | 0.08 | 0.05 | 0.18 |
| jar c10 | 0.83 | 0.03 | 0.06 | 0.94 |
latent rel. dist: seed 0.70-1.40, blur 0.11-0.35, drop 0.44-0.95.
-> old prefix (between sink chunk 0 and last 12 latents) matters (dropping it = as different as a new seed on jar c10) but only coarsely: 4x lower res is below the seed floor. => mixed-resolution prefix.
B (x0 at step i vs final, DINO): after the 6 high-expert steps still 0.27-0.83; converges below ~0.1 only around step 11-12 (low expert). Colour settles by ~step 5-7.
-> layout/semantics are NOT fixed by the high expert alone: explains handoff failure. Keep low-expert steps.
Also: Base's own seed variance is huge (jar c6 0.94) — "looks like Base" must be judged against Base's distribution, not one rollout.
### 2:50 AM — launched probe round 2 (blur4_r6, blur4_r3, blur8_r12) and implemented mixed-res Base forward (specwm/base_mixed.py, spec_v0 --corr-mixed K:R).
### 3:39 AM CDT — probe round 2 + mixed-forward unit test
Round 2 (DINO dist to full-prefix result; seed floor castle .27/.32, jar .94/.83 for c6/c10):
| case | blur4, keep last 6 | blur4, keep last 3 | blur8, keep last 12 |
| castle c6 | .05 | .05 | .07 |
| castle c10 | .31 | .28 | .23 |
| jar c6 | .15 | .13 | .12 |
| jar c10 | .05 | .19 | .07 |
-> 4x low-res old prefix with last 6 latents full-res is below the seed floor everywhere; keep-last-3 starts to cost on jar c10.
Unit test base_mixed.mixed_forward: all-k=1 split vs Base forward rel .0024 (correct); k=2 middle segment -> .08 on last latent.
Launched exp5_mixed on GPUs 4,7 (clean): smoke, then --corr-mixed 4:6 on all 6 cases (castle4321, jar9999 first), then 4:12 on castle4321/jar9999.
Expected tokens at chunk 10: 33 latents -> ~13 (4:6) / ~19 (4:12).
### 4:35 AM CDT — mixed 4:6 first results
- smoke jar s1234 61f always: 488 s (≈ plain 16-step at this short prefix, as expected); video: single jar, sand pile grows (matches the StEvo target) — no red-liquid swap (handoff had it).
- castle 4321 full (sem best + mixed 4:6): **673 s vs 936 s** (16-step). Corrections: c1 75, c2 98, c3 137, c10 154 s (16-step had c7 = 286 s).
  Video: castle survives through the middle like Base (16-step lost it at once and drifted to a forested headland); later blue sky/open sea, trees at the very end.
- Change: mixed path only when there is an old part (L-R > 3) — early chunks use the plain path (mixed had ~10% overhead there); RoPE factors precomputed once per forward.
### 4:40 AM CDT — mixed 4:6 jar 9999: 825 s vs 1534 s (16-step); corrections flat 146-158 s (16-step: 244/344/600 s)
BUT: chunk 6 (a Base mixed correction, post-score 0.6 -> accepted) turned the sand into dark red liquid; it persisted to the end. 16-step kept sand.
Hypothesis: 4x low-res old prefix loses material/texture (granular sand vs liquid) — DINO in the probe does not see that; same failure as ctx 18 (old prefix dropped).
Queued exp6 after exp5: 2x low-res (2:6, 2:12) on jar 9999 / castle 4321. exp5 still runs 4:12 on both at its end.
### 4:48 AM CDT — mixed 4:6 castle 1234: 630 s vs 673 s. Same plan as 16-step; chunks 1-3 identical (plain path now used when no old prefix), chunk 8 correction 148 s vs 334 s. Video ~same as 16-step (identical first half; end slightly clearer sky + a few trees).
### 5:22 AM CDT — mixed 4:6 jar 1234: 670 s vs 1338 s (-50%). Same plan; corrections c9 152 vs 396 s, c12 175 vs 600 s.
Video: identical to 16-step through chunk 8; c9 mixed correction keeps sand (16-step had coffee-coloured sand there); **last chunk (c12, mixed correction) turns into yellow liquid + white foam** — 2nd material swap from a 4x mixed correction (jar 9999 c6 red liquid). Supports the texture-loss hypothesis; 2x variants queued (exp6).
Verifier note: colour term uses only S/V shifts, not hue -> liquid swaps pass the post-correction check.
### 5:37 AM CDT — mixed 4:6 castle 9999: 663 s vs 727 s (one extra correction at c12). c9 150 vs 387 s.
Video: same through c8; after the c9 mixed correction the castle stays whole and a big green wave approaches at the end — closer to Base 9999 (castle standing + wave at the end) than 16-step (castle fragments + mountains/forest).
### 5:57 AM CDT — mixed 4:6 jar 4321: 838 s vs 1347 s; c9 152 vs 405 s, c12 175 vs 600 s.
Video: same through c8; **after the c9 mixed correction the scene changes (plates, napkin, bowl of yellow liquid, plant), then a white-salt jar** — severe.
Summary 4:6 over 6 cases: time -7% to -50% (avg 1090 -> 717 s); castle 3/3 same-or-closer to Base; jar 3/3 material/scene swaps after a mixed correction. => 4x is too coarse for the small-object task.
Checked a view-aware variant (full-res for prefix frames whose camera yaw matches the new chunk): the action is a monotonic 360 deg yaw (0,36,...,180,-144,...,0 per chunk), so for c9 (yaw -108..-72) no earlier view matches except recent chunks; chunk 0 (sink, full-res) already covers the return view at c11. Not worth it here.
### 6:15 AM CDT — mixed 4:12 castle 4321: 1106 s (6 corrections; c7 225 s, c10 231 s). Video follows the 16-step path (forested headland).
Caveat found: the first 4:6 castle-4321 run used the mixed path already at c3 (before the plain-path fix) -> tiny numeric difference -> different trajectory from c4 on. So its "castle survives" was partly a different random path, not only the method. Rollouts are chaotic: single-seed quality comparisons are noisy (Base's own seed spread is also large). Speed numbers are solid; quality needs more seeds or a distribution-level check.
### 6:48 AM CDT — mixed 4:12 jar 9999: 1550 s (no faster; c6 225, c12 258 s) and liquid again (blue/brown). Jar mixed runs: 4/4 liquid or scene swap; 16-step jar: 0/3.
Open question: the probe's blur variants (low-res upsampled back to the full token grid) were harmless, but the real mixed forward uses fewer tokens with strided RoPE (out of distribution for Base). Which one causes the swaps — information loss or OOD tokens?
Paused exp6 (2:6/2:12 full runs) and launched probe round 3 on GPUs 4 (jar 1234) / 7 (castle 4321): mix4_r6, mix2_r6, mix4_r12 = the real mixed-token path, same noise stream as the blur variants; compare with blur4_r6 / blur2 / blur4 from rounds 1-2 and look at the frames for material swaps.
### 7:12 AM CDT — probe round 3: the real mixed-token path is the culprit (not information loss)
DINO dist to full-prefix result, same noise: jar c10 blur4_r6 .05 vs mix4_r6 .95 / mix2_r6 .89 / mix4_r12 .90; castle c10 .31 vs .83/.89/.36; castle c6 .05 vs .57/.29/.13.
Frames: mix variants hallucinate a salad bowl (jar c10), trees (castle c10), liquid (jar c6); blur variants ~= full. Even 2x merging breaks -> Base cannot read merged/strided low-res tokens (OOD), though it tolerates low-res *content* on the normal grid.
Next (round 4, launched): token subsample (sub2: real full-res patches, every 2nd token in h and w, true positions) and temporal subsample (tsub2/tsub3: every 2nd/3rd old latent frame, true positions).
### 7:40 AM CDT — probe round 4: every token-dropping variant changes the result
DINO dist to full-prefix (same noise): sub2 jar .04/.86, castle .76/.44; tsub2 jar .17/.94, castle .23/.22; tsub3 jar .06/.85, castle .68/.32 (c6/c10). blur4 (no speed-up) jar .15/.05, castle .05/.31.
Frames: at jar c10 the full-prefix result is the turned-away empty table; every dropped variant re-draws the jar (tsub2: sand, tsub3: red liquid, sub2: odd cup). tsub2 is the least harmful (castle closest).
=> Base needs its whole prefix on the normal grid; training-free token reduction costs quality. Speed must come from elsewhere or from adapting Base (e.g. a LoRA trained to read compressed memory).
Round 5 launched: CFG truncation (cfg8 = CFG on first 8 of 16 steps, cfg6 = only the 6 high-expert steps) and flow reuse (skip forwards at steps 9, 11, 13).
### 8:08 AM CDT — probe round 5: CFG truncation and flow reuse are near-free
DINO dist to full result (same noise), jar c6/c10, castle c6/c10: cfg8 .08/.04/.08/.30; cfg6 .11/.09/.11/.22; reuse(9,11,13) .02/.05/.06/.10 — all far below the seed floor, same scene/material in the frames.
Time per correction vs full grid (c10): cfg8 -25%, cfg6 -31%, reuse -18%. Combined cfg6+reuse: 19 instead of 32 forwards (~-40%).
Launched exp7 on GPUs 4,7: best sem + --cfg-steps 6 --flow-reuse 9,11,13 on all 6 cases.
### 8:24 AM CDT — cfg6+reuse castle 4321: 771 s vs 936 s (6 corrections vs 5). Per correction: c1 51 vs 67, c2 52 vs 86, c3 73 vs 121, c4 94 vs 156 s (-24..-40%).
Video: same kind of path as 16-step (headland, pine forest), greyer/cloudier sky; no new artifacts.
### 8:30 AM CDT — user: use GPUs 4,5,6,7. GPUs 5,6 were free (lx2995 jobs gone) -> second worker on 5,6 running the same exp7 bundle (run() skips cases already running/done).
### 8:45 AM CDT — cfg6+reuse jar 9999: 988 s vs 1534 s (-36%). Per correction c6 147 vs 244, c12 360 vs 600 s.
Video: single jar, sand kept to the end, no liquid (mixed 4:6/4:12 had red/brown liquid here), and no coloured straws (16-step had blue/green straws). Clean.
### 9:00 AM CDT — cfg6+reuse jar 1234: 842 s vs 1338 s (-37%); same plan; c9 241 vs 396, c12 362 vs 600 s.
Video: single jar, sand colour consistent throughout (16-step had coffee-coloured sand at c9-12). Clean.
### 9:10 AM CDT — cfg6+reuse castle 1234: 678 s vs 673 s (different path: corrections at c7/c10 instead of c8; c7 177 s, c10 265 s).
Video: a dark car/boat + pier building appear at c9 and stay (c11 adds a castle). They come from **accepted Fast drafts** (c9 score 0.87, c11 0.65 < thr 1.0), not from a Base correction -> verifier miss (new object, small global DINO change), independent of the correction speed-up.
### 9:30 AM CDT — exp7 (cfg6 + reuse 9,11,13) complete
| case | 16-step s | cfg6+reuse s | by eye vs Base |
| castle 1234 | 673 | 678 | car/boat + pier from accepted drafts (verifier miss) |
| castle 4321 | 936 | 771 | ~ 16-step path (headland/forest), greyer sky |
| castle 9999 | 727 | 419 | castle persists like Base; odd dark cavity in the castle at the end |
| jar 1234 | 1338 | 842 | clean, sand colour consistent (better than 16-step) |
| jar 4321 | 1347 | 837 | very Base-like: static jar filling with sand (16-step: stick + coffee sand) |
| jar 9999 | 1534 | 988 | clean, single jar, sand kept |
| avg | 1093 | 756 (-31%, ~27% of Base 70-step) | jar 3/3 clean; castle 1 artifact from drafts |
Launched exp8 (cfg4 + reuse 7,9,11,13) on jar 9999 (4,7) and castle 4321 (5,6) to use the last ~30 min.
### 9:55 AM CDT — exp8 (cfg4 + reuse 7,9,11,13)
- jar 9999: 846 s (cfg6+reuse 988, 16-step 1534); per correction c1 42, c12 304 s. Video ~identical to cfg6+reuse (clean sand jar).
- castle 4321: 1002 s (7 corrections -> slower overall than cfg6's 771); per correction c1 43, c4 80, c11 256 s. Video: headland path, foggy; a small driftwood/figure at c5. Similar quality to cfg6.
- Per-correction cost cfg4+reuse ≈ -35..-50% vs 16-step; the total is dominated by how many chunks get corrected (path-dependent).

## End of window summary (for the user)
Best training-free setting: sem gate (thr 1.0, recent ref, retry 1) + 16-step RePaint with CFG on the first 6 steps only + flow reuse at steps 9,11,13:
avg 756 s over 6 cases (16-step: 1093 s; Base 70-step ~2800 s), jar 3/3 clean, castle: one accepted-draft artifact (verifier miss).
Dead ends tonight: LoRA 4-step (drift/creatures), KV-reuse RePaint (drift, slower), high-expert handoff (layout not fixed until ~step 11), mixed/merged low-res prefix tokens and token/temporal dropping (Base cannot read them -> liquid/scene swaps).
Next candidates: (1) verifier for new objects (castle 1234 car came from accepted drafts), (2) fewer corrections (verifier threshold/schedule), (3) adapting Base to compressed memory (LoRA trained on mixed-token prefixes) if the 200-400 s target stays.

# Sep 24 afternoon
### 12:48 PM CDT — option C: mixed-memory LoRA training (GPUs 4,5 high / 6,7 low), 400 steps, data = 44 Base 70-step rollouts of S00/S01/S02/S05 (jar/castle held out); LoRA (rank 32, attn+FFN) active only on merged old-prefix tokens; distill to full-prefix teacher flow on the new chunk.
Step-1 relative flow error from merging (no LoRA): high 0.077, low 0.005. Step 40: EMA high 0.023, low 0.0066.
### 1:35 PM CDT — action-aware scheduling (user OK'd GPUs 2,3): --act-theta: force a Base correction when camera yaw moved >= theta since the last corrected chunk. Yaw per chunk end: 0,24,60,96,132,168,-156,-120,-84,-48,-12,-24,-60. theta 90 -> forced 6, 9; theta 72 -> 5, 7, 9.
Launched exp9 on 2,3: act90 then act72, with cfg6+reuse, all 6 cases.
### 1:55 PM CDT — guidance ideas (user): implemented --autoguide w (Base correction: CFG uncond replaced by Fast velocity on the new chunk, v = v_fast + w(v_base,cond - v_fast); autoguidance with Fast as the bad model; ~1 s Fast forward instead of a full-window Base uncond) and --histguide w (Fast drafting: v = v_full + w(v_full - v_nomem), v_nomem = Fast attending only to the current chunk; history guidance).
Queued exp10 on GPUs 2,3 after exp9: smokes, then ag2 / hg1 / ag3 / hg0.5 on castle 4321 + jar 9999 (on top of cfg6+reuse).
### 2:05 PM CDT — act90 castle 1234: 1472 s vs 678 s (cfg6+reuse sem only). 8 corrections: forced 6, 10 (anchor moved to c7 by a sem correction) + sem 7, 11, 12 — forcing did not reduce the sem corrections.
Video: the car/pier is gone (good), but the castle keeps re-appearing in new views (c7-c11) — Base, seeing the castle in the prefix, redraws it; Base 70-step itself had washed it away. Not closer to Base, 2x slower.
### 2:10 PM CDT — lx2995 started train_pg.py on GPUs 0-3 (~63 GB each); act90 jar 9999 OOMed on GPU 2. Stopped the 2,3 workers (exp9 remaining + exp10 queue) to avoid a chain of OOM failures. LoRA training on 4-7 continues (step 110/400).
### 4:26 PM CDT — mixed-memory LoRA trained (400 steps, 3.4 h on 2+2 GPUs). Final loss EMA high 0.0069 (from 0.077), low 0.0054 (from ~0.005).
Probe (held-out tasks, same noise; DINO dist to full-prefix result, lower = better):
| case | mix4_r6 no LoRA | mix4_r6 + LoRA | blur4_r6 (no speed-up, reference) |
| jar c6 | .16 | .07 | .15 |
| jar c10 | .95 | .91 | .05 |
| castle c6 | .57 | .15 | .05 |
| castle c10 | .83 | .37 | .31 |
Frames: LoRA removes the hallucinations (no salad bowl, no palm tree; castle scenes restored). Jar c6 sand turns paler; jar c10 still draws the jar where the full-prefix result has the turned-away table (camera-direction info from the old prefix still lost).
Chain started: exp10 (guidance) on 6,7; exp11 (mixed 4:6 + LoRA + cfg6+reuse full runs) on 4,5.
### 4:45 PM CDT — autoguide smoke (jar 1234, 61 f, all 4 chunks corrected, cfg6 + autoguide w=2): 300 s total, denoise 258 s (full-CFG 16-step smokes were ~470-490 s). Video: single jar, sand, stable — no artifacts.
### 4:50 PM CDT — history guidance smoke (Fast alone, w=1): video destroyed from chunk 2 (purple/green rings, textures) — the 4-step distilled Fast cannot take guidance of this size. Dropped hg runs; exp10b = autoguide w=2, 3 on castle 4321, jar 9999, jar 1234, castle 1234 (GPUs 6,7).
### 5:05 PM CDT — mixed 4:6 + LoRA + cfg6+reuse castle 4321: **527 s** (cfg6+reuse 771, 16-step 936). Corrections c1 47, c2 53, c3 74, c4 94, c5 95, c9 102 s — late correction cost now flat (~100 s vs 201 s at c8 without mixing).
Video: similar path to cfg6+reuse (open beach, pine islands at the edges, greyer sky), no hallucination (plain mixed 4:6 had a giant fortress/odd path). Same quality class as cfg6+reuse.
### 5:15 PM CDT — mixed 4:6 + LoRA + cfg6+reuse jar 9999: **563 s** (cfg6+reuse 988, 16-step 1534). Corrections flat: c6 99, c9 104, c12 119 s.
Video: the c6 mixed correction still turns the sand into red liquid for that chunk, but the c9 correction brings sand back and it stays sand to the end (no-LoRA mixed: liquid persisted to the end). Partial fix: 1 bad chunk out of 13.
### 5:20 PM CDT — autoguide w=2 castle 4321: 1458 s (7 corrections, 4 retries: post-correction scores 1.66/1.2/1.43/2.14 stay above thr). Per correction cheap (c1-c3 40-56 s) but the video is over-cooked: pink haze from chunk 1, sand spray, washed textures. Dropped w=3; exp10c = ag2 jar 9999 + ag1.5 castle 4321 / jar 9999.
### 5:35 PM CDT — mixed 4:6 + LoRA + cfg6+reuse jar 1234: **463 s** (cfg6+reuse 842, 16-step 1338); c9 104, c12 119 s.
Video: identical through c8; after the c9 mixed correction the jar is empty with clear water (sand gone) for c9-c11; c12 shows sand again, paler. Material swap weakened (no-LoRA: yellow liquid + foam) but not fixed. Jar: 2/2 with a transient material error; castle 1/1 fine.
### 5:50 PM CDT — mixed 4:6 + LoRA + cfg6+reuse castle 1234: 555 s (cfg6+reuse 678). Corrections flat 100-117 s after warm-up.
Video: no car/pier (different path from c7 on); foamy surf, low hills, ends on flat water and beach — closer to Base's foam + flat sand than any other method on this seed.
### 5:55 PM CDT — autoguide w=2 jar 9999: 923 s (cfg6+reuse 988), 7 corrections (one more); per correction cheaper (c6 108 vs 147, c12 264 vs 360 s). Video clean: sand throughout; at chunk ~4 the camera briefly shows only table (like Base turning away), jar returns. So autoguide w=2 is fine on jar, over-cooked on castle (pink haze).
### 6:05 PM CDT — autoguide w=1.5 castle 4321: 772 s (= cfg6+reuse 771), 7 corrections. Video: less haze than w=2 but grainy spray and noisy textures, murky water at the end; not better than cfg6+reuse. Autoguidance so far: no net gain (jar OK, castle degraded).
### 6:15 PM CDT — mixed 4:6 + LoRA + cfg6+reuse castle 9999: 961 s (cfg6+reuse 419 with only 4 corrections). 7 corrections, 3 of them needed a retry (first-try post scores 1.19 / 2.22 / 1.76 > thr) — per correction still ~100 s, the time comes from count + retries.
Video: castle disappears after chunk 5, flat beach and spray, a small dark object at ~c10; Base 9999 keeps the castle to the end -> further from Base than cfg6+reuse on this seed.
### 6:40 PM CDT — seam analysis + new direction
Frame-to-frame change at chunk boundaries: Base 70-step ~ same as inside chunks; ours jumps 3-4x (35-50 vs 12) exactly when entering a Base-corrected chunk, with a 20-30 mean-colour shift: the correction snaps back to a first-frame-like view (jar centred) while the accepted drafts had the camera turned away. Chunk-wise RePaint from pure noise ignores the prefix's camera state (layout is fixed at high noise where the prefix is ~noise). Same effect appeared in the Base-prefix probe (jar c6).
User direction: no separate planning pass; Base must run in parallel from Fast's earliest noisy state. => --corr-from-draft K: Base correction starts from the draft's state entering node K (K=1: sigma .94) and runs only the remaining lower-noise steps (no high-expert/CFG steps). Expected: seams gone (state carries the camera), ~60% fewer Base forwards, and it enables Fast/Base parallelism (Base starts after Fast's first node).
Queued exp12 (fd1 on 6 cases; fd2 and fd1+mixlora on castle 4321/jar 9999) on both pairs after the current runs.
### 9:00 PM CDT — exp12 from-draft results (Base continues from Fast's node-1 state, sigma .94; cfg6+reuse)
Time: castle 371/269/230 s, jar 354/236/640 s (avg 350 s, Base 70-step 12%). Seams: max boundary jump 15-31 (cfg6+reuse 29-48; Base 15-24); jumps into corrected chunks 8-28 = Base's own range -> cuts gone.
Quality: Fast's drift passes through untouched — castle 1234 ends under water, castle 9999 a person walking, jar 1234 jar tips over + flowers, jar 4321 concrete wall/doors, jar 9999 a deer + hedge. fd2 (sigma .83) castle 4321: bushes/dunes; fd1+mixlora castle 4321 (207 s): foam. 
=> At sigma .94 the Fast state already fixes the scene; Base's low-noise steps only polish. Semantic control needs Base to own the high-noise steps (from noise). Parallelism must therefore hide latency (Base runs from noise concurrently with Fast), not skip Base's high steps.
### 9:45 PM CDT — strict temporal metrics (scripts/temporal_eval.py) after the user's verdict that all results look bad next to Base
jar (static scene): Base 70-step flicker 4-14, ssim(4 frames apart) .71-.94; ours (cfg6+reuse etc.) flicker 20-23, ssim .54-.59; Fast alone flicker 48-55, ssim .31-.35.
Per chunk: Base-corrected chunks flicker 6-8 (≈ Base 70-step 4-5), accepted Fast chunks 25-47 (5-8x worse). => the jitter/broken backgrounds the user sees are the accepted Fast chunks; every Fast chunk left in the video breaks consistency.
Camera check: action IS passed (window emb == full emb prefix; camera-ablation probe changes results at jar c10 / castle c6); the jar c6 re-centering happens with or without camera (Base RePaint habit).
Launched exp13 always16 (every chunk corrected, plain 16-step, full CFG) = quality ceiling; queued exp14 --polish S:N (accepted Fast chunks re-noised to sigma .77 and refined by Base's low-noise steps of a 24-step schedule; no CFG) to remove Fast jitter without cuts.
### 10:40 PM CDT — ceiling (always16) verdict, 2 cases: castle 4321 3472 s, jar 9999 3556 s (> Base 70-step). Very stable (jar flicker 8.6 < Base 13.9, drift <= .04) but (1) seams remain (jar boundary jumps 8-25 vs inside 3; castle 2-4x) and (2) the correction ignores the camera action: jar stays centred for all 13 chunks while Base 70-step turns away — chunk-wise RePaint conditioned on frame 0 plans "variations of frame 0", not "continuation of the current view". Cancelled the remaining 4 always16 runs (user OK). Polish runs (exp14) start now.
Next (user OK): seam fixes — A seam polish (joint low-noise re-denoise of the boundary latents), C noise continuity (AR(1)-correlated noise field shared by Fast and Base), D last-frame conditioning (ctx window, poses from the current time).
### 11:40 PM CDT — user: compare (1) latent-level correction, (2) Base-guided Fast, (3) Base K/V memory tokens for Fast (needs training), (4) Fast history guidance, on GPUs 0-3 (user re-allowed 0-3).
Implemented (2): --bguide w --bguide-nodes: at Fast's nodes, x0 <- x0_fast + w (x0_base - x0_fast), x0_base = Base's one-step x0 for the same chunk state with the prefix re-noised to that sigma (existing base_x0_at; fixed its t0c NameError). Fast keeps drawing on its own cache -> no seam; Base pulls the layout at the high-noise nodes.
Launched exp16 on 0,1 / 2,3: smoke, then bg1_n01, bg05_n0123, bg1_n0123, hg0.2, hg0.4 on castle 4321 / jar 9999 (policy never = guidance on every chunk, no corrections). 4-7 keep polish -> seam runs.
### 11:50 PM CDT — polish (sigma .77, 24-step schedule -> 6 low steps, no CFG) jar 4321: 1269 s; Fast chunks' flicker 28-41 vs 32-47 without polish (Base 4-5) — barely helps, 30-100 s per chunk. Dead end at this sigma; higher sigma brings back Base's planning (snap). Cancelled the remaining polish runs -> exp15 (ctx18 / noise field / seam) starts on 4-7.
Base-guided Fast smoke re-examined: inside-chunk motion 1.8 (Base 11.5) = frozen chunks, jumps 29 / 12 into chunks 1 / 2 -> "stutter". x0 blending at sigma~1 pulls toward Base's conditional mean (static) and re-anchors every chunk. The flicker metric rewarded the frozen video; add motion-vs-Base and boundary/inside ratio to the eval.
### 12:05 AM CDT (Sep 25) — Base-guided Fast full runs: castle 4321 bg1_n01 553 s, bg05_n0123 987 s — both freeze (inside-chunk motion 2 vs Base 16 from chunk ~5 on; boundary/inside up to 16x). Option 2 rejected at every w tried. Killed bg1_n0123; hg 0.2/0.4 continue on 0,1. Launched kv_probe (option B linear probe: Base K/V -> Fast K/V ridge R2 per layer) on GPU 2.
### 12:25 AM CDT — seam smoke (ctx18 + noise field + seam) is BROKEN: colour-block noise from chunk 1 on. Suspect the AR(1) noise field (rho .9: near-identical noise across frames at high sigma = OOD). Isolation smokes (nc only / seam only / ctx only) queued on 0,1 after hg. Paused the 4,5 exp15 worker (its nc/seam runs would inherit the bug); 6,7 continues al_ctx18 (D only, no new code). kv_probe OOMed on one GPU -> relaunched on 4,5 (Fast on 4, Base on 5).
### 12:45 AM CDT — K/V linear probe (option B), 6 training videos (first 4 train, last 2 held out; same scene family), chunks 2/4/6, Base low expert t=0 window forward vs Fast commit pass:
raw cosine Base vs Fast K 0.59-0.91, V 0.36-0.80 (shared lineage); ridge Base->Fast on held-out: K R2 .47-.77 (cos .78-.94), V R2 .30-.61 (cos .59-.86); layer 0 ~1.0 (patch-embed dominated).
=> a linear translator explains roughly half to three quarters of Fast's K/V from Base's; the rest needs a nonlinear translator and/or a Fast-side LoRA. Feasible; held-out is not a different scene, so treat as optimistic.
### 1:00 AM CDT (Sep 25) — D (al_ctx18) jar 9999: 2498 s; identical to always16 through c6, then boundary jumps 15-25 (vs 6-8), warm colour drift, still no camera turn. Chunk-wise Base (any conditioning) does not express the action; only the joint 161-frame generation does. => Base chunk outputs are unfit as teachers; teachers = Base 70-step rollouts (teacher forcing) and long-window Base continuations from Fast states (on-policy).
User OK: start the training pipeline (A: Fast LoRA distillation; B: K/V translator). Data: Base 70-step rollouts for 20 more StEvo tasks (seed 1234), 6 single-GPU workers on GPUs 2-7 (queue exp/queue/qdata.txt). Cancelled exp15.
### 1:50 AM CDT — isolation smokes: the AR(1) noise field (--noise-corr) is the broken piece (colour-block garbage from chunk 1); seam-only and ctx-only run fine but do not reduce boundary jumps (8.1x / 7.8x vs 5.3x baseline). All three dropped.
Fast LoRA trainer (A): model_fast wrote grad-carrying K/V into the cache in place -> graph retained across steps -> OOM. Patched model_fast.py: cache writes .detach(), attention uses cat(past cache, current chunk k/v) so k/v LoRA still gets gradients (inference values identical). Peak 64 GiB, 13 s per chunk-step (4 nodes). Step-1 losses per node: 0.82 / 0.42 / 0.22 / 0.12 (rel. MSE of x0).
Launched run1 on GPUs 0,1: 44 Base rollouts (S00/S01/S02/S05), 3 epochs, rank 32 attn+FFN, lr 5e-5.
### 2:00 AM CDT — user: "끝나면 학습이랑 알아서 싹 진행해줘" -> autonomous: run1 -> eval (Fast alone + LoRA, 6 cases) -> stevo cache -> run2 (64 videos, resume) -> on-policy data -> B translator.
### 2:25 AM CDT — run1 step 100: loss EMA 0.40 -> 0.15; per-node (0/1/2/3) e.g. 0.06/0.05/0.03/0.02 at chunk 4, 0.28/0.16/0.10/0.06 at chunk 6. ~17 s/step, ~75 min/epoch/rank -> 3 epochs finish ~5:30 AM CDT.
### 4:30 AM CDT — 20 StEvo Base-70 rollouts done (22 tasks with base70 incl. jar/castle). run1 at step 470 (ep 1), EMA 0.117.
Launched: evalfl_ck470 (Fast alone + step-470 LoRA, 6 cases) on 2,3 / 4,5; fastroll_* (Fast rollouts of the 20 training tasks, for on-policy data) on 6,7; stevo_cache (latents of the 20 rollouts) on GPU 7.
### 5:00 AM CDT — evalfl_ck470 (Fast alone + step-470 LoRA, all 4 nodes trained): metrics look great (jar flicker 9-11 vs 48-55 raw Fast; ssim .69-.72) but the videos COLLAPSE: after chunk 2 everything turns into a featureless sand/wood texture, motion 3-5 (Base 9-16), scene gone. Cause: x0-MSE at the high-noise nodes (sigma 1.0 / .94) regresses Fast's sample-like prediction to the conditional mean (blur); the AR rollout then feeds blur to itself. Same mean-seeking failure as Base-guided Fast.
=> run2_n23 (GPUs 2,3): LoRA trained only at nodes 2,3 (sigma .83/.63, where the conditional mean is sharp), 64 videos, 2 epochs. High-noise planning stays Fast's own. For nodes 0/1 a distribution-level objective (DMD-style with Base as the real score, or consistency targets) would be needed — not MSE.
### 7:20 AM CDT — run1 final (3 epochs, EMA 0.102) eval: same collapse as ck470 (flat blurry wood/sand/sea planes from chunk 3; motion 2-10). All-node x0-MSE distillation is rejected.
On-policy continuation generation started on GPUs 0,1,6,7 (20 tasks x chunks 3/6/9, Base 40-step 81-frame I2V from Fast's last prefix frame). run2_n23 (nodes 2,3 only) running on 2,3.
Objective for nodes 0/1 (to decide after run2): (a) consistency targets = Base's deterministic denoise from the SAME noise z (sample-like, no mean collapse; but chunk-wise Base planning = snap-to-frame-0 risk; ~100 s per sample, precompute), (b) DMD-style distribution matching with Base as the real score (needs a fake-score copy), (c) leave nodes 0/1 to Fast.
### 9:15 AM CDT — the Claude session was restarted (host memory pressure); all workers/child processes died with it (run2 eval never ran; on-policy stopped at 12/60 samples). Relaunched detached (setsid): evalfl_run2_n23 on 2,3 / 4,5; on-policy for the remaining 16 tasks on 0,1,6,7.
### 9:45 AM CDT — root cause of the two session deaths: host RAM OOM, not the server. Our user cgroup hit memory.peak 774 GB (host 755 GB), oom_kill 4: six Base processes (4 onpolicy_gen + 2 spec_v0) each pinning both experts (2 x 28 GB) + loading peaks. onpolicy_gen now keeps both experts on the GPU (no pinned copies). Relaunched eval (2,3 / 4,5) and on-policy (0,1,6,7).
### 11:50 AM CDT — run2_n23 (nodes 2,3 only) eval: collapse again (flat planes; jar 4321 frozen). x0-MSE at any sigma blurs stochastic detail and the AR rollout compounds it. MSE distillation of the few-step generator is dead; DMD/Self-Forcing-style distribution matching is the principled alternative (2-3 days). Proposed to the user.
On-policy data complete: 60 samples (20 tasks x chunks 3/6/9), Base 40-step 81-frame continuations from Fast's last prefix frame.
User asked about B (K/V translator): not trained yet, only the linear probe. Before training B: oracle test = Fast with a 41-latent cache (nothing evicted) on the 6 cases (never_win41, running on 2-5) — if drift persists with full memory, B cannot fix it.
### 12:40 PM CDT — full-cache oracle (never_win41): identical drift and flicker to the 18-latent window -> memory is not the cause; B (K/V translator) cannot fix drift. User's direction: a LoRA that RECEIVES Base information (speculative-decoding style). Agreed plan, stage 1 = Base-hint LoRA with deterministic pairing:
- pair_gen.py (GPUs 4-7, 4 shards, ~8 h): per (video, chunk) fixed noise z0 -> Base 16-step RePaint (cfg6+reuse, all noise seeded => deterministic) -> saves z0, hint0 (one-step x0 at sigma~1), z1/sig1 (teacher state at the first low step), target (final x0).
- hintlora_train.py: Fast LoRA (attn+FFN, active only for t >= 900 i.e. nodes 0/1) + fine-tuned patch_embedding; the hint enters through the y channel of the current chunk (mask 0.5, image-latent channels = hint0); loss = MSE(z0 -> target), MSE(z1 -> target). The target is a deterministic function of the input, so MSE cannot average across samples (the collapse mechanism of run1/run2).
- spec_v0 --hint-lora: one Base CFG forward per chunk at sigma 1 (prefix re-noised) gives the hint; nodes 2/3 are plain Fast; y restored before the commit pass.
### 2:20 PM CDT — hint-LoRA run1 ck150 (116 pairs, 40 min) eval: jar 9999 snaps at chunk 1 to a different (dark) jar and freezes; castle 4321 smears into a gradient. Bug found: the fine-tuned patch_embedding was applied to EVERY call (commit pass, nodes 2/3), contaminating the cache; the LoRA was gated but the PE was not. Fixed (GatedPE: original weights unless the hint is active) in trainer + spec_v0; run2 launched fresh on 0,1. Hint cost at inference 6-41 s per chunk (CFG, full prefix) -> 331 s per rollout; cut later (cfg 1 / window).
### 3:45 PM CDT — hint-LoRA run2 step 300 (265 pairs, EMA 0.125; node-0 losses vary 0.05-0.68 by chunk). ck300 saved; eval deferred (host RAM 617/755 GB with 4 pair_gen + 2 trainers + 2 plan runs).
New direction (user, after MilliVid): Base plans the WHOLE clip jointly at low resolution (cheap, action-following, no chunk seams); Fast renders 480p chunks conditioned on the plan through the hint channel. Running Base 16-step full-clip at areas 184320/122880/99840/61440 on jar 9999 / castle 4321 (GPUs 2,3) to check plan quality and cost.
### 4:15 PM CDT — Base 16-step whole-clip generation at low resolution (161 frames, CFG 5): time / peak GiB
area 184320 (576x304): 294-301 s / 48; 122880 (464x256): 211 s / 44; 99840 (416x224): 184 s / 42; 61440 (320x176): 142-153 s / 39. Not proportional to tokens (weight streaming dominates at small token counts) — the cheapest plan is ~150 s.
### 4:30 PM CDT — low-res whole-clip Base plans (16-step): 576x304 (area 184320, ~300 s) follows Base 70-step closely on both cases (jar: camera turns away, no jar re-appearance; castle: washed away, beach); 464x256 borderline (jar re-appears / tank-like castle); 416x224 drifts; 320x176 garbage. => plan resolution = 576x304.
Decision: stop pair_gen (chunk-hint pairs) and the chunk-hint LoRA run2 (its ck300 eval on 2,3 finishes for the record). New stage: PLAN-hint LoRA — hint = the whole-clip low-res plan (train: blurred/downsampled GT chunk latents from the 64 Base rollouts; test: Base 576x304 plan latents upsampled), Fast renders 480p chunks; nodes 0/1 LoRA + gated patch embedding as before.
### 4:20 PM CDT — chunk-hint LoRA ck300 (PE-gated fix) eval: same as ck150 — jar snaps to a different dark jar and freezes (motion 2.7, boundary 15x), castle smears. The per-chunk one-step-x0 hint (a blurry conditional mean at sigma~1) makes Fast render a static blurry version of it. Chunk-hint line closed. Caution for the plan-hint line: the plan is also blurry, but it carries motion; nodes 2/3 stay plain Fast for texture.
### 5:30 PM CDT — PLAN-HINT works. plan300 (57 min of training, blur-0.68 hints) + 576x304 Base plans, Fast renders 480p with 0 Base forwards per chunk (71-83 s render + ~300 s plan):
All 6 cases follow the plan: jar 1234/4321/9999 turn away to table+wall like Base (no houses, rooms, people); castle 4321 castle washed away -> beach + hills like the plan; castle 1234/9999 beach with tree line (plan-following). No hallucinated objects, boundary jumps 1.2-1.8x on jar (Base 1.8-2.3), 4-7x on castle.
Remaining gaps: Fast-level texture jitter (jar flicker 32-34 vs Base 14; castle 12-21 vs 15-25 OK); jar renders show more motion than Base (22 vs 8.5) — wood grain swims; to_base DINO 0.31-0.83 (plan itself 0.36-0.55).
Next: keep training plan1 (more steps), try LoRA on all 4 nodes with the hint (plan2), cut plan cost (cfg6+reuse -> ~200 s), then verifier-free pipeline = plan + render.

## 9/25 18:30 CDT — plan cost reduction + plan700 / plan2_all
- New `specwm/plan_gen.py`: whole-clip low-res Base plan (576x304, 16 UniPC steps, shift 10, guide 5, both experts resident on the GPU, no offload),
  `--cfg-steps K` (CFG only on the first K steps) and `--reuse i,j,k` (skip those step forwards, reuse the previous flow). Saves video.mp4 + plan_lat.pt + plan.json (gen_s).
- Bundles: `plang_<task>:<seed>:<tag>` (tag cr = cfg6 + reuse 9,11,13; full = plain 16-step, timing baseline w/o offload), `evalplg_<ck>:<tag>` renders with plan `plan_<tag>_s<seed>`.
- Queued qplang.txt on GPUs 2,3 / 6,7 after the plan700 eval (6 x cr + 1 x full).
- Running: plan1 (GPUs 4,5, step ~740, ema 0.019), plan2_all (GPUs 0,1, LoRA on all 4 nodes, tmin 0), evalplan_plan700 (GPUs 2,3 / 6,7).

## 9/25 19:35 CDT — plan700 eval, GATE BUG found & fixed, colour drift diagnosed
- plan700 ≈ plan300 on every metric (same plan → same render; more steps of the node-0/1 LoRA do not reduce jitter).
- BUG: fast_x0 switched the hint LoRA + fine-tuned patch embedding on for chunk 0 (no hint yet) → chunk 0 was destroyed (castle: grey frames 4-8, first boundary 48-53, DINO drift 0.77-0.87). Fixed: LoRA on only while a hint is written (HL["hint"]).
  plf_plan700_b16_st1 (fixed): castle drift(max) 0.23-0.39 (was 0.77-0.87), first boundary 33-47 (chunk 0 = plain Fast look → plan look jump).
  --start 0 (hint chunk 0 too, keeping frame 0's image conditioning): castle first boundary 23-28, maxboundary 1.9-3.7x (st1 2.8-5.9x); chunk-0 motion 18-21 (jittery, un-trained case). Jar: 31→13 first boundary.
- --no-base 1: single-GPU render (Fast only, 71-74 s) — plan+render runs on ONE GPU; six renders now take 2 GPUs x 8 min.
- plan_cr (cfg6 + reuse 9,11,13, experts resident): 133-135 s (vs 290 s offloaded 16-step); same content/colour as base16 (motion 6.6 vs 9.5, lum/sat tracks). => plan+render ≈ 210 s on one GPU.
- COLOUR DRIFT: renders go dark+saturated within the first chunk (castle 4321 sat 50→90-110, lum 130→108; plan stays sat 55-60 = Base). Plain Fast doesn't (sat 50 until chunk 5). => hint train/test mismatch: train hint = latent-space down/up (plan_blur 0.68), test hint = VAE(cubic-upscaled plan video).
  Fix in flight: specwm/hint_cache.py builds matched hints for the 64 cached rollouts (decode → 576x304 → hint_lo = VAE at low res; hint_pix = VAE of the cubic-upscaled video); hintlora_train --hint-src pix|lo; spec_v0 --plan-hint-mode lo (latent upsample) as a test-time probe with plan700.
- Running: plan2_ck175 (all-node LoRA) st0 renders on GPU 3; plan700 lo-mode renders on GPU 7; plan700 renders on the cr plans on GPUs 2/6; hint_cache on GPU 7.
### 9/25 20:05 CDT — colour drift: cause narrowed to the un-LoRA'd nodes seeing the hint
- plan2_ck175 (LoRA on all 4 nodes, tmin 0, 175 steps) st0: colour drift GONE (castle sat 53-60 = plan/Base; jar 66-91 vs Base 80-105), castle drift(max) 0.16-0.33, maxboundary 1.5-3.1x — but jitter unchanged/worse (jar motion 22-24, flicker 37-39; castle flicker 25 vs Base 15-25) and frame 0 brightened (lum 144-149 vs image 125: nodes 2/3 LoRA + chunk-0 hint alters frame 0).
- lo-mode hint (latent upsample of the low-res plan latents) with plan700: jar motion 17-19 (pix-mode 22), flicker 27-30 (34-37), colour drift unchanged (sat 100+). Latent upsample = closer to the plan_blur training hint → less jitter.
- Interpretation: with the node-0/1 LoRA, nodes 2/3 (plain Fast) still receive the hint y (mask 0.5 + plan latents) they were never trained on → saturation/darkening. Test: --hint-low plain (plain nodes see the original y) with plan700, st0 → plh_ runs on GPUs 3/7. Also plan2_ck175 st1 (frame-0 check) on GPUs 2/6.
- plan1 stopped at step 1188 (ema 0.020, checkpoint plan1175); plan3 = resume plan1175 with pixel-matched hints (hint_cache.py, --hint-src pix) on GPUs 4,5; plan2_all continues on 0,1.
### 9/25 21:05 CDT — --hint-low plain FIXES the colour drift and halves the jitter (no retraining)
plh_plan700_b16_st0 (plan700 LoRA on nodes 0/1 sees the plan hint; nodes 2/3 = plain Fast with the ORIGINAL y; chunk 0 hinted):
  jar  motion 12.3/17.7/14.5 (plf 22-23; Base 8.4/2.5/9.2), flicker 21/30/25 (plf 34-37; Base 13/4/14), maxboundary 1.6-1.8x, drift 0.40-0.43, sat 77-88 (= Base)
  castle motion 11.2/9.5/6.1 (Base 9.6/15.4/12.9), flicker 20/18/13 (Base 15/25/21), maxboundary 1.8/2.1/3.1x, drift 0.22-0.39, sat 50-63 (= plan)
=> best render so far. plan2_ck175 st1: colour OK but jitter unchanged (jar 22-23 / 37) → all-node LoRA not needed; the hint must simply be hidden from the plain nodes.
Deployable pipeline = plan_cr (133 s) + plh render (73 s) ≈ 206 s on ONE GPU. Rendering it now (GPUs 2,3); plan3_ck325 (pix-matched hints, ema 0.0066) with hint-low plain on GPUs 6,7.
### 9/25 21:45 CDT — v2 pipeline measured; plan3 (pix-matched hints) is the best LoRA
plh_plan700_cr_st0 (plan_cr 133 s + render 73 s = 206 s, one GPU): jar motion 11.5/14.9/12.8 flicker 21/26/22, castle 10.9/8.9/5.2 flicker 19/18/12; maxboundary 1.6-3.4x; colour = Base. to_base jar1234 0.60 (b16 plan: 0.34) — the cr plan differs from Base more on that case.
plh_plan3_ck325_b16_st0 (pix-matched hints, resumed from plan1175, 325 steps): jar motion 10.8/14.0/9.9 flicker 18/24/17 (Base 13/4/14), castle 10.1/7.8/5.3 flicker 19/16/11 (Base 15/25/21); maxboundary 1.6-2.9x; drift 0.31-0.40; lum/sat track Base. => best render; matched hints reduce jitter further.
Next: plan3 + cr plans (GPUs 2,3) = candidate final v2; plan3 keeps training (eval a later ckpt); lineup site rebuilt with pl2/plancr.
### 9/25 22:00 CDT — v2 FINAL candidate: plan_cr (133 s) + plan3_ck325 render with --hint-low plain --start 0 (73 s) = 206 s, one GPU
  jar    motion 9.4/10.5/8.0 (Base 8.4/2.5/9.2), flicker 16.5/20.1/16.6 (Base 13.4/4.3/13.9), maxboundary 1.8/2.2/2.4x, drift 0.45/0.30/0.77 (jar 9999 one chunk), to_base 0.39/0.80/0.48
  castle motion 10.4/8.3/5.3 (Base 9.6/15.4/12.9), flicker 18.2/16.2/11.0 (Base 15.1/25.1/20.6), maxboundary 2.3/2.6/3.1x, drift 0.35/0.32/0.41, to_base 0.43/0.56/0.70
  Lineup site: pl2 = this run, plancr = its plan. plan3 continues (ck525 eval with cr queued on 6,7); plan2_all continues (colour fixed by hint-low instead; keep for the record).
### 9/25 22:00 CDT — plan3 ck525 ≈ ck325 (jar motion 8.9/11.7/9.4, flicker 16/21/17; castle 10.4/8.3/5.1) → matched-hint LoRA has plateaued (ema 0.006). plan3_ck325 stays the reference.
plan2_all stopped (colour drift is solved by --hint-low plain; all-node LoRA gave no jitter gain) → GPUs 0,1 released.
Next: plan4 = resume plan3_ck525 and ALSO train chunk 0 with the hint (frame 0 = image, frames 1-2 = plan), the one untrained situation left (chunk-0 motion 18-21, castle first boundary 2-3x).
### 9/25 23:15 CDT — plan4_ck200 (chunk 0 trained with the hint, resumed from plan3_ck525) ≈ plan3: jar motion 8.7/11.5/8.3 flicker 15.5/21.4/16.6, castle 10.2/8.3/5.1; chunk-0 motion still 13-18 (Base ~8-9) — hinting chunk 0 costs a little motion there but keeps the first boundary small (9-22). Render side has plateaued.
Remaining gap = plan vs Base-70 sampling differences (jar 4321: plan pans away, Base static; jar 9999: plan brings the jar back, Base shows a teal pot). Test: 24-step plan (cfg 8 + reuse 14,17,20, tag c24) → does to_base/drift improve, at what cost? Queued on GPUs 2,3 with plan3_ck325 renders.
### 9/25 23:50 CDT — 24-step plan (c24: 24 UniPC, CFG on 8, reuse 14,17,20): 197-202 s (+65 s over cr)
Render (plan3_ck325, hint-low plain, st0) to_base: jar 0.29/0.54/0.37 (cr 0.39/0.80/0.48), castle 0.50/0.59/0.73 (cr 0.43/0.56/0.70). The render's to_base tracks its plan's within ~0.03 on every case → the plan is the only knob left for Base-likeness; the render is faithful.
Jitter unchanged (jar motion 9.4/9.8/8.2, castle 11.5/15.2/6.3 — castle 4321 follows a livelier plan, motion 12.7).
Options: v2-fast = cr plan + render = 206 s; v2-close = c24 plan + render = 271 s (closer to Base on jar 4321, otherwise equal). Both within the 200-400 s target on ONE GPU.
Plan4 (chunk-0 trained) keeps training on 6,7 (epoch 2) for the record; plan3 finished 3 epochs.
### 9/26 00:10 CDT — plan4 final (ck806, 2 epochs with chunk 0): jar motion 9.1/11.9/9.3 flicker 16.5/21.5/17.2, castle 10.3/8.1/5.0 — same as plan3_ck325/plan4_ck200. Render LoRA line is saturated; all GPUs 0-7 free.

## 9/26 01:30 CDT — DATA GENERATION for the next training round (user: "데이터 생성 해보자"; professor's meeting: low-res/soft transfer + Base→Fast KV translator)
specwm/plandata_gen.py: per training video, (1) plan = Base 576x304 16-step cfg6+reuse (the test-time plan, ~180 s here — the idle expert is parked on the host, swaps cost), (2) target = Base 480p refinement of the upscaled plan: VAE(cubic-up plan) noised to sigma s_start, low expert denoises the rest of a 30-step shift-10 schedule with CFG.
Sweep on the held-out jar 1234 (refine s / sharpness (Base 4.9, plan 6.1 at low res) / to_base / flicker (Base 13.4)):
  s0.6: 3 steps, 126 s, sharp 3.9, to_base 0.32, flicker 11.2 | s0.75: 6 steps, 241 s, 4.0, 0.31, 11.4 | s0.9: 14 steps, 553 s, 4.3, 0.26, 12.8
=> s0.9 chosen (closest to Base in detail and DINO; still a little softer than Base 70). All targets follow the plan, flicker at Base level.
Launched: 64 videos, 6 shards on GPUs 2-7 (~12 min/video → ~2.2 h). Output: specwm_out/plandata/{cache,hints,video}; train with hintlora_train --cache plandata/cache --hints plandata/hints --hint-src pix.
Cheaper refine config to test later: 20-step schedule from 0.9 (~9 steps) with CFG on 4 → ~290 s.
### 9/26 04:10 CDT — plandata complete: 64/64 pairs (plan ~180 s + refine ~550 s each; 6 GPUs, 2.6 h). specwm_out/plandata/{cache,hints,video}.
plan5 launched (GPUs 4,5): resume plan4_ck806, --cache plandata/cache --hints plandata/hints --hint-src pix --chunk0 1, 3 epochs. Eval at ck200 with cr plans (auto-queued on 2,3).
### 9/26 05:10 CDT — plan5_ck200 (real plans → refined targets): smoother but SOFTER
  jar motion 8.4/9.2/7.3 (plan3 9.4/10.5/8.0), flicker 15/18/16 (16.5/20/16.6), to_base 0.34/0.80/0.47 (≈), castle to_base 0.39/0.53/0.67 (0.43/0.56/0.70), ssim4 up.
  Sharpness (Laplacian, Base 4.9 / 10.4 / 4.2): plan3 5.3 / 8.5 / 4.8, plan5 4.1 / 6.7 / 3.8 → the refined targets (sharp 4.3 < Base 4.9) taught softness; crops show blurred wood grain.
  Fix in flight: plan6 = mixed data (64 old sharp-GT pairs + 64 new plan/refined pairs, specwm_out/mixdata symlinks) on GPUs 6,7; plan5 continues to ck500 (auto eval). Sharpness is now part of the eval.
### 9/26 06:05 CDT — plan5_ck500: same as ck200 (jar motion 8.2/9.4/6.6, flicker 15/19/16; sharp 4.0/5.8/3.7 vs Base 4.9/15.9/4.2; castle sharp 6.9/6.8/4.9 vs Base 7.8/10.3/8.5). Soft targets → soft renders; the plan5 line is closed unless targets get sharper (60-step refine from 0.9 ≈ 18 min/video, or high-frequency-weighted loss).
scripts/quick_eval.sh RUN GPU = motion/temporal/sharpness on the six cases. Waiting for plan6 (mixed data) ck400.
### 9/26 06:50 CDT — plan6_ck400 (mixed: 64 sharp-GT pairs + 64 real-plan/refined pairs): sharpness back to Base level on jar (4.7/6.8/4.3 vs Base 4.9/15.9/4.2), castle 7.7/7.3/5.1 (Base 7.8/10.3/8.5); motion/flicker/to_base ≈ plan3 (jar 8.7/10.7/8.3, to_base 0.39/0.78/0.34; castle 0.48/0.56/0.69).
=> mixed data fixes plan5's softness; net effect vs plan3 is small. The node-0/1 y-channel LoRA is saturated regardless of data. Remaining sharpness gap only on the high-detail castle cases (splash, sand grain) = Fast texture capacity at low noise.

## 9/26 08:20 CDT — two training tracks started (user: "KV 변환기, on policy 둘 다 해보자")
On-policy (texture LoRA, DAgger with the plan):
- specwm/plan_rollout.py: Fast + plan6_ck400 LoRA rolled out on the 64 training videos (plan hints, hint-low plain, chunk 0 hinted) → specwm_out/onpol2/rollout/<name>.pt (lat, per-node z and x0). Done (64).
- specwm/onpolicy2_gen.py: for chunks 2,5,8,11 Base low expert RePaints chunk c given FAST'S prefix (5-chunk window, prefix re-noised each step) starting from the plan hint at sigma 0.9 (30-step schedule) → onpol2/samples/<name>_c<c>.pt (prefix, target, hint, z, x0n). Running on GPUs 2-5.
- next: texture LoRA on nodes 2/3 (gate t < 900) trained from the rollout's node-2 input z toward the on-policy target; inference flag --tex-lora.
KV translator (professor's idea):
- wan/modules/model_fast.py: WanSelfAttention.mem = (k, v) prepended to the KV cache (plan memory).
- specwm/plan_kv.py: capture Base low expert's post-norm pre-RoPE K/V at blocks 0,8,16,24,32 on the low-res plan (t=0), every 2nd latent frame (14.7k tokens/layer); Translator = per-block 5120x5120 linears (K identity init, V zero init); Fast RoPE at plan positions mapped to Fast's grid.
- specwm/kvlora_train.py: joint LoRA (nodes 0/1) + translator, teacher forcing on mixdata (124 videos), Fast on cuda:2r, Base low on cuda:2r+1; 7 s/step, peak 73 GB. Smoke OK (identity-V init started at loss 0.39 → switched V to zero init).
- kv1 launched on GPUs 0,1 (1 rank, no y hint, 2 epochs ≈ 6 h; eval at ~step 400). plan5 stopped (soft); plan6 final eval queued on 6,7 after its training ends.
### 9/26 09:20 CDT — plan6 final (ck1612, 2 epochs of mixed data) went soft again: sharp jar 4.1/5.7/3.7, castle 6.7/6.8/4.7 (ck400: 4.7/6.8/4.3, 7.7/7.3/5.1); motion/flicker/to_base unchanged. Longer training on the mixed set converges to the soft refined targets → plan6_ck400 stays the reference hint LoRA (early stop). The texture question moves to tex1 (on-policy) and kv1.
### 9/26 12:10 CDT — both tracks training
- tex1 (texture LoRA nodes 2/3, on-policy targets, 248 samples, GPUs 2,3): step-1 loss 0.23; 16 s/step; ck100 → evaltex (plan6_ck400 + tex1) queued on 6,7.
- kv1 (KV translator + LoRA nodes 0/1, no y hint, mixdata, GPUs 0,1): fixed a devB memory blow-up (expandable segments + cache release; devB steady 35 GB, dev 65 GB); loss ema 0.38 → 0.28 by step 46 (plain-Fast teacher-forcing level is ~0.38; the y-hint LoRA reached 0.04-0.06 on the same data). 9.3 s/step, 4.2 h/epoch. ck400 → evalkv on 4,5 (spec_v0 --kv-mem --plan-lat; Base low expert on dev1 reads the plan; run name plk_*).
- plan6_ck400 = reference hint LoRA (later checkpoints go soft).
### 9/26 13:40 CDT — tex1_ck100 (texture LoRA, nodes 2/3, MSE toward Base RePaint targets from the PLAN hint) FREEZES the video: motion 1.3-2.8, flicker 3-5, boundary 4-6.6x, to_base 0.6-0.87, sharp below Base. Same mean-seeking collapse as every low-noise regression toward a *different sample*. tex1 stopped.
Fix: DAgger labelling on the SAME state — Base denoises Fast's actual node-2 input z2 (sigma 0.83, prefix RePaint) → the target is the expert's answer for the learner's exact input (low-variance mapping, MSE is sound). onpolicy2_gen --init-node 2 → onpol2/samples_z2 (GPUs 2,3,6,7), then tex2.
kv1: ema 0.165 at step 120 (pure KV route learning; y-hint route was 0.05). ck400 eval queued on 4,5.
### 9/26 14:40 CDT — KV route works end to end (spec_v0 --kv-mem --plan-lat --resident; render 66-68 s + one Base low-res pass). kv1_ck400 (loss ema ~0.17 then):
  jar motion 4.6/5.0/6.6 (Base 8.4/2.5/9.2), flicker 8.9/10.7/11.3 (Base 13/4/14), to_base 0.48/0.79/0.49, sharp 5.4/6.9/4.5 (Base 4.9/15.9/4.2)
  castle motion 6.1/4.9/3.4 (Base 9.6/15.4/12.9), flicker 10/9/7, to_base 0.54/0.67/0.75, sharp 5.7/5.5/4.0 (Base 7.8/10.3/8.5); maxboundary 1.9-3.4x; no Base call per chunk, no y hint.
  => smoother/less motion than Base and than the y-route (plan6_ck400: jar 8.7/10.7/8.3), to_base slightly worse. Early checkpoint; kv1 at step 875 has ema 0.087 → ck875 eval queued.
### 9/26 15:30 CDT — kv1_ck875 (ema 0.087): jar motion 5.8/6.6/7.7 flicker 11/12/13, to_base 0.44/0.77/0.52 (y-route plan6_ck400: 0.39/0.78/0.34); castle motion 6.6/5.4/3.8, to_base 0.53/0.64/0.72 (y-route 0.48/0.56/0.69); sharp jar 5.8/7.4/4.9 (Base 4.9/-/4.2), castle 6.5/5.5/4.4 (Base 7.8/10.3/8.5); maxboundary 1.7-3.2x.
  Improving with training (ck400 → ck875 on every axis); still behind the y-route on castle fidelity and detail. Contact sheet at ck400: layout follows the plan, colours drifted pink; ck875 colour check below. Next eval at ck1600 (end of epoch 1).
### 9/26 16:20 CDT — kv1_ck875 with --kv-low plain (memory only while the KV LoRA is on, plain commit): castle to_base 0.39/0.57/0.71 (= y-route 0.48/0.56/0.69, better on 1234), drift 0.17-0.41, maxboundary 1.3-2.7x (best so far), colour drift halved (sat 49→60 vs 46→85) but jitter at the plan700 level (jar motion 11.6-14, flicker 19-25) since nodes 2/3 are plain Fast.
  With the memory at all nodes ("mem"): smooth (motion 5-8) but saturation drifts (nodes 2/3 never trained with the memory). => kv2: resume kv1_ck875, LoRA on all 4 nodes (tmin 0) with the memory, GPUs 4,5; eval ck400 on 6,7.
  tex2 (same-state DAgger targets) training on 2,3: step-1 loss 0.177 (tex1 0.226); ck100 eval queued on 6,7.
### 9/26 17:10 CDT — tex2_ck100 (same-state targets, but re-noised with FRESH noise at sigma 0.81) also freezes (jar 1234 motion 1.9, flicker 4.9). Cause: the learner's input noise ≠ the noise the target was denoised from → stochastic map → MSE mean → static. Deterministic pairing needed (as with pair_gen): store the exact init latent z_init and train on it. onpolicy2_gen now saves z_init; regenerating chunks 5,11 (128 samples) on GPUs 2,3 → tex3 (4 epochs) → ck100 eval on GPU 6.
kv1_ck1600 renders relaunched on 6,7 (the first attempt collided with kv2 on 4,5).
### 9/26 18:15 CDT — kv1_ck1600 (end of epoch 1, memory at all nodes) jar: motion 4.8/4.6/6.8, flicker 10/11/12, to_base 0.47/0.76/0.51, sharp 5.1/6.2/4.5 = ck875 → the node-0/1 KV route has plateaued (jar); castle renders pending (GPU 6,7 collisions between eval workers — now serialized in one queue).
kv2 (LoRA all nodes with memory, resumed from kv1_ck875): step 515, ema 0.071 (node losses 0.075/0.061/0.045/0.028). ck400 eval in the same queue.
### 9/26 19:45 CDT — kv2_ck400 (LoRA on ALL nodes with the memory, MSE): blurry and near-static (sharp 2.8-3.9 vs Base 4-16, motion 2.7-4.9, boundary up to 4.6x) → stopped.
LESSON (now seen 5 times: plan2, kv2, tex1, tex2, chunk-hint): any MSE/x0-regression training of Fast's LOW-noise nodes (2/3) blurs or freezes the output — at sigma ≤ 0.83 the conditional mean of x0 has no high-frequency detail, and the distilled few-step model must commit to details deterministically (LingBot trained it with distribution matching). Only the high-noise nodes (0/1, layout) can be trained with MSE. => keep nodes 2/3 as plain Fast in every route; a texture improvement needs a distribution-matching loss (DMD/adversarial), not regression.
kv1_ck1600 with memory at all nodes: castle to_base 0.59/0.66/0.77, sharp 5.4-5.5, motion 3-5 (too smooth). Evaluating kv1_ck1600 --kv-low plain (memory only at nodes 0/1) as the KV route's proper configuration.
### 9/26 20:20 CDT — KV route, proper config (kv1_ck1600, memory only at nodes 0/1, plain nodes 2/3): jitter at Base level (jar motion 7.7/7.9/7.0 vs Base 8.4/2.5/9.2, flicker 14.5/16.9/15.2; castle 9.1/6.9/4.9, flicker 16/14/11), to_base castle 0.53/0.54/0.64 (y-route 0.48/0.56/0.69), jar 0.44/0.77/0.47 (y-route 0.39/0.78/0.34), maxboundary 2.1-3.5x, sharp slightly under Base (jar 4.1/5.6/3.8). Colour: castle tracks the plan; jar 1234 DESATURATES (83→37, plan 70) — the memory alone carries layout better than colour.
=> KV route ≈ y-route, complementary failure modes → kv3 = y hint + KV memory (resume kv1_ck1600 translator + plan6_ck400 LoRA, --y-hint 1) on GPUs 4,5; eval bundle evalkvy (spec_v0 --hint-lora + --kv-mem on the same ckpt, run plky_*).
### 9/26 21:00 CDT — tex3_ck100 (texture LoRA, deterministically paired same-state targets; loss ema 0.06 vs 0.2 before): SHARPNESS reaches/exceeds Base (jar 5.2/7.1/4.7 vs Base 4.9/15.9/4.2; castle 8.4/8.2/5.4 vs 7.8/10.3/8.5) — first texture gain — but motion inside chunks drops (2.5-4.2 vs Base 8.5) and boundaries jump 3.3-5.8x; to_base castle 0.39/0.73/0.80. Checking whether the targets themselves are low-motion (RePaint chunk given the prefix) or the LoRA freezes texture.
  Target check (40 samples, latent space): inside-chunk motion target 0.33 vs Fast rollout 0.37 vs refined whole-clip target 0.36; boundary jump identical (0.336). => the paired targets are NOT low-motion; the texture LoRA (100 steps) reproduces spatial detail but flattens the chunk temporally. Waiting for the final tex3 checkpoint (4 epochs) before judging; if it stays flat, regression at low noise is closed for good and texture needs a distribution-matching loss.
### 9/26 22:30 CDT — kv3_ck400 (y hint + KV memory; LoRA from plan6_ck400, translator from kv1_ck1600; hint-low plain, kv-low plain): jar motion 8.5/7.1/8.0 (Base 8.4/2.5/9.2), flicker 15/15/15 (y-route 15.6/20.7/17.6), to_base 0.46/0.78/0.53 (y-route 0.39/0.78/0.34); castle motion 8.8/7.5/4.6, to_base 0.45/0.55/0.69 (y-route 0.48/0.56/0.69), drift 0.21-0.47, maxboundary 1.7-3.2x (y-route 2.0-3.5x); sharp 4.2/5.5/3.5 (y-route 4.7/6.8/4.3).
  => smoother jar (flicker -25% on two seeds) and slightly tighter boundaries; slightly softer; Base-likeness equal. Marginal gain, not a step change. kv3 continues (ck1200 eval queued).
  Colour (sat): kv3 jar 1234 85→56 (plan 70, y-route 73), jar 9999 follows the plan's fall; castle OK. The KV memory carries a desaturation bias (kv1 gated: 83→37) that the y hint only partly cancels.
### 9/26 23:10 CDT — tex3 final (ck248, 4 epochs, ema 0.039): identical to ck100 — sharp (jar 5.5/7.2/4.9, castle 8.4/8.5/5.5 ≥ Base on 4 of 6) but temporally flat (motion 2.2-4.1, boundaries 3.4-5.2x, to_base castle 0.45/0.74/0.81). Regression at nodes 2/3 is closed even with deterministic pairing: it learns spatial detail, not temporal structure. Last cheap probe: texture LoRA at 0.5 / 0.25 strength (--tex-scale) — does partial strength keep some sharpness without freezing?
kv1 finished (2 epochs, ema 0.066). kv3 at step ~700 (ema 0.034), ck1200 eval queued.
### 9/26 23:40 CDT — kv1 final (ck3224, gated memory): to_base jar 0.42/0.78/0.34, castle 0.51/0.56/0.65 (= or better than the y-route) but softer (sharp jar 3.5/4.8/3.1, castle 4.4/4.5/3.2) and slower (motion 5-6.6): the second epoch over-smooths, like plan6's. Sweet spot for the KV route = kv1_ck1600 gated. Early stopping is the rule for every LoRA here (plan6_ck400, kv1_ck1600).
### 9/27 00:05 CDT — kv3_ck1200 (y hint + memory): to_base jar 0.43/0.74/0.51, castle 0.55/0.58/0.65; motion 6.4-8.4; sharp 3.3-4.6 (soft) → over-trained past ck400 (sharp 4.2/5.5/3.5). kv3 stopped; combined-route reference = kv3_ck400.
SUMMARY of the two tracks: KV translator works (layout/motion via plan memory, ≈ y-route; colour weaker; best gated at nodes 0/1); combined y+KV = smoothest jar and tightest boundaries, same Base-likeness. On-policy texture LoRA: deterministic pairing gives Base-level sharpness but flattens time; regression at low-noise nodes is closed. Early stopping (~400 steps after a resume) is the rule.

## 9/27 01:40 CDT — plan speed track (user "ㄱㄱ"): public 4-step LoRAs as training-free planners (576x304, no CFG, shift 5)
- lightx2v 1022 4-step (l4): 41 s/plan (cr 133 s). Plan to_base jar 0.22/0.38/0.52 (cr 0.31/0.80/0.43), castle 0.43/0.55/0.63 (cr 0.43/0.58/0.71); motion 9.5-15 (closer to Base's than cr's 7.5-8.9). Renders (plan6_ck400): jar to_base 0.31/0.49/0.41 (cr 0.39/0.78/0.34), castle 0.35/0.58/0.53 (cr 0.48/0.56/0.69); sharpness at Base level (castle 7.6/9.1/6.1). BUT by eye: castle 4321 l4 plan turns the castle into rocky ruins and the render adds towers + a returning wave — hallucination the DINO metric does not penalise. Jar 1234 l4 follows Base (table edge + wall) better than cr.
- Seko 4-step (sk4): jittery plans (flicker 26-38) → renders jitter (18-35). Rejected. l4 6-step (l6): between.
- Verdict: the public 4-step LoRA is a 3x cheaper planner with Base-like motion but it hallucinates content on the castle — not shippable as is. Next (training): adapt the merged 4-step planner to our plan distribution — LoRA on both Base experts at low res, MSE at the 4 schedule nodes toward the 16-step cr plans (64 plandata plans as targets), i.e. domain-adapt the few-step planner ("planlora").
- related_work.md draft written (paper/related_work.md).
### 9/27 03:00 CDT — planner adaptation (planlora1): LoRA rank 64 on the LOW expert (public 1022 4-step LoRA merged first), x0-regression at schedule nodes 1,2,3 (sigma 0.94/0.83/0.63; node 0 = pure noise skipped: MSE there would average the plan's motion away) toward 124 low-res reference clips (64 cr plans + 60 Base-70 rollouts downscaled). 2 ranks on GPUs 0,1, ~60 s/clip, 3 epochs. Smoke: node losses 0.49-0.75 / 0.10-0.27 / 0.04-0.13 / 0.02-0.07 at nodes 0-3.
  plan_gen --planlora <plan_lora.pt>; bundle tag pa_<ck>. ck60 → plans + scoring + renders queued on GPUs 2-4.
### 9/27 04:30 CDT — planlora1 crashed at step 13 (rank-1 OOM on a 29.8k-token clip); ck10 evaluated anyway: plans smoother (motion 5-9 vs l4 9.5-15) and LESS Base-like than l4 (jar to_base 0.41/0.65/0.54 vs l4 0.22/0.38/0.52; castle 0.50/0.72/0.68 vs 0.43/0.55/0.63) — after 10 steps it just drifts toward the cr-plan look. Relaunched from ck10 (skip >28.8k-token clips, expandable segments); ck70 eval queued. If the adapted planner does not beat l4 by ck70, the adaptation is not worth it: the reference plans (cr) are themselves less Base-like than the public 4-step plans on the jar.
### 9/27 06:40 CDT — planlora1_ck70: plans to_base jar 0.49/0.71/0.55, castle 0.55/0.75/0.75 (l4: 0.22/0.38/0.52, 0.43/0.55/0.63; cr: 0.31/0.80/0.43, 0.43/0.58/0.71), motion 5-8 (l4 9.5-15, Base 8.4-15); renders jar 0.33/0.69/0.38, castle 0.41/0.63/0.62 (l4 renders 0.31/0.49/0.41, 0.35/0.58/0.53). => MSE adaptation toward our 16-step plans makes the 4-step planner smoother and LESS Base-like on every case; stopped. The public 4-step LoRA (l4, 41 s) stays the best cheap planner; cr (133 s) the safe one.
PLAN-SPEED TRACK SUMMARY: v2-fast = l4 plan 41 s + render 73 s ≈ 115 s (Base-like motion, best jar fidelity, hallucination risk on castle 4321); v2-safe = cr plan 133 s + render 73 s ≈ 206 s. A planner that is both cheap and hallucination-free needs distribution-matching distillation, not regression.

## 9/27 16:30 CDT — follow-ups on GPUs 0-3 (user: "4개써서 후속 해보자")
- Hallucination-mitigated fast planner: l4c = public 4-step LoRA at scale 0.8, 6 steps, CFG 3 on the first 2 steps (66 s); l8c = scale 0.7, 8 steps, CFG on 3 (85 s).
  BY EYE (castle 4321): l4 plan → rocky ruins, render → towers + returning wave (hallucination); l4c and l8c plans → castle washed away, beach + hills (the cr/Base-like outcome), renders clean. DINO to_base does not see this (l4c plan 0.63 vs l4 0.55) — metric blind spot.
  Metrics: l4c plans jar 0.37/0.56/0.54, castle 0.39/0.63/0.74; renders jar 0.33/0.63/0.48, castle 0.40/0.60/0.71; motion 7-10 (Base-like); sharp jar 4.1/4.0/5.5.
  => v2-fast should use l4c: 66 + 73 ≈ 140 s, no hallucination on the known-bad seed. l8c adds 20 s for no visible gain.
- Windowed (streaming) plans: plan_gen --window 81 had an infinite loop on the 1-frame tail (fixed: stop at frames-1, min window 5) — the first attempt burned 4 h on GPUs 0-3. Windows take 55-60 s each (cr) → rerunning crw/l4w on GPU 0 with renders.
- Five StEvo held-out prompts page: https://claude.ai/artifact/C5hm8PX8eEGMqZcCs8jHmV ; five easy demo-scene prompts (easy_* tasks appended to tasks_camera.json, prompted Base bundle data70p_) running on GPUs 4-7.

## 9/27 evening — results of the follow-ups + easy scenes
- Streaming plans (2 x 81-frame windows, second window = I2V from the first's last low-res frame with pose offset): cr 123 s (68 + 55, no cheaper than 133 s), l4 36-39 s. Plans lose the whole-clip context: jar to_base 0.70/0.05/0.69, castle 0.60/0.73/0.31 (cr: 0.31/0.80/0.43, 0.43/0.58/0.71), and the renders on crw are extremely noisy at the seam (sharp 17-19 vs Base 4.9). => windowed planning in this form is worse; whole-clip plans stay. Only value: first frame ~30 s earlier.
- Easy demo scenes (5, seed 1234; page https://claude.ai/artifact/DXCqTSUQhzm3gWdo1BbjjK): Fast alone is much better here (to_base 0.20-0.34) than on StEvo (0.39-0.71) — these are LingBot's home turf. v2-safe 0.13-0.23, v2-fast 0.10-0.22, v2-fast-cfg similar; v2 wins 4/5, Fast alone wins lake (0.20 vs 0.23). Lake: every plan loses the tree that Base keeps in frame (object permanence at low res). Stonehenge: v2-safe follows Base's pan-to-sunset-and-back. Dragon: Fast alone plausible, v2-fast-cfg odd (low over grass).
- kv4 (all 40 blocks, time-local memory current+next chunk, per-head+low-rank translator): smoke peaks 84-90 GB → OOM; with frame stride 2 in the window + bf16 LoRA maths → CUDA illegal memory access (debugging).
- plandata2 (100 new clips: 40 new scenes x {R s1234, L s4321} + 20 x forward s777) generating on GPUs 0-3 (~12 min/clip → ~5 h).
### 9/27 22:30 CDT — kv4 launched (GPUs 4-7, 2 ranks): memory on 20 blocks (even), time-local window = current + next plan chunk (6 latent frames, ~4.1k tokens), Translator2 (per-head linear + rank-256 mixing, K identity / V zero init, on the Base GPU; bf16 maths), LoRA nodes 0/1, mixdata 124 clips, 2 epochs. Smoke: 8-9 s/step, peak 80 GB. 40 blocks did not fit (88-90 GB) and an async cross-device race gave illegal memory accesses at clip boundaries until explicit synchronize() calls were added.
  Eval plan: ck400 and ck1600 with --kv-low plain (gated to nodes 0/1) on GPUs 0,1 after plandata2 finishes.
### 9/28 01:00 CDT — plandata2 complete: 100 clips (40 new StEvo scenes x {right-pan s1234, left-pan s4321} + 20 x forward s777), specwm_out/plandata2; mixdata2 = 224 clips (symlinks, suffix _p2).
kv4 (mixdata, 124 clips) on GPUs 4-7 (2 ranks) — third launch after two multi-GPU NCCL fixes (device_id binding; translator collectives via the bound GPU). kv5 = same config on mixdata2 (224 clips), 1 rank on GPUs 0,1. GPUs 2,3 reserved for the ck400 evaluations (gated memory, cr plans). Extra held-out scenes for the expanded eval set: block_on_ramp, clay_drying, tea_dilution, wine_color_fabric, yoyo_drop (Base-70 not yet generated).

## 9/28 02:30 CDT — overnight: professor's action items
- Item 6 (one-step check cost): Base low expert, one CFG-free forward at 480p: 12-latent window (9 prefix + chunk, 18.4k tokens) 3.74 s, peak 44 GB; 21-latent window (32.3k tokens) 7.53 s, peak 50 GB (≈ linear in tokens). Batched forwards (B noised copies in one call) FAIL in the camera-conditioned Base forward (plucker embedding path expands batch 1 only) → parallel checks = separate forwards; the cheap way to check several noise levels is the low-res plan (576x304, 684 tokens/frame: 5x fewer tokens).
- Item 4 (protocol): Base-vs-Base DINO distance across seeds is 0.29-0.75 (mean over chunks; max-chunk up to 0.99) → 'distance to the same-seed Base' cannot define drift. New criterion (scripts/drift_rate2.py): nearest-neighbour distance to the Base batch (8 seeds) per chunk, threshold = 95th pct of Base leave-one-out. Running 8 seeds x 2 prompts: base70s_ bundle (new seeds 11,22,33,44,55), fastonly, plang cr/l4c, renderpl, specstep.
- Items 2/3 (speculative sampling): spec_v0 --specstep implemented (per-position Gaussian acceptance at nodes 0,1,2, residual resampling). First smoke crashed for a trivial reason (SPECWM_NO_FLASH not set outside the queue worker); rerun queued on GPUs 2,3.
### 9/28 04:35 CDT — speculative step made EXACT (reflection maximal coupling)
- Replaced the residual rejection loop (≤4 tries then fall back to a target sample = biased when tries run out) with the reflection maximal coupling per latent position: draft z = muD + s·eps kept w.p. min(1, N(z;muT,s²)/N(z;muD,s²)); else z' = muT + s·(eps − 2(e·eps)e), e = (muD−muT)/|muD−muT|. Output is exactly N(muT, s² I) and the acceptance equals 1 − TV (maximal).
- CPU check (16-dim, 200k positions): |d| = 0 / 0.34 / 1.12 / 3.35 → accept 1.000 / 0.865 / 0.575 / 0.093 vs theory 1.000 / 0.867 / 0.576 / 0.094; output mean error ≤ 0.006, covariance error ≤ 0.010.
- spec_v0 now also logs spec_dmean_<node> = mean per-position |muD − muT| / s (how far the draft kernel is from the target kernel).
- Launched workflow wf_30fe665c-19a (literature on speculative diffusion + code audit + 3 independent designs of the unbiased frame-level AR sampler with adversarial proof/speed reviews + eval-protocol and verifier-design-space reviews + synthesis + completeness critic). Agents are read-only.
### 9/28 06:20 CDT — exact speculative sampler: the TARGET (CFG-5 Base kernel) collapses
- Workflow wf_30fe665c-19a (15 agents) done; sections saved in paper/overnight/ (email draft, repro plan, item-3 design, item-4 protocol, item-5 design space, code audit, critic). Key conclusions: --specstep is exact w.r.t. the hybrid chunk-AR target T_H^c9 (Base CFG-5 RePaint kernels at nodes 0-2, Fast node 3, Fast chunk 0), NOT Base-70; per-position acceptance 0.72-0.91 but whole-chunk acceptance <= 1e-25, so no lookahead/speed-up is possible with Fast drafts; Base-70 official pipeline wall-clock is 2,737-2,845 s at 464x848 on 1 GPU (the ~620 s figure is 360x512 + WorldCache k=4).
- By eye (4 finished runs, jar/castle x s1234/s4321): the exact spec output brightens and washes out to white/green after ~4 chunks. Luminance at frames 0/24/60/100/152: jar s4321 124/136/178/212/204 (Base 119/118/120/123/131), castle s1234 130/177/246/244/224 (Base 124/191/194/200/186). Because the sampler is exact, this is the target kernel (one-step CFG-5 x0 at sigma 1.0/0.94/0.83, re-noised, compounding through the prefix/anchor frame), consistent with the earlier bg1_n01 (Base-guided nodes) freeze.
- Action: stopped spending GPU on more CFG-5 seeds; GPUs 2,3 now run a target-kernel CFG sweep (1, 2.5) on s1234; GPUs 6,7 (after kv4) run target-only null arms (CFG 5 and CFG 1) for the equality-in-law check.
### 9/28 06:35 CDT — target-kernel CFG sweep (exact spec sampler, jar s1234): CFG 1 fixes colour but kills the camera
- CFG 1 (one Base forward per check; 232 s/clip vs 349 s at CFG 5): luminance 125/122/121/121/126/127/139/138/133/137, saturation 76-90 (frame 0: 125 / 80) — no wash-out. But the jar stays centred for all 153 frames: the camera command (pan away and back) is ignored, while Base-70 pans to the table and a white wall. Same failure as the earlier chunk-wise Base corrections ("snap to the prefix").
- => the chunk-AR target built from Base's one-step kernel is not Base-like at either CFG (5: colour collapse; 1: frozen view). Exactness w.r.t. such a target is not useful; the target definition is the open problem, not the coupling.
- Blind 3-rater drift labels (38 clips, Fleiss kappa 0.72, 87% unanimous): Base-70 passes only 3/10 (jar: material swap / new objects; castle: overcast -> sunny); Fast 0/5; exact spec (CFG 5) 0/5 (colour/texture collapse); v2-safe 0/6; v2-fast-CFG 2/6; v2-fast 2/6. No arm differs significantly from Base (Boschloo p >= 0.38; n = 2-7 per cell).
- CFG 1 castle s1234 (218 s): no wash-out (lum 130-179, sat 21-50 ~ Base 20-60), castle destroyed as in Base, but from chunk ~4 the frames are hazy and grainy (noise-like texture, little detail); CFG 2.5 running.
### 9/28 07:20 CDT — target-kernel CFG sweep complete (exact spec sampler, s1234) + null arm + one-step cost grid
| target CFG | jar: lum @ frames 0/60/100/152 | jar camera pans like Base? | castle: lum @ 0/60/100/152 | castle look | clip time |
|---|---|---|---|---|---|
| Base-70 (ref) | 119/107/127/128 | yes (table → white wall) | 124/194/200/186 | castle destroyed, beach, sky | 2,777-2,829 s |
| 5 | 125/186/189/157, sat → 198 (green) | no | 130/246/244/224 | washes out to white/green by chunk 4 | 341-349 s |
| 2.5 | 125/153/188/199 | no (jar stays centred) | 130/210/222/222 | over-exposed, hazy | 342-345 s |
| 1 | 125/126/139/137 | no (jar stays centred) | 130/179/137/177 | colour OK, hazy + grainy | 218-232 s |
- Target-only null arm (--bguide 1 at nodes 0-2, same ctx 9 / CFG 5), jar s1234: washes out the same way (lum 125/168/211/212) → the collapse is the target kernel's, not the coupling's (equality in law expected; n = 1 only a qualitative check).
- One-step Base cost grid (batch 1, one GPU, both experts resident): cost is linear in tokens, identical for high/low expert, CFG doubles it. 480p: 6/12/21-latent windows = 1.66/3.73/7.50 s (CFG 3.32/7.47/14.99 s), peak 74/78/85 GB. 576x304: 0.70/1.46/2.83 s (CFG 1.40/2.91/5.65 s), 2.2-2.6x cheaper. Batched forwards still fail (camera path).
- kv4 finished (step 1612); ck800 and ck1612 evaluations queued on GPUs 3,2.
### 9/28 08:30 CDT — blind labels batch 2 (73 clips total) + kv4 evaluation
- Blind 3-rater rubric, batches 1+2 (73 clips, Fleiss kappa 0.66, 88% unanimous). Strict no-drift, pooled jar+castle: Base-70 3/16, Fast 0/16, exact spec (CFG 5) 0/5, null arm 0/4, v2-safe 2/10, v2-fast-CFG 3/10, v2-fast 2/6 — the strict rubric counts Base's own behaviour (jar: new blue pot / material swap; castle: overcast -> sunny) as drift, so nothing differs from Base (min p 0.11).
- Exploratory (post hoc) 'severe' binary = teleport | colour collapse | texture collapse | freeze (failures Base rarely makes): clean Base 12/16, Fast 0/16 (teleport 16/16, p<0.001), exact spec 0/5 (p=0.004), null arm 0/4 (p=0.009; same colour+texture collapse profile as the exact sampler), v2-safe 3/10 (p=0.031 vs Base; p=0.036 better than Fast), v2-fast-CFG 4/10 (p=0.094; p=0.009 vs Fast), v2-fast 5/6. v2's castle failures = plan replaces the sea with a grassy/forested plain (teleport) + sunny sky. Scripts: scripts/blind_score.py, scripts/blind_cats.py, scripts/blind_report.py, scripts/fill_item4.py.
- kv4 (20 blocks, time-local memory, Translator2) with --kv-low plain on cr plans, DINO to_base mean of 6 cases: ck800 0.536, ck1612 0.565 vs v2-safe 0.541, kv1_ck875 0.584; seams (max boundary ratio mean) 2.73 / 2.62 vs 2.40 / 1.88 (Base 1.88). By eye: castle s1234 kv4 removes v2-safe's walking person; jar s9999 kv4 turns the white wall into a lake/shore. => KV route = y-hint level; plan quality is the bottleneck. kv5 (mixdata2, 224 clips) stopped at ~step 690 at 3:47 AM for GPU; ck575 eval partial (jar s1234/s4321 to_base 0.39/0.79; s9999 OOM).
- Report: paper/action_items_report.md -> https://claude.ai/artifact/5Ufm4yR4HAJViNa9KUByho
### 9/28 08:42 CDT — blind batch 3 (85 clips, kappa 0.69) + pre-registered confirmatory batch launched
- Final exploratory 'severe-clean' (8 seeds x 2 prompts): Base 12/16 (jar 7/8, castle 5/8), Fast 0/16, v2-safe 5/16 (jar 4/8, castle 1/8; p=0.016 vs Base, p=0.017 vs Fast), v2-fast-CFG 6/16 (jar 4/8, castle 2/8; p=0.037 vs Base, p=0.006 vs Fast), exact spec 0/5, null arm 0/4. Strict rubric: Base 3/16, v2-safe 3/16, v2-fast-CFG 3/16, Fast 0/16. The old DINO-NN rule called v2-safe 14/16, v2-fast-CFG 16/16, Fast 4/16 drift-free -> it misses plan teleports.
- PREREG: paper/overnight/prereg/PREREG_item4.md (severe-clean primary endpoint, rater script hashes fixed, Holm over 5 tests). Queue exp/queue/qprereg.txt (192 jobs: cr/l4c plans, Fast, v2 renders, Base-70 for seeds 101-116 x jar/castle), workers w4-7_prereg on GPUs 4-7 from 08:40 CDT. Base-70 is last in the queue (~6-7 h).
### 9/28 12:05 PM CDT — SDVG baseline (user: "이 베이스라인을 돌려서 항상 우리 비교실험할때 추가해줘", paper arXiv 2604.17397)
- Repo gustn9609/archive cloned (archive/). Branch verdict: `dev` = the paper's method (ImageReward min-frame router, fixed τ / dynamic quantile, force-reject block 0, target re-denoises rejected blocks from the same noises, VAE cache clone/restore); target/draft = target-only/draft-only rows; ablation_* = random/force-reject ablations; realtime = upstream. Details: paper/overnight/sdvg_baseline.md.
- Ported the router into spec_v0 as `--policy ir:TAU` (+ `--ir-prompt`, `--start 1 --warmup 1`): draft = Fast, router = ImageReward-v1.0 on every decoded frame (min), rejected chunk → Base 16-step RePaint. Bundle sdvg_<task>:<seed>:<tau> → sdvg_t<tau>_ir. Smoke (castle s1, 61 frames, τ −0.7): plan DCDCC, ir_min 1.51 / −1.69 / −2.23, 383 s.
- Calibration on existing videos (scripts/ir_pool.py, exp/kveval/ir_pool.json; 8 seeds × 2 tasks, 12-frame chunks): min-frame ImageReward sits at the model floor ≈ −2.28 for 64 % of Fast chunks, 19 % of Base chunks; min ≥ −0.7 for Fast 5 %, Base 33 %, v2-safe 27 %; avg-score median Fast −2.18, Base −1.40. So τ = −0.7 ≈ 'always Base' with Fast drafts; τ = −2.5 accepts everything. Sweep queued: τ ∈ {−0.7, −2.2, −2.5} × jar/castle × s1234/4321/9999 on GPUs 0,1 + 2,3 (exp/queue/qsdvg.txt).
### 9/28 12:25 PM CDT — disk cleanup + original SDVG reproduction set up
- Cleanup (user-approved): frames_all.npy (37.5 GB), onpol2/samples* (40 GB), hintlora kv1/kv2/kv3 + smoke checkpoints (~20 GB), non-mp4 files of the 9/14–9/21 experiment dirs (~18 GB), hf/Qwen3-VL-32B (63 GB). /home free 76 → 248 GB. All 1,051 spec_v0 videos and old mp4/md/json kept.
- Original SDVG (user: "원본 코드를 그대로 재현"): archive checked out at dev; .venv via uv (py3.11, torch 2.8.0+cu128, SDPA attention — no sage/flash wheels for Blackwell), image-reward installed; wan_models/Wan2.1-T2V-1.3B (T5 + VAE + 1.3B), checkpoints/self_forcing_dmd.pt (drafter), checkpoints/krea-realtime-video-14b.safetensors (target). Runner archive/sdvg_ours.py: MODE=sdvg|target|draft (target = tau 1e9 with force-reject, draft = tau −1e9 without), our two scene prompts (exp/irprompt_*.txt), seeds 1234/4321/9999, 13 blocks (153 frames) at 832×480, outputs specwm_out/sdvg_orig/<task>/<tag>_s<seed>/video.mp4 + sdvg.json (accepted/rejected/scores/gen_s). Launchers scripts/sdvg_orig_launch.sh wait for the ported-SDVG sweep on GPUs 0–3 to finish (sdvg on 0,1; target then draft on 2,3).
- Pitfall hit again: `pkill -f <script>` killed my own shell (command line contained the pattern) — use pgrep + explicit PIDs.
### 9/28 12:55 PM CDT — original SDVG launch fix
- First launch crashed at load (no wan_models/Wan2.1-T2V-14B: the wrapper builds the 14B architecture via from_pretrained before loading the Krea weights). Fix: downloaded only Wan2.1-T2V-14B/config.json and patched archive/utils/wan_wrapper.py to use from_config when the folder has no base weights; verified on meta tensors that krea-realtime-video-14b.safetensors covers all 1,095 parameters exactly (strict load), so the model is identical.
- The launcher had started early: the earlier wait patch was never applied (my pkill killed the shell that would have written it). Rewritten scripts/sdvg_orig_launch.sh waits for WORKER_DONE in both w01_sdvg/w23_sdvg logs + batch4.json. The early crash happened before any GPU allocation; the ported sweep kept running.
- 16-case metrics (exp/kveval/metrics16.json): Base / Fast / v2-safe / v2-fast-CFG — DINO to Base –/0.672/0.559/0.521; start-view distance at frame ~133 0.664/0.927/0.668/0.827; ImageReward avg −0.70/−1.85/−0.87/−1.35; |motion−Base| –/13.4/4.0/3.3; |flicker−Base| –/23.4/6.8/5.5; v2-safe vs v2-fast-CFG not significant on any metric (paired p 0.07–0.46); jar favours v2-safe on loop/IR, castle favours v2-fast-CFG.
### 9/28 3:35 PM CDT — direction 2 (exact test in a low-frequency subspace): band probe, jar/castle s1234, CFG 5 and CFG 1 targets
- spec_v0 --band-probe 1 (logging only) inside the exact sampler: gap = (muD − muT)/s per node, projected onto k×k block means per channel and latent frame (k = 2, 4, 8, 16, whole frame); exact acceptance of a whole frame tested only in that subspace = 2Φ(−|P gap|/2). Summary: scripts/band_summary.py → exp/kveval/band_summary.json. (First launch crashed: the probe variable `g` shadowed the sampler RNG; renamed.)
- log10 P(accept one latent frame), full / 8×8 blocks / 1 value per channel: CFG 1 node 0 jar −6.1/−4.2/−2.9, castle −3.0/−2.1/−1.0; node 1 jar −16/−6.2/−3.0, castle −12.5/−6.8/−2.1; node 2 ≈ −54/−13/−4.5. CFG 5 node 0 jar −139/−132/−105, castle −67/−65/−33; node 1 −32..−23/−21..−18/−11..−7; node 2 −96..−87/−50..−34/−23..−12.
- Gap energy is low-frequency: CFG 5 node 0 keeps 94–98 % of its energy in 8×8 block means and 62–66 % in the per-channel frame mean (a global colour/brightness disagreement = the CFG wash-out); CFG 1: 57–69 % (node 0) down to 16–25 % (node 2) in 8×8.
- => Restricting the exact test to low frequencies does not make whole-frame acceptance feasible: even with 16 numbers per frame the draft and target disagree by ≈ 3–9 σ (CFG 1) and far more at CFG 5. Draft and target disagree in content (layout/colour), not in detail. Needs a draft trained toward the target (L_D) and/or a target closer to Fast (same-family AR target) — next: measure the Fast vs causal-pretrain gap.
### 9/28 4:20 PM CDT — SDVG port sweep complete (18 clips) + blind labels (3 independent raters, one agent per rater; κ over all 103 clips 0.69)
| arm (jar+castle × s1234/4321/9999) | severe-clean (blind) | strict no-drift | DINO to Base | start-view dist @ ~133 | max seam ratio | time |
| Base-70 | 5/6 | – | – | 0.593 | 1.88 | 2,804 s |
| Fast alone | 0/6 | 0/6 | 0.649 | 0.873 | 1.55 | 63 s |
| v2-safe | 1/6 | – | 0.541 | 0.917 | 2.40 | 206 s |
| v2-fast-CFG | 2/6 | – | 0.525 | 0.903 | 3.04 | 139 s |
| SDVG port τ −0.7 (paper default; 3–11 of 12 chunks to Base) | 2/6 | 1/6 | 0.443 | 0.330 | 5.72 | 2,578 s |
| SDVG port τ −2.2 (1–4 chunks to Base) | 4/6 | 1/6 | 0.426 | 0.448 | 2.33 | 803 s |
| SDVG port τ −2.5 (only the forced first chunk) | 3/6 | 1/6 | 0.520 | 0.835 | 1.64 | 142 s |
- τ −0.7 failures: camera never turns (freeze, jar s1234/s9999: Base corrections snap back to the start view), sand → amber liquid (jar s4321), smeared frames (castle). Seams 5.7×. Castle τ −2.2/−2.5 s9999 identical videos (same decisions from the same seed). n = 6 per arm: no difference is significant.
### 9/28 4:30 PM CDT — original SDVG reproduction complete (archive dev, T2V, our two prompts, s1234/4321/9999, 13 blocks 832×480)
- Mean time per clip: target-only (Krea 14B) 70.4 s, SDVG τ −0.7 31.1 s (2.27×), draft-only (Self-Forcing 1.3B) 27.3 s. SDVG accepted 12/13 blocks on every clip (block 0 forced); ImageReward min-frame 0.07–1.41 on these static-camera T2V clips (vs ≈ −2.28 floor on our camera-pan clips) — the router passes almost everything here. Paper: 1.59× at 73 % acceptance on A6000.
- By eye (s1234): SDVG shares block 0 with the target, then follows the drafter — castle stays standing (target destroys it), jar sand level behaves like the draft. No camera control or start image possible with this model.
- Files: specwm_out/sdvg_orig/<task>/<arm>_s<seed>/{video.mp4, sdvg.json}; paper/0928/item2_speculative_baseline/sdvg_original_reproduction (+ side_by_side target | SDVG | draft).
### 9/29 5:20 AM CDT — Guandao's whiteboard, offline: Base scores a WHOLE clip with ONE step (lingbot-world/specwm/base_verify.py, GPU 5, 2.3 h)
- 103 blind-labelled clips (66 severe), first 153 frames, σ 0.3/0.5/0.7/0.95, conditional Base, start image + camera, same noise for all methods of a case.
- Base-only reconstruction error does not separate failures: AUC (severe) 0.49–0.63 over σ × {mean, max}; best σ 0.95 max 0.63 (0.66 without exact-spec/null arms). Normalising by the same case's Base-70 clip or by the clip's own σ-0.3 error does not help (0.40–0.64).
- Why: washed-out / blurry clips (exact spec, null arm) get the LOWEST error — flat images are easy to reconstruct (classic likelihood-vs-complexity trap). Hence the whiteboard's p/q: subtract the draft's own one-step error on the same frames and noise.
- Next (running on GPU 5): lingbot-world-v2/specwm/fast_verify.py (Fast teacher-forced one-step error at its node σ 0.9373/0.8331/0.6247, same noise recipe) → base_verify at the same σ → scripts/pq_verify_summary.py (d = err_Base − err_Fast). Smoke: Fast 51 s per clip.
- GPUs 6,7: SDVG fresh seeds (≈ 22 min/clip incl. model loading) → then 18 blend runs (--corr-blend low:4 / soft:0.5 / low:8).
### 9/29 8:56 AM CDT — p/q verifier (whiteboard min(p/q)): d = err_Base − err_Fast per latent frame, same clip and noise, Fast teacher-forced (fast_verify.py) vs Base whole-clip one step
- All 103 clips: AUC severe 0.52–0.60 (washed-out exact-spec/null clips fool both models equally). Without them (n = 88): Base only 0.58–0.60 → p/q mean **0.68–0.70** (σ 0.62/0.83), σ 0.94 p/q ratio 0.69 severe / **0.76 strict drift**; Fast-only error is anti-correlated (0.42–0.46).
- => subtracting the draft's own score removes much of the complexity confound; the signal is moderate (below DINO+colour 0.90 on the older chunk labels, different label set). A flatness/colour check is needed for washed-out outputs.
- Running (GPU 5, ~4 h): out-of-sample scoring of the 128 fresh-seed clips (exp/kveval/prereg_clips.json) → fast_verify_prereg.json + base_verify_prereg.json; AUC once their blind labels exist.
### 9/29 12:55 PM CDT — out-of-sample p/q scores on the 128 fresh-seed clips (labels pending), d = err_Base − err_Fast (mean over latents 1..38)
| σ | Base-70 | Fast alone | v2-safe | v2-fast-CFG |
| 0.9373 | −0.0237 | +0.0167 | −0.0225 | −0.0296 |
| 0.8331 | −0.0119 | +0.0120 | −0.0007 | −0.0064 |
| 0.6247 | −0.0052 | +0.0061 | +0.0014 | −0.0009 |
- Paired with the same-case Base-70 clip: Fast alone is less likely in 32/32 cases at every σ. v2 at σ 0.94 (layout level) ≈ Base (v2-safe 18/32, v2-fast-CFG 11/32 less likely), but at σ 0.62 (detail level) v2 is less likely in 31/32 (safe) and 28/32 (fast-CFG) → the plan fixes layout; v2's remaining gap to Base is texture/detail. The p/q score separates methods cleanly even though per-clip failure detection is only moderate.
- GPUs 4,5 now run the second blend worker (first run jar s1234 low:4 started 12:51 PM); 6,7 finish SDVG (31/32) then join.
### 9/29 1:30 PM CDT — PRE-REGISTERED item-4 result (fresh seeds 101–116 × jar/castle = 32 clips per arm; 3 blind raters, 12 agent batches of 40 in shuffled order; Fleiss κ 0.85, 144/160 unanimous)
| arm | severe-clean ↑ | rate [95% CI] | strict no-drift |
| A Base-70 | 28/32 (jar 13/16, castle 15/16) | 0.88 [0.72, 0.95] | 21/32 |
| D Fast alone | 3/32 (0/16, 3/16) | 0.09 [0.03, 0.24] | 0/32 |
| E1 v2-safe | 22/32 (10/16, 12/16) | 0.69 [0.51, 0.82] | 12/32 |
| E2 v2-fast-CFG | 24/32 (12/16, 12/16) | 0.75 [0.58, 0.87] | 16/32 |
| F SDVG port τ −2.2 (amendment) | 18/32 (9/16, 9/16) | 0.56 [0.39, 0.72] | 6/32 |
Registered tests (Boschloo, Holm over 5): E1 vs A p 0.080 (Holm 0.16, CMH 0.13); E2 vs A p 0.22 (Holm 0.22, CMH 0.34); E1 vs D, E2 vs D, D vs A all p < 0.0001. Amendment: F vs A p 0.006 (CMH 0.014), F vs E2 p 0.13.
=> v2 is clearly better than Fast and NOT distinguishable from Base-70 at n = 32 (the exploratory batch had suggested v2 < Base; that did not replicate). SDVG port is significantly worse than Base. Caveat: rater setup differs from the exploratory batches (agent per 40 sheets vs one agent per sheet), so compare only within this batch.
- Out-of-sample p/q verifier on these 128 clips (A, D, E1, E2): AUC severe 0.80–0.82 (Base-only 0.69–0.75), strict drift 0.71–0.76 → the p/q correction holds on unseen seeds.
### 9/29 2:40 PM CDT — smooth draft→verify→redraw (user + labmate idea, modified): spec_v0 --policy pq (GPUs 0–3, user-approved; hl35496 Blender jobs ~4 GB share GPUs 0/2)
- pq_score: both models re-denoise the SAME partly noised drafted chunk (σ of Fast node 2 = 0.8331) in one step; Fast with its committed prefix in the KV cache, Base with the last 18 prefix latents re-noised to σ (no CFG); d = err_Base − err_Fast (chunk mean).
- On reject (d > τ) Fast REDRAWS the chunk with fresh noise (up to R), keeps the lowest d → every frame comes from Fast's own rollout, so no Base frame is spliced in (no seam by construction). Fallback options: best | base | base_low4.
- Calibration (--pq-teacher, bundle pqcal_): teacher-forced scoring of Base-70 and Fast clips, seeds 101–104 × jar/castle. First case jar s101: Base-70 chunk d −0.035…+0.0007, Fast d rises 0.0007 → 0.039 as it drifts.
- Chain: τ = Base-70 chunk p75 → pqgen on jar/castle × s1234/4321/9999 with R = 3 and R = 7 (exp/queue/qpqgen.txt).
### 9/29 4:10 PM CDT — pq verifier calibration (v1 Base target, teacher-forced, seeds 101–104 × jar/castle, 8 Base-70 + 8 Fast clips)
- Per-chunk d (err_Base − err_Fast at σ 0.8331): Base-70 mean −0.023 … −0.002 by chunk; Fast −0.0005 → +0.042 (rises as it drifts).
- Chunk-level AUC (Fast vs Base-70 chunks) 0.943; clip-level AUC (max over chunks) 1.000. τ = Base-70 p75 = 0.0005 rejects 25 % of Base-70 chunks and 96 % of Fast chunks (p90: 10 % / 81 %).
- Velocity-difference score |v_Base − v_Fast|² (logged for the last clips only): chunk AUC 0.88 on 24 chunks — full comparison comes from the v2-bid calibration.
- Smooth generation (pqgen, τ 0.0005, R = 3 and 7, fallback best) running on GPUs 0–3; v2 bid (139 GB, fp32, lingbot-world-v2/lingbot-world-v2-14b-bid; links in bid_as_v1_cam resolve) downloaded after deleting LongLive + 5 unused venvs (user-approved, ~90 GB); v2-bid calibration (pqcalb_) running on GPUs 6,7. /home free 60 GB.
### 9/29 4:30 PM CDT — Guandao's frame-jump fixes (latent-level soft / low-frequency update of Base corrections), 6 cases × 4 variants, blind 3 raters (24/24 unanimous)
| variant | seam at Fast↔Base switches | max seam | DINO to Base | flicker | severe-clean (blind) |
| replace (cfg6reuse_semc100) | 1.67× | 3.62× | 0.491 | 19.6 | 6/6 |
| soft 0.5 | 1.54× | 3.15× | 0.567 | 22.6 | 0/6 (teleport 6/6) |
| low-freq 4×4 | 1.47× | 3.97× | 0.680 | 29.3 | 1/6 |
| low-freq 8×8 | 1.37× | 2.70× | 0.748 | 34.4 | 1/6 |
=> Blending at the latent level trims the seam by 8–18 % but keeps the draft's content, so Fast's teleports/new objects survive; Base-likeness collapses. Same lesson as fd1 (Base continuing from the draft): the draft's layout must not be kept when the chunk was rejected. KV-level (not latent-level) soft update untested.
### 9/29 4:25 PM CDT — switch the target to LingBot v2 14B bid (user: "다시 다운받은 v2로 넘어가자")
- v1-target pq generation (τ 0.0005 = v1 Base-70 p75) stopped after 2 runs: plans DDDFFFFFFFFFF / DFRFFFFFFFFFF — almost every chunk failed all 3 redraws (kept the best Fast draft, 'F'). With the v1 target Fast almost never passes (96 % of Fast chunks rejected in calibration).
- Now: v2-bid calibration (pqcalb_, 16 clips) on all 4 GPU pairs → τ_v2 = its Base-70 p75 → queue qv2: semb_ (sem draft→verify→Base-correction pipeline with v2 bid as corrector, 6 cases; does same-family correction reduce the frame jump?) + pqgenb_ (smooth redraw with v2 target, R = 3 and 7, 6 cases each), 4 workers. Watcher self-match bug avoided with pgrep "[p]qcalb_".
### 9/29 6:55 PM CDT — v2-bid target results (6 cases each)
- v2-bid calibration: chunk AUC 0.932 (v1 0.943), clip AUC 1.00; τ_v2 (Base-70 p75) = 0.0010 → rejects 24 % Base-70 / 91 % Fast chunks; velocity-difference score AUC 0.82 (error difference is better).
| method | seam at Fast↔Base switches | other boundaries | DINO to Base | flicker | time |
| Fast alone | – | 1.14× | 0.649 | 39.7 | 63 s |
| replace, v1 corrector | 1.67× (max 3.62×) | 1.26× | 0.491 | 19.6 | 756 s |
| replace, v2-bid corrector | 1.88× (max 7.25×) | 1.41× | 0.504 | 24.0 | 792 s |
| v2 verify + Fast redraw R3 | – | 1.17× | 0.705 | 45.3 | 566 s |
| v2 verify + Fast redraw R7 | – | 1.16× | 0.716 | 43.1 | 1027 s |
- Same-family corrector does NOT shrink the jump (worse max seam). Redraw is seamless but Fast passes only chunks 1–3 (plans DDDRFFFF…/DFFFF…): once the prefix drifts, fresh noise from the same prefix does not escape → quality stays at Fast level.
- => the jump comes from HOW the correction is made (Base re-decides the chunk from a noisy prefix and snaps back), not from the model family. Next (running, 4 pairs): Base correction that sees the prefix CLEAN through its y/mask channel (--corr-mode mask / both), v1 corrector, 6 cases each.
### 9/29 8:40 PM CDT — clean-prefix Base correction (--corr-mode both / mask, v1 corrector, 6 cases each)
- Seam at Fast↔Base switches: replace 1.67× (max 3.62×) → both 1.02× (max 2.34×), mask 1.15× (max 4.69×); other boundaries 1.09× (Fast 1.14×); flicker 19.7 / 20.8 (replace 19.6). The frame jump is gone with y/mask + repaint.
- But DINO to Base 0.655 / 0.622 (replace 0.491, Fast 0.649); by eye (s1234): jar table turns into open ocean, castle → sea + forested shore; Base smoothly CONTINUES whatever it is shown (incl. early drift) — continuity without correction. Blind labels below.
### 9/29 9:40 PM CDT — plan + render + p/q verify + Fast redraw (pqplan, τ 0.0005, R 3, v1 target), 6 cases
| method | DINO to Base | max seam | flicker | motion | time |
| Fast alone | 0.649 | 1.55× | 39.7 | 23.7 | 63 s |
| draft→verify→Base replace | 0.491 | 3.09× | 19.6 | 12.2 | 756 s |
| v2-fast-CFG (plan + render) | 0.525 | 3.04× | 13.7 | 8.5 | 72 s (+67 s plan) |
| + p/q verify + redraw | 0.498 | 3.21× | 13.4 | 8.4 | 359 s (+67 s plan) |
- Plans: castle s1234 all 12 chunks accepted first try; jar s4321 DDDDDFFRRRDRD, jar s9999 DDDDRRDDFRRDD (redraws now DO pass: 10 of 25 rejected chunks recovered), castle s4321/9999 mostly F (no recovery). With the plan anchoring the content, the verifier accepts most chunks and redraws succeed sometimes; without a plan (Fast alone as draft) nothing ever passed after chunk 3.
- By eye (jar s4321, castle s9999): the redraw output ≈ v2-fast-CFG (same plan drives both), both differ from Base-70 (jar: Base keeps the jar centred, plan pans over the table; castle: plan turns to a grassy shore under blue sky). Seams: max ratio 3.2× comes from chunk 0 → 1 (plain Fast chunk 0 → hinted chunk 1), same as v2. Blind labels pending.
- Blind (3 raters): pqplan severe-clean **6/6** (jar 3/3 clean, castle 3/3 only mild lighting change, i.e. the plan's sunny sky), strict 3/6. v2-fast-CFG on the same 6 seeds: severe-clean 2/6 (jar s4321 texture/colour collapse; castle teleports). Caveat: different rater batches; 6 clips.
### 9/29 10:25 PM CDT — ALL-v2 decision (user: "아니 다 v2로 돌려야지"): draft = v2 causal-fast, target/planner/verifier = v2 14B bid, reference = v2 bid 70-step
- plan_gen.py / base_verify.py now take SPECWM_BASE_CKPT; spec_v0 --base-ckpt. Bundles: base70v2_ (v2 bid 70-step reference, bench_base_official --ckpt bid_as_v1_cam), plangv2_<task>:<seed>:l4c (v2-bid low-res plan), v2render_ (Fast + plan6_ck400 hint LoRA on the v2 plan), pqplanb_<task>:<seed>:0.0010:3 (v2 plan + render + p/q with v2 target τ 0.0010 + redraw).
- Caveat: the hint LoRA plan6_ck400 was trained on v1 plans; v2 plans may look different (check by eye). Queue qv2all (18) on pairs 0,1 / 2,3 after pqplan2; qbase70v2 (6 × ~47 min) on single GPUs afterwards.
### 9/29 10:45 PM CDT — SDVG original, stricter τ (13 blocks, s1234/4321/9999)
- τ 1.5: accepted 0/13 (jar s1234, s9999, castle s9999), 3–4/13 (others) → 55–71 s per clip (target-only 70 s). τ 1.3 running.
### 9/29 11:00 PM CDT — plan + render + p/q verify + redraw on fresh seeds (28 of 32; 4 OOM'd at load on GPUs 0/1 when another user's job filled the GPU, re-queued in qv2all)
- Blind (3 raters, 26/28 unanimous): severe-clean 20/28 (jar 10/13, castle 10/15), strict 11/28. Same seeds, earlier batch: v2-fast-CFG 24/32 (0.75), Base-70 28/32. => the 6/6 on s1234/4321/9999 did NOT carry over; on fresh seeds the verify+redraw step is at best equal to plain v2-fast-CFG (0.71 vs 0.75). Redraws recover some chunks but also cost ~4x render time (199–517 s vs 72 s).
- SDVG original τ sweep (13 blocks): τ −0.7 31 s (12/13 accepted), τ 1.3 51 s (5.8/13), τ 1.5 63 s (1.8/13), target 70 s. Added to paper/0930_ppt/sdvg_original_3methods (5-way side-by-side).
### 9/30 12:50 AM CDT — AR vs bidirectional (same Wan2.1 family, same prompts, seed 1234) for the 9/30 meeting
- Bidirectional Wan2.1 T2V 1.3B (the SDVG drafter's teacher) run via the fork's WanT2V after two fixes (undefined rope_params_riflex → original rope; direct flash_attention → attention() SDPA fallback): 81 frames / 16 fps, 50 steps, CFG 5, 130–132 s per clip on one GPU. AR 1.3B (drafter only) 26–28 s / 150 frames, AR 14B (target only) 68–77 s / 150 frames; 30 s clips: AR 1.3B 140–144 s, AR 14B 339–342 s, SDVG τ −0.7 183 s (tokyo, 46/61) and 274 s (bigsur, 19/61), τ 1.3 256 s (24/61) / 338 s (0/61).
- Folder paper/0930_ppt/ar_vs_bidirectional (+ sdvg_original_3methods with the τ sweep, sdvg_original_30s). Bidirectional 14B (Krea's teacher, 57 GB) not run: disk 53 GB free.
- v2 chain: castle s9999 plan was lost to an OOM (another user's job on GPU 0) → re-running plan + pqplanb + v2render for castle 9999 on GPUs 4,5; base70v2 references generating on GPUs 0–3.
### 9/30 1:10 AM CDT — cleanup #3 (user-approved): frames_all.npy again (9.9 GB), hintlora plan1–5/plan300/700/ck150/300/run1/2/timing/planlora1*/tex1–3 (12.7 GB; kept plan6_ck400, kv4*, kv5*), stevo 9/14–17 baselines except reground (15 GB), onpol2/rollout, mixlora, fastlora (8.4 GB), ~/.cache/uv (7.8 GB) → /home free 53 → 100 GB. Downloading Wan2.1-T2V-14B model shards (57 GB, bidirectional teacher of Krea 14B) into archive/wan_models/Wan2.1-T2V-14B (T5/VAE/tokenizer symlinked from the 1.3B folder); then bid_t2v.py --size 14B on GPU 6 for the 4 prompts.
### 9/30 1:55 AM CDT — ALL-v2 results (6 cases, reference = LingBot v2 14B bid, 70 steps, 2,757–2,850 s per clip)
| arm | DINO to v2 reference | start-view dist @ ~133 (ref 0.277) | |Δmotion| | |Δflicker| | max seam |
| v1 Base-70 | 0.399 | 0.593 | 2.6 | 4.5 | 1.88× |
| Fast alone | 0.710 | 0.873 | 12.3 | 22.0 | 1.55× |
| v1 plan + render (v2-fast-CFG) | 0.668 | 0.903 | 3.1 | 5.3 | 3.04× |
| v2 plan + render | **0.394** | **0.310** | 3.0 | 5.0 | 2.11× |
| v2 plan + render + p/q(v2, τ 0.0010) + redraw | 0.395 | 0.311 | 3.0 | 5.0 | 2.20× |
- With the v2 bid as planner, the render lands as close to the v2 reference as v1 Base-70 itself is (0.394 vs 0.399) and closes the camera loop like the reference (0.31 vs 0.28); the v1 plan pipeline measured against v2 is far (0.67, loop 0.90). The p/q verify + redraw step adds nothing on top of the v2 plan (0.395 vs 0.394). Blind labels pending (paper/overnight/blind_v2, 18 sheets).
- Bidirectional Wan2.1 14B (Krea's teacher) generated after cleanup #3: 645–650 s per 81-frame clip on one GPU (SDPA); added to paper/0930_ppt/ar_vs_bidirectional (4-way side-by-side). /home free 47 GB.
- By eye (v1v2_plans strip): the v2 bid reference keeps the jar centred (camera command weakly followed) and turns the sand into yellow liquid on s1234; v2 plans on castle come out pale/over-exposed (lightx2v LoRA + CFG settings were tuned on v1) and the render follows. "Closer to the v2 reference" therefore also inherits the v2 reference's own quirks. Blind labels pending.
- Blind (3 raters, 16/18 unanimous), all-v2 6 cases: v2 bid 70-step reference severe-clean 5/6 (jar s1234/s4321 material swap = sand → liquid, mild; castle s1234 teleport + sunny); v2 plan + render 3/6 (jar s4321 teleport, jar s9999 teleport + new object, castle s4321 colour collapse = the pale plan); v2 plan + render + p/q + redraw 3/6 (identical decisions). Same 6 seeds, v1 pipeline: v2-fast-CFG 2/6, plan+verify+redraw 6/6 (small-n). => switching to the v2 family makes the render track its reference (DINO 0.39) but does not raise the blind pass rate; the v2 plan inherits the v2 bid's own quirks (jar material swap, pale castle).
### 9/30 2:05 AM CDT — plan + render + p/q verify + redraw, fresh seeds: FINAL 32/32 rated (the 4 re-run clips: jar s114 teleport, jar s115 material swap only, jar s116 clean, castle s113 new object + lighting only)
- Blind (3 raters, 25/32 unanimous): severe-clean **23/32 (0.72)**, jar 12/16, castle 11/16. Same seeds, earlier batch: v2-fast-CFG 24/32 (0.75), v2-safe 22/32, Base-70 28/32, SDVG τ −2.2 18/32, Fast 3/32.
- Paired vs v2-fast-CFG (same plan drives both): both clean 21 / only pq clean 2 / only v2 clean 3 / both bad 6 → no gain; vs Base-70: 20 / 3 / 8 / 1. Time ~4× v2-fast-CFG (199–517 s vs 72 s per clip).
- Conclusion (v1 and v2 target alike): chunk-level p/q verification + Fast redraw adds nothing on top of the plan; the plan decides the outcome. Next lever = plan-level verification / regeneration or re-tuning the planner for the v2 target.
