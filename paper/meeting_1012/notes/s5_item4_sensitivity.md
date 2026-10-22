# Action item 4: Sensitivity analysis and combination (speed-up next to quality, per technique)

**Headline:** The speed levers multiply, and they give three operating modes. Speed-ups are quiet steady ratios from the quiet timing pass (facts.md §15: jar + sandcastle, steady s/chunk, generation only; teacher 2.939 s/chunk = 4.08 fps). Quality is paired over the 86 bench-43 clips.
- **Ours: teacher-parity mode** (`cw07_safe`, chunk-wise adaptive 0.7 + safe commit): 6.80 fps, **1.67×** the teacher, Δ quality −0.07 ± 0.06, Δ drift −3.3 ± 3.1 (both within noise).
- **Ours: quality mode** (`ours832`, 832 2-step fused + partial commit): 9.74 fps, **2.39×** (cloud GPU only), Δ quality −0.30 ± 0.08.
- **Ours: realtime mode** (`ours704`, 704 2-step fused + partial commit): 15.73 fps, **3.85×** (cloud GPU only), Δ quality −0.62 ± 0.09.

The same arms on bench-43 under load give 1.72× / 2.48× / 3.99× from raw s/chunk (listed, rounded: 1.74× / 2.5× / 4.1×), within ~5 % of the quiet ratios. **Correction:** the often-quoted teacher "4.9 fps quiet" is a whole-clip average (153 frames / 31.2 s generation, whole clip, including the cheap early chunks before the KV window fills), not a steady fps. On the steady definition the teacher runs at 4.08 fps quiet / 3.9 under load (3 % apart). The earlier "1.4× / 2.0× / 3.2× against the quiet teacher" divided by the 4.9 and are withdrawn.

Chunk-wise adaptive is free on both prompt sets (12×3: 0.00 ± 0.11; bench-43: +0.03 ± 0.05) at 1.45× (quiet). The commit shortcut on top of it raises the speed-up to 1.66–1.67× (quiet; bench-43 under load, raw s/chunk: 1.49× → 1.72–1.74×). Its quality cost against chunk-wise alone is small and borderline (−0.10 ± 0.06 safe / −0.13 ± 0.07 fused + partial, paired, 86 clips, < 2 SE), and both stay within noise of the teacher. Beyond parity, resolution and the 2-step commit shortcut set the quality cost: at 2 steps, fused + partial 1 costs about as much as one resolution step (832 → 704), and the resolution cost gets steep below 576. The quality and realtime fps count the cloud GPU only (device step assumed to run in parallel on a separate device). With the device step run serially on the same GPU they are 8.9 / 13.6 fps (2.25× / 3.44×; measured only under load, on bench-43). Figure: **f11_modes** (panel a plots bench-43 under-load fps; its title, mode labels and table lead with the quiet ratios 1.67× / 2.39× / 3.85× and give the under-load ratios 1.74× / 2.5× / 4.1× in parentheses; re-rendered 10/12 08:00 CDT).

**Setup (applies to every table below)**
- Teacher: LingBot-World v2 14B causal-fast, 4 steps, 832×480. Every arm runs on the same inference stack. GPU: RTX PRO 6000 Blackwell (96 GB).
- Speed: fps = 12 frames / steady seconds per chunk (median over clips of the per-clip median after warm-up). Real time is 16 fps (one chunk = 0.75 s).
  - Preferred: the quiet timing pass (facts.md §15; jar + sandcastle, no other generation jobs). Generation only: decode is excluded from the timed interval. The teacher and chunk-wise arms ran with no decode at all (spec_early --no-video). The ours / 2-step arms streamed the full-VAE decode on the same GPU in the same runs (res_split_infer --stream-decode 1), but cloud_chunk_s times the cloud step only (their whole-loop steady chunk interval with the decode inside is 1.204 s at 704 and 1.876 s at 832). These runs also give the §13 streaming rows (10.8 / 6.9 fps). Teacher and chunk-wise arms: chunk-start differences from chunk 5 on (the whole per-chunk loop, head included). 2-step arms, incl. ours 832 / 704: cloud_chunk_s from chunk 6 on (cloud GPU only; the device step is assumed to run on a separate device in parallel, facts §1/§6).
  - Bench-43 under load (7 GPU jobs; facts §12): same definitions, on the quality prompts. Shown as a check next to the quiet value, or alone where no quiet value exists; always labelled.
  - Teacher: 4.08 fps quiet (2.939 s/chunk) / 3.9 under load (3.038 s/chunk). The often-quoted 4.9 fps is whole-clip (README §N: 153 frames / 31.2 s, 10 prompts, seed 1234). It is not steady, and no ratio in these notes uses it.
  - Ours 832 / 704 run serially on the same GPU (cloud + device step): 8.9 / 13.6 fps (under load, facts §12). On one shared GPU with the full VAE, ours 704 = 10.1 fps (composed from the under-load 0.761 s/chunk cloud, facts §6).
  - Every fps cell states its timing condition: quiet, under load, not recorded (facts.md §5 'timing load'), composed from an under-load value, or, for the facts §13 streaming runs, "no other generation jobs" (quiet-equivalent; the VLM judge ran on another GPU).
- Quality judge: Qwen3-VL-32B on 8 frames per clip, at 448 px unless stated otherwise. Exception: the StEvo-Bench subset in the Caveats uses its own protocol (facts §14: 32 frames, 3-vote majority, criteria control / physics / state).
  - VLM quality is on a 1–10 scale (↑). It has not been validated against humans.
  - VLM drift is on a 0–100 scale (↓). It has clip-level AUROC 0.82 against human severe-drift labels.
- Prompt sets:
  - **12×3**: 12 prompts × 3 seeds = 36 clips per arm, with ± = SE over clips. All 12×3 arms share the same 36 prompt–seed pairs, so the 12×3 Δs below are paired.
  - **bench-43**: 38 StEvo-Bench camera-WM tasks + 5 easy scenes = 43 prompts × seeds 1234/4321 = 86 clips per arm. Its Δ values are paired against the teacher, or arm vs arm where stated (commit policy vs `cw07_sep`).
  - The quiet timing pass runs on jar + sandcastle, which is neither quality prompt set.
- Mode names used in every section: **ours: teacher-parity mode** = `cw07_safe`, **ours: quality mode** = `ours832` (ours 832), **ours: realtime mode** = `ours704` (ours 704).

---

## 1. One lever at a time on the 12×3 matrix (same judge; paired over the same 36 clips per arm unless noted)

| lever (context) | fps ↑ before → after | speed-up ↑ | VLM quality ↑ before → after | Δ quality ↑ (paired) | VLM drift ↓ before → after | Δ drift ↓ (paired) | boundary jumps > 1.5× ↓ before → after |
|---|---|---|---|---|---|---|---|
| steps 4 → 2 (832, separate commit) | 4.08 → 7.98 (both quiet) | <u>1.95×</u> | 7.64 → 7.53 | −0.11 ± 0.10 | 45.4 → 50.8 | +5.4 ± 4.3 | 2 % (s1234) → – |
| chunk-wise adaptive, learned head, 2.9 steps (vs 4-step, 832; 34 shared clips) ¹ | 4.08 → 5.94 (both quiet) | 1.45× | 7.68 → 7.68 | **0.00 ± 0.11** | 43.4 → 42.9 | −0.4 ± 3.1 | 2 % (s1234) → – |
| fused commit (832, 2-step) | 7.98 (quiet) → 11.8 (not recorded) | 1.47× ² | 7.53 → 7.28 | −0.25 ± 0.10 | 50.8 → 49.4 | <u>−1.4 ± 3.1</u> | – → 7 % (s1234) |
| fused commit (704 cloud, native) | 12.50 → 19.04 (both quiet) | 1.52× | 7.22 → 7.08 | −0.14 ± 0.14 | 54.2 → 49.3 | **−4.9 ± 3.3** | 3 % → 12 % |
| + partial commit, last 1 latent frame (on 704 fused) | 19.04 → 15.73 (both quiet) | 0.83× | 7.08 → 6.86 | −0.22 ± 0.15 | 49.3 → 53.5 | +4.2 ± 3.0 | 12 % → 4 % |
| + partial commit, last 2 latent frames (on 704 fused) | 19.04 (quiet) → 13.7 (not recorded) | 0.72× ² | 7.08 → 7.08 | **0.00 ± 0.17** | 49.3 → 52.9 | +3.6 ± 3.5 | 12 % → 4 % |
| net: fused + partial 1 vs separate (704) | 12.50 → 15.73 (both quiet) | 1.26× | 7.22 → 6.86 | −0.36 ± 0.13 | 54.2 → 53.5 | −0.7 ± 3.3 | 3 % → 4 % |
| resolution 832 → 704 (14B 2-step, separate, native) | 7.98 → 12.50 (both quiet) | 1.57× | 7.53 → 7.22 | −0.31 ± 0.10 | 50.8 → 54.2 | +3.3 ± 3.0 | – → 3 % |
| resolution 704 → 672 (fused) | 18.7 → 22.4 (both quiet, README V3 run) | 1.19× | 7.08 → 6.89 | −0.19 ± 0.12 | 49.3 → 55.4 | +6.1 ± 4.2 | 12 % → 12 % |
| resolution 832 → 576 (14B 2-step, separate, native) | 7.98 (quiet) → 21.5 (not recorded) | **2.69×** ² | 7.53 → 6.83 | −0.69 ± 0.12 | 50.8 → 56.9 | +6.1 ± 4.6 | – |
| resolution 832 → 448 (448×256; 14B 2-step, separate, native) | 7.98 (quiet) → no quiet value (diag run 32.7, load not recorded; s4) | – | 7.53 → 6.17 | −1.36 ± 0.16 | 50.8 → 58.2 | +7.4 ± 4.1 | – |
| resolution 832 → 384 (384×224; 14B 2-step, separate, native) | 7.98 (quiet) → no quiet value (diag run 48.0, load not recorded; s4) | – | 7.53 → 5.06 | −2.47 ± 0.20 | 50.8 → 66.9 | +16.1 ± 5.3 | – |
| SR: 1.3B refiner on the 704 cloud (vs native) | 12.50 → 12.50 (quiet; refiner on the device, cloud-bound) | 1.00× | 7.22 → 6.72 | −0.50 ± 0.11 | 54.2 → 57.1 | +2.9 ± 2.1 | – |
| decoder: TAEHV vs full VAE (704 fused + partial 1) | 15.73 → 15.73 (own device GPU: cloud-bound, quiet); 10.1 → 15.5 (one shared GPU; composed from the under-load 0.761 s/chunk cloud, facts §6) | 1.00× / 1.53× | 6.86 → 6.31 | −0.56 ± 0.13 | 53.5 → 60.8 | +7.4 ± 3.0 | 4 % → 3 % |

Bold marks the best value and underline the second, ranked within the speed-up, Δ quality and Δ drift columns. In the Δ quality column, two rows tie at 0.00 (chunk-wise adaptive and partial 2), so they take both the first and second places and no other row is underlined.

fps = 12 / steady s/chunk. Speed-ups are ratios of steady s/chunk (fps ratio for 576, which has only an fps). The quiet values come from the quiet timing pass (facts §15): teacher 2.939, chunk-wise 2.021, 2-step 832 separate 1.504, 704 separate 0.960, 704 fused 0.630 and 704 fused + partial 1 0.763 s/chunk. The teacher and chunk-wise use chunk-start differences; the 2-step arms use cloud_chunk_s. Cells without a quiet value keep their older value and condition:
- 832 fused 1.02 s/chunk (11.8 fps) and partial 2 0.874 s/chunk (13.7 fps): facts §5, no recorded load.
- 576 separate 21.5 fps: README §U via facts §3, no recorded load.
- Refiner: the refiner runs on the device, so the arm runs at the 704 separate-commit cloud rate, 12.50 fps quiet (0.960 s/chunk, facts §15). facts §3 lists 12.7 for both 704 arms; that is the older value of the same arm, with no recorded load.
- 704 → 672: a same-run quiet pair from README V3 (0.640 → 0.536 s/chunk), a different run from facts §15.

The timing prompts (jar, sandcastle) are not the 12 quality prompts, and no fps comes from the 36-clip VLM run. The 10/10 run's 4.9 / 6.5 / 8.3 fps that this table used before were whole-clip averages (153 frames / gen s, including the cheap early chunks) and are no longer used.

Δ = mean ± SE of the per-clip paired differences (after − before) over the same 36 prompt–seed pairs (chunk-wise: 34). The Δs are computed from `paper/1012/vlm_scores_3seed.json`, `vlm_scores_fused.json` and `vlm_scores_diag.json`; the 832 fused arm is labels f832_n0_500 (s1234) + f832_s4321 + f832_s9999, and 448/384 are dg448_cloud / dg384_cloud. facts.md lists the paired SEs only for steps 4 → 2, chunk-wise and the refiner (§3) and for TAEHV (§6). Boundary-jump column: the 832 arms (teacher, 832 fused) are seed 1234 only (12 clips); the 704/672 arms are 36 clips.

¹ Paired on the 34 clips both arms share (facts.md §3 'Chunk-wise paired': Δ quality +0.00 ± 0.11, 7 better / 20 equal / 7 worse; Δ drift −0.4 ± 3.1 → tied).
- The 'chunk-wise adaptive' label in `paper/1012/vlm_scores_3seed.json` has 34 clips (seed 1234: 10, 4321: 12, 9999: 12). The two missing clips are sand_filling_up_jar s1234 and sandcastle_wave_impact s1234. On those, the teacher scored drift 75 / 85 and quality 7.0 / 7.0, both worse than its mean. Comparing the 34-clip arm mean with the 36-clip teacher mean (7.68 vs 7.64, 42.9 vs 45.4) therefore flatters chunk-wise.
- The old '43 ± 4.5' was README R's SE across the 3 seed means; the per-clip SE is 5.2 (42.9 ± 5.2).
- Speed: the quiet pass gives 1.45× (chunk-wise 0.7, separate commit, 2.94 mean steps on jar + sandcastle; facts §15). Bench-43 under load gives 1.49× from raw s/chunk (1.51× listed; 2.91 steps), within 3 %. The 1.33× used before (10/10 run, 4.9 → 6.5 fps) was a ratio of whole-clip averages and is superseded.
- Bench-43 confirms the quality on 86 paired clips (§4, `cw07_sep`): Δ quality +0.03 ± 0.05 (12 better / 65 equal / 9 worse), Δ drift −0.6 ± 3.2.

² Mixed conditions: the 'before' cell is quiet (facts §15) and the 'after' cell has no recorded load condition (facts §5 / README §U). On the steady basis load slowed the teacher by 3 % (2.939 quiet vs 3.038 s/chunk under load, different prompts; facts §15), and facts §0 allows ≈ 10–20 %, so these ratios are approximate.

- **Noise floor:** the paired SE of a single-lever Δ is 0.10–0.20 for quality and 2.1–5.3 for drift. On quality these are below 2 paired SE: steps 4 → 2, chunk-wise, fused at 704, partial 1 or 2 alone, and 704 → 672.
- **Clear quality costs (more than 2 paired SE):**
  - Resolution: 704 costs −0.31, 576 −0.69, 448 −1.36 and 384 −2.47 (drift +16.1 at 384).
  - Fused commit: −0.25 at 832. The jump-free variant at 704 (fused + partial 1 vs separate) costs −0.36.
  - The 1.3B refiner costs −0.50 at zero speed-up.
  - TAEHV costs −0.56 at the 448-px judge and −1.06 ± 0.14 at the 832-px judge.
- **Fused commit is not free.**
  - Paired over the same 36 clips, fused at 832 costs −0.25 ± 0.10 quality (2 better / 24 equal / 10 worse).
  - The jump-free variant at 704 (fused + partial 1) costs −0.36 ± 0.13 against separate commit (2 / 23 / 11). That is as large as the 832 → 704 resolution cost (−0.31 ± 0.10, 2 / 22 / 12).
  - Fused alone also causes chunk-boundary jumps, including hard scene cuts in a few clips: 12 % of boundaries at 704 and 7 % at 832 (s1234). Partial commit (last 1 latent frame) brings this back to 4 % (19/432; separate commit: 3 %, 15/432).
  - The speed cost of partial 1 at 704 is now measured with both runs quiet: 0.630 → 0.763 s/chunk (+0.133 s/chunk; 19.04 → 15.73 fps, 0.83×; facts §15). These two medians cover 6 vs 5 clips; on the same 5 clips it is 0.640 → 0.763 s/chunk (≈ +0.12 s/chunk; s2 §b, derived).
  - Figures: f02_partial_commit and f02b_jump_frames.
- **Decoder (f09_decoder):** the full VAE takes 426 ms/chunk and TAEHV takes 12.8 ms/chunk at 704.
  - With a separate device GPU, the 426 ms decode is shorter than the ≈ 0.76 s cloud chunk (0.763 quiet / 0.761 under load), so the full VAE costs no fps.
  - TAEHV only pays off when the cloud and device share one GPU: 10.1 → 15.5 fps (composed from the under-load cloud chunk, facts §6).
  - Ours keeps the full VAE.

## 2. Levers measured with their own protocol (not ranked: the protocols differ)

| lever | measured on | speed effect ↑ | quality evidence | verdict |
|---|---|---|---|---|
| systems stack: exact patches → + compile → + FP8 (4-step) | 5 prompts, 14B, e2e s per clip | 82.5 → 70.9 → 66.1 → 56.5 s (1.16× → 1.07× → 1.17×; 1.46× total) | Blind test of the full S7 stack (3 raters, 24 vs 24 clips): not worse than the original. Quality 3.78 vs 3.65, severe-clean 9/24 vs 3/24. S7 also includes 2 steps. | keep |
| systems stack: T5/VAE pipeline → + SageAttention 2 (after 2 steps) | same | 41.5 → 30.7 → 27.7 s (1.35× → 1.11×). Sage kernel 8.05 → 5.54 ms (1.45×); per forward 0.550 → 0.467 s/chunk (1.18×). | same blind test | keep |
| full ladder S0 → S7 (incl. 2 steps) | same | 82.5 → 27.7 s (2.98×) | same blind test | keep |
| parallelization: sequence parallel, 2 GPUs over PCIe | 14B 2-step 832, e2e s per clip | 18.5 → 18.0 s = 1.03× (bf16 comm, Sage) / 18.5 → 16.8 s = 1.10× (fp8 comm); at 576 SP (7.4 s) is slower than 1 GPU (< 1×) | not measured (fp8 comm is lossy, so not exact) | drop on this machine |
| parallelization: commit offload to a 2nd GPU | jar/castle | faster cloud loop | hallucinated scenes in 6/6 clips | drop |
| parallelization: cloud/device pipelining (decode on the device) | 704 fused + partial 1 | full VAE (426 ms) hidden under the cloud chunk (0.763 s quiet / 0.761 s under load) | no quality cost | keep |
| verifier + redraw (14B drift head, AUROC 0.88 on unseen scenes) | 6 scenes × 2 seeds, 832 | time ×2 | redrawn chunks are flagged again: no gain | drop (detector only) |
| draft + ImageReward verifier (SDVG), τ −0.7 / −2.2 | bench-43; jar + sandcastle | bench-43, steady, under load (no quiet value): 3.9 → 3.1 / 6.7 fps = 0.79× / 1.70× (3.038 vs 3.83 / 1.79 s/chunk; whole-clip 3.5 / 6.4 fps). Jar + sandcastle (facts §13, one GPU, streaming decode inside, no other generation jobs = quiet-equivalent, VLM judge on another GPU): 2.71 / 3.53 vs teacher 3.80 steady fps | Δ quality +0.00 ± 0.07 / −0.84 ± 0.10; Δ drift +1.9 ± 3.0 / +19.7 ± 3.4 (bench-43) | baseline; accept-all ceiling ≈ 7.6 fps (composed from the τ −2.2 bench-43 per-chunk costs, under load; facts §2) |

Combinations that did **not** compound (closed, one line each):
- Draft-then-Finish: the result stays at the draft's quality level (6.2 vs teacher 7.6).
- Adaptive 576 → 832 switch: degenerates into scene routing. Its gain is proportional to the time spent at 832.
- Resolution distillation at 576 collapsed, and RFT/DPO had no effect.

## 3. Do the levers compound? Same lever measured in two contexts (12×3, fps)

| lever | at 832 (2-step) | at 704 |
|---|---|---|
| fused vs separate commit | 7.98 (quiet) → 11.8 (not recorded) (1.47× ²) | 12.50 → 19.04 (both quiet) (1.52×) |
| + partial commit 1 on fused | 11.8 → 9.7 (0.82×; same s1234 run, not recorded) | 19.04 → 15.73 (both quiet) (0.83×) |
| resolution 832 → 704, holding the commit type | separate 7.98 → 12.50 (both quiet, 1.57×) · fused 11.8 (not recorded) → 19.04 (quiet) (1.62× ²) · fused + partial 9.74 → 15.73 (both quiet, 1.61×) | |

Ratios are of steady s/chunk (832 fused 1.02 and 832 fused + partial 1.24 s/chunk: facts §5, same seed-1234 run; all other values facts §15).

- Each lever's speed-up moves by at most 0.05× across contexts, so **speed-ups multiply.** This is expected because the cost is about linear in tokens × forwards: a KV commit is one full forward, and a partial commit is about 1/3 of a forward.
- **Quality depends on resolution and on the commit type.** The commit variants span 7.28–7.53 at 832 and 6.86–7.22 at 704 (832 fused + partial 1, s1234 only: 7.5). At 704 the jump-free variant (fused + partial 1) costs −0.36 ± 0.13 against separate commit (paired), as much as the 832 → 704 step (−0.31 ± 0.10).
- **The commit shortcut also stacks on chunk-wise adaptive (832; §4).**
  - Speed vs the teacher, quiet: 1.45× (separate commit) → 1.66× (fused + partial 1) or 1.67× (safe commit). That is 2.021 → 1.771 / 1.764 s/chunk, ×1.14 / ×1.15 on top of chunk-wise alone.
  - Bench-43 under load agrees: 1.49× → 1.74× / 1.72× from raw s/chunk (listed 1.51× → 1.77× / 1.74×).
  - Quality, paired arm vs arm (86 clips, facts.md §12): the commit policy costs −0.13 ± 0.07 (fused + partial − separate, 10 better / 56 equal / 20 worse; drift +0.1 ± 3.0) and −0.10 ± 0.06 (safe − separate, 10 / 57 / 19; drift −2.6 ± 2.8). That is borderline (≈1.7–1.9 paired SE, twice as many clips worse as better) and smaller than the 2-step 704 cost (−0.36 ± 0.13). Against the teacher both stay within noise (−0.09 ± 0.06 / −0.07 ± 0.06).
- Lever-by-lever fps bridge from the teacher to ours: **f04_speed_bridge**. The three modes side by side: **f11_modes** (panel a plots bench-43 under-load fps; its title, mode labels and table lead with the quiet ratios 1.67× / 2.39× / 3.85× and give the under-load ratios 1.74× / 2.5× / 4.1× in parentheses; re-rendered 10/12 08:00 CDT).

## 4. Combined configurations: direction 1 and our three operating modes (quality on bench-43, 86 clips per arm; speed from the quiet timing pass, with bench-43 under load as a check)

Direction 1 (facts.md §12) runs the 14B 4-step schedule with chunk-wise adaptive steps. After each step, a learned head (head14b_v3) predicts per-token convergence, and the whole chunk stops once ≥ 70 % (`cw07`) or ≥ 50 % (`cw05`) of its tokens are predicted converged. The chunk is then committed with one of three policies: separate (clean full pass), partial (fused + clean pass on the newest latent frame), or safe (partial if the chunk stopped at step ≥ 3, full clean pass if it stopped at step 1–2). There is no reference pass, so the fps include the head's cost.

| config | levers stacked on the systems stack | mean steps (bench-43 · quiet pass) | steady s/chunk, quiet ↓ | fps, quiet ↑ | speed-up vs teacher, quiet ↑ ³ | bench-43 fps, under load (speed-up, listed) | VLM drift ↓ | VLM quality ↑ | Δ quality vs teacher (paired) ↑ | Δ drift vs teacher (paired) ↓ | boundary jumps > 1.5× ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| teacher 14B 4-step 832 | none | 4.00 · 4.00 | 2.939 | 4.08 | 1.00× | 3.9 (1.00×) | 51.0 ± 3.0 | <u>7.51 ± 0.08</u> | – | – | **1 %** |
| `cw07_sep`: chunk-wise 0.7, separate commit | chunk-wise adaptive (learned head) | 2.91 · 2.94 | 2.021 | 5.94 | 1.45× | 5.9 (1.51×) | <u>50.3 ± 3.1</u> | **7.55 ± 0.08** | **+0.03 ± 0.05** (12 / 65 / 9) | <u>−0.6 ± 3.2</u> | 2 % |
| `cw07_pc1`: chunk-wise 0.7, fused + partial 1 | chunk-wise + fused/partial commit | 3.01 · 3.15 | 1.771 | 6.78 | 1.66× | 6.9 (1.77×) | 50.5 ± 3.2 | 7.42 ± 0.07 | −0.09 ± 0.06 (10 / 59 / 17) | −0.5 ± 3.1 | **1 %** |
| **ours: teacher-parity mode** = `cw07_safe`: chunk-wise 0.7, safe commit | chunk-wise + safe commit (partial 1 if stopped at step ≥ 3, else full clean pass) | 3.01 · 3.10 | 1.764 | 6.80 | 1.67× | 6.8 (1.74×) | **47.7 ± 3.2** | 7.44 ± 0.08 | <u>−0.07 ± 0.06</u> (10 / 60 / 16) | **−3.3 ± 3.1** | **1 %** |
| `cw05_pc1`: chunk-wise 0.5, fused + partial 1 | looser stop + fused/partial commit | 2.61 · 2.47 | 1.495 † | 8.03 † | 1.97× † | 7.0 † (1.79×) | 53.2 ± 3.0 | 7.34 ± 0.09 | −0.17 ± 0.07 (9 / 57 / 20) | +2.2 ± 3.1 | 2 % |
| **ours: quality mode** = `ours832` (ours 832) | 2 steps (0/500) + fused + partial 1 | 2.00 · 2 | <u>1.232</u> (cloud only) | <u>9.74</u> | <u>2.39×</u> | 9.8 (2.5×) | 53.5 ± 3.0 | 7.21 ± 0.08 | −0.30 ± 0.08 | +2.5 ± 3.0 | 2 % |
| **ours: realtime mode** = `ours704` (ours 704) | + 704 cloud (max_area 270336), device = full-VAE decode + bicubic up | 2.00 · 2 | **0.763** (cloud only) | **15.73** | **3.85×** | 15.8 (4.1×) | 56.9 ± 2.7 | 6.90 ± 0.08 | −0.62 ± 0.09 | +5.9 ± 3.4 | 2 % |

Bold marks the best value and underline the second in each ranked column. Mean steps and the bench-43 under-load column are descriptive and not ranked. In the boundary-jump column, three rows tie at 1 % (teacher, `cw07_pc1`, `cw07_safe`), so all three are bold and no cell is underlined. Quiet columns: facts.md §15 (jar + sandcastle, generation only; raw `lingbot-world-v2/specwm_out/timing_pass/<task>/<arm>_s*/loop.json`). All other columns: bench-43. Δ = mean ± SE of the per-clip paired differences vs the teacher over 86 clips. Better / equal / worse counts are in parentheses; facts.md has them only for the chunk-wise arms. The drift differences among the top four rows are all within about 1 SE (drift SE ≈ 3). Bench-43 sources: `paper/1012/bench43_summary.json` (keys teacher, cw07_sep, cw07_pc1, cw07_safe, cw05_pc1, ours832, ours704), per-clip `paper/1012/vlm_scores_bench43.json`, mean steps and chunk timing in `lingbot-world-v2/specwm_out/bench43/*/<arm>_s*/loop.json`. The ours 832 / 704 s/chunk count the cloud GPU only (the teacher and chunk-wise rows count the whole per-chunk loop). Gen s per 10 s clip (bench-43, under load): teacher 35.8, ours 832 16.1, ours 704 10.7; not reported for the chunk-wise arms. Figure: **f11_modes** (panel a plots bench-43 under-load fps; its title, mode labels and table lead with the quiet ratios 1.67× / 2.39× / 3.85× and give the under-load ratios 1.74× / 2.5× / 4.1× in parentheses; re-rendered 10/12 08:00 CDT).

† `cw05_pc1` speed cells are not a stable speed. With fused + partial commit, the per-chunk time is set by the stop step: ≈ 1.24 / 1.75 / 2.26 s at step 2 / 3 / 4, the same in both passes. A median over clips therefore picks the modal stop step.
- Quiet: the 6 clip medians fall into two modes. 3 clips are at 1.23–1.26 s/chunk (stop at step 2) and 3 at 1.74–1.77 s (step 3). The median of 6, 1.495 s, is the midpoint between the two modes.
- Bench-43: 36 of 86 clip medians are in 1.0–1.5 s and 49 in 1.5–2.0 s, so the median 1.722 s lands in the step-3 mode.
- On a mean basis (mean over clips of the per-clip mean of diff(chunk_start)[5:], recomputed from the loop.json files named in facts §12 / §15), `cw05_pc1` is faster than `cw07_pc1` in both passes: bench-43 1.554 vs 1.739 s/chunk (teacher 2.984), quiet 1.436 vs 1.810 (teacher 2.941). The two passes do not really disagree.

What direction 1 shows:
- **Chunk-wise adaptive alone (`cw07_sep`) keeps teacher quality on 43 prompts:** Δ quality +0.03 ± 0.05 (12 better / 65 equal / 9 worse), Δ drift −0.6 ± 3.2, at 1.45× quiet (bench-43 under load: 1.49× from raw s/chunk; 2.91 mean steps). This matches the 12×3 result (0.00 ± 0.11 on 34 clips, §1).
- **The commit policy adds speed at a small, borderline quality cost.**
  - Speed, quiet: fused + partial (`cw07_pc1`) gives 1.66× and safe commit (`cw07_safe`) 1.67×, from 1.45× for `cw07_sep`. Bench-43 under load agrees: 1.74× / 1.72× from 1.49× (raw s/chunk).
  - Paired against `cw07_sep` (86 clips), quality drops by −0.13 ± 0.07 (10 / 56 / 20; drift +0.1 ± 3.0) and −0.10 ± 0.06 (10 / 57 / 19; drift −2.6 ± 2.8): borderline, ≈1.7–1.9 paired SE.
  - Against the teacher both stay within noise: −0.09 ± 0.06 (10 / 59 / 17) and −0.07 ± 0.06 (10 / 60 / 16). Boundary jumps stay at 1 %, the teacher's level.
- **Teacher-parity mode = `cw07_safe`** (facts.md §12: `cw07_safe`, or `cw07_pc1`).
  - It costs no speed against `cw07_pc1`: quiet 6.80 vs 6.78 fps (bench-43 under load 6.8 vs 6.9).
  - It falls back to a full clean commit for chunks that stopped at step 1–2.
  - It has the lowest drift of all arms (47.7; Δ −3.3 ± 3.1), but that is within about 1 SE.
- **The looser stop (`cw05_pc1`) saves time, but its median speed is not a stable number, and it costs quality.**
  - Steps: 2.47 (quiet) and 2.61 (bench-43); in the steady chunks 5–11 of bench-43, 2.62 vs 3.00 for `cw07_pc1`.
  - Per-chunk time is set by the stop step (≈ 1.24 / 1.75 / 2.26 s at step 2 / 3 / 4 for fused + partial), and a median over clips picks the modal step (footnote †). The quiet 1.495 s/chunk (8.03 fps, 1.97×) is the midpoint between 3 clips at step 2 and 3 clips at step 3; the bench-43 1.722 s/chunk lands in the step-3 mode. Neither is a speed any clip runs at.
  - On a mean basis (mean of per-clip means of diff(chunk_start)[5:]) it is faster than `cw07_pc1` in both passes: bench-43 1.554 vs 1.739 s/chunk, quiet 1.436 vs 1.810. So the two passes do not really disagree. The other arms' median timings match across the passes within ~1 % (2.021 vs 2.045, 1.771 vs 1.748, 1.764 vs 1.764, 1.232 vs 1.225, 0.763 vs 0.761 s/chunk).
  - Quality: −0.17 ± 0.07 (9 / 57 / 20; more than 2 SE), drift +2.2 ± 3.1.
  - Not used as a mode: the quality cost exceeds 2 SE.
- **Mean steps:** on bench-43, 3.01 with fused/partial or safe commit and 2.91 with separate commit. These are slightly higher than in the 10/8 runs (2.9), because exits are counted per chunk on 43 different prompts (facts.md §12). The quiet pass on jar + sandcastle gives 3.15 / 3.10 / 2.94 (`cw07_pc1` / `cw07_safe` / `cw07_sep`) and 2.47 (`cw05_pc1`).
- **The three modes trade speed for quality in steps:** teacher-parity 1.67× at −0.07 ± 0.06 → quality 2.39× at −0.30 ± 0.08 → realtime 3.85× at −0.62 ± 0.09.
  - These are quiet steady speed-ups; the quality and realtime fps count the cloud GPU only.
  - Bench-43 under load: 1.72× / 2.48× / 3.99× from raw s/chunk (listed 1.74× / 2.5× / 4.1×).
  - Only the teacher-parity mode stays within noise of the teacher on quality; the quality mode's drift (+2.5 ± 3.0) is also within noise.

What the speed columns measure:
- **Quiet steady fps (preferred; facts.md §15)** = 12 / steady s/chunk in the quiet timing pass. **The headline uses this column.**
  - Setting: jar + sandcastle, 2 lanes on 2 GPUs with no other generation jobs (VLM judges ran on other GPUs). Generation only: decode is excluded from the timed interval. The teacher and chunk-wise arms ran with no decode at all; the ours / 2-step arms streamed the full-VAE decode on the same GPU in the same runs, but cloud_chunk_s times the cloud step only. These runs also give the facts §13 streaming rows.
  - Excluded clips: the per-task warm-up clip. For the res_split arms (ours 832 / 704, 704 separate) castle s1234 is excluded instead, because it paid a cold compile.
  - Teacher and chunk-wise arms: median over clips of median(diff(chunk_start)[5:]), i.e. the whole per-chunk loop. Ours: median of cloud_chunk_s[6:], i.e. the cloud GPU only.
- **Bench-43 under load (check; facts.md §12)** = the same definitions on the 86 bench-43 clips with 7 GPU jobs running. Teacher 3.038 s/chunk (3.9 fps listed, 3.95 raw).
  - Ours and the chunk-wise arms (except `cw05_pc1`, whose median is not a stable speed, †) take nearly the same s/chunk in both passes, while the teacher is 3 % slower under load (3.038 vs 2.939). So the under-load ratios of the three modes run 3–4 % higher than the quiet ones.
- **Not like-for-like for ours (both passes).**
  - The device step runs inside the same per-chunk loop (device_chunk_s median 0.122 s/chunk on bench-43) but is left out, i.e. assumed to run in parallel on a separate device (facts.md §1/§6).
  - Counted serially on the same GPU (measured only under load, bench-43): 1.348 / 0.884 s/chunk → 8.9 / 13.6 fps (2.25× / 3.44× vs 3.038 s/chunk; facts.md §12).
  - End to end on one GPU with the streaming full-VAE decode competing (facts.md §13; jar + sandcastle; whole-clip output fps, not steady): realtime 10.8, quality 6.9, vs the optimized teacher 4.2. Ours n = 5 (jar ×3 + castle ×2; castle s1234 excluded for a cold compile); teacher n = 6 from the sdvg_profile run. Both ran with no other generation jobs (quiet-equivalent; VLM judge on another GPU).
- **The teacher's 4.9 fps** (README §N: 10 prompts, seed 1234) = 153 frames / 31.2 s, a whole-clip average that includes the cheap early chunks before the KV window fills. It is not a steady fps, and no ratio here uses it.
- **Gen s** = median whole-clip generation seconds on bench-43 (under load). It includes chunk 0 and the warm-up chunks but excludes prepare and decode (teacher: + 8.6 s full-VAE decode + 6.3 s prepare). Its ratios are lower: ≈2.2× (35.8 / 16.1) and ≈3.3× (35.8 / 10.7).
  - Reason 1: gen s includes chunk 0 and the warm-up chunks.
  - Reason 2: for ours it also includes the device step (gen s ≈ cloud s + device s, e.g. 10.66 ≈ 9.19 + 1.45 s at 704, medians; facts.md §12).
- Cloud sum (bench-43): ours 832 = 14.6 s, ours 704 = 9.2 s.

³ How the speed-up column is built:
- **Quiet (in the table):** teacher 2.939 s/chunk ÷ the arm's steady s/chunk (facts.md §15). This gives 1.45× / 1.66× / 1.67× / 1.97× † / 2.39× / 3.85× for `cw07_sep` / `cw07_pc1` / `cw07_safe` / `cw05_pc1` / ours 832 / ours 704 († median of two modes, not a stable speed; see the † note under the table).
- **Bench-43 under load (check):**
  - Listed: rounded fps ÷ 3.9 (facts.md §12 convention) → 1.51× / 1.77× / 1.74× / 1.79× / 2.5× / 4.1×.
  - Raw: teacher 3.038 s/chunk ÷ 2.045 / 1.748 / 1.764 / 1.722 / 1.225 / 0.761 s/chunk → 1.49× / 1.74× / 1.72× / 1.76× / 2.48× / 3.99×. Source: `lingbot-world-v2/specwm_out/bench43/*/{uniform4,cw07_sep,cw07_pc1,cw07_safe,cw05_pc1,pc1_832,pc1_704}_s*/loop.json`.
  - The raw ratios are within 5 % of the quiet ones for every arm except `cw05_pc1` (1.76× vs 1.97×), whose two medians fall in different stop-step modes (†); on a mean basis it is faster than `cw07_pc1` in both passes.
- **Ours 832 / 704 count the cloud GPU only** in both passes (cloud_chunk_s); the device step that runs in the same loop is assumed to run in parallel on a separate device (facts.md §1/§6). Counted serially on one GPU (under load only): 1.348 / 0.884 s/chunk → 8.9 / 13.6 fps → 2.25× / 3.44× (facts.md §12). The teacher and chunk-wise ratios already count the whole loop on one GPU.
- **Withdrawn:** the "against the quiet teacher" ratios 1.4× / 2.0× / 3.2× in earlier versions. They divided under-load steady fps by the teacher's whole-clip 4.9 fps, a different basis on a different prompt set (facts.md §2/§15).

By scene group (bench-43, quality ↑ / drift ↓):

| group | teacher quality ↑ | teacher drift ↓ | ours 832 quality ↑ | ours 832 drift ↓ | ours 704 quality ↑ | ours 704 drift ↓ |
|---|---|---|---|---|---|---|
| easy scenes (5 prompts, n = 10) | **8.00** | **21.5** | <u>7.90</u> | <u>25.0</u> | 7.60 | 36.5 |
| tabletop & physics (38 prompts, n = 76) | **7.45** | **54.9** | <u>7.12</u> | <u>57.2</u> | 6.80 | 59.6 |

Drift is given to one decimal from `paper/1012/vlm_scores_bench43.json`. facts.md rounds these to 22 / 25 / 36 and 55 / 57 / 60. Per-group numbers for the direction-1 arms (incl. the teacher-parity mode) are not in facts.md.

- Ours 832 stays within 0.10 (easy) and 0.33 (tabletop) quality of the teacher.
- Per clip, ours 704's drift increase is larger on easy scenes: +15.0 (21.5 → 36.5, n = 10) against +4.7 on tabletop & physics (54.9 → 59.6, n = 76). But tabletop & physics has 76 of the 86 clips, so it accounts for most of the overall +5.9 (about 4.2 of the 5.9 points).
- Full comparison with the student, SDVG, Matrix-Game 2.0 and Waypoint: **f05_pareto_bench43** and **f06_paired_deltas**. Resolution curve: **f08_lowres_curve** (incl. 448 and 384; no fps).

## 5. Sensitivity ranking (one slide)
- **Free:**
  - Systems stack: 1.46× at 4 steps, 2.98× for the full ladder; the blind test was not worse.
  - Chunk-wise adaptive (learned head, frac 0.7): 1.45× (quiet; bench-43 under load 1.49× from raw s/chunk). Quality on both prompt sets:
    - 12×3: Δ quality 0.00 ± 0.11 and Δ drift −0.4 ± 3.1 vs the teacher on the 34 shared clips.
    - Bench-43 (`cw07_sep`): Δ quality +0.03 ± 0.05 (12 / 65 / 9) and Δ drift −0.6 ± 3.2 over 86 clips.
- **Small, borderline cost on top of chunk-wise:** the commit policy.
  - Speed: fused + partial (`cw07_pc1`) raises the speed-up to 1.66× and safe commit (`cw07_safe`) to 1.67× (quiet). Bench-43 under load, raw s/chunk: 1.49× → 1.74× / 1.72×.
  - Quality (bench-43): −0.10 ± 0.06 (safe) / −0.13 ± 0.07 (fused + partial) vs `cw07_sep`, paired, 86 clips (< 2 SE); still within noise of the teacher (−0.07 ± 0.06 / −0.09 ± 0.06). Boundary jumps stay at 1 % (teacher 1 %).
  - Chunk-wise + safe commit is **ours: teacher-parity mode**.
- **Cheap:** steps 4 → 2 gives 1.95× (quiet) for −0.11 ± 0.10 quality and +5.4 ± 4.3 drift (12×3, paired).
- **Costly:**
  - Fused + partial 1 at 2 steps (the jump-free variant): net 1.26× at 704 (quiet) for −0.36 ± 0.13 quality (paired, 36 clips). Fused alone at 832: 1.47× (mixed conditions, §1 ²) for −0.25 ± 0.10 (paired). Fused alone also gives boundary jumps > 1.5× at 12 % of boundaries at 704, including hard scene cuts in a few clips.
  - On chunk-wise (≈3 steps) the same shortcut gives less speed (×1.14–1.15, quiet) and costs less quality: −0.13 ± 0.07 (fused + partial) / −0.10 ± 0.06 (safe) vs `cw07_sep`. That is borderline (≈1.7–1.9 paired SE) vs chunk-wise alone and within noise of the teacher (above). Both are paired arm vs arm, but on different prompt sets and resolutions (12×3 at 704; bench-43 at 832).
  - Resolution: 704 gives 1.57× (quiet) for −0.31 ± 0.10, and 576 gives 2.69× (mixed conditions, §1 ²) for −0.69 ± 0.12. 448 costs −1.36 quality / +7.4 drift, and 384 costs −2.47 quality / +16.1 drift (vs 832 2-step, 12×3; no quiet fps, diag-run cloud times only, s4).
  - For scale, 576 fused + partial runs at 25.66 fps quiet, 6.28× the teacher (facts.md §15); facts.md has no quality score for that exact arm.
  - TAEHV: only pays on one shared GPU, 1.53× (composed from the under-load 0.761 s/chunk cloud, facts §6) for −0.56 (judge 448) / −1.06 (judge 832).
- **Not adopted:** the looser chunk-wise stop (`cw05_pc1`, frac 0.5).
  - Quality: −0.17 ± 0.07 (9 / 57 / 20; more than 2 SE) and +2.2 ± 3.1 drift.
  - Speed: it stops earlier (2.47 steps quiet, 2.61 bench-43) and is faster than `cw07_pc1` on a mean basis in both passes (bench-43 1.554 vs 1.739 s/chunk, quiet 1.436 vs 1.810). Its median speed is not a stable number: per-chunk time is set by the stop step (≈ 1.24 / 1.75 / 2.26 s at step 2 / 3 / 4), so the quiet median 1.495 s (1.97×) is a midpoint between two modes and the bench-43 median 1.722 s sits in the step-3 mode (§4 †).
- **No speed or negative:**
  - 1.3B refiner: 1.00×, −0.50.
  - Verifier + redraw: ×2 time, no gain.
  - Sequence parallel: 1.03× (bf16 comm) / 1.10× (fp8 comm, lossy); < 1× at 576.
  - Commit offload: 6/6 clips hallucinated.
  - SDVG at the paper τ: 0.79× on the teacher's steady basis (bench-43, under load). On jar + sandcastle (facts.md §13: one GPU, streaming full-VAE decode on the same GPU, no other generation jobs) it runs at 2.71 vs the teacher's 3.80 steady fps (decode included; not comparable with the generation-only 4.08).
- **Combined = our three operating modes** (speed: quiet steady, teacher 4.08 fps; quality: bench-43, 86 paired clips; **f11_modes**: panel a plots bench-43 under-load fps; its title, mode labels and table lead with the quiet ratios 1.67× / 2.39× / 3.85× and give the under-load ratios 1.74× / 2.5× / 4.1× in parentheses):

| mode | config | fps, quiet ↑ | speed-up vs teacher, quiet ↑ | bench-43 fps, under load (speed-up, listed) | Δ quality vs teacher (paired) ↑ | Δ drift vs teacher (paired) ↓ |
|---|---|---|---|---|---|---|
| ours: teacher-parity mode | `cw07_safe`: chunk-wise adaptive 0.7 + safe commit | 6.80 | 1.67× | 6.8 (1.74×) | **−0.07 ± 0.06** | **−3.3 ± 3.1** |
| ours: quality mode | `ours832`: 832 2-step fused + partial commit | <u>9.74</u> | <u>2.39×</u> | 9.8 (2.5×) | <u>−0.30 ± 0.08</u> | <u>+2.5 ± 3.0</u> |
| ours: realtime mode | `ours704`: 704 2-step fused + partial commit | **15.73** | **3.85×** | 15.8 (4.1×) | −0.62 ± 0.09 | +5.9 ± 3.4 |

Bold = best, underline = second within this table; the bench-43 under-load column is a check and not ranked. Speed comes from the quiet timing pass (facts.md §15: jar + sandcastle, steady s/chunk, generation only; teacher 2.939 s/chunk = 4.08 fps). The bench-43 under-load ratios (raw s/chunk: 1.72× / 2.48× / 3.99×) agree within ~5 %. The quality and realtime fps count the cloud GPU only (in both passes). With the device step run serially on the same GPU they are 8.9 / 13.6 fps (2.25× / 3.44×), measured only under load on bench-43 (footnote ³).

### Caveats
- All 12×3 arms share the same 36 prompt–seed pairs, so the single-lever Δs are paired (chunk-wise: 34). The bench-43 deltas vs the teacher are paired over 86 clips. Single-lever deltas below 2 paired SE (quality SE 0.10–0.20, drift SE 2.1–5.3) are within noise.
- Direction 1 (bench-43), commit-policy cost paired arm vs arm (facts.md §12, from `paper/1012/vlm_scores_bench43.json`, 86 pairs): `cw07_pc1` − `cw07_sep` −0.13 ± 0.07 quality (10 / 56 / 20), +0.1 ± 3.0 drift; `cw07_safe` − `cw07_sep` −0.10 ± 0.06 quality (10 / 57 / 19), −2.6 ± 2.8 drift. Borderline (≈1.7–1.9 paired SE) vs chunk-wise alone; within noise of the teacher.
- The teacher-parity mode's drift gain (−3.3 ± 3.1) is about 1 SE; read it as "no worse than the teacher", not as better.
- Timing:
  - Speed and quality come from different prompts. The speed-ups are quiet steady ratios from the timing pass on jar + sandcastle (facts.md §15), while quality comes from 12×3 or bench-43.
  - The same arms on bench-43 under load give raw ratios within 5 %: 1.49× / 1.74× / 1.72× / 2.48× / 3.99× vs 1.45× / 1.66× / 1.67× / 2.39× / 3.85×. The exception is `cw05_pc1` (1.76× vs 1.97×): per-chunk time is set by the stop step, and the two medians fall in different stop-step modes (§4 †). On a mean basis it is faster than `cw07_pc1` in both passes (bench-43 1.554 vs 1.739, quiet 1.436 vs 1.810 s/chunk).
  - Teacher fps: 4.08 quiet / 3.9 under load on the steady definition (3 % apart). The "4.9 fps quiet" in older notes and figures is a whole-clip average (153 frames / 31.2 s generation, whole clip, including the cheap early chunks before the KV window fills; README §N, 10 prompts). It is not comparable to any steady fps, and the earlier 1.4× / 2.0× / 3.2× "against the quiet teacher" are withdrawn.
  - Ours 832 / 704 fps count the cloud GPU only in both passes; the device step in the same loop (median 0.122 s/chunk on bench-43) is assumed to run in parallel on a separate device (facts.md §1/§6). Serially on one GPU: 8.9 / 13.6 fps (2.25× / 3.44×), measured only under load. End to end on one GPU with streaming decode: 10.8 / 6.9 output fps (whole-clip; facts.md §13; ours n = 5: jar ×3 + castle ×2, castle s1234 excluded for a cold compile; teacher 4.2 n = 6 from the sdvg_profile run; both with no other generation jobs, VLM judge on another GPU). The teacher and chunk-wise fps count the whole loop on one GPU.
  - The partial-commit cost at 704 is now measured with both runs quiet: 0.630 → 0.763 s/chunk (0.83×; 6 vs 5 clips, ≈ +0.12 s/chunk on the same 5 clips, s2 §b). The older 18.7 (quiet) → 15.8 (under load) comparison mixed conditions and is no longer used.
  - Mixed-condition cells remain in §1/§3 (footnote ²): 832 fused 11.8 and partial 2 13.7 (facts.md §5), and 576 separate 21.5 (README §U), all with no recorded load condition. 704 → 672 is a quiet same-run pair from README V3 (18.7 → 22.4), a different run from facts.md §15.
  - The 12×3 chunk-wise 1.33× used before (10/10 run, whole-clip 4.9 → 6.5 fps) is superseded by the quiet 1.45×.
- How the speed-up ratios were computed:
  - Quiet: teacher 2.939 s/chunk ÷ the arm's s/chunk (facts.md §15). Between two non-teacher arms: the ratio of their steady s/chunk.
  - Bench-43, listed: rounded under-load fps ÷ 3.9 → 1.51× / 1.77× / 1.74× / 1.79× / 2.5× / 4.1×. Raw: 3.038 s/chunk ÷ the arm's s/chunk → 1.49× / 1.74× / 1.72× / 1.76× / 2.48× / 3.99× (ours cloud only; 2.25× / 3.44× with the device step serial).
  - SDVG on the teacher's steady basis: 3.038 / 3.83 = 0.79× and 3.038 / 1.79 = 1.70× (bench-43, under load). The older ≈0.89× / ≈1.62× divided whole-clip SDVG fps (3.5 / 6.4) by the steady teacher fps.
  - The systems-stack and SP ratios use e2e seconds per clip, not fps.
- The VLM judge sees 8 frames at 448 px. It did not see the chunk-boundary jumps, which is why the jump metric is shown separately. Its quality prompt has not been checked against humans. Judging at 832 px penalizes low-res and TAEHV more.
- Systems-stack quality comes from one blind test of the whole S7 stack, which also includes 2 steps. Compile, FP8 and Sage are not isolated.
- 12×3 chunk-wise row: the raw file has 34 of 36 clips. Missing are jar/castle s1234, where the teacher scored worse than its mean, so only the paired comparison on the 34 shared clips is valid.
- The 832 fused + partial 1 row on 12×3 is a single seed (12 clips). The 832 boundary-jump numbers in §1 (teacher, 832 fused) are s1234 only.
- Verifier + redraw was measured on 6 scenes × 2 seeds, not on a bench.
- Physics: there is no physics metric per lever.
  - The StEvo-Bench subset (facts.md §14; 38 tasks, seed 1234, 249-frame videos, local Qwen3-VL-32B judge, not comparable to the paper's leaderboard) covers whole arms only. Its judge protocol differs from the Setup's 8-frame, 448-px quality/drift judge: 32 frames, 3-vote majority, StEvo criteria control / physics / state.
  - Physical inaccuracy ↓: ours quality mode 0.55, ours realtime mode 0.68, teacher 0.68, SDVG τ −0.7 0.71, Matrix-Game 2.0 0.79, Waypoint 0.92, student 1.00. The s.e. is ≈ 0.05–0.08, so differences < ~0.15 are not significant.
  - State evolution and task success are at the floor for every arm. The teacher-parity mode was not run on StEvo.
- 448/384 have no quiet fps. Their diag runs logged 0.367 / 0.250 s/chunk cloud-only (≈ 32.7 / 48.0 fps, load not recorded; s4 footnote ³, derived, not in facts.md).
- For the direction-1 arms facts.md has no gen s per clip and no per-group scores. Their per-frame latency columns (first frame, action→frame, miss rate) stay n/a, because the timing pass ran them through spec_early without streaming decode, so only chunk-start times exist (facts.md §13). The head's cost is inside their fps (no reference pass).

### 발표 메모 (Korean)
- 기법마다 속도와 품질을 같이 놓고 보면, 속도 향상은 거의 곱해지고 품질은 주로 해상도가 결정합니다. 다만 2-step에서는 fused + partial commit도 704에서 품질을 −0.36 (paired) 깎아서, 해상도 한 단계(832→704, −0.31)와 비슷한 비용입니다.
- 속도 배수는 이제 quiet timing pass(facts §15, jar + sandcastle, steady s/chunk)의 비율을 씁니다. 중요한 정정: 지금까지 인용한 teacher "4.9 fps (quiet)"는 clip 전체 평균(153 frame / 31.2 s, KV window가 차기 전의 싼 초반 chunk 포함)이라 steady 값이 아닙니다. steady 기준 teacher는 quiet 4.08 fps / under load 3.9 fps로 3 %밖에 차이가 안 납니다. 그래서 예전에 말한 "quiet teacher 기준 1.4× / 2.0× / 3.2×"는 철회합니다.
- 공짜 레버는 시스템 스택과 chunk-wise adaptive입니다. chunk-wise는 quiet 1.45×(bench-43 under load 1.49×)이고, 두 프롬프트 세트 모두에서 teacher와 품질이 같습니다(12×3 공통 34개 클립 0.00 ± 0.11, bench-43 86개 클립 +0.03 ± 0.05).
- direction 1 결과: chunk-wise 위에 commit 정책(fused + partial / safe)을 얹으면 quiet 기준 1.45× → 1.66× / 1.67×로 빨라집니다(bench-43 under load, raw s/chunk 기준 1.49× → 1.74× / 1.72×). 품질은 chunk-wise 단독(cw07_sep)과 paired로 비교하면 −0.13 ± 0.07 / −0.10 ± 0.06 (86 clips, 약 1.7–1.9 SE)으로 작지만 경계선 수준의 비용이고, teacher 대비로는 −0.09 / −0.07 ± 0.06으로 noise 안입니다. 경계 점프도 teacher와 같은 1 %.
- 0.5 threshold(cw05)는 median 속도 숫자를 그대로 말하지 않습니다. fused + partial에서는 chunk 시간이 멈춘 step으로 정해지고(step 2 / 3 / 4에서 약 1.24 / 1.75 / 2.26 s), clip들의 median은 가장 많은 step 쪽을 고릅니다. quiet pass의 1.495 s/chunk(1.97×)는 step 2 clip 3개와 step 3 clip 3개의 중간값이고, bench-43의 1.722 s는 step 3 쪽에 떨어진 값입니다. 평균으로 보면 두 측정 모두 cw07_pc1보다 빠릅니다(bench-43 1.554 vs 1.739, quiet 1.436 vs 1.810 s/chunk). 그래도 품질이 −0.17 ± 0.07(2 SE 초과)이라 모드로는 쓰지 않습니다.
- 그래서 모드를 세 개로 정리합니다(속도는 quiet steady, teacher 4.08 fps 기준 / 품질은 bench-43 paired): 'ours: teacher-parity mode'(cw07_safe) 6.80 fps, 1.67×, 품질 −0.07 ± 0.06 · 'ours: quality mode'(ours832) 9.74 fps, 2.39×, −0.30 ± 0.08 · 'ours: realtime mode'(ours704) 15.73 fps, 3.85×, −0.62 ± 0.09. bench-43 under load 비율(raw s/chunk 1.72× / 2.48× / 3.99×, 표기값 1.74× / 2.5× / 4.1×)과 5 % 안에서 일치합니다. 그림은 f11_modes입니다. 패널 a 축은 bench-43 under-load fps이지만, 제목·라벨·표는 quiet 비율 1.67× / 2.39× / 3.85×를 먼저 쓰고 under-load 비율 1.74× / 2.5× / 4.1×는 괄호에 넣었습니다(08:00 CDT 다시 그림). 말할 때도 quiet 비율을 씁니다.
- partial commit의 속도 비용도 이제 같은 quiet 조건에서 쟀습니다: 704에서 0.630 → 0.763 s/chunk(19.04 → 15.73 fps, 0.83×). 단 두 값은 6 vs 5 clip의 median이라, 같은 5 clip끼리는 0.640 → 0.763, 약 +0.12 s/chunk입니다. 예전의 18.7(quiet) → 15.8(under load) 비교는 조건이 섞여 있어서 더 이상 쓰지 않습니다.
- quality / realtime mode의 fps는 cloud GPU만 센 값입니다(device step은 별도 기기에서 병렬로 돈다고 가정). 한 GPU에서 직렬로 돌리면 8.9 / 13.6 fps(2.25× / 3.44×, bench-43 under load에서만 측정)입니다. teacher-parity mode의 drift −3.3 ± 3.1은 1 SE 정도라 "teacher보다 낫다"가 아니라 "teacher만큼"으로 말합니다.
