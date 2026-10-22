# 1012 — results since the 10/7 action-item list (feedback-driven diagnostics)

All 14B runs: LingBot-World v2 14B causal-fast, 832x480, FP8 + SageAttention + ring KV, 4-step schedule (timesteps 0/250/500/750), 1 GPU, same code path
(`specwm/spec_early.py`). "Error vs dense 4-step" = mean per-token gap ||x0 − x0_dense4|| / σ (σ = median token norm of the dense output) with the
SAME noise and context, i.e. how far a sampler's chunk is from what the 4-step sampler would have produced there. Scenes: jar, castle, urban × seeds 4321/9999
unless noted. Times are 14B denoising seconds per 9.6 s clip (commit ≈ 6 s and prepare/decode excluded).

## A. Dense baselines incl. the missing uniform 3-step (item 1 of the feedback)
| Arm | Steps / token | Error vs dense 4 ↓ | 14B denoise (s) | gen incl. commit (s) |
|---|---|---|---|---|
| uniform 4-step (reference) | 4.00 | 0 | 25.0 | 31.1 |
| **uniform 3-step** | 3.00 | **0.162** | 18.3 | 24.3 |
| uniform 2-step | 2.00 | 0.222 | 12.2 | 18.3 |
| early exit ε 0.24 (learned head, sparse K/V reuse) | 2.38 | 0.219 | 17.3 | 23.4 |

Dense 3-step dominates the current early exit: same time, 0.162 vs 0.219 error. (Steady 5 prompts, seed 4321: early 2.32 steps / 36.7 s e2e vs uniform 2 32.9 / uniform 3 — see all_runs.csv.)

## B. Where the early-exit error comes from (feedback 5.1: oracle / learned × dense recompute / sparse reuse)
| Arm | Steps | Error ↓ (p90) | Denoise (s) |
|---|---|---|---|
| oracle mask + dense recompute (no speed-up) | 2.15 | **0.159** (0.176) | 24.4 |
| learned mask + dense recompute | 2.30 | 0.170 (0.197) | 24.3 |
| oracle mask + sparse K/V reuse | 2.14 | 0.234 (0.277) | 15.6 |
| learned mask + sparse K/V reuse (= early exit) | 2.38 | 0.219 (0.252) | 17.3 |

- The stop decision is fine (oracle + recompute reaches 3-step error at 2.15 steps; the head adds only +0.011).
- **Reusing the K/V of frozen tokens is the loss** (0.159 → 0.234). Not an attention-kernel artifact: SageAttention does give 3–7 % different outputs when queries are a subset of keys, but switching sparse steps to flash attention left the error unchanged (0.234).

## C. Attempts to reduce the K/V-reuse loss (items 1–2)
| Variant (sparse, learned unless noted) | Steps | Error ↓ | Denoise (s) | vs dense 3-step (0.162 @ 18.3 s) |
|---|---|---|---|---|
| no exit before step 2 | 2.61 | 0.185 | 18.6 | worse |
| no exit before step 3 | 3.17 | 0.155 | 22.4 | slightly better error, 4 s slower |
| last-step refresh (final step dense, x0 kept) | 2.37 | 0.217 | 18.4 | worse (refresh at the end cannot fix the intermediate steps) |
| min-exit 2 + last-step refresh | 2.64 | 0.182 | 20.3 | worse |
| oracle, min-exit 2 / 3 | 2.57 / 3.18 | 0.188 / 0.149 | 18.4 / 22.4 | |
| **1.3B model**, oracle + dense recompute vs + sparse reuse | 3.08 / 3.27 | 0.113 / 0.181 | 5.7 / 5.7 | the same loss on the model Speculative Forcing reports on |

No variant beats the dense time–error frontier. The loss is inherent to token freezing with stale K/V (also on the 1.3B), so VBench-level metrics in the paper did not see it.

## D. Head operating point (item 3)
False-exit rate P(true gap > ε | stopped) at ε 0.24 = 16 % (p90 of stopped tokens' gap 0.26, p99 0.38); per step 17 / 8 / 4 %. "Predicted depth 2.18 ≈ oracle 2.15" means the same amount stopped, not the same tokens.

## E. Noise calibration of the cross-model gap (item 4; 6 scenes, same context)
| Proposal for the 14B 4-step output | Mean gap ↓ | Within ε 0.24 | On 14B-easy tokens | On 14B-hard tokens |
|---|---|---|---|---|
| 14B with a different noise (intrinsic variability) | 0.405 | 34 % | 60 % | 10.6 % |
| 1.3B draft, same z0 | 0.607 | 7 % | 13 % | 0.8 % |
| 14B own step-1 estimate | 0.308 | 44 % | – | – |

The draft is much farther from the target than another 14B sample of the same chunk, especially where the 14B needs more steps. Correct wording: "under the shared-noise latent-gap criterion the draft's acceptable share (7 %) is far below the 14B's own noise variability (34 %)"; it is a statement about paired verification, not about output distributions.

## F. Figures and videos
- fig_cost_closeness.png — cost (14B forwards / token) vs error to the 4-step output for every sampler and variant; the dashed line is the 14B's own different-noise variability.
- fig_paths_dino_<scene>_s4321.png — rollout paths in DINOv2 CLS space (shared PCA), similarity to the first frame, similarity to the target frame at the same time. fig_paths_<scene>_s4321.png — the same in VAE-latent PCA (less readable).
- videos/<scene>__seed<s>__u2-u3-early-u4.mp4 — uniform 2 | uniform 3 | early exit | uniform 4, side by side.
- all_runs.csv — every run (arm, task, seed, steps, times, error).

## G. Status of the remaining items
- StEvo-Bench (85 physics tasks, 253 frames, seed 1234; judge Qwen3-VL-32B 3-vote majority; modified protocol — compare only within our runs):

| Arm | Observation ctrl % ↑ | Action ctrl % ↑ | Control success % ↑ | Physics plausible % ↑ | State progress % ↑ | Task success % |
|---|---|---|---|---|---|---|
| 14B uniform 2-step | 63.5 | 76.5 | 54.1 | 38.8 | 1.2 | 0.0 |
| 14B early exit 0.24 (2.3 steps) | 65.9 | 78.8 | 56.5 | 48.2 | 1.2 | 1.2 |
| 14B uniform 4-step | 56.5 | 72.9 | 49.4 | 41.2 | 1.2 | 1.2 |
| 14B chunk-wise adaptive 0.7 (2.9 steps) | 67.1 | 72.9 | 55.3 | 42.4 | 4.7 | – |
| (Sept run) 14B official 4-step | 62.4 | 70.6 | 50.6 | 32.9 | 2.4 | 0.0 |
| (Sept run) 1.3B draft | 42.4 | 71.8 | 32.9 | 1.2 | 0.0 | 0.0 |

  One seed, 85 tasks: differences of ~10 points are within noise (uniform 4 < uniform 2 on physics shows the spread). Read as "acceleration preserves physics/control", not as an improvement. Per-task judgements: specwm_out/stevo/runs/<arm>/.
- Chunk-wise adaptive steps (all-or-nothing per chunk, no K/V freezing): queued for the next free GPU.
- Blind set (80 clips) at paper/1006_early_exit/blind_1006; a 3-step arm should be added before rating.

## H. 10/8 — chunk-wise adaptive steps, anchor split, branch test, best-of-N (all jar/castle/urban(/lake) × seeds 4321/9999)
### H1. Chunk-wise adaptive steps (same learned head; a chunk stops when >= frac of its tokens are predicted converged; no token freezing, K/V always fresh)
| Arm | Mean steps | Error vs dense 4 ↓ | 14B denoise (s) |
|---|---|---|---|
| uniform 2 / 3 / 4 | 2 / 3 / 4 | 0.222 / 0.162 / 0 | 12.2 / 18.3 / 25.0 |
| token early exit (learned, sparse) | 2.38 | 0.219 | 17.3 |
| **chunk-wise learned, frac 0.7** | 2.92 | **0.146** | **17.8** |
| chunk-wise learned, frac 0.9 | 3.41 | **0.070** | 20.7 |
| chunk-wise learned, frac 0.5 | 1.99 | 0.221 | 12.1 |
| chunk-wise oracle, frac 0.7 / 0.5 | 2.56 / 1.83 | 0.164 / 0.233 | 15.5 / 11.1 |
First adaptive variant on or above the dense frontier (3-step error at 85–97 % of its time; 0.07 error at 3.4 steps). Specwm/early_exit.py `chunk_frac`.

### H2. Anchor split (cloud = 14B step 1, device = 1.3B finishes) and velocity guidance
| Proposal | Gap to 14B final ↓ (σ) | Within ε 0.24 ↑ | Device steps | Cloud forwards |
|---|---|---|---|---|
| 1.3B draft alone | 0.527 | 8 % | 4 | 0 |
| velocity guidance w 0.5 / 1.0 at step 1 | 0.458 / 0.398 | 11 / 16 % | 4 | 1 |
| anchor + 1.3B 3 steps | 0.399 | 16 % | 3 | 1 |
| **anchor + 1.3B last 2 steps** | **0.345** | **26 %** | 2 | 1 |
| anchor only (14B step 1) | 0.302 | 43 % | 0 | 1 |
| floor: 14B finishing its own anchor, other re-noise / 14B different noise | 0.291 / 0.332 | 47 / 40 % | 0 | 4 |
One cloud forward + two device steps lands at the 14B's own noise variability. specwm/anchor_probe.py.

### H3. Branch test (future-state impact of one approximated chunk; prefix dense, 3 dense chunks after)
| Branch at c0 | Local gap ↓ | Future gap f=1/2/3 ↓ | Future DINO cos to A ↑ |
|---|---|---|---|
| A′ noise-only floor | 0.290 | 0.40 / 0.46 / 0.50 | 0.78 / 0.77 / 0.75 |
| oracle mask + dense recompute | 0.158 | 0.18 / 0.26 / 0.35 | 0.94 / 0.92 / 0.87 |
| learned + sparse (token early exit) | 0.205 | 0.25 / 0.34 / 0.43 | 0.91 / 0.88 / 0.85 |
| uniform 2-step | 0.223 | 0.30 / 0.38 / 0.48 | 0.85 / 0.82 / 0.78 |
Future divergence is monotone in the local gap; no evidence (at this scale) of future impact beyond the local error. specwm/branch_probe.py.

### H4. Best-of-N 1.3B candidates (2-step, N up to 8)
Oracle-best gap to the 14B final improves only 0.52 → 0.49 σ (same or independent z0); candidate diversity (0.30–0.40 σ) is smaller than the distance to the target. Anchor distance predicts the final gap with r = 0.99 (e_T: 0.79). specwm/candidates_probe.py.

## I. 10/8 — absolute verifiers and the per-scene distillation pilot
### I1. Is a model's own denoising error a drift detector? (240 human-labelled clips, balanced severe / not; teacher-forced context)
| Score (per chunk → clip by max / mean) | Clip AUC vs severe drift, max ↑ | mean ↑ |
|---|---|---|
| 14B denoising error e_T, σ 0.3 / 0.5 / 0.7 | 0.60 / 0.57 / 0.56 | 0.52 / 0.55 / 0.55 |
| 1.3B denoising error e_D, σ 0.3 / 0.5 / 0.7 | 0.50 / 0.49 / 0.48 | 0.44 / 0.46 / 0.46 |
| 14B anchor distance d1 (step-1 estimate vs the chunk) | 0.51 | 0.56 |
| (ref) learned drift critic (human labels) | 0.84 | |
Density-style scores from either model do not see what human raters call drift (AUC ≈ chance); as an *absolute* (human-plausibility) verifier they need supervision. Against the TARGET they are moderately informative (velocity_probe, 288 chunks, Spearman with the gap between the draft's and the 14B's final chunk): velocity disagreement ‖v_T−v_D‖ 0.65, 14B denoising error e_T 0.64, 1.3B's own error e_D 0.61 (free), contrast log(e_T/e_D) 0.0 — all far below the anchor distance d0 (0.95, same 14B cost) and below the learned cross head (0.75, free). So velocity is a usable but dominated verifier; the open question is making d0 cheaper (early-layer / low-res anchor, cheap_anchor_probe.py). specwm/denoise_err_labels.py.

### I2. Per-scene paired distillation of the 1.3B (LoRA r64 on attn+FFN, 4-node x0 matching with the 14B's own noise sequence; jar + castle, 16 train seeds each, 1,152 steps, 90 min)
| Held-out seed 7 | Mean gap to 14B final ↓ (σ) | Within ε 0.24 ↑ | On 14B-easy tokens | On 14B-hard tokens |
|---|---|---|---|---|
| 1.3B stock (6-scene calibration) | 0.607 | 7 % | 13 % | 0.8 % |
| **1.3B + scene LoRA, jar/castle** | **0.483** | **18 %** | 35 % | 2.7 % |
| 1.3B + the same LoRA, unseen scenes (urban, lake) | 0.476 | 12 % | 18 % | 2.4 % |
| floor: 14B different noise (jar/castle) | 0.415 | 36 % | 64 % | 11.9 % |
Paired alignment improves (7 → 18 %) but stays well above the 14B's own variability; the gain carries partly to unseen scenes. Blur must be checked before this counts (earlier MSE LoRAs blurred). specwm/draft_distill4.py, noise_calib.py --d-lora.
Blur check (fig_lora_blur_check.jpg; pure-draft videos, seed 7): castle (train scene) — hallucinations gone, sharp, 14B-like; jar (train scene) — the 14B's room/wall drift mode but smeared textures; urban (unseen) — ghosted/blurred. The paired-gap gain does not yet translate into usable video outside one scene; needs a perceptual/distribution term and more data. Videos: specwm_out/lora_blur/.

### I3. Target-based verifiers, cheaper than a full 14B step (6 scenes × 2 seeds, 144 chunks; ρ = Spearman with gap(draft final, 14B final))
| Signal | Cost (14B layers / 40) | Pooled ρ | Within-clip ρ mean (min) |
|---|---|---|---|
| anchor distance d0 (14B step-1 estimate vs draft) | 40 | 0.97 | 0.87 (0.62) |
| ridge anchor from layer-30 / 20 / 10 / 5 hidden | 30 / 20 / 10 / 5 | 0.91 / 0.87 / 0.39 / 0.32 | – |
| logit lens (untrained head) L5–30 | 5–30 | 0.13–0.29 | 0.25 |
| draft stability ‖x̂0^(2)−x̂0^(1)‖ (free) | 0 | 0.86 | 0.55 (−0.30) |
| (velocity_probe, 288 chunks) ‖v_T−v_D‖ / e_T / e_D / log(e_T/e_D) | 40 / 40 / 0 / 40 | 0.65 / 0.64 / 0.61 / 0.0 | – |
| (cross head on draft hidden) | 0 | 0.75 | – |
d0 is the only signal that ranks chunks within a rollout; free signals mostly encode scene difficulty. Early layers (<20) carry no usable anchor; at 20–30 the saving is ≤ 50 %. In the anchor-split design the cloud computes d0's forward anyway, so verification is free there. specwm/cheap_anchor_probe.py.

## J. 10/8 PM — anchor-split pipeline, 10-seed path figures, where a 14B step goes, three levers, commit offload

### J1. Anchor-split 2-GPU pipeline (`specwm/spec_pipe.py --anchor`, `exp/queue/anchor_pipe.sh`, out `specwm_out/anchor_pipe`)
cloud GPU = 14B step 1 (anchor) + verify (token gap to the anchor) + commit; device GPU = 1.3B last `--anchor-dsteps` steps from the anchor. 6 scenes × seeds 1234/4321/9999; steady clips (the first clip of every scene recompiles, excluded).

| arm | e2e ↓ median (13-chunk scenes) | rain / lake | accepted | gap to anchor (σ) | T_anchor / T_commit / wait device / 14B denoise (s) |
|---|---|---|---|---|---|
| target-only 2-step (14B) | 21.6 | 27.1 / 27.4 | – | – | 0 / 6.6 / 0 / 13.2 |
| **anchor + 1.3B 1 step** | **18.2** | 22.1 / 22.7 | 100 % | 0.245 | 6.3 / 6.6 / 1.6 / 1.3 |
| *anchor + 1.3B 2 steps* | *19.3* | 23.5 / 24.0 | 100 % | 0.333 | 6.3 / 6.6 / 2.8 / 1.3 |
| anchor + 1.3B 2 steps, τ 0.45 | 19.2 | 23.6 / 24.0 | 97.7 % | 0.331 | 6.3 / 6.6 / 2.8 / 1.8 |

Quality (anchor_probe + `hmode1`, `specwm_out/anchor_probe_d1`, jar/castle × 4321/9999, gap to the 14B 4-step output, same noise & context):

| proposal | token gap ↓ (σ) | within ε ↑ | device 1.3B steps | cloud 14B fwd |
|---|---|---|---|---|
| 1.3B draft | 0.586 | 4 % | 4 | 0 |
| anchor + 1.3B 2 steps | 0.382 | 19 % | 2 | 1 |
| anchor + 1.3B 1 step | 0.353 | 29 % | 1 | 1 |
| anchor only (14B step 1, blurry) | **0.334** | **37 %** | 0 | 1 |
| 14B, different noise (floor) | 0.385 | 35 % | 0 | 4 |

Caveat: the token gap rewards blur (the step-1 estimate is closest of all). Sharpness (Laplacian variance of decoded frames, jar / castle): 14B 4-step 456/277, anchor+2 steps 326/184, anchor+1 step 167/176 ≈ 14B 2-step 192/181. Frame grid: `fig_anchor_pipe_frames.jpg` (frames 30, 54). On jar s4321 the 14B 4-step itself loses the jar by frame 30 and teleports to a building by frame 54 while the 1.3B keeps the jar — the 14B is an unstable reference on these seeds.

### J2. 10-seed path figures (`scripts/viz_paths10.py`, `fig_paths10_<scene>.png`)
Seeds 1234, 4321, 9999, 11, 22, 33, 44, 55, 66, 77 (new arms via `exp/queue/paths10.sh`). Left: shared PCA, thin = seeds, thick = per-frame mean over seeds. Right: relative latent distance to the same-seed 14B 4-step frame (mean ± s.e.), grey dashed = 14B 4-step vs 14B 4-step with another seed (90 pairs). x-axis fixed: 3 latent frames per chunk (the old figure divided by 12 and showed 3.3 chunks).

| arm | jar ↓ | castle ↓ |
|---|---|---|
| 14B 4-step, different seed (reference's own spread) | 0.981 | 0.891 |
| 1.3B draft only | 0.936 | 0.971 |
| 14B uniform 2-step | 0.875 | 0.752 |
| 14B uniform 3-step | *0.822* | *0.689* |
| ours: token early exit (2.3 steps) | **0.805** | 0.695 |
| ours: chunk-wise adaptive (2.9 steps) | 0.841 | **0.690** |

Reading: rollouts are chaotic — the reference's own different-seed distance reaches ≈1.0 by chunk 3–4, so same-seed fidelity is only resolvable in the first 3–4 chunks; there ours ≈ 3-step < 2-step, draft diverges from chunk 1. On castle the draft stays above the grey band to the end (different distribution, not different noise).

### J3. Where a 14B step goes (`specwm/step_bench.py`, chunk 4 of jar, 24-frame window, FP8 + compile)

| forward | active tokens | time per chunk ↓ | vs dense |
|---|---|---|---|
| dense step, Sage | 100 % | **0.467 s** | 1.00 |
| dense step, flash | 100 % | 0.550 s | 1.18 |
| sparse step 75 / 50 / 25 / 10 % (flash) | | 0.409 / 0.317 / 0.137 / 0.070 s | 0.87 / 0.68 / 0.29 / 0.15 |
| KV commit (clean pass) | 100 % | 0.469 s | 1.00 |

Step cost is ≈ linear in active tokens (attention is a minor share); early exit's cost is simply its mean step count (2.55) × 1.1 (flash) + 1 commit per chunk. Real levers: exit earlier, Sage on sparse steps, and the commit (= one full forward per chunk, the floor of every 1-GPU sampler).

### J4. Three levers (jar/castle × 4321/9999)
A. Commit overlap quality (`specwm/overlap_probe.py`, chunks 3/6/9): step 1 of chunk c+1 on chunk c's last-denoise K/V, steps 2–4 clean → gap **0.291σ, 57 % within ε** vs 14B noise floor 0.318 / 46 %; all-stale (skip commit) 0.403 / 27 %. Jar chunk 6 is above the floor (0.47–0.54), castle always below.
B. Early exit ε sweep (sparse, flash): ε 0.32 → 1.87 steps, gap 0.266 (dense 2-step: 2.00, 0.222); ε 0.40 → 1.52 steps, 0.315 (dense 1-step 0.296). Token early exit is below the dense frontier at every ε → closed.
C. Sage on sparse steps (`SPECWM_SAGE_SPARSE=1`): gap 0.217 → 0.213, denoise 18.1 → 17.2 s. Free; keep on.

### J5. Commit offload on 2 cloud GPUs (`specwm/spec_commit_pipe.py`, `exp/queue/commit_pipe.sh`, out `specwm_out/commit_pipe`)
rank 0 denoises (K uniform steps, never runs the clean pass), rank 1 commits every finished chunk and ships its K/V ring slots (0.96 GB/chunk, NCCL over PCIe, GPUs 2/3). jar/castle × 3 seeds, steady clips:

| 14B sampler | gen s (13 chunks) ↓: 1 GPU | 2 GPUs, commit offloaded | wait for K/V | chunk-1 gap to the 1-GPU run (same noise) |
|---|---|---|---|---|
| uniform 2-step | 18.5 | **13.0** | 0.1 | 0.413 (0.224 with `--sync-first 1`: the image chunk 0 committed locally) |
| uniform 3-step | 24.6 | **19.0** | 0.1 | 0.42 |
| uniform 4-step | 31.4 | **25.1** | 0.1 | 0.40 |

The commit (6 s/clip) is hidden completely; transfer is invisible. e2e in this script includes a non-streaming decode (~15 s) — compare gen. Later-chunk gaps (0.5–0.9) are AR divergence; the clean per-context measure is J4-A.

### J6. Can the commit be hidden in the anchor split? No (`specwm/anchor_stale_probe.py`, out `specwm_out/anchor_stale`, jar/castle × 4321/9999, chunks 3/6/9)
In the anchor pipeline the cloud's only forward on chunk c is the σ=1 anchor, so hiding commit(c) means running anchor(c+1) on that σ=1 K/V:

| chunk c+1 variant | gap to 14B 4-step ↓ (σ) | within ε ↑ | gap to the clean-K/V version |
|---|---|---|---|
| 14B anchor on clean K/V (today's pipeline) | **0.338** | **38 %** | – |
| 14B anchor on the σ=1-stale K/V of chunk c | 0.648 | 2 % | 0.592 |
| anchor clean + 1.3B 2 steps | 0.385 | 20 % | – |
| anchor stale + 1.3B 2 steps | 0.638 | 4 % | 0.531 |
| 14B 4-step, different noise (floor) | 0.365 | 43 % | – |

Worse than the 1.3B draft alone (0.586). The AR chain anchor(c) → device(c) → commit(c) → anchor(c+1) is strict, so the anchor split's cloud floor is 2 full forwards per chunk (0.94 s, 1.25× realtime). Commit offload only works where the cloud ran a late denoise step (J4-A/J5): cloud-only 2-step with offloaded commit = 2 forwards on the critical path, 13.0 s gen, gap 0.222 — it dominates the anchor split (17.1 s gen, 0.382) on both axes; the anchor split's remaining merit is cloud compute per chunk (2 vs 3 forwards), not latency.

### J7. Commit offload breaks rollouts (videos, not metrics) — `fig_offload_vs_1gpu_frames.jpg`, `fig_offload_stalecopy_frames.jpg`
jar/castle × 3 seeds, frames 30/66/120, 1-GPU 2-step vs offloaded 2-step: the offloaded arm injects people, cars, bottles in **6/6 clips** (the 1-GPU arm drifts to buildings/beaches but never to people). Same with `--stale copy` (chunk c-1's clean K/V copied into chunk c's slots for step 1): 6/6 again, now with prompt-relevant objects (jars, sandcastles + waves) plus people. Any stand-in for chunk c at step 1 (noisy K/V or a duplicated chunk at the wrong position) reads as a scene cut → the model falls back on the text and invents content. The per-chunk token gap (J4-A: 0.29σ, inside the 14B noise floor) did not catch this: it is a one-chunk, same-context statistic, and the failure is semantic and cumulative. → Commit offload closed; the cloud floor stays at K+1 full forwards per chunk. Low-res is re-run with synchronous commits (J8).

### J8. Low-res cloud with synchronous commits (`spec_commit_pipe.py --sync-first 99`, 14B 2-step, jar/castle × 4321/9999, out `specwm_out/commit_pipe_lr{576,448}` + `_sr`)

| cloud resolution | gen ↓ (13 chunks) | per chunk | vs realtime 0.75 s | SR (Real-ESRGAN x4v3) | videos (`fig_lowres_sync_frames.jpg`) |
|---|---|---|---|---|---|
| 832×480, 1 GPU | 19.9 s | 1.53 s | 2.0× | – | drifts to buildings / beaches (no people) |
| **576×320 + SR** | **6.9 s** | **0.53 s** | **0.71× (realtime)** | 4.1 ms/frame (0.6 s/clip) | same drift pattern, softer texture, 0/4 people |
| 448×256 + SR | 4.6 s | 0.35 s | 0.47× | 2.5 ms/frame | 1/4 clips with a person (jar s4321 f66) |
| (offloaded commit at 576, J7 failure mode) | 5.2 s | 0.40 s | – | – | people in most clips |

Forward cost is ≈ linear in tokens, so 576×320 (0.44× the tokens) puts the dense 2-step under realtime with the commit kept synchronous — the first configuration that is realtime without touching the AR chain. Open: a quality measure at low res (the 14B head / drift critic are trained at 832×480), and whether 3-step at 576 (≈ 10 s, 1.0× realtime) looks better than 2-step at 832.

### J9. A drift meter for low-res clips (`specwm/critic_lowres_eval.py`, `critic_lowres_control.py`; out `specwm_out/critic_lowres{,_ctrl}`)
Full-res latent drift critic (drift/noprev_noplanrc, human severe-drift labels, canonical 56×104 latent grid; flag = logit > 0.05). Routes: generated latents as they are, or decoded video → VAE re-encode (for low-res: Real-ESRGAN to 832×480 first).

Control (full-res clips degraded exactly as the low-res pipeline would; any rise = domain shift, not drift):

| full-res clip, degraded | mean logit | chunks flagged | Δ vs original |
|---|---|---|---|
| original 832×480 (VAE round trip; latent↔video Spearman 0.98) | 0.29 | 38 % | 0 |
| ↓576 → VAE at 576 (latent route) | 0.69 | 41 % | +0.39 |
| ↓576 → SR → VAE at 832 (SR route) | 0.22 | 38 % | **−0.08** |
| ↓448 → VAE at 448 (latent route) | 1.17 | 45 % | +0.88 |
| ↓448 → SR → VAE at 832 (SR route) | 0.21 | 38 % | **−0.09** |

→ the SR route is shift-free; use it. Scores (jar/castle × 3 seeds, SR route for low-res, video route for 832):

| arm | mean logit ↓ | chunks flagged ↓ | clips ≥1 flagged |
|---|---|---|---|
| 1.3B draft 832 | **−1.56** | **19 %** | 67 % |
| 14B 2-step 832 | *0.39* | *40 %* | 67 % |
| 14B 4-step 832 | 2.06 | 55 % | 100 % |
| 2-step offload 832 (J7) | 2.13 | 55 % | 100 % |
| 2-step 576 sync + SR | 3.02 | 65 % | 100 % |
| 2-step 576 offload + SR | 3.42 | 69 % | 100 % |
| 2-step 448 sync + SR | 4.29 | 73 % | 100 % |

`fig_critic_lowres_frames.jpg` (frames at chunk centres with logits): the critic fires exactly when the original scene is gone (jar → blue building, beach → grass/room) and stays low while the beach persists; at 576 the scene is lost by chunk 3 in 3/4 clips vs later at 832. So the 576 realtime configuration drifts earlier than 832 2-step. Note the 14B 4-step reference itself is flagged more than 2-step on these seeds (it teleports), and the 1.3B is the most scene-stable — the critic agrees with the frame grids, not with "closer to the 14B = better".

### J10. More steps at 576 do not buy the drift back (`exp/queue/lowres_steps.sh`, critic SR route, jar/castle × 3 seeds)

| arm | gen ↓ (13 chunks) | per chunk | chunks flagged ↓ | mean logit ↓ |
|---|---|---|---|---|
| 14B 2-step 832 | 19.9 s | 1.53 s | **40 %** | **0.39** |
| 14B 3-step 832 | 27.3 s | 2.10 s | *51 %* | *1.17* |
| 14B 4-step 832 | 34.8 s | 2.68 s | 55 % | 2.06 |
| 2-step 576 + SR | **6.9 s** | **0.53 s** | 65 % | 3.02 |
| 3-step 576 + SR | 9.2 s | 0.71 s | 77 % | 4.62 |
| 4-step 576 + SR | 11.5 s | 0.88 s | 67 % | 3.72 |

Resolution, not step count, sets the drift level: at 576 every step count sits at 65–77 % flagged (vs 40–55 % at 832), and at both resolutions more steps do not reduce the flagged share (n = 6 clips, ±10 pts noise). The 14B appears to lose the scene more readily when it renders more detail. So the realtime 576 configuration costs roughly +25 pts of flagged chunks on these two scenes, and no sampler change at 576 recovers it — the fix has to come from the model side (e.g., per-resolution fine-tune, or a low-res-aware drift verifier + redraw) or from keeping the cloud at 832 and cutting cost elsewhere.

### J11. 576 vs 832 on the 5 steady prompts (`exp/queue/lowres_prompts.sh`, 14B 2-step, seeds 1234/4321)
Timing: 832 gen 18.0–18.4 s, 576 gen 6.8–6.9 s (same ratio as jar/castle). The drift critic is **not usable here**: it was trained on jar/castle clips (test-arm AUC 0.63) and flags 86 % (832) / 92 % (576) of chunks with logits 6–7 on every prompt — out of distribution, not drift. By eye (`fig_prompts_832_vs_576.jpg`, frames 6/54/102/150): both resolutions keep the scene on rain, stonehenge, pouring and lake; domino grows standing people in **both** arms; 576 is softer but not more drifted. So the +25 pt drift penalty of 576 seen on jar/castle does not show on the easy prompts — the realtime 576 configuration is scene-dependent, which is exactly what a scene-adaptive resolution policy would exploit, but it needs a verifier that generalizes across scenes; the current latent critic does not (known since 10/6).

## K. 10/9 — SOTA speed check vs. ours (see `sota_comparison.md`, video `videos/sota_cmp10.mp4`)
Faster open models exist: Waypoint-1.5-1B 30 fps at 1024x512 (5.4 s / 153 frames) and Matrix-Game 2.0 20.5 fps at 640x352 (7.5 s), vs ours 6.9 s at 576x320 (+SR) and the 14B teacher 31 s; both are 1–2 B game-trained keyboard/mouse models that render photos as game worlds and do not take text or camera trajectories (Matrix-Game 3.5 does, 5 B, 20 fps claimed, needs 40 GB of weights — not run, disk). Same 10 prompts / camera paths (converted and flow-calibrated), timings re-verified in fresh processes.
Also: 10-prompt × 8-arm comparison video `videos/cmp10.mp4` (section J table in the chat log), VLM drift judge: clip-level AUROC 0.82 on 32 unseen scenes (chunk-level 0.66, 0.27 s/call).

### K2. 10-seed path figures on 4 easier scenes (`fig_paths10_easy_lake.png`, `_easy_stonehenge_sunset`, `_easy_urban_rain`, `_pouring_water_into_cup`; arms now include anchor split and 576+SR)
Mean relative latent distance to the same-seed 14B 4-step frame (all frames, 10 seeds) ↓:

| arm | lake | stonehenge | urban rain | pouring |
|---|---|---|---|---|
| 14B 4-step, different seed (reference's own spread) | 0.712 | 0.639 | 0.874 | 1.047 |
| 1.3B draft | 0.932 | 0.783 | 0.898 | 1.254 |
| 14B 2-step | 0.591 | 0.539 | 0.789 | *0.965* |
| 14B 3-step | **0.503** | **0.469** | **0.704** | 0.963 |
| ours: token early exit (2.3 steps) | 0.531 | 0.491 | *0.761* | **0.958** |
| ours: chunk-wise adaptive (2.9 steps) | *0.524* | *0.482* | 0.786 | 0.966 |
| ours: anchor split (2 GPU) | 0.718 | 0.593 | 0.830 | 1.017 |
| ours: 576×320 + SR (re-encoded; carries a ~0.37 SR/VAE re-encode offset already at the shared first frame — not comparable) | 0.973 | 0.822 | 1.019 | 1.148 |

On these scenes the reference is stable enough that the comparison is informative over the whole clip: all 14B same-resolution arms (2/3-step, early exit, chunk-wise) stay *below* the teacher's own different-seed spread for all 13 chunks (they are closer to the same-seed teacher than another teacher sample is), the anchor split sits on the different-seed curve (teacher-level in the weak sense), and the draft is above it from chunk 1. The low-res arm's latent distance is dominated by the SR→VAE re-encode offset and must be judged by video/critic instead.
### K3. Matrix-Game 3.5 (same inputs as ours: image + camera path + text) — see `sota_comparison.md` §5
171.8 s per 169-frame clip at 1280×704 (0.9 fps; denoise alone ≈ 3.6 fps) vs ours 6.9 s; follows the camera path but drifts to game/corridor scenes within 1–2 s on photos and tabletop scenes. Tier-1 deletions done 10/9 (gap_dump_v3, gap_dump_cross, stevo/reground, loop1005fix, plandata(2), e1repro, agree_main; +43 GB).

## L. 10/9 PM — scene-general drift verifier (direction 1) + resolution split + SP
### L1. Verifiers on human severe-drift labels, scene-disjoint
`specwm/dino_drift_eval.py` (DINOv2-L first-frame cosine), `specwm/drift_feat_dump.py` + `drift_head_train.py` (14B hidden features from the commit pass, 3×5 grid + mean pooling, MIL head, 2-fold over the 32 non-jar/castle scenes), out `specwm_out/dino_drift`, `exp/drift_feat` (620 clips), `specwm_out/drift_head`.

| verifier | input / cost | AUROC on scenes never trained on ↑ | AUROC jar/castle held-out seeds ↑ |
|---|---|---|---|
| latent critic (10/6) | latents, 10 ms | 0.63 | 0.84–0.98 |
| VLM clip-level (Qwen3-VL-32B) | 8 frames, 0.4 s/clip | 0.82 | 0.71 |
| DINOv2 zero-shot, min cosine to frame 0 | decode + 85 ms/chunk | 0.82 (any-drift 0.90) | 0.70 |
| **14B hidden head, layer 20** | free at commit, ~1 ms | **0.88** (0.86 / 0.91) | **0.89** |
| 14B hidden head, layer 30 | | 0.84 | 0.84 |
| 14B hidden head, L20+L30 | | 0.89 | 0.84 |

Epoch chosen on the held-out fold (optimistic by ≲0.02; confirm with a 3-way split). Hand-made internal signals (velocity, anchor gap) could not see drift (AUC 0.5–0.6); a head on hidden states with external labels can.

### L2. Resolution split refiner (`reslora_train.py`, `res_split_infer.py`; data `res_data` 32 scenes × 2 seeds; `reslora/{r900,r850,r850_930}`)
Held-out relative error vs the 14B 832 chunk: upsampled input 0.084 → 1-node σ 0.357: 0.033, σ 0.468: 0.038, 2-node σ 0.468→0.273: **0.027** (2,088 steps). Pipeline on 1 GPU: cloud 6.8 s + device 2.8 s (1-node) / 4.3 s (2-node) per clip. Visually (`fig_refiner_crops.jpg`, `fig_ressplit_frames.jpg`) the refiner adds detail but leaves a fine grid texture on water/walls that Real-ESRGAN does not have; not yet a clear win. Figure: `fig_resolution_split.png/.svg`, site `paper/1012/site/`.

### L3. Sequence parallel (2 GPUs, PCIe): 14B 2-step 832 18.5 → 18.0 s (Sage) / 16.8 s (fp8 comm); at 576 7.4 s (slower than 1 GPU). September's 12.0 s not reproduced; NCCL needed an NVML preload (`tools/nvml584`, driver 595.84 vs userspace 595.99 until reboot). Closed on this machine.

### L4. Verify + redraw with the drift head (`specwm/spec_verify.py`, out `specwm_out/verify`; 6 scenes × seeds 4321/9999, 14B 832, τ = −3.25 = p95 of chunk logits on clean clips)

| arm | gen s ↓ | chunks still flagged after (of 12) ↓ | mean head logit chunks 1–10 ↓ | redrawn chunks |
|---|---|---|---|---|
| 14B 2-step, scored only | 19.9 | 6.8 | 0.24 | 0 |
| 14B 4-step, scored only | 33.0 | 6.6 | 0.38 | 0 |
| 2-step + verify → resample (2 tries) | 44.1 | 6.8 | −0.09 | 7.0 |
| 2-step + verify → dense 4-step redraw | 41.3 | 6.8 | 0.31 | 6.9 |

Per scene the head orders scenes as expected (urban rain −7 < lake −5 < pouring 1.7 < castle 2–5 < domino 3.7 < jar 6) and fires on 0/12 (rain), 2–4/12 (lake), 9–10/12 (jar, castle, pouring). **Redrawing a flagged chunk — with fresh noise or with the full 4-step sampler — does not lower the score**: once the scene starts to go, every alternative chunk the 14B draws from that context is also flagged (the drift lives in the model + prompt, not in one unlucky chunk; 4-step drifts as much as 2-step). The verifier is a detector, not a repair, and chunk-level rejection sampling doubles the time for nothing. Position artifact to fix: the last two chunks always score very low (−8…−13), so a per-chunk-index threshold is needed. Next: (i) backtracking test (reject → roll back k chunks) to close the question, (ii) use the head as a training signal (direction 5) rather than an inference-time gate on these scenes, (iii) as an inference gate it still pays on mild scenes (lake) and as a scene-level switch.

## M. 10/10 — scalability: the recipe on a second base model (Matrix-Game 2.0, 1.8B causal, keyboard/mouse; `tools/mg2/run_mg2.py --res --steps`)
Same 10 prompts / camera paths. Patched the repo's hard-coded 880 tokens-per-frame (`FSEQ`) so other resolutions run. Drift measured label-free with DINOv2 (`scripts/dino_score_videos.py`, cos to frame 0, min over chunks).

| model / config | gen s ↓ (153 frames) | fps ↑ | DINO min cos ↑ | chunks < 0.5 ↓ |
|---|---|---|---|---|
| LingBot 14B 4-step 832 | 31.2 | 4.9 | 0.297 | 59 % |
| LingBot 14B 2-step 832 | 18.5 | 8.3 | 0.283 | 51 % |
| LingBot 14B 2-step 576 + SR | 6.9 | 22 | 0.170 | 52 % |
| LingBot 1.3B 4-step 832 | 14.7 | 10.4 | 0.238 | 65 % |
| MG2 3-step 352×640 (as released) | 7.5 | 20.5 | 0.149 | 67 % |
| MG2 2-step 352×640 | 6.5 | 23.6 | 0.136 | 70 % |
| MG2 3-step 256×448 | 5.3 | 28.8 | 0.100 | 72 % |
| MG2 2-step 256×448 | 4.5 | 34.3 | 0.114 | 75 % |

Transfer: the two speed levers apply unchanged (steps 3→2: −13 %; resolution 0.51× tokens: −30 %; both: −40 %, 34 fps). MG2 gains less than the 14B because its per-step denoise is small next to fixed costs (eager VAE decoder, CLIP, action modules). Quality trend also replicates: fewer steps barely change drift, lower resolution loses the scene earlier (DINO min 0.149 → 0.100, same pattern as LingBot 0.283 → 0.170). Not yet transferred: the drift head (needs labels on MG2 rollouts → VLM pseudo-labels) and the resolution-split refiner (LoRA on MG2 itself as its own device model).

### M2. Drift head transferred to Matrix-Game 2.0 (`specwm/mg2_head_train.py`; data `specwm_out/mg2/head_data` = 36 scenes × 2 seeds + 10 prompts, labels = VLM clip drift score ≥ 70, `paper/1012/vlm_scores_mg2.json`)

| MG2 features (context pass, blocks 10 / 20, 1536-d) | AUROC vs VLM label, held-out scenes ↑ | Spearman vs VLM score ↑ | AUROC on the 10 comparison prompts ↑ |
|---|---|---|---|
| block 10 | 0.77 | 0.34 | 0.90 |
| block 20 | 0.76 | 0.36 | 0.90 |
| both | 0.73 | 0.25 | 0.85 |

Same head, same pooling, no hand-tuning; 84 clips and VLM labels (themselves 0.82 vs humans), so noisier than the LingBot result (0.88 on 620 human-labelled clips), but the recipe (hidden states + external labels → scene-general head) transfers.

## N. 10/10 — concrete baseline matrix (10 prompts, seed 1234, same start image + camera path; `paper/1012/vlm_scores.json`, `scripts/dino_score_videos.py`, `scripts/vlm_clip_score.py`)
TTFF = per-request prepare + first chunk (computed from steady gen/13; prepare 2.0 s LingBot with T5 on GPU, 2.6 s for 2-GPU arms, 1.5 s 1.3B, 0.5 Waypoint, 1.6 MG2, 8 MG3.5). VLM = Qwen3-VL-32B clip judge (drift 0–100 ↓, quality 1–10 ↑). Sharpness (Laplacian variance) is listed but not comparable across renderers: game textures and the refiner's grid noise inflate it.

| arm | group | resolution | GPUs | gen s ↓ | fps ↑ | TTFF s ↓ | VLM drift ↓ | VLM quality ↑ | DINO min cos ↑ |
|---|---|---|---|---|---|---|---|---|---|
| 14B 4-step (teacher) | teacher | 832×480 | 1 | 31.2 | 4.9 | 4.4 | 40 | 7.7 | 0.297 |
| 14B 3-step | cloud-only | 832×480 | 1 | 24.6 | 6.2 | 3.9 | 44 | 7.7 | 0.283 |
| 14B 2-step | cloud-only | 832×480 | 1 | 18.5 | 8.3 | 3.4 | 42 | 7.5 | 0.283 |
| **ours: chunk-wise adaptive** | cloud-only | 832×480 | 1 | 23.7 | 6.5 | 3.8 | **36** | **7.8** | 0.316 |
| ours: token early exit | cloud-only | 832×480 | 1 | 22.2 | 6.9 | 3.7 | 40 | 7.6 | **0.332** |
| **ours: 14B 2-step 576 + ESRGAN** | cloud-only | 576→832 | 1 | 7.7 | 19.9 | 2.6 | 60 | 6.2 | 0.170 |
| hybrid: anchor split | hybrid | 832×480 | 1+1 | 18.0 | 8.5 | 4.0 | 43 | 7.0 | 0.245 |
| hybrid: 576 cloud + 1.3B refiner (2-node) | hybrid | 576→832 | 1+1 | 10.1 | 15.2 | 3.4 | 64 | 5.7 | 0.123 |
| edge: 1.3B 4-step | edge-only | 832×480 | edge | 14.7 | 10.4 | 2.6 | 54 | 6.6 | 0.238 |
| edge: 1.3B 2-step 576 + ESRGAN | edge-only | 576→832 | edge | 5.4 | 28.2 | 1.9 | 76 | 5.0 | 0.193 |
| edge SOTA: Waypoint-1.5-1B | edge-only | 1024×512 | edge | 5.4 | 28.3 | 0.9 | 50 | 5.5 | 0.562* |
| edge SOTA: Matrix-Game 2.0 (3-step) | edge-only | 640×352 | edge | 7.5 | 20.5 | 2.2 | 57 | 7.2 | 0.149 |
| edge SOTA: Matrix-Game 2.0 + our recipe (2-step 448) | edge-only | 448×256 | edge | 4.5 | 34.3 | 1.9 | 72 | 5.8 | 0.114 |
| cloud SOTA: Matrix-Game 3.5 distilled | cloud-only | 1280×704 | 1 | 171.8 | 0.9 | 21.2 | 68 | 6.4 | 0.076 |

\* Waypoint barely moves on long camera sweeps (under-rotates), which keeps its first-frame similarity high.

Reading: (1) at 832 the teacher-level claim holds — chunk-wise adaptive and early exit match or beat the 4-step teacher on every quality column at 1.3–1.4× speed; (2) the realtime configuration (576 + ESRGAN, 20 fps) costs +20 VLM drift points and 1.5 quality points vs the teacher, and still beats every edge-only model that runs at a similar frame rate on drift (60 vs 72–76) except Waypoint's quasi-static 50; (3) the learned refiner currently *hurts* (64 / 5.7 vs 60 / 6.2 for ESRGAN) — the grid noise and its own K/V drift outweigh the added detail, so the hybrid row is not yet a win; (4) the two open-weight SOTA models trade scene retention for speed exactly like ours, and Matrix-Game 3.5 is slow and drifts. Missing for a paper: human blind ratings, Waypoint at matched 360p, measured (not computed) TTFF, more seeds.

### N2. Refiner retrained with a pixel loss (`reslora/r850_930_pix`: 2 nodes σ 0.468→0.273, latent MSE + 0.5·(L1 + LPIPS-vgg) on one decoded frame per chunk, input noise 0.05, 4 epochs, val 0.0261)
| arm | gen s | fps | VLM drift ↓ | VLM quality ↑ | DINO min ↑ |
|---|---|---|---|---|---|
| ours: 14B 2-step 576 + ESRGAN | 7.7 | 19.9 | 60 | 6.2 | 0.170 |
| hybrid: 576 + refiner (latent loss only, N) | 10.1 | 15.2 | 64 | 5.7 | 0.123 |
| **hybrid: 576 + refiner (pixel + LPIPS loss)** | 6.8 cloud + 4.3 device (parallel) | ≈ 15–22 | **53** | **6.9** | 0.142 |
| 14B 2-step 832 (reference for the row) | 18.5 | 8.3 | 42 | 7.5 | 0.283 |
The grid noise is gone (`fig_refiner_crops2.jpg`); the hybrid row now beats the ESRGAN realtime row on both VLM columns — first hybrid win. Training lesson: SageAttention has no backward; LoRA training with it active diverges (all three 14B runs did) — use SDPA for training.

### N3. Drift-reduction fine-tune of the cloud model (track A) — negative
Data `specwm_out/rft_data` (36 scenes × 4 seeds, 14B 2-step 576, 4 candidates per chunk scored by the drift head, lowest kept): best-of-4 lowers the mean chunk logit 1.67 → 0.56 but the flagged share only 0.79 → 0.77 (selection among 4 rarely finds a clean continuation on drifting scenes — same lesson as L4). LoRA self-distillation toward the kept candidates (`drift_rft_train.py`, rank 32, SDPA, lr 3e-5, val 0.036 → 0.031) and Diffusion-DPO (winner vs worst candidate; diverged twice, the loser term is unbounded) gave:

| arm (10 prompts) | VLM drift ↓ | VLM quality ↑ | DINO min ↑ |
|---|---|---|---|
| 576 base, bilinear upsample (no refiner) | – | – | 0.189 |
| 576 + RFT LoRA, bilinear | 64 | 3.6 | 0.189 |
| 576 + pixel refiner | 53 | 6.9 | 0.142 |
| 576 + RFT LoRA + pixel refiner | 54 | 6.9 | 0.176 |

No drift reduction from the fine-tune. With this data the post-training route is closed; drift at 576 remains a model/data-level property. Pitfall log: SageAttention has no backward (training diverged until switched to SDPA); DPO on x0-errors needs a bounded loser term.

## O. 10/10 night — (i) scene-adaptive resolution, (ii) mid resolutions, (iii) human blind study
12 prompts (cmp10 + jar/castle), seed 1234, VLM = Qwen3-VL-32B clip judge (drift ↓, quality ↑; n = 12, single seed, noise ≈ ±5 drift points), DINO = min cosine to frame 0. Timing re-measured steady (second pass) after raising torch.compile's recompile limit: the earlier runs had silently fallen back to eager on the second aspect ratio (that was the +25 % on the stevo prompts at 576, 8.6 vs 6.9 s).

| arm | gen s ↓ (steady) | fps ↑ | VLM drift ↓ | VLM quality ↑ | DINO min ↑ |
|---|---|---|---|---|---|
| 14B 4-step 832 (teacher) | 31.2 | 4.9 | 47 | 7.6 | 0.247 |
| 14B 2-step 832 | 18.5 | 8.3 | 47 | 7.5 | 0.240 |
| 576 cloud + pixel refiner | **6.9** | **22.3** | 57 | 6.8 | 0.113 |
| 640 cloud + pixel refiner | ≈ 9.2 | ≈ 16.6 | 56 | 7.1 | 0.171 |
| 704 cloud + pixel refiner | ≈ 11.4 | ≈ 13.4 | 54 | 7.1 | 0.182 |
| adaptive 576→832, τ 4.8 (6/12 switched) | 16.6 (12.0 est. with T5 reuse) | 9.2 | 54 | 7.0 | 0.115 |
| adaptive 576→832, τ 0.64 (8/12 switched) | 20.1 (15.6 est.) | 7.6 | 49 | 7.2 | 0.128 |

(i) Adaptive switch (`res_split_infer.py --switch-tau`): the head at 576 flags every tabletop/physics prompt by chunk 2–5 and no landscape prompt, so the policy degenerates into scene routing (hard scenes → 832 2-step). The switch rebuilds the 14B's 832 K/V from the device's refined 832 chunks (sink + window, ≤ 8 commits); measured cost 11 s, of which ~6 s is re-encoding the prompt with T5 on CPU (reusable) — ≈ 4.5 s without it. Quality moves toward 832 2-step in proportion to time spent at 832; no free lunch.
(ii) Mid resolutions trace the same curve: each step up in resolution buys a few drift points and ~0.3 quality for ~25–60 % more cloud time; the gap to 832 (drift 47) stays.
(iii) Blind pairwise study, every candidate vs the teacher (`scripts/blind_build.py`, page https://claude.ai/artifact/HyFxbLBV5Ab1ouMPnLXXdW, answers in its `ratings` collection; mapping in `paper/1012/blind/mapping.json`, not published): 90 trials = 10 prompts × 9 candidates (14B 2-step, chunk-wise, 576+ESRGAN, 576+refiner, adaptive τ 0.64, 704+refiner, 1.3B edge, Matrix-Game 2.0, Waypoint). Raters need Contributor access to save.

## P. 10/11 — why does 576 drift more? Early-anchor test (`res_split_infer.py --hi-first K`)
Cloud generates the first K chunks at 832, then steps down to 576 (K/V rebuilt from the 832 chunks resampled to the 576 grid; T5 context cached, step-down cost 3.4–4.1 s) and continues 576 + pixel refiner. 12 prompts.

| arm | share of clip at 832 | cloud s | VLM drift ↓ | VLM quality ↑ | DINO min ↑ |
|---|---|---|---|---|---|
| 576 + refiner (K = 0) | 0 % | 6.9 | 57 | 6.8 | 0.113 |
| K = 1 | 8 % | 10.8 | 64 | 6.7 | 0.131 |
| K = 2 | 15 % | 11.5 | 58 | 6.8 | 0.147 |
| K = 3 | 23 % | 12.5 | 60 | 6.6 | 0.167 |
| K = 5 | 38 % | 14.6 | 51 | 7.0 | 0.194 |
| 832 2-step | 100 % | 18.5 | 47 | 7.5 | 0.240 |

Anchoring the first 1–3 chunks at 832 does not carry over: once the cloud runs at 576 it loses the scene at its own rate, and quality/drift move in proportion to the share of the clip generated at 832 (DINO min rises roughly linearly with K). The extra drift at low resolution is a persistent property of the model at that resolution, not an early layout decision — so no K/V or scheduling trick at inference recovers it; it would take training the cloud model at the low resolution on real video, or a base model that is robust there.

## Q. 10/11 — measured latency, and the decoder is a hidden bottleneck
Per-chunk cloud/device times recorded in `res_split_infer.py` (`cloud_chunk_s`, `device_chunk_s`; 4 prompts × steady pass, `specwm_out/latency_split`). Link: half RTT + 0.3 MB at 100 Mbps. Prepare = 2.0 s (T5 on GPU + image encode).
**The full Wan2.1 VAE decode costs 631 ms per chunk at 832×480 (bf16)** — more than the whole 576 cloud chunk (553 ms) — so with it the device, not the cloud, sets the frame rate. The tiny decoder TAEHV (`taew2_1`, github madebyollin/taehv, `tools/taehv`) decodes the same latents in **19 ms per chunk** (33×), at PSNR 28.3 dB / LPIPS 0.149 against the full VAE.

| config | decoder | link RTT | TTFF ↓ | steady fps ↑ | action → frame p50 ↓ |
|---|---|---|---|---|---|
| 576 cloud + refiner | full VAE | 0 / 100 ms | 3.31 / 3.36 s | 12.1 | 1568 / 1618 ms |
| **576 cloud + refiner** | **TAEHV** | 0 / 50 / 100 ms | **2.70** / 2.72 / 2.75 s | **21.7** | **956** / 981 / 1006 ms |
| 704 cloud + refiner | TAEHV | 0 / 100 ms | 2.94 / 2.99 s | 12.7 | 1351 / 1401 ms |
| 14B 2-step 832, one GPU | TAEHV / full VAE | – | ≈ 3.4 s | 8.3 / 5.8 | ≈ 1.45 / 2.05 s |

All earlier "fps" figures in this README are generation-only (decode excluded or overlapped); the TAEHV rows are the honest end-to-end numbers. Network RTT adds only its one-way half per chunk (latents are 0.3 MB).

Resolution distillation (section R, running): first checkpoint (rank 32, step 408) VLM drift 68 / quality 6.2, DINO min 0.160 vs 576 + refiner 57 / 6.8 / 0.113 — worse on VLM, slightly better on DINO; later checkpoints pending.

## R. 10/11 — 3-seed matrix (12 prompts × seeds 1234 / 4321 / 9999; `paper/1012/vlm_scores_3seed.json`)
VLM = mean over 36 clips (± s.e. across the three seed means); fps = end-to-end with TAEHV decode where the arm decodes Wan latents (section Q), generation-only for the game models (their own decoders).

| arm | GPUs | fps ↑ | VLM drift ↓ | VLM quality ↑ | DINO min ↑ |
|---|---|---|---|---|---|
| 14B 4-step 832 (teacher) | 1 cloud | 4.9 | 45 ± 0.7 | 7.6 | 0.229 |
| **chunk-wise adaptive 832** | 1 cloud | 6.5 | **43 ± 4.5** | **7.7** | **0.234** |
| 14B 2-step 832 | 1 cloud | 8.3 | 51 ± 2.1 | 7.5 | 0.220 |
| 704 cloud + 1.3B pixel refiner | cloud + device | 12.7 | 57 ± 1.3 | 6.9 | 0.154 |
| 576 cloud + 1.3B pixel refiner | cloud + device | 21.7 | 60 ± 1.6 | 6.5 | 0.137 |
| edge: LingBot 1.3B 4-step 832 | device | ≈ 10 | 59 ± 1.0 | 6.6 | 0.202 |
| edge: Matrix-Game 2.0 (640×352) | device | 20.5 | 58 ± 1.3 | 7.1 | 0.130 |
| edge: Waypoint-1.5-1B (1024×512) | device | 28.3 | 58 ± 2.1 | 5.1 | 0.400* |

With three seeds the single-seed "hybrid win" (N2: 53 / 6.9) shrinks: the 576 hybrid is at the level of the edge-only models (drift 58–60), not above them, and Matrix-Game 2.0 scores higher on VLM quality. Teacher-level quality is reached only at 832 (chunk-wise adaptive: equal to the teacher on every column at 1.3×). * Waypoint barely follows long camera sweeps, which inflates first-frame similarity.

## S. 10/11 — resolution distillation (832 behaviour → 576 cloud): negative
`specwm/resdistill_train.py`, data `res_data` (144 14B 832 2-step rollouts, 36 scenes × 4 seeds, downscaled + re-encoded to the cloud's 576 grid), teacher-forced LoRA on the 14B at 576, flow-matching x0 loss on the downscaled 832 chunks. Evaluated per checkpoint (576 cloud + LoRA + pixel refiner, 12 prompts, seed 1234; base row = 58 / 6.8 / DINO 0.113):

| run | step | VLM drift ↓ | VLM quality ↑ | DINO min ↑ |
|---|---|---|---|---|
| rank 32, nodes σ 1.0 + 0.833, lr 3e-5 | 408 / 804 | 68 / 86 | 6.2 / 4.8 | 0.160 / 0.121 |
| rank 64 + FFN, same loss, lr 2e-5 | 408 / 804 | 69 / 86 | 5.8 / 4.8 | 0.147 / 0.091 |
| rank 32, node σ 0.833 only, lr 1e-5 | 408 / 1200 | 60 / 62 | 5.8 / 6.4 | 0.126 / 0.124 |

With σ = 1 in the loss the few-step generator collapses (tabletop scenes turn flat grey: random-noise x0 regression teaches the mean); without it the model stays intact but drift does not move. Without paired noise across resolutions, plain fine-tuning cannot transfer the 832 dynamics into the 576 few-step model. Closed, together with RFT/DPO (N3): post-training on generated data does not fix low-resolution drift in this model.

## T. 10/11 — Draft-then-Finish A/B (closed: finish inherits the draft's world)
`lingbot-world-v2/specwm/draft_finish.py`, outputs `specwm_out/df_ab`. Device 1.3B drafts each chunk at 832 (2 steps, nodes 0/500; its K/V holds the 14B's finished chunks with a one-chunk lag). Cloud 14B re-noises the draft to σ_f and runs ONE x0 forward at 832. Two ways to commit: *separate* adds a clean t=0 commit pass (2 forwards per chunk); *fused* keeps the K/V that the finishing forward wrote (1 forward). 12 prompts, seed 1234, full-VAE video, same Qwen3-VL judge as section R. The restarted judge was re-checked on the baselines: teacher 47 / 7.6 and draft 60 / 6.3, matching the earlier scores of 46.7 / 7.58 and 60 / 6.33.

| arm | cloud forwards / chunk ↓ | cloud s / chunk ↓ | pipelined fps ↑ | VLM drift (0–100) ↓ | VLM quality (1–10) ↑ | DINO min cos ↑ |
|---|---|---|---|---|---|---|
| 14B 4-step teacher (ref) | 5 | — | 4.9 (e2e) | **47** | **7.6** | — |
| 1.3B draft alone (ref) | 0 | — | ≈10 | 60 | <u>6.3</u> | — |
| DF fused σ_f 0.625 | 1 | **0.51** | **23.6** | 60 | 6.2 | <u>0.195</u> |
| DF separate σ_f 0.625 | 2 | 1.02 | 11.8 | <u>55</u> | 6.2 | **0.201** |
| DF fused σ_f 0.468 | 1 | **0.51** | **23.5** | 56 | 6.2 | 0.169 |
| DF separate σ_f 0.468 | 2 | 1.02 | 11.8 | 58 | 6.2 | 0.160 |
| DF fused σ_f 0.273 | 1 | **0.51** | **23.6** | 57 | 5.9 | 0.149 |

Pipelined fps = 12 frames / max(cloud, device + TAEHV), measured on one GPU for both roles.
- **Fused commit works**: across σ_f, fused and separate commits give the same quality (6.2 vs 6.2) and drift within noise. Reusing the noisy-input K/V does not cause the hallucinations that commit offload did, so 1 cloud forward per chunk is enough. This is a reusable finding.
- **Quality stays at the draft's level (6.2 vs draft 6.3; goal ≥ 7.2)**, including the separate-commit upper bound. Frames (`pouring_water`, `domino`) show that the finished video follows the draft's trajectory: same camera path, same scene breakup. The 14B re-texture does not restore the teacher's world. Even at σ_f 0.625, the layout is already fixed by the 1.3B.
- By the pre-set decision rule (A < 7.2 → close), the direction is closed. If the draft–finish split is revisited, the cloud must own the high-σ steps (layout), i.e. the reverse split, which is the low-res plan + device refiner already measured in section R (576: 21.7 fps, quality 6.5).

## U. 10/11 — fused commit on the 14B's own sampler and on the low-res split (seed 1234, 12 prompts)
`res_split_infer.py --fused [--cloud-nodes]`: drop the clean commit pass and keep the K/V written by the last denoising forward. Outputs in `specwm_out/fused`. Separate-commit references are the same-seed arms from `cmp10` / section R. fps = 12 / max(cloud s/chunk, device + TAEHV) per chunk, measured on one GPU.

| config | commit | cloud s / chunk ↓ | fps ↑ | VLM drift ↓ | VLM quality ↑ | DINO min cos ↑ |
|---|---|---|---|---|---|---|
| 14B 4-step 832 (teacher) | separate | 2.45 | 4.9 | <u>46.7</u> | **7.58** | — |
| 14B 2-step 832 (0,500) | separate | 1.45 | 8.3 | 47.1 | <u>7.50</u> | — |
| 14B 2-step 832 (0,500) | **fused** | 1.02 | 11.8 | 54 | 7.4 | 0.245 |
| 14B 2-step 832 (0,750) | separate | 1.53 | 7.9 | 49 | 7.3 | **0.256** |
| 14B 2-step 832 (0,750) | **fused** | 1.01 | 11.9 | **42** | 7.1 | 0.226 |
| 768 + pix refiner | **fused** | 0.83 | 14.5 | 58 | 7.1 | <u>0.258</u> |
| 704 + pix refiner | separate | 0.95 | 12.7 | 54 | 7.08 | — |
| 704 + pix refiner | **fused** | 0.64 | 18.7 | 67 | 6.6 | 0.152 |
| 576 + pix refiner | separate | 0.56 | <u>21.5</u> | 57.5 | 6.75 | — |
| 576 + pix refiner | **fused** | 0.37 | **31.3** | 68 | 6.6 | 0.090 |
| Matrix-Game 2.0 (ref, 3-seed) | — | — | 20.5 | 58 | 7.1 | 0.130 |

- Removing the commit pass saves exactly one forward per chunk (−0.43 s at 832, −0.31 s at 704, −0.19 s at 576).
- **At 832, fused costs little:** (0,500) goes 7.50 → 7.4 quality and 47 → 54 drift, at 1.4× the fps. That gives 2.4× teacher speed at 7.4 quality, better than 704 separate on both axes.
- **At low resolution, fused hurts drift** (704: 54 → 67; 576: 57.5 → 68), and quality drops to 6.6. The low-res 14B is already near its drift limit, and the noisy-input K/V pushes it over.
- Single seed. 3-seed confirmation of the 832 (0,500) fused and 768 fused arms is running.

### U2. 3-seed confirmation (seeds 1234 / 4321 / 9999, 36 clips per arm; ± = SE)
| config | fps ↑ | VLM drift ↓ | VLM quality ↑ |
|---|---|---|---|
| 14B 4-step 832 teacher (separate) | 4.9 | **45.4 ± 5.2** | **7.64 ± 0.16** |
| 14B 2-step 832 (separate) | 8.3 | 50.8 ± 5.0 | <u>7.53 ± 0.14</u> |
| 14B 2-step 832 **fused** | <u>11.8</u> | <u>49.4 ± 5.4</u> | 7.28 ± 0.15 |
| 768 + pix refiner **fused** | **14.5** | 54.7 ± 4.8 | 7.03 ± 0.15 |
| 704 + pix refiner (separate) | 12.7 | 56.7 ± 4.7 | 6.89 ± 0.13 |
| 576 + pix refiner (separate) | 21.5 | 60.1 ± 4.5 | 6.53 ± 0.18 |
| Matrix-Game 2.0 | 20.5 | 58 | 7.1 |
- With 3 seeds, fused at 832 matches separate on drift (49.4 vs 50.8). Its quality is 0.25 lower, which is within about 1.5 SE, at 1.4× the fps. The single-seed drift penalty (54 vs 47) was seed noise.
- 768 fused beats 704 separate on all three axes (fps 14.5 vs 12.7, drift 54.7 vs 56.7, quality 7.03 vs 6.89). It is close to MG2 on quality, but at 14.5 vs 20.5 fps.

### U3. Periodic refresh (fused, with a clean commit every K chunks; 3 seeds × 12 prompts) and the quality–fps chart
| config | cloud forwards / chunk ↓ | fps ↑ | VLM drift ↓ | VLM quality ↑ | DINO min cos ↑ |
|---|---|---|---|---|---|
| 704 separate (ref) | 3 | 12.7 | 56.7 ± 4.7 | **6.89 ± 0.13** | — |
| 704 fused + refresh 3 | 2.33 | <u>16.9</u> | **52.8 ± 4.6** | <u>6.81 ± 0.15</u> | 0.153 |
| 704 fused + refresh 2 | 2.5 | 15.9 | 56.8 ± 4.4 | 6.67 ± 0.20 | <u>0.164</u> |
| 768 fused (ref) | 2 | 14.5 | <u>54.7 ± 4.8</u> | **7.03 ± 0.15** | **0.201** |
| 768 fused + refresh 3 | 2.33 | 13.0 | 59.3 ± 4.4 | 6.69 ± 0.23 | 0.185 |
| 576 separate (ref) | 3 | 21.5 | 60.1 ± 4.5 | 6.53 ± 0.18 | — |
| 576 fused + refresh 2 | 2.5 | **26.8** | 61.9 ± 4.3 | 6.58 ± 0.20 | 0.119 |
fps for the refresh arms uses the mean cloud time per chunk, because refresh chunks are slower.
- Refresh does not move quality. At every resolution, the separate, fused and refresh variants are within about 1 SE of each other (704: 6.89 / 6.81 / 6.67; 576: 6.53 / 6.58). The single-seed "fused hurts at low res" result (U) was seed noise. **The commit type is a free speed lever; resolution sets quality** (576 ≈ 6.55, 704 ≈ 6.8, 768 ≈ 6.9, 832 ≈ 7.3–7.5).
- Chart: `fig_pareto_quality_fps.png`. No configuration reaches ≥ 20 fps and ≥ 7.2. Matrix-Game 2.0 (20.5 fps, 7.1) is the only point near that corner. Our closest points are 576 fused + refresh 2 (26.8 fps, 6.58), which is faster but scores lower, and 768 fused (14.5 fps, 7.03), which scores about the same but is slower.

## V. 10/11 — Diagnostic: the device refiner, not the low-res 14B, loses quality
`res_split_infer.py --save-cloud` also decodes the cloud's own low-res latents (Wan VAE, native resolution) from the same run (`specwm_out/fused/*/dg{704,576}_s*`, separate commit, 3 seeds × 12 prompts). Judge at its usual 448-px input, then again at 832-px input (`VLM_W=832`; the low-res video is bicubic-upsampled by the judge's resize).

| judge input | output | VLM drift ↓ | VLM quality ↑ |
|---|---|---|---|
| 448 px | 704 cloud, native | **54.2 ± 5.0** | **7.22 ± 0.15** |
| 448 px | 704 cloud + 1.3B refiner → 832 | 57.1 ± 4.8 | 6.72 ± 0.14 |
| 448 px | 576 cloud, native | <u>56.9 ± 4.7</u> | <u>6.83 ± 0.14</u> |
| 448 px | 576 cloud + refiner → 832 | 60.3 ± 4.7 | 6.58 ± 0.18 |
| 832 px | 14B 4-step 832 teacher | <u>43.2 ± 5.0</u> | **7.78 ± 0.14** |
| 832 px | 14B 2-step 832 | **42.4 ± 4.9** | <u>7.22 ± 0.18</u> |
| 832 px | 704 cloud, bicubic up | 48.3 ± 5.2 | 7.11 ± 0.21 |
| 832 px | 704 cloud + refiner → 832 | 53.5 ± 4.9 | 6.58 ± 0.23 |
- Paired by clip, the refiner costs **−0.50 ± 0.11 quality at 704** (quality per clip: 1 better, 18 equal, 17 worse) and −0.25 ± 0.12 at 576. It also adds about 3 drift points.
- Even with the judge at 832 px, where blur from upsampling is visible, the plain bicubic-upsampled 704 cloud (7.11) is close to the 14B 2-step at 832 (7.22) and well above the refined video (6.58).
- So the 1.3B refiner was degrading the 14B's output, not adding detail. The low-res 14B itself is good.

### V2. Drop the refiner: 704 cloud with fused commit, plain upsampling on the device (3 seeds × 12 prompts)
Cloud output is 704×384 (latent 48×88). Matrix-Game 2.0 outputs 640×352, so the two resolutions are matched. fps uses the clean cloud time of section U (0.64 s/chunk fused at 704). Device work is TAEHV decode at 704 (about 15 ms/chunk) plus a bicubic upsample.

| config | fps ↑ | judge 448: drift ↓ | judge 448: quality ↑ | judge 832: drift ↓ | judge 832: quality ↑ |
|---|---|---|---|---|---|
| 14B 2-step 832 (separate) | 8.3 | 50.8 ± 5.0 | **7.53 ± 0.14** | **42.4 ± 4.9** | **7.22 ± 0.18** |
| 704 cloud, separate, no refiner | 12.7 | 54.2 ± 5.0 | <u>7.22 ± 0.15</u> | 48.3 ± 5.2 | <u>7.11 ± 0.21</u> |
| **704 cloud, fused, no refiner** | <u>18.7</u> | **49.3 ± 4.7** | 7.08 ± 0.16 | <u>47.8 ± 4.5</u> | 6.89 ± 0.21 |
| 576 cloud, fused, no refiner | 29 (est.) | 60.3 ± 4.6 | 6.58 ± 0.18 | 55.6 ± 4.7 | 6.28 ± 0.18 |
| 704 cloud + refiner (previous path) | 12.7 | 56.7 ± 4.7 | 6.89 ± 0.13 | 53.5 ± 4.9 | 6.58 ± 0.23 |
| Matrix-Game 2.0 (640×352) | **20.5** | 58.4 ± 4.7 | 7.08 ± 0.17 | 54.6 ± 5.0 | 7.03 ± 0.15 |
- Paired against MG2 on the same prompt and seed (36 pairs):
  - 448-px judge: drift −8.6 ± 5.8, quality +0.03 ± 0.23
  - 832-px judge: drift −7.5 ± 5.9, quality −0.14 ± 0.24
- **At matched resolution the two are tied on quality and drift, within 1.5 SE.** Ours runs at 18.7 vs 20.5 fps. Our model is a general camera-conditioned 14B world model; MG2 is a game-specific model.
- Recommended realtime path from here: 704 cloud, 2-step, fused commit, with the device doing TAEHV decode and bicubic upsampling. The 1.3B refiner is retired.

### V3. 10/11 — measured timing for 704 fused, and 672×384 (exclusive GPU, `t704` / `fc672` in `specwm_out/fused`)
- 704 fused, no refiner: first chunk 0.46 s, steady 0.640 s/chunk (median of the per-clip medians over chunks 6–12) → **18.7 fps measured**. TTFF ≈ 2.0 s prepare (T5 on GPU, section Q) + 0.46 + TAEHV ≈ 2.5 s. Three of 12 clips had a first-time aspect ratio and paid a torch.compile recompile (up to 4.8 s on chunk 0, 1.2–1.6 s/chunk). These runs encode T5 on CPU (cold prepare 20–23 s), so their prepare times are not TTFF.
- 672×384 fused, no refiner (3 seeds × 12): 0.536 s/chunk → 22.4 fps. Judge 448: drift 55.4 ± 4.6, quality 6.89 ± 0.15. Judge 832: 51.7 / 6.69. This is below 704 (49.3 / 7.08): shrinking the cloud slides along the same quality–speed curve. 704 stays the realtime config.
- Status doc for the professors: https://claude.ai/code/artifact/7af40e3e-00cc-4b07-83f7-bb56a3da069b

## W. 10/12 — fused commit causes chunk-boundary jumps; partial commit removes them
User spotted frame jumps in all "ours" arms on the comparison site. `scripts/boundary_jump.py`: d = consecutive-frame mean abs diff (grey, 208 px); at each chunk boundary (frame 8|9, then every 12 frames) r = d[b] / mean of the 4 neighbouring steps; control = the same ratio mid-chunk. Example: `easy_urban_rain` 704 fused s1234 hard-cuts to a different corridor at frame 128→129 (a boundary).

| arm (3 seeds × 12 unless noted) | cloud s/chunk ↓ | fps ↑ | boundary ratio ↓ | boundaries > 1.5× ↓ | mid-chunk control | VLM drift ↓ | VLM quality ↑ |
|---|---|---|---|---|---|---|---|
| 14B 4-step teacher 832 (s1234) | 2.45 | 4.9 | 1.10 | 2 % | 1.09 | 45.4 | 7.64 |
| 704 separate commit | 0.95 | 12.7 | 1.14 | 3 % | 1.11 | 54.2 | **7.22** |
| 704 fused | **0.640** | **18.7** | 1.24 | 12 % | 1.09 | **49.3** | 7.08 |
| 704 fused + partial commit, last 1 latent frame | <u>0.761</u> | <u>15.8</u> | 1.15 | 4 % | 1.10 | 53.5 | 6.86 |
| 704 fused + partial commit, last 2 latent frames | 0.874 | 13.7 | **1.14** | 4 % | 1.08 | 52.9 | <u>7.08</u> |
| 832 2-step fused (s1234) | 1.02 | 11.8 | 1.19 | 7 % | 1.09 | 54 | 7.4 |
| 832 2-step fused + partial 1 (s1234) | 1.24 | 9.7 | **1.11** | **0 %** | 1.11 | 52.5 | 7.5 |
| 672 fused | 0.536 | 22.4 | 1.25 | 12 % | 1.12 | 55.4 | 6.89 |
| 576 fused | 0.37 | ≈ 29 | 1.30 | 18 % | 1.09 | 60.3 | 6.58 |
- Fused commit leaves the newest frames' K/V computed from a noisy input; the next chunk then starts from a slightly different state, and in a few cases from a different scene. The VLM judge (8 frames per clip) and the 3-seed averages did not see this.
- **Partial commit** (`res_split_infer.py --partial-commit K`): after the fused step, re-run a clean t = 0 pass on only the last K latent frames of the chunk, via the token-subset path `fast_patch.ACTIVE`. With K = 1 the boundary statistics return to the separate-commit level, for +0.12 s per chunk at 704 (15.8 fps vs 12.7 separate). VLM differences between the 704 variants are within ~1.5 SE.
- The realtime claims in V2/V3 (18.7 fps tie with Matrix-Game 2.0) hold only for the jumpy fused variant. The jump-free realtime config is 704 + partial commit 1: 15.8 fps.

### W2. Decoder ablation on the jump-free realtime config (704 fused + partial commit, 36 clips; `scripts/taehv_ablation.py`)
| decoder (runs on the device) | ms / chunk at 704 ↓ | pipelined fps (cloud 0.761 s/chunk) ↑ | judge 448: drift ↓ | judge 448: quality ↑ | judge 832: drift ↓ | judge 832: quality ↑ | boundary jumps > 1.5× ↓ |
|---|---|---|---|---|---|---|---|
| full Wan2.1 VAE (bf16) | 426 | 15.8 (device idle 46 % of each chunk) | **53.5** | **6.86** | **51.1** | **6.81** | 4 % |
| TAEHV taew2_1 (fp16) | **12.8** | 15.8 | 60.8 | 6.31 | 62.2 | 5.75 | **3 %** |
- Paired by clip, TAEHV vs full VAE: quality −0.56 ± 0.13 (judge 448) and −1.06 ± 0.14 (judge 832); drift +7.4 ± 3.0 and +11.1 ± 3.2. **The tiny decoder costs a clear amount of quality.** All earlier VLM numbers (ours and baselines) were scored on full-VAE videos, while the fps figures assumed TAEHV.
- At 704 the full VAE (426 ms) is faster than the cloud chunk (761 ms). On a device with a GPU of this class, the full VAE therefore does not lower the pipelined fps; TAEHV only matters for weaker devices, or when the device also runs a model (e.g. the old refiner, section Q). On one shared GPU (serial): cloud + VAE = 1.19 s → 10.1 fps, cloud + TAEHV = 0.774 s → 15.5 fps.
- Official 4-step schedule check: `wan/image2video.py generate()` passes timesteps_index [0, 250, 500, 750] to the causal-fast sampler, i.e. timesteps 999 / 937 / 833 / 624 (σ 1.0 / 0.937 / 0.833 / 0.625). The model predicts x0 at every node, so the last node does not need to reach σ = 0. Ours uses indices [0, 500] = timesteps 999 / 833.
