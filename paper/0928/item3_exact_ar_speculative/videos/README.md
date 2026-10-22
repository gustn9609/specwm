# Item 3 — exact frame-level (AR) speculative sampler: videos

File names: `<task>_s<seed>_<arm>.mp4`. All clips: same start image, same camera trajectory (back 3 steps, 360° right turn, then left), empty text prompt.

| arm | what it is |
|---|---|
| exactspec_cfg5 | `spec_v0 --specstep 1 --spec-nodes 0,1,2 --ctx 9`: at Fast nodes 0/1/2 the draft transition is kept or reflected per latent position (reflection maximal coupling, De Bortoli et al. ICML 2025), so each transition is exactly the target's; target = Base one-step CFG-5 x0 re-noised, with 9 prefix latents re-noised to the same σ (RePaint); chunk 0 and node 3 = Fast |
| exactspec_cfg2p5, exactspec_cfg1 | same sampler, target built with Base CFG 2.5 / 1 (seed 1234 only) |
| nullarm_cfg5, nullarm_cfg1 | the same target kernel without any draft (`--bguide 1 --bguide-nodes 0,1,2 --ctx 9`); an exact sampler must match it in distribution |
| ref_base70_target, ref_fast_alone | Base-70 and Fast alone with the same seed, for comparison |

`side_by_side/` (start here): `<task>_s<seed>_base_fast_exactspec_nullarm.mp4` = Base | Fast | exact spec (CFG 5) | null arm (CFG 5) for jar/castle s1234 and s4321; `jar_s9999_base_fast_exactspec.mp4` (no null arm for that seed); `<task>_s1234_target_cfg_sweep.mp4` = Base | exact spec with CFG-5 / 2.5 / 1 targets.
The first, biased implementation (rejection with ≤ 4 retries) was not kept as video.
`per_chunk_acceptance.json` and the per-run `.json` files hold the per-chunk acceptance rate at each node (spec_acc_k) and the mean draft–target gap (spec_dmean_k).

## Per-position acceptance (mean over chunks)

| case | arm | chunks | accept node 0 | accept node 1 | accept node 2 | time per clip |
|---|---|---|---|---|---|---|
| castle_s1234 | exactspec_cfg1 | 12 | 0.96 | 0.92 | 0.83 | 218 s |
| castle_s1234 | exactspec_cfg2p5 | 12 | 0.92 | 0.91 | 0.83 | 342 s |
| castle_s1234 | exactspec_cfg5 | 12 | 0.82 | 0.91 | 0.85 | 341 s |
| castle_s4321 | exactspec_cfg5 | 12 | 0.78 | 0.88 | 0.82 | 342 s |
| jar_s1234 | exactspec_cfg1 | 12 | 0.95 | 0.91 | 0.81 | 232 s |
| jar_s1234 | exactspec_cfg2p5 | 12 | 0.89 | 0.91 | 0.82 | 345 s |
| jar_s1234 | exactspec_cfg5 | 12 | 0.72 | 0.86 | 0.78 | 349 s |
| jar_s4321 | exactspec_cfg5 | 12 | 0.73 | 0.89 | 0.80 | 347 s |
| jar_s9999 | exactspec_cfg5 | 12 | 0.71 | 0.88 | 0.79 | 348 s |

Whole-chunk acceptance is the product over ~295k latent positions, ≤ 10^-25 in every case, so drafts can never be accepted in blocks and there is no speed-up. What to look for in the videos: the exact sampler and the null arm wash out to white/green the same way at CFG 5 (the target, not the coupling, is the problem); CFG 2.5 over-exposes; CFG 1 keeps colour but the camera stays on the jar (jar) or the image turns hazy and grainy (castle).
