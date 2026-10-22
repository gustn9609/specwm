## Audit: what `spec_v0 --specstep` is (state at 05:32 CDT, 9/28)

**Code.**
- `specwm/spec_v0.py:466-490`, inside `draft_chunk`.
- Target mean from `base_x0_at` (:329-333).
- Conditioning from `base_cond` (:194-202).

**What it samples.** Chunk 0 is plain Fast, because `outs` is empty and `--start 1`. For each chunk c ≥ 1:
1. z_0 ~ N(0, I), from the CPU generator `g` seeded by `--seed`.
2. For k = 0, 1, 2 (σ 0.9998 → 0.9373 → 0.8331 → 0.6247):
   - Fast proposes z' = (1−s')·x0_F(z_k) + s'·ε.
   - Base supplies μT = (1−s')·x0_B, where x0_B comes from one Wan2.2-A14B CFG-5 call (2 forwards) on [(1−s_k)·prefix + s_k·ξ ; z_k] at t = 1000·s_k. Node 0 uses the high expert, nodes 1–2 the low expert (boundary 947).
   - Each latent position keeps the draft with probability min(1, N(z';μT)/N(z';μD)). Otherwise it is reflected to μT + s'(ε − 2(e·ε)e). This is De Bortoli Alg. 4 applied per position (https://arxiv.org/abs/2501.05370).
3. Node 3 is never verified; the step sits inside `if j < len(NODES)-1`. Fast's x0_F(z_3) is committed.

**The runs in flight use a variant.** `scripts/q173_bundles.sh:156` adds `--ctx 9 --resident`.
- The Base window is the last 9 prefix latents plus the chunk.
- For c ≥ 4, Base's clean conditioning image is the VAE decode of prefix latent j = 3c−9, a Fast frame from chunk c−3, with poses offset by 4j.
- I call this target **T_H^c9**. The full-prefix T_H in the design documents is not what is running.

**Honest claim.** `--specstep 1 --spec-nodes 0,1,2 --policy never` produces an exact sample of the hybrid chunk-AR chain T_H^ctx:
- chunk 0 and each chunk's final denoise come from Fast;
- the three stochastic transitions per chunk are Base-A14B CFG-5 RePaint kernels.

It is not exact with respect to Base-70, Base's true conditionals, or any Base-only sampler. The Fast draft does not change the output law, and the sampler is not faster than running the target alone.

**First results (2 of 16 queued runs finished).**

| run (s1234, T_H^c9) | acc. node 0 ↑ | acc. node 1 ↑ | acc. node 2 ↑ | whole-chunk ‖Δ‖ ↓ (lower bound) | wall-clock s ↓ | peak GB ↓ |
|---|---|---|---|---|---|---|
| jar | 0.58–0.82 (0.72) | 0.62–0.91 (0.86) | 0.58–0.84 (0.78) | ≥ 31 | 349 | 78.7 |
| castle | **0.76–0.92 (0.82)** | **0.88–0.94 (0.91)** | **0.77–0.91 (0.85)** | **≥ 21** | **341** | **78.6** |

- The logged mean acceptance matches 2Φ(−|Δ_p|/2) to within 0.008 in all 6 cells.
- Whole-chunk acceptance is ≤ 10^-25.
- The videos have not been viewed yet.
- The remaining 14 runs are on GPUs 2 and 3, at about 11.5 minutes each.

**Bugs and hazards**

| # | severity | where | issue | fix |
|---|---|---|---|---|
| 1 | HIGH (claim) | :471-473, :812 | Node 3 and chunk 0 are pure Fast, so the output is exact only for the hybrid chain. | Name the target in every claim. |
| 2 | HIGH (target quality) | :332 | Same-σ RePaint gives prefix SNR 0.0002 / 0.067 / 0.20, which leads to snap-back and an ignored camera (NIGHT3:136, 148, 160). The closest measured relative, `bg1_n01` (Base at nodes 0–1, ctx 0), froze: motion 2 vs 16 for Base. With ctx 9 the anchor image is a Fast frame, so the target may inherit Fast drift (UNVERIFIED). | Judge T_H^c9 by eye first. |
| 3 | MEDIUM | :332 (ξ), :482 (u) | ξ and u come from the default CUDA generators and are never seeded. **Correction to the audit:** PyTorch seeds these non-deterministically (per the main-branch CUDAGeneratorImpl.cpp; the installed 2.11.0 was not checked separately). Seeds are therefore independent, but runs cannot be reproduced and the spec and target-only arms cannot be paired. | Use seeded per-(seed, chunk, node, stream) generators. |
| 4 | MEDIUM (ops) | q173_bundles.sh:156 | `--resident` peaks at 78.7 GB out of 94.97 GiB, so it fits only if the GPU is not shared. The 04:41 CDT full-prefix runs hit CUDA out-of-memory because another process held 57.45 GiB (w23_night.log:133-154). | Check for co-tenants before launching; do not use `--resident` with the full prefix. |
| 5 | MEDIUM | :446-463 | `--hint-lora` without a plan video duplicates the node-0 CFG call (about 211 s per clip). specstep plus a hint is not an exact plan→render. | Do not combine them. |
| 6 | LOW | :333 | `TM['denoise']` also counts the cond time. | Use `t0c`. |
| 7 | LOW | :332 | The target's CFG silently follows `--bguide-cfg`. | Add a separate `--spec-cfg`. |
| 8 | LOW | policy loops | specstep is silently ignored under most policies. Under sem, always, every, verify and sched, rejected chunks are replaced by Base, which is biased. The legacy `--policy spec` is also biased. | Assert `policy == never`, and that bguide and hint are off. |
| 9 | LOW (docs) | :10; action_items_status.md | The help text still says "residual"; `--spec-tries` is dead code; the status doc (01:50 CDT) still describes the loop with at most 4 tries; "2.5 s per forward" should be 3.74 s; "5x fewer tokens" should be 2.25x per latent frame; "6 sink + 18 window" is really 18 frames in total, including the 6 sink frames (image2video.py:369). | Update the text. |
| 10 | hazard (UNVERIFIED) | :200 | With ctx 9, one latent frame is decoded on its own even though it encodes a group of 4 frames, which may be off-distribution. `cond_cache` ignores the prefix content: that is safe for in-order sampling but wrong for any lookahead. | Assert that no lookahead is used. |
| OK | verified | :465-486 | ε recovery, σ indexing, the fp32 log-ratio, one u per position and the reflection are all correct. A CPU check agrees with theory (0.865 vs 0.867). | Add residual logging. |