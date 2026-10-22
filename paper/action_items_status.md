# Action items — status (auto-maintained; last update: 2026-09-28 01:50 CDT)

Legend: DONE / PARTIAL / IN PROGRESS (overnight) / OPEN. Details and numbers in `lingbot-world-v2/specwm_out/spec_v0/NIGHT3_NOTES.md`.

## 1. Test first on the two failure prompts; no speed optimization yet — DONE
All methods (correction policies, guidance, hand-off, K/V memory, distillation, plan → render) were developed and judged on `sand_filling_up_jar` and `sandcastle_wave_impact` (seeds 1234/4321/9999) before any speed work. Speed work (CFG-6 + flow reuse, single-GPU render, 4-step planner) came after consistency was reached. Lineup: https://claude.ai/artifact/L4B8p5w4YzZMP2TPCihmUX

## 2. Reproduce the speculative diffusion baseline — PARTIAL → IN PROGRESS
No email was sent (I cannot send mail). Implemented instead the standard continuous speculative-sampling rule at the draft's denoising nodes (`spec_v0 --specstep 1`): draft proposes z_{k+1} ~ N((1-σ)x0_D, σ²); target gives x0_T at the same state (one Base forward, prefix re-noised); each latent position is accepted with min(1, p_T/p_D) and rejected positions are resampled from the residual of the target kernel (rejection sampling, ≤4 tries). Running tonight on the two failure prompts; acceptance rates and drift metrics will be in the notes.

## 3. Unbiased frame-level (AR) speculative sampling — IN PROGRESS
Same algorithm as item 2 applied per chunk (12 frames) at the AR draft's own nodes; unbiased per position under the factorised Gaussian kernels of both samplers. Earlier training-free variants (replace / KV replace / soft blend / hand-off) were all rejected for seams or freezing (notes, 9/24–25).

## 4. Evaluation protocol (many seeds, binary drift rate) — IN PROGRESS
`scripts/drift_rate.py` + `scripts/base_selfdist.py`. Tonight: 8 seeds × 2 failure prompts for Base-70, Fast alone, v2-safe, v2-fast(CFG) and the speculative sampler; drift := DINO distance to Base above the Base-vs-Base seed floor (calibrated tonight) or a chunk jump above the Base maximum; Wilson 95% CIs.

## 5. Verifier & correction design space — PARTIAL
Explored: target-likelihood proxy (one-step Base x0 distance at σ=1 and at the draft node), semantic DINO gate (with retries), low-resolution verification/correction (360p/240p checks, 576×304 whole-clip plan = the winner), K/V-cache reuse and a learned Base→Fast KV translator, Base-guided nodes, hand-off. Not done: learned lightweight verifier, physics-aware reward.

## 6. One-step Base check cost / parallel checks on noised frames — PARTIAL → IN PROGRESS
Measured: one Base forward on a 12-latent window at 480p ≈ 2.5 s (used by the `sem`/`x0` policies). Tonight: `specwm/onestep_profile.py` times batched forwards (B = 1, 2, 4, 8 noised copies in one call) to see whether several noise levels / seeds can be checked for the price of one.
