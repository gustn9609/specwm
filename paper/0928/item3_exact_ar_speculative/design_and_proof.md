## 3. Unbiased frame-level AR speculative sampling

### 3.1 Bottom line
- **An exact sampler exists and is already implemented.** `--specstep` is SPEC-AR: window γ = 1 with per-position reflection maximal coupling. It is exact **with respect to a named chunk-AR target T_H^ctx, not Base-70.**
- **With Fast as the draft it cannot beat the target alone.**
  - On fresh T_H^c9 states the probability of accepting a whole chunk is ≤ 10^-25, so lookahead is never accepted.
  - As coded it runs at ≈ 0.91× of a lean target-only rollout (estimated). With Fast and Base running concurrently it would be ≈ 1.0× at best.
- **At nodes 0–2 the draft does not change the output law.** The output has the same law as the target-only rollout; the draft only decides which realisation you get (58–94% of positions are kept).
- **The only exact speed-up is systems parallelism of the target itself.** Estimated ≈ 1.7–2.7× on 4 GPUs (UNVERIFIED). This is not speculation.
- **plan→render stays the practical method** (114–206 s, biased). It should be judged with the item-4 protocol.

### 3.2 Target
Notation:
- Chunks c = 0..12. Each chunk has d = 16×3×58×106 = 295,104 dimensions (P = 18,444 positions × 16 channels).
- Node σ = 0.9998 / 0.9373 / 0.8331 / 0.6247.
- X_<c = the committed prefix.

**T_H^ctx** (ctx = 0: full prefix and the start image; ctx = 9: the runs in flight):
- **Chunk 0:** Fast's 4-node sampler.
- **For c ≥ 1:**
  - z_0 ~ N(0, I).
  - For k = 0, 1, 2: z_{k+1} ~ K_k(·|z_k, X_<c) = E_ξ N((1−s_{k+1})·m_k, s_{k+1}² I).
    - m_k = the chunk slots of the x0 from one Base-A14B CFG-5 call on [(1−s_k)·X_window + s_k·ξ ; z_k] at t = 1000·s_k.
    - High expert iff t ≥ 947.
    - ctx = 9: the window is the last 9 prefix latents. For c ≥ 4 the y-channel image is the VAE decode of prefix latent j = 3c−9, with poses offset by 4j.
  - X_c = x0_Fast(z_3; KV cache of X_<c). This step is deterministic and is committed.

| property | preserved? |
|---|---|
| joint law of the 13 committed chunks under T_H^ctx | **yes** |
| law of any video functional f (drift indicator, motion, DINO) is the same as under a target-only rollout | **yes** (equality in law, not pathwise) |
| Base-70 whole-clip law | no. Base is a bidirectional 161-frame joint, so no chunk-AR kernel reaches it |
| Base's true conditional p_B(X_c \| X_<c) | no. Same-σ RePaint gives prefix SNR 0.0002 / 0.067 / 0.20 at nodes 0/1/2 |
| a Base-only sampler | no: node 3 and chunk 0 are Fast (the T_B variant would need new code) |

Asymptotically exact conditional samplers exist (SMC, e.g. TDS, https://arxiv.org/abs/2306.17775), but they cost particles × Base forwards.

### 3.3 Algorithm
```
# SPEC-AR (γ = 1) = --specstep after the P0 fixes
X[0] = Fast4Node(chunk 0)
for c in 1..12:
    z = randn(G(seed, c, 'z0'))                        # CPU generator; common random numbers with Fast-only
    for k in 0, 1, 2:
        s1  = sigma[k+1]
        muD = (1-s1) * FastX0(z, node k, KV = X[<c])
        xi  = randn(G(seed, c, k, 'xi'))
        muT = (1-s1) * BaseX0_cfg5(window(X[<c], ctx, xi, sigma[k]), z, t = 1000*sigma[k])[chunk]
        eps = randn(G(seed, c, k, 'eps'));  u = rand(G(seed, c, k, 'u'))   # u: one per position
        zt  = muD + s1*eps
        logr_p = -(|zt-muT|_p^2 - |zt-muD|_p^2) / (2*s1^2)            # per position, 16 channels
        e_p = (muD-muT)_p / |muD-muT|_p
        z = where(log u_p <= min(0, logr_p), zt, muT + s1*(eps - 2*(e_p·eps)*e_p))
        log: acc_p, |Δ_p|, stats of r = (z-muT)/s1, mean(e·r)
    X[c] = FastX0(z, node 3, KV = X[<c]);  commit(X[c])
# Null arm (target-only): z = muT + s1*eps, with no coupling.
# It already exists as: --policy never --bguide 1 --bguide-nodes 0,1,2 (+ the same --ctx / --resident)
```

- **Lookahead (PathSpec-AR, γ ≥ 2):**
  - Fast drafts γ transitions ahead; Base verifies each one on its own GPU; the whole state is coupled in order; the loop stops at the first rejection.
  - It is exact only if every speculative branch has its own Fast KV cache. The shared cache rolls destructively on the first forward of a new chunk, and a snapshot costs ≈ 21–23 GiB.
  - At acceptance ≤ 10^-25 it always stops at the first transition, so do not build it.

### 3.4 Proof sketch
**Assumptions**
- **A1.** Draft and target use the same isotropic variance s'² (both are x0-renoise on the shared grid).
- **A2.** μD and μT depend only on the history (plus a fresh ξ), never on the current (ε, u).
- **A3.** ξ, ε and u are fresh and mutually independent.
- **A4.** A target mean is used only at the realised state and with the true prefix.
- **A5.** The deterministic maps are identical in both arms: node-3 finish, commit and KV eviction. This holds in order.
- **A6.** Chunk 0 has the same law in both arms.
- **A7.** Exactness is with respect to the kernel as computed in bf16.
- **A8.** Adaptive choices never look at the current (ε, u).

**Lemma 1** (De Bortoli Prop. 3.1, after Bou-Rabee et al.)
- The whole-state reflection coupling outputs exactly N(μT, s'² I).
- P(keep draft) = 2Φ(−‖Δ‖/2) = 1 − TV, which is the maximum over all couplings. Here Δ = (μD − μT)/s'.

**Lemma 2** (per position)
- The Gaussians factorise over positions, so applying Lemma 1 per position with independent (ε_p, u_p) yields exactly N(μT, s'² I).
- P(every position kept) = Π_p (1 − TV_p) ≤ 1 − TV_whole. Per-position coupling therefore maximises *reuse*, not *lookahead*.

**Theorem**
- Given the history and ξ_t, the next state is exactly N(μT(S_t, ξ_t), s'² I). Integrating over ξ gives the kernel K_t.
- Markov induction (Ionescu-Tulcea) makes the path law equal to the target path law.
- The deterministic finish then gives the T_H^ctx video law.

**Corollaries**
- **C1.** E_spec[f] = E_target-only[f] for every video functional f.
- **C2.** Exactness breaks under any of the following:
  - legacy `--policy spec`;
  - τ ≠ 1 (De Bortoli §F.1);
  - SDVG-style thresholds;
  - Base regeneration on rejection;
  - `--specstep` under any policy other than `never`;
  - `--specstep` combined with a hint or plan. The law then stays K_T's, so this is *not* an exact plan→render.

**Extension to ctx = 9 (mine; the review checked the ctx = 0 path line by line).** μT depends on the committed prefix through `base_cond`, whose cache is keyed by (F, j). In order, that is a deterministic function of the history, so A2 and A4 still hold.

### 3.5 Acceptance and speed with our numbers
Fresh T_H^c9 data: seed 1234, chunks 1–12, `specstep_n012c9_never`, finished at 05:14 / 05:25 CDT.

| prompt · transition σ→σ' | per-position acceptance ↑ (mean) | mean \|Δ_p\| ↓ | whole-chunk ‖Δ‖ ↓ (lower bound) | log10 P(whole chunk kept) ↑ (upper bound) |
|---|---|---|---|---|
| jar · 0.9998→0.937 | 0.58–0.82 (0.72) | 0.47–1.10 | ≥ 64 | ≤ −222 |
| jar · 0.937→0.833 | 0.62–0.91 (<u>0.86</u>) | 0.23–1.14 | ≥ 31 | ≤ −53 |
| jar · 0.833→0.625 | 0.58–0.84 (0.78) | 0.42–1.38 | ≥ 57 | ≤ −179 |
| castle · 0.9998→0.937 | 0.76–0.92 (0.82) | 0.19–0.61 | <u>≥ 26</u> | <u>≤ −39</u> |
| castle · 0.937→0.833 | 0.88–0.94 (**0.91**) | **0.16–0.30** | **≥ 21** | **≤ −25.7** |
| castle · 0.833→0.625 | 0.77–0.91 (0.85) | <u>0.22–0.60</u> | ≥ 30 | ≤ −51 |

- **Lookahead requirement.** Keeping the whole chunk with probability ≥ 0.5 needs ‖Δ‖ ≤ 1.35, i.e. an RMS per-position |Δ_p| ≤ 1.35/√18,444 = 0.0099. The measured means are 16–140× too large, and the RMS is at least the mean.
- **Legacy full-prefix states** (biased chain, review-corrected numbers): ‖Δ‖ = 19–84 at σ 0.937 and 47–203 at σ 0.833, so P ≤ ~10^-21.
- **Cause.** Fast is DMD-distilled, so its x0 looks like a sample. Base is undistilled with CFG 5, so its x0 is a posterior mean. The two path measures barely overlap, even if the video marginals matched.
- **Caveat (review 1).** The bound is taken against one ξ-component of the kernel, and the mixture could be closer to the draft. What is proven is "component-wise coupling cannot look ahead", not "no exact scheme can".

**Speed.** S = E[L̂] / (1 + γ·C_D/C_T), with E[L̂] = 1.
- C_D ≈ 0.9 s per Fast forward.
- C_T ≈ 7.0 s per CFG call on a 12-latent window (251 s / 36 calls).
- So S ≈ 0.89 when run sequentially and ≈ 1.0 when Fast runs concurrently with Base.
- The literature agrees: Soen et al. get only 1.22× at K = 50 even with ideal parallel verification (https://arxiv.org/abs/2606.13426). We have 3 stochastic steps per chunk.

| sampler (161 frames, 480p) | exact w.r.t. | GPUs | wall-clock s ↓ | source |
|---|---|---|---|---|
| Fast only | – | 1 | **61–63** | measured |
| plan→render v2 | none (biased) | 1–2 | <u>114–206</u> | measured |
| T_H0^c9 target, CFG split + node-0 prefetch (systems parallelism) | T_H0^c9 (a different target) | 4 | ≈ 130–180 | estimate, UNVERIFIED |
| target-only T_H^c9, lean | T_H^c9 | 2 | ≈ 317 | estimate (349 − 36 × 0.9) |
| **SPEC-AR `--specstep` (c9)** | T_H^c9 | 2 | 341–349 | measured, 2 runs |
| target-only T_H, full prefix | T_H | 2 | ≈ 740 | estimate; legacy analogue 750 s |
| Base-70 whole clip | Base-70 | 1 | 2,766–3,111 | measured, 8 runs (the brief's ~600 s is not in any log) |

### 3.6 What survived review and what did not

| claim | verdict | correction |
|---|---|---|
| SPEC-AR (in order, γ = 1, per position) is exact w.r.t. T_H | **survived** (checked line by line) | also holds for ctx 9 (my extension, argued above) |
| draft does not affect the law at nodes 0–2; target-only is the null | **survived** | – |
| no speed-up with Fast drafts; block verification cannot help | **survived** | numbers corrected (below) |
| "whole-state acceptance ≈ 1e-75"; ‖Δ‖ 37–148; relative L2 39–47% / 31–33% | numbers wrong | ≤ 1e-21 (legacy) and ≤ 10^-25.7 (fresh). The relative-L2 figures were shifted by one node. |
| reuse of 59–96% | wrong | 58–94% (fresh data) |
| "T_H0 ≈ T_H because the change is below bf16 resolution" | **refuted** | bf16 rounding dithers the 1.5e-4 prefix term, so T_H0 is a different target. Under c9, node 0 also sees the prefix through the y-channel image. |
| "ξ and u streams are identical across seeds" | **refuted** | the default CUDA generator is seeded non-deterministically. Runs are independent but not reproducible and cannot be paired. |
| "any exact scheme gains no lookahead"; "free per-position TV map = likelihood proxy" | overclaim | proven only for component-wise coupling. The TV map is the TV to one ξ-component; it is not a likelihood, and it is not free (3 CFG calls per chunk). |
| CS-AR and PathSpec-AR with the existing Fast cache | **biased** (violates A5: the shared KV cache rolls) | needs separate caches; dropped. CS-AR also gains nothing on 4 GPUs. |
| null test tolerances ±0.0003 / ±0.0005 per transition | wrong | per transition: ±0.0018 (mean) and ±0.0026 (var); pooled over 36 transitions: ±0.0003 / ±0.0004. It is also blind to a stale μT, so recompute μT independently. |
| "queued runs sample T_H"; "no output yet" | outdated | they sample T_H^c9 with `--resident`; 2 runs are done. |
| "peak 78.7 GB of 80 GB" (review 2) | wrong | the GPU has 94.97 GiB; the OOMs came from a co-tenant process holding 57.45 GiB. |
| second design (LSAR, speed angle) | reached the same no-go | its adversarial verdicts were not in my input (UNVERIFIED) |

### 3.7 Minimal experiment
- **E0 (now, no GPU):** watch the 2 finished videos, jar and castle `specstep_n012c9_never_s1234`, next to `base70_s1234` and Fast-only `never_s1234`. Question: is T_H^c9 Base-like enough to be worth being exact to?
- **E1 (no code change):** a target-only null arm with `--policy never --bguide 1 --bguide-nodes 0,1,2 --ctx 9 --resident` on seeds 1234 / 4321 / 9999 × 2 prompts.
  - Cost: ≈ 345 s per run, about 35 min on one GPU pair.
  - Purpose: a cost baseline and a low-power equality check.
- **E2 (after the P0 fixes):**
  - P0 fixes:
    - seeded ξ and u generators;
    - asserts: `policy == never`, bguide/hint off;
    - residual logging;
    - independent recomputation of μT on random transitions;
    - `--spec-cfg`;
    - `t0c` timing fix.
  - Then run 2 runs and check the residual mean and variance against the per-transition tolerances. With seeded streams, spec and target-only become paired, so McNemar can be used.
- **Decision rule:** if T_H^c9 is not Base-like by eye, or its item-4 drift rate is not better than Fast-only, report item 3 as: "exact w.r.t. a named but poor target; with Fast drafts a speed-up is impossible, shown by a bound and by measured numbers". Then stop spending GPU time on it.