## 2. Reproducing the speculative-diffusion baseline (De Bortoli, Galashov, Gretton, Doucet, ICML 2025)

### 2.1 What the baseline is
Paper: https://arxiv.org/abs/2501.05370. There is no official code; I checked the arXiv v2 HTML and the PMLR PDF.
- **Target.** A K-step stochastic Euler–Maruyama chain of the reverse SDE with churn ε > 0. Each transition is N(m^q(y), σ_k² I).
- **Draft.** Either an independent cheaper model, or the paper's main choice: the *frozen target* draft, which reuses the target drift computed at the start of the window for all L steps.
- **Verification.** One batched target call per window, then in-order reflection maximal coupling (Alg. 4), stopping at the first rejection.
- **Exactness.** Exact only with respect to that discretised stochastic chain. The paper says it is not directly applicable to deterministic samplers.
- **CIFAR-10 numbers** (Table 1, 100-step target; re-checked in the v2 HTML):

| setting | exact? | FID ↓ | target NFE ↓ |
|---|---|---|---|
| speculative, ε = 0.25, τ = 1 | yes | <u>2.35</u> | <u>39.31</u> |
| speculative, ε = 0.25, τ = 2 (the headline "35 calls") | **no** (§F.1: τ ≠ 1 is not exact) | **2.34** | **35.40** |
| 100-step target, ε = 0.25 | reference | 2.45 | 100 |
| 30-step target, ε = 0.25 | reference | 7.68 | 30 |

- The window L used for Table 1 is not stated in the main text (UNVERIFIED; asked in the email).
- NFE counts one batched call as one evaluation, and the paper reports no wall-clock times.

### 2.2 Plan
**Stage A: validate our coupling code in their setting.** Do this first; it is cheap.
- Setting: CIFAR-10 pixel DDPM, 100-step EM with ε = 0.25, frozen target draft, τ = 1.
- Pass criterion: FID ≈ 2.35 at ≈ 39 NFE, against 2.45 for the plain 100-step chain.
- Code: the only public implementation of this chain verifier is https://github.com/marcellobullo/tree-specdiff (Bullo et al., https://arxiv.org/abs/2609.17691); its K = 1 case is De Bortoli's chain.
- Missing details: the checkpoint and L. Use the authors' answers if they arrive; otherwise sweep L ∈ {5, 10, 20}.
- Cost: UNVERIFIED, probably a few GPU-hours. Run it only when a GPU is free (GPUs 4–7 are busy tonight).

**Stage B: their method in our setting.**

| variant | draft | target (exactness w.r.t.) | verification | expected | status |
|---|---|---|---|---|---|
| **B0 = `--specstep`** | Fast (their option (i), independent draft) | T_H^ctx: Base-A14B CFG-5 RePaint kernels at nodes 0–2; node 3 and chunk 0 are Fast | per latent position, window L = 1 | exact; **no speed-up** (whole-chunk acceptance ≤ 10^-25, see item 3) | **implemented; 2 of 16 runs done** |
| B1 = Fast draft + their Alg. 3 window L ≥ 2 | Fast | T_H^ctx | whole state; one Base GPU per drafted state (batched Base fails in the camera path) | E[accepted lookahead] = 1, so B1 = B0 minus the per-position reuse; also needs a separate Fast KV cache per branch (≈21–23 GiB) | do not build |
| B2 = the literal baseline on Base (frozen-target self-speculation) | Base drift frozen at the window start | Base with a *stochastic* K-step EM sampler over the whole clip (16×41×58×106 ≈ 4.0M dims). This is not the deterministic UniPC Base-70 | one GPU per drafted state, so L ≤ 4 on GPUs 4–7 | at best ≈ 1.2–1.5×. Soen et al. report 1.22× at K = 50 and 1.55× at K = 100 on an ImageNet LDM with ideal TPU parallelism (https://arxiv.org/abs/2606.13426); Bullo et al. report that exact coupling efficiency degrades with dimension | not started; each run costs about one Base-70 clip (≈2,800 s on 1 GPU) or more |
| B3 = frozen draft on the chunk-AR target (3 transitions per chunk) | Base x0 from node k reused at node k+1 | T_H^ctx | whole state | ≈ 0 acceptance: Base's x0 would have to move by less than about 1.7% (relative L2) between σ 0.94 and 0.83, but NIGHT3 probe B shows it is still far from final after the high-noise steps (DINO 0.27–0.83) | do not build |

**Metrics.** Their FID-vs-NFE becomes, for us:
1. sequential Base rounds per chunk (a batched or parallel round counts as 1);
2. wall-clock time and GPU-seconds;
3. per-position and whole-state acceptance, and E[L̂];
4. an exactness check: residual statistics, plus the spec batch against a target-only batch;
5. quality against the Base-70 batch (item 4), and watching the videos.

### 2.3 What `--specstep` already reproduces

| element of the paper | in `--specstep`? | evidence |
|---|---|---|
| Alg. 4 reflection maximal coupling | yes, per latent position (16 dims) | spec_v0.py:480–486. CPU check: acceptance 1.000 / 0.865 / 0.575 / 0.093 vs theory 1.000 / 0.867 / 0.576 / 0.094 |
| Prop. 3.1 (acceptance = 1 − TV, output exactly the target) | yes, per position | Runtime: the logged mean acceptance equals 2Φ(−\|Δ_p\|/2), evaluated at the logged mean \|Δ_p\|, to within 0.008 in all 6 (prompt, node) cells of the two finished runs |
| τ = 1 (exact) | yes | τ ≠ 1 is not implemented, which is correct for exactness |
| independent draft (option i) | yes: Fast | — |
| stochastic target chain with the same σ as the draft | yes, but only 3 transitions per chunk and a RePaint target | x0-renoise on the shared shift-5 grid |
| frozen-target draft | no | — |
| window L > 1 (Alg. 3 lookahead) | no | pointless at the measured Δ |
| batched parallel verification | no | batched Base forwards fail in the camera path |
| whole-state acceptance | no (per position) | whole-state acceptance ≤ 10^-25 |
| FID-vs-NFE evaluation | no | replaced by the item-4 batch protocol |

**Bottom line for item 2.** We reproduce the paper's *verifier* (coupling and exactness) at the transition level, and the CIFAR-10 run (Stage A) is what would confirm our code against theirs. We do not reproduce its *speed-up mechanism* (lookahead), and our numbers say that mechanism cannot work with Fast as the draft. The faithful video-scale version is B2, whose best-case gain in the literature is ≈ 1.2–1.5× at K = 50–100.