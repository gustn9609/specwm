## 5. Verifier and correction design space

### 5.0 Summary
- **Detection has never been the bottleneck; correction has.**
  - Every per-chunk Base correction left seams or snapped the view back toward frame 0.
  - Every soft injection of Base into Fast froze the video.
  - The method that works (plan→render) has no verifier at all.
- **Latent target-likelihood checks do not detect drift** (AUC 0.19–0.60). The best cheap signal is on decoded frames: DINO + colour, AUC 0.900 at ≈ 0.5 s per check.
- **Exact tests only work per position.** A chunk-level likelihood ratio is all-or-nothing.
- **In v2 the verifier has a different job.** It should accept or reject a cheap plan, or trigger a Fast re-draw, never a Base patch.

### 5.1 Tried (measured)
Each row is mapped to the professor's four categories: likelihood = target-likelihood proxy, low-res = low-resolution verification, learned = learned lightweight verifier, physics = physics-aware reward.

**(a) Per-chunk offline benchmark.** 68 jar/castle videos, 816 chunks labelled by eye (364 drifted); `vbench/auc.md`, NIGHT2.

| verifier | category | cost per check ↓ | AUC, 68 videos ↑ | AUC, 6 Fast-only probe videos ↑ |
|---|---|---|---|---|
| DINO + colour/80 (the deployed `sem` gate) | – | **≈ 0.5 s** | **0.900** (castle 0.914) | – |
| DINO nearest neighbour to the Base-70 frame bank | – | ≈ 0.5 s | <u>0.888</u> | **0.960** |
| DINO distance to the start frame | – | ≈ 0.5 s | 0.872 | 0.864 |
| HSV colour shift | – | ≈ 0.5 s | 0.857 | <u>0.909</u> |
| logistic Base-vs-Fast discriminator on DINO features (leave-one-seed-out) | learned | ≈ 0.5 s | 0.832 | 0.765 |
| one-step Base x0 distance, 360p, no CFG (latent) | likelihood + low-res | 0.85–0.88 s | – | 0.596 |
| one-step Base x0 distance, 240p / 360p with CFG | likelihood + low-res | – | – | 0.391 / 0.378 |
| one-step Base x0 distance, 480p + CFG | likelihood | 3.5–9.8 s | – | 0.454 |
| one-step Base x0 distance, block-causal KV, CFG / no CFG | likelihood | 0.85–3.9 s | – | 0.222 / 0.190 |

**(b) Clip-level signals.** 9/17–18; positives are Fast or corrupted clips, negatives are Base.

| signal | category | cost per clip ↓ | Fast vs Base ↑ | teleport ↑ | shuffle ↑ | freeze ↑ |
|---|---|---|---|---|---|---|
| VGGT camera-rotation jump (max) | – (geometry) | <u>≈ 4 s</u> | **0.81** | **1.00** | **1.00** | 0.31 |
| Base diffusion loss at σ 0.24 | likelihood | ≈ 6 s | <u>0.75</u> | <u>0.69</u> | <u>0.81</u> | 0.31 |
| V-JEPA 2 surprise | – | **≈ 2 s** | 0.69 | 0.62 | 0.75 | 0.31 |

- No signal detects freezing: all score 0.31, below chance.
- VGGT is unchanged at 128p (low-res works). The diffusion-loss verifier degrades with resolution: 0.75 → 0.69 → 0.56.

**(c) Other measured signals**

| verifier | category | result | verdict |
|---|---|---|---|
| Base diffusion loss as a physics prior (LikePhys, 11 scenarios) | physics + likelihood | PPE 0.51 (chance 0.50) | a texture prior, not a physics prior |
| chunk-level Monte-Carlo likelihood ratio (`mc`) | likelihood | acceptance either 1 or 1e-93 to 1e-125 | degenerate at 295k dims |
| exact per-position speculative test (`--specstep`, T_H^c9) | likelihood (exact) | per-position acceptance 0.58–0.94; whole chunk ≤ 10^-25; 3 Base CFG calls per chunk (≈ 24 s) | exact, but it never saves a Base call and costs more than a whole plan→render |
| VLM judge (Qwen3-VL-32B, StEvo protocol, n = 85) | physics / semantics | separates in aggregate (control 65.5% vs 50.6%, physics 52.4% vs 32.9%), but the pairwise quality judge prefers Fast 41:18 and was order-consistent on only 51–77 of 85 pairs | unreliable pairwise |
| VisionReward, CLIP | reward | rate Fast ≥ Base (S02: 0.097 vs 0.047) | not Base-likeness |
| online gates on the two prompts | – | one-step x0 gate fired on 12 of 12 chunks (fixed threshold) or 8–10 (adaptive); `sem` gate missed a second jar, spoon + coffee-coloured sand (score 0.90), a car/pier (0.87 / 0.65) | detector blind spots: new or duplicate objects, material swaps, intended state changes |

**Correctors** (see the notes for the full table):
- 16-step Base RePaint: seams and snap-back.
- Base continuing from Fast's σ 0.94 state: Fast's drift passes through.
- Base-guided Fast nodes: frozen.
- x0-MSE LoRAs: blur or freeze.
- KV translator: roughly equal to the hint-channel route.
- **plan→render: works** (115 / 140 / 206 s).

### 5.2 Not yet tried (ranked)

| rank | candidate | category | plug-in point | cost per check ↓ | why | main risk |
|---|---|---|---|---|---|---|
| **1** | **offline bake-off of decoded-frame signals** on existing labelled data (816 chunks, the online misses, the l4 castle-4321 tower hallucination): object persistence with expected-outcome rules, camera-trajectory consistency vs the command, render-vs-plan DINO, a targeted VLM yes/no rubric | – / physics / learned | offline, then P0–P2 | seconds per clip; **0 generation cost** | targets the 3 blind spots; doubles as the choice of item-4 primary function | VLM–human agreement; Base's own loop-closure rate is low |
| **2** | **speculative planning:** l4 plan (41 s) as the draft, a plan verifier (VGGT rotation / VLM rubric / object persistence at 576×304), fall back to the cr plan (133 s) | low-res | P0 (plan) | ≈ 2–4 s per plan | v2's failure is plan hallucination, which DINO misses | needs verifier recall on plan hallucinations; only a few positives |
| **3** | **learned probe on Fast's node-1 x0 or features** (σ 0.94, where Fast's layout is already fixed) → Fast re-draw with new noise; variant: Discriminator Rejection Sampling (Azadi et al., ICLR 2019, https://arxiv.org/abs/1810.06758), which is approximately target-distributed only if the discriminator is near-optimal | learned | P1 | ≈ 0 (a small head) | the check sees the final layout at a quarter of the chunk cost; a re-draw keeps continuity | needs node-1 states dumped (re-generation); labels are per chunk |
| 4 | render-vs-plan latent consistency (x0 at node 1 downsampled to the plan grid) | low-res | P1 / P2 | ≈ 0 | cheap drift-from-plan alarm | plan and render differ legitimately in detail |
| 5 | SDVG-style ImageReward (min over frames) threshold (https://arxiv.org/abs/2604.17397) | reward | per chunk | ≈ 1 s | the published baseline for draft-verify video diffusion (1.59× at 98.1% VisionReward) | biased by design; our reward models prefer Fast |
| 6 | physics-specific VLM rubric (VideoPhy-style) | physics | clip | seconds | covers item 5's "physics-aware" category | LikePhys was at chance; our failures are semantic, not physical |
| 7 | parallel one-step Base checks at plan resolution (item 6) | likelihood + low-res | P0 | 684 vs 1,537 tokens per latent frame: 2.25× fewer, not the 5× in NIGHT3 | early detection | raw latent distances are at chance; batched forwards fail, so each check is a separate forward |

**Next three experiments.**
1. Run the rank-1 bake-off, CPU/GPU inference on existing videos only.
2. Speculative planning with the winning verifier.
3. The node-1 probe with Fast re-draw.

Deprioritise one-step Base likelihood checks. The LLM precedent (Judge Decoding, https://arxiv.org/abs/2501.19309) supports trading exactness for a learned judge, but any such verifier must be scored together with motion vs Base, the boundary/inside ratio, and watching the video.