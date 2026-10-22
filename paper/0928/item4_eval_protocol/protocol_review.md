## 4. Evaluation protocol

### 4.1 Verdict on the current criterion (`scripts/drift_rate2.py`)
The rule is: per-chunk nearest-Base DINO distance, with τ = the 95th percentile of the leave-one-out (LOO) Base distances over 8 seeds. It is **not a valid 5% test**, for four reasons.
1. **Line 24.** With 8 values, `np.percentile(loo, 95)` = x(7) + 0.65·(x(8) − x(7)). Only the maximum lies above τ, so the Base no-drift rate is 7/8 by construction; it is never measured. The realised false-positive rate is ≈ 11–16% and varies a lot between prompts (0.6–31%).
2. **Circularity (line 31).** The same Base clips set τ and serve as references. Method clips are scored against 8 references but LOO against 7, which biases the result in favour of the method. Scoring `base70` as a method gives distance 0.
3. **Line 16.** Chunk means of L2-normalised features are not re-normalised. The score is then a mean pairwise distance that penalises panning or flicker and rewards freezing.
4. **Line 23.** Taking the min per chunk allows chimera matches (seed A in chunk 3, seed B in chunk 7).

**Other defects:**
- Line 21 drops missing Base seeds silently.
- Windows are offset by 3 frames from the generated chunks.
- `drift_rate.py`'s docstring says the thresholds were "never exceeded" by Base, but Base jar s1234 and s9999 have chunk jumps of 0.48 and 0.59, above the 0.45 threshold.

### 4.2 Recommended protocol
**1. Pre-register one primary binary function, and validate it before looking at any method arm.**
- Positives are the hand-labelled cases in NIGHT3_NOTES:
  - `fd1_cr_semc100`: castle under water, a person, the jar tips over, wall/doors, deer;
  - `mixed46` jar material swap;
  - `fl_ck470` texture collapse;
  - `bg1_n01` freeze.
- Negatives are the Base-70 clips.
- Requirements: sensitivity ≥ 0.8 and Base pass rate ≥ 0.85.
- The function must encode the expected outcomes (StEvo): the castle *should* be obliterated, leaving flattened wet sand and foam; the jar's sand pile should grow. Losing the castle is therefore not drift.
- Prefer functions that need no Base calibration set:

| function | drift := | calibration | main risk |
|---|---|---|---|
| **VLM yes/no rubric** (per chunk, frame 0 as reference, blind to method) | any "bad" answer about our logged drift types: material swap, new person/animal/vehicle, environment change, jar tipped, featureless texture | none | measure VLM–human κ on ≥ 40 labelled clips |
| **camera-trajectory consistency** (RAFT, VGGT or MegaSaM yaw vs the command: +3°/frame to 360° at frame 132, then −3°/frame) | yaw error above a tolerance at chunk boundaries, or no reversal | a tolerance | Base's own compliance varies with the seed |
| **object persistence with expected outcome** (OWLv2 / Grounding-DINO + SAM 2) | jar/funnel missing, duplicated or changing class while in view (frames 0–25, 119–145) | detector threshold | Base does not close the loop (`to_start` 0.79–0.98 at chunk 11) |
| freeze / seam / collapse detectors | inside-chunk motion below the Base minimum; boundary/inside ratio; `hf_energy` collapse | thresholds from separate Base seeds | catches bg1-style freezes that DINO-NN rewards |
| DINO-NN (only if kept) | trajectory-level NN distance > conformal τ | **≥ 19 separate** Base seeds (5% level) | apply the bug fixes above; run an A/A test |

**2. Arms.** Use the same seed list per prompt.

| arm | what it is | flags |
|---|---|---|
| A | Base-70 (reference) | existing `base70` bundle |
| B | target-only T_H^c9 (null arm for exactness) | `--policy never --bguide 1 --bguide-nodes 0,1,2 --ctx 9 --resident` (the review's version omitted `--ctx 9`; it must match C) |
| C | `--specstep` | queued (`specstep_n012c9_never`) |
| D | Fast only (negative control) | `never` |
| E | plan→render (cr, l4c) | queued plan/render arms |

**3. Tests.**
- **Per prompt:** Boschloo exact unconditional test (`scipy.stats.boschloo_exact`), with the Newcombe hybrid-score 95% CI for the difference.
- **Pooled over the two prompts:** Cochran–Mantel–Haenszel stratified by prompt.
- **McNemar:** only for arms that share random numbers. That requires the seeded ξ/u fix; a shared seed *label* does not pair Fast-based arms with Base.
- **Claim that the spec sampler is exact:** use TOST equivalence, or better the per-transition residual test from item 3.
- **Multiple comparisons:** Holm correction.
- **Secondary analyses:**
  - Mann–Whitney or permutation tests on the continuous score;
  - Kaplan–Meier / log-rank for the time to the first drifting chunk (links to item 6).
- **By eye:** watch the videos (the user's rule).

**4. Comparisons and their meaning.**
- C vs B = exactness; the difference must be zero.
- B vs A = the gap between the target kernel and Base.
- C vs A and E vs A = the headline result.

### 4.3 Seeds needed (detect 0.90 vs 0.60 no-drift, α = 0.05 two-sided, power 0.80)

| design | n per arm per prompt ↓ |
|---|---|
| one-sided, normal approximation | **25** |
| arcsine (Cohen h = 0.726) | <u>30</u> |
| normal, pooled variance | 32 |
| exact Boschloo | 32 |
| exact Fisher (power 0.800 at 36; saw-tooth, so use 38) | 36–38 |
| two prompts pooled with CMH (homogeneous effect) | ≈ 16 per prompt |
| non-inferiority, margin 0.15, both at 0.9 | 50 |
| TOST equivalence, margin 0.15 | 69 |

- **Current 8 seeds:** Fisher power 0.095, Boschloo power 0.195. The Fisher values (0.095 / 0.748 / 0.800 / 0.830 at n = 8 / 32 / 36 / 38) were re-verified by exact enumeration on the CPU; the Boschloo values come from the protocol review.
- **Cost.** Base-70 takes ≈ 2,800 s per clip on one GPU (measured range 2,766–3,111 s).
  - With a calibration-free function: 32 per prompt, i.e. 24 more for the jar and 29 more for the castle, ≈ 41 GPU-h.
  - With a DINO-NN function: also 19 calibration seeds per prompt, ≈ 30 more GPU-h.
  - Spec and target-only arms: ≈ 345 s × 2 GPUs per clip. Fast only: 62 s. plan→render: 114–206 s.

### 4.4 Status at 05:32 CDT
| arm | jar clips done | castle clips done |
|---|---|---|
| Base-70 | 5 (1234, 4321, 9999, 11, 22); s33/s44 dirs empty | 3 |
| specstep c9 | 1 of 8 (s4321 running) | 1 of 8 |
| Fast only | 3 | 2 |
| target-only (arm B) | **not queued** | **not queued** |

### 4.5 Caveats
- **The camera action pans 360°.** The start content is out of view from about frame 25 to frame 119, so most content is legitimately hallucinated. That is why Base varies so much between seeds (DINO 0.29–0.75).
- **The Base batch looks bimodal:** some seeds stay nearly static (jar s4321: `to_start` ≤ 0.05; inferred from DINO, not checked by eye). A frozen output can therefore pass a nearest-neighbour test.
- **Known blind spots:**
  - the flicker metric rewarded a frozen video;
  - VisionReward and the pairwise VLM judge prefer Fast over Base;
  - DINO-to-Base missed the l4 plan's hallucinated towers.
- **Seeds are independent.** ξ and u come from non-deterministically seeded CUDA generators, so batches are i.i.d. but not reproducible.