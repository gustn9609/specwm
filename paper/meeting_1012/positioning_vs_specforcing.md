# Positioning vs. Speculative Forcing (arXiv 2610.22347, NAVER AI Lab / KAIST, posted 2026-10-15) — 2026-10-16

## 1. What they have (read from the PDF)
- Token-wise early commitment inside ONE frozen few-step AR generator: intermediate x̂0^(k) = draft; a 4.11 M-param conv head on the last block's hidden state predicts rel_k(i) = ‖x̂0^(k)(i) − x̂0^(K)(i)‖/σ_t; accept if ≤ ε (0.24 for world model); accepted tokens frozen with their KV; later steps forward only active tokens; clean pass at t=0 still refreshes the whole chunk (merged with the next chunk's first pass). Training-free variant (history similarity at step 1, draft stability later).
- Models: LongLive 1.3B and **LingBot-World 2.0 1.3B causal-fast** only. Throughput on one H100, DiT only, VAE excluded.
- Numbers (Table 3, LingBot 1.3B, 200 random-camera scenes, 253 frames): 4-step 18.1 fps → OursH 2.55 mean steps, 24.5 fps (**1.35×**), VBench quality 82.65 vs 82.62, RPE equal, MUSIQ-Drift 2.84 vs 3.09 (lower). Uniform 2-step = 81.68 (clearly worse).
- Ablation (Table 5): **chunk-wise acceptance with the same head, calibrated to the same mean steps: 82.54 / drift 3.21 vs token-wise 82.65 / 2.84** → they claim chunk < token.
- Limitations (D.1/D.2): speed-up depends on the backbone and domain; realized speed-up is bounded by max_i k*(i) per chunk, not the mean; ideal 5/(k̄+1) because of the clean pass.
- No: larger models, device memory, two machines, latency (TTFF / action→frame), external baselines (ABot/MG2/Waypoint), trained students.

## 2. Overlap map

| claim | Speculative Forcing | ours (facts) | verdict |
|---|---|---|---|
| learned verifier on frozen-generator features predicting the remaining gap | yes, token-wise, 1.3B, AUROC ≈ 0.9 | yes (10/6 14B head AUROC 0.93–0.98; step-1 head §47) | **same idea, they are public first** → cannot be our headline |
| adaptive step allocation per chunk | ablation only (worse than token-wise at matched mean steps) | main device policy (§12 cw07 +0.03 at 1.5×; §47 adapt25 −0.05 at 15.3 fps) | contested; our defence must be wall-clock, not FLOPs |
| token-wise on a large (14B) model | not done | done 10/6–10/8: 2.3–2.5 steps/token but **slower than uniform 2-step and dominated by dense 3-step** (stale KV of frozen tokens; step cost linear in active tokens; max k* bound) | our evidence = their D.2 limitation, measured |
| intermediate estimates as drafts for speculative verification | yes (framing) | yes (self-speculation §35) + negative result that a separate 1.3B drafter is not a usable proposal (7 % tokens within ε, exact coupling gives no speed-up) | complementary; our negative result is new |
| two-machine split of the denoising schedule, KV re-anchoring feedback | no | yes: device 2-step + cloud finish + KV feedback; cloud stream **+0.07 vs teacher** at 15.7 fps throughput, device teacher parity at 11.1 fps (§35, §47c) | **ours only** |
| device fit of a 14B world model (24 / 32 GB) | no | yes (§18–19: GPTQ-4 + FP8 KV + window 12 = 17 GiB, −0.14) | **ours only** |
| latency protocol (TTFF, action→frame, miss rate) and edge-only SOTA comparison at matched resolution | no | yes (§13, §42, baselines_ondevice.md) | **ours only** |
| trained 2-step student (DMD) | no (generator frozen) | stage 4 running (R0/R1/R2) | ours if it works |
| evaluation currency | VBench 6-dim + RPE (DA3) + MUSIQ drift, 200 scenes | Qwen3-VL judge (quality prompt not human-validated) + StEvo subset | **we must add theirs** |

## 3. Position that can be accepted
Headline = **serving a 14B interactive world model on device-class hardware**: "self-speculative edge–cloud pipelining with KV re-anchoring". Claims in order:
1. The few-step schedule of an AR video model is a pipeline whose intermediate estimates are valid states; the only state to synchronize between machines is the KV cache. Device shows its own k-step draft, cloud finishes the schedule and re-anchors the device's KV (verify-then-re-anchor). Result: cloud stream above the single-GPU teacher at 3.9× its throughput; device stream at teacher parity at 11 fps, −0.05 at 15 fps; 1 s correction lag.
2. The device policy is chunk-level step allocation with a step-1 verifier. Positioned explicitly against Speculative Forcing: at 14B scale token-wise commitment cuts token evaluations (2.3–2.5 steps) but not wall-clock (max-k* bound, stale-KV loss), chunk-level allocation converts the same verifier signal into wall-clock (+0.57 over uniform 2-step at −0.4 fps). Their method is also run as a baseline on the 14B in the same table and shown complementary (it could run on the device inside a chunk).
3. The 14B fits 24/32 GB with the KV window as the main lever; the cloud/device split is justified exactly below 24 GB or above ~11 fps on a 24 GB card.
4. Fair baselines (ABot, MG2, Waypoint, Astra) with the §7 checklist; our own metrics plus their VBench/RPE/Drift protocol on the same clips.
5. Analysis section: why draft-model speculative decoding fails for video (cross-model gap) and why MSE post-training lines fail (§46/§48/§49) — the negative results that explain the design.
6. If stage 4 works: trained 2-step device student at teacher quality = the generator-side contribution they do not have.

What NOT to lead with: "a learned verifier for adaptive denoising depth" (incremental to them and contradicted by their Table 5 unless the wall-clock argument is front and center).

## 4. Required work before submission (ordered)
1. Reimplement Speculative Forcing (token-wise, ε 0.24, their smoothing) on the 14B at 704/832 and on the 1.3B; put it in the main table with wall-clock fps on the same GPU (early_exit.py exists; bench-43 + judge ≈ 1 day).
2. Add their metrics to bench-43 arms: VBench 6-dim aggregate, RPE via Depth Anything 3 (weights already in tools/mg35), MUSIQ-Drift (≈ 1 day).
3. Their-style benchmark: 200 scenes × 5 random camera segments × 253 frames (we have the 253-frame StEvo path; add random-camera generation) so the comparison is in their currency (1 day gen on 4 GPUs).
4. Human blind rating of 40–80 clips for the quality claims the VLM judge carries.
5. Related-work paragraph citing them as concurrent (arXiv 10/15); our 10/6 head is not a priority claim, say "independently".
6. Stage 4 go/no-go (today); portability of the split to ABot (same 4-node schedule) as the scalability evidence Guandao asked for.
