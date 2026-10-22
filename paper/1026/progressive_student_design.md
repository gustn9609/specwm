# Progressive 2-step device student + low-res verifier + reject-to-cloud — design (10/22)

User's framing (10/22): the device (<= a 4090) runs a progressive 2-step sampler — step 1 at LOW resolution, step 2 at FULL resolution —
a verifier on the low-res step decides accept / reject per chunk; accepted chunks finish on the device, rejected chunks are corrected by the cloud.
Order agreed: (0) random-gate curve + oracle gate -> (1) distill the progressive student (Vista) -> (2) verifier head on the low-res hidden state
-> (3) reject -> cloud = two_machine with a finite --max-lag. (0) and (3) are queued (gen/queue/12, tier 12: rt2c1hyb32L1, rt2c1hyb32d, rt2c1hyb32o50/o25/o12).

## 0. Facts this design is built on (measured)
- Content is decided at step 1 (action decided at step 1, 10/13; half-res step 1 changes the scene, 10/20 multires pilot) -> verify after step 1.
- Half-res step costs 0.09 s vs 0.51 s full-res at 832-class (multires pilot table 1, mr2 = 2.12x faster than the 2-step, 10.0 vs 7.9 fps) but
  without training mr2 drifts (+18 judge drift, -20 % sharpness).
- Same-model gating: the step-1 verifier head is accurate (AUROC 0.93-0.98, facts §39) and the drift head generalises (AUROC 0.88 unseen scenes);
  end-to-end gating with LOW-RES drafts failed (§40: accepted chunks soft, accept path 0.32 s) -> the accepted path must end with a full-res step.
- Cloud must not continue from a step-1 x0 (-0.26, 10/16): on reject the cloud refines the finished device draft in its own context (current c1hyb).
- 10/13 step-1 x0-MSE LoRA blurred -> the student loss must be distribution matching (dmd14.py, Self-Forcing DMD at the 2-step schedule), not MSE.
- Low-res tokens mixed into the full-res KV failed (9/26) -> the half-res step keeps its OWN pooled KV (multires_pilot.py: pool_mat / pool_noise).

## 1. Student sampler (deployment = training rollout)
chunk c: z_lo = noise on the half grid (28x52 for 832-class; 20x40 for 640-class) -> step 1 at sigma 1.0 with the half-res KV (pooled history)
-> x0_lo -> [VERIFIER reads the layer-20 hidden of this pass] -> up(x0_lo) to the full grid -> re-noise to sigma 0.833 -> step 2 at full res with the
full KV -> x0 -> commit (full KV) + pooled commit (half KV). Accepted: done (device ~0.6 s/chunk at 832-class on the PRO 6000 vs 1.1 s for the 2-step).
Rejected: the device still finishes the chunk (keeps real time) and sends the finished x0 + action; the cloud (original 14B, 4-node) refines it at
sigma 0.625 in its own context and returns it; the device re-commits (KV re-anchor, progressive 0.25->0.5->1 blend = 10/21 best).
Device budget: 24 GB class = GPTQ-4 + FP8 KV + window 12 (+ the pooled half KV, ~1/4 of the full KV).

## 2. Training (step 1, Vista or a free local pair)
Base: specwm/dmd14.py (generator = 14B causal-fast + LoRA 'gen', free-running rollouts, random stop step per chunk; real score = bidirectional
14B on cuda:1; fake score = same base + LoRA 'fake'). Changes:
  (a) rollout(): step j = 0 runs on the half grid (multires_pilot.denoise lo_steps = 1 mechanics: prepare_res at the half grid, pooled noise,
      own KV), its x0 is up()-sampled and re-noised for step 1 at full res; stop step k in {1, 2} -> k = 1 means the sample is up(x0_lo)
      (teaches the half-res step to put the right content down), k = 2 the finished chunk.
  (b) the LoRA is node-gated: gen_lo (active at the half-res step) and gen_hi (full-res step), both rank 64 q/k/v/o/ffn; the fake critic unchanged.
  (c) DMD gradient as dmd14 (x0_fake - x0_real at a random shift-5 sigma on the whole clip, backprop through the stop step of --grad-chunks chunks).
  (d) the device CONFIG during rollouts = the deployed one (window 12, FP8 KV; GPTQ-4 is applied after training = QAT not needed for a first pass).
Data: the frozen task sets' scenes are held out; train on the 47-scene distill pool (specwm_out/distill_data47 scenes) with random action streams.
Compute: dmd14 = 2 GPUs (gen + real score), 81-frame clips, ~2000 steps; one step ~ rollout 7 chunks x (0.09 + 0.51 s) + critic + replay ~ 25 s
-> 2000 steps ~ 14 h on 2 PRO 6000; Vista 8 GPUs: 4 runs in parallel (lr / rank / p_stop ablations) in the same wall time.
Go / no-go after step (0): if the ORACLE gate at 25 % applied is within 0.1 judge / 3 scene-persistence points of the every-chunk hybrid, the
accept path is worth a verifier; if not, the verifier cannot win and the student alone (speed) is the deliverable.
Eval: 60-s mg2_long / neutral_832_long (scene persistence, judge drift, RPE) and the 10-s frozen sets vs mx32 / mx24 / rt2 / teacher; speed on the
24 GB class, and on a real 4090 if one is available (the PRO 6000 cap is a memory proxy only).

## 3. Verifier (step 2): conv/MLP head on the layer-20 hidden of the HALF-RES step (drift_head_train.py recipe: 16 pooled tokens, channel
standardisation, lr 1e-4). Labels from the student's own rollouts: (i) rel = |x_cloud - x_student| / |x_student| of the cloud refinement of the
finished chunk (the oracle gate's quantity), (ii) the drift-head logit of the finished chunk. Budget B -> threshold at the (1-B) quantile.
Measured: send fraction, applied fraction, quality vs B against the random gate (lower bound) and the oracle gate (upper bound) from step (0).

## 4. Reject -> cloud (step 3): two_machine --gate head on the half-res pass (gate_score hook) + cloud --max-lag 1 (rt2c1hyb32L1 measures the
graceful-degradation effect now, with the stock 2-step device). Collaboration metrics: cloud GPU-s / clip, latent MB / clip, correction lag,
applied fraction, and quality when the cloud is slowed (inject --delay on the cloud: 0 / 0.5 / 1 s).

## 5. Open items for the user
- Vista access (host / account / scheduler) — not in this environment; the local pair 2+3 is the fallback (one run at a time, ~14 h, only when
  no other user is on the pair).
- A 4090 for the device timing row.
