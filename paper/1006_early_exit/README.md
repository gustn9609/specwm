# Token-wise early exit for the 14B target (Speculative Forcing head) — 10/6

Head: `scripts/head_train.py` on `specwm_out/gap_dump_v3` (90 scenes, 14B 4-step rollouts; features = last-block hidden, channel-standardized, random-projected 5120->1536, 2x2 pooled; label = per-token gap to the 4-step output). Test on 11 unseen scenes (incl. jar, castle): AUROC@eps0.24 = 0.933 / 0.961 / 0.976 (k = 1/2/3), Spearman 0.88 / 0.90 / 0.91. Predicted-policy mean depth 2.18 vs oracle 2.15 at eps 0.24.
Inference: `specwm/early_exit.py` (fast_patch.ACTIVE = active-token forwards, frozen K/V for committed tokens, 9x9 spatial + temporal smoothing), `specwm/spec_early.py` (1 GPU), `spec_pipe.py --t-early EPS` (2-GPU pipeline).

## 1 GPU, same code path (jar/castle x seeds 4321, 9999; FP8 + Sage + ring KV; prepare ~2.7 s, decode ~8.5 s not overlapped here)
| Arm | mean steps/token ↓ | 14B gen ↓ (s) | denoise / verify / commit (s) | e2e ↓ (s) |
|---|---|---|---|---|
| uniform 4-step | 4.00 | 31.3 | 25.2 / 0 / 6.1 | 45.9 |
| early exit eps 0.16 | 3.18 | 27.0 | 20.9 / 0.1 / 6.1 | 41.5 |
| early exit eps 0.24 (bucket 512) | 2.52 | 23.6 | 17.5 / 0.1 / 6.1 | 38.2 |
| early exit eps 0.24 (bucket 128, castle only) | 2.1-2.4 | 20-21 | 14-15 / 0.1 / 6.0 | - |
| uniform 2-step | 2.00 | 18.4 | 12.3 / 0 / 6.0 | 32.8 |

## 2-GPU pipeline (draft on GPU B, target on GPU A; streamed decode; jar/castle x seeds 4321, 9999)
| Arm | e2e ↓ (s) | 14B denoise / commit (s) | draft chunks accepted (of 12) | mean steps (14B chunks) |
|---|---|---|---|---|
| 1.3B draft only (1 GPU) | 15.4 | - | 12 | - |
| 14B target only, uniform 2-step | 21.6 | 12.1 / 6.1 | 0 | 2.0 |
| 14B target only, early exit 0.24 | 25.9 | 16.3 / 6.1 | 0 | 2.48 |
| force-first 3 + latent critic 0.05, 2-step target | 18.7 | 3.9 / 6.0 | 8.2 | 2.0 |
| force-first 3 + latent critic 0.05, early-exit target | 22.9 | 7.4 / 6.0 | 7.0 | 2.70 |
| accept-all, force-first 3 | 17.1 | 2.2 / 6.0 | 10 | - |

Quality (own inspection; blind rating pending): early exit 0.16/0.24 ~ uniform 4-step (same drift types, incl. people appearing on castle at seed 4321 for BOTH); uniform 2-step shows more structural artifacts (glass building, car). No validated metric separates them yet.
Videos: specwm_out/early_cmp/<task>/<arm>_s<seed>/video.mp4, specwm_out/pipe_ff3/<task>/<arm>_s<seed>/video.mp4
