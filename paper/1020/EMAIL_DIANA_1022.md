Subject: Edge-cloud world model — quick status (10/22)

Hi Diana,

Short update on the edge-cloud world-model line after the 10/20 meeting with Guandao. Details are in paper/1020/REPORT_DIANA.md.

Where we are
- Framing: squeezed 14B on a 24 GB GPU (GPTQ-4 + FP8 KV, window 12, 2 steps) + the original 14B on the cloud re-anchoring the device's KV cache. Literature check done: no prior edge-cloud split of an interactive video world model.
- Established: device-only tiers lose the scene within ~20 s on 60-s rollouts; device + cloud keeps it for the full minute (scene-persistence 53-69 % -> 87-98 %), camera error 3-4x lower, drift and VBench better; the cloud helps only inside the same model family (1.3B device does not work).
- Guandao's cloud low-res + device SR baseline is scored: full-res quality at 1.3x fps, but cloud-only generation.

Action items (this week)
1. Finish the ~1,000-clip run and scoring: quality vs cloud fraction (50 / 25 / 12.5 / 0 % of chunks sent), 60-s cells at n = 12, edge-only 10-s rows for the 32 / 24 GB tiers.
2. Re-run 34 hybrid clips contaminated by GPU sharing (slow cloud -> corrections aged out); add "cloud keeps up / drops stale chunks" to the collaboration metrics.
3. Write up the cloud-fraction curve and the 60-s matrix with the scene-persistence metric.
4. Next phase (needs lab GPUs): config-aware distillation of the device model and a learned re-anchor transition.

Best,
[name]
