# f13 latency table (Chenfeng's sdv2_latency method; our port scripts/wm_latency.py)

Finding: Ours realtime mode has the lowest action→frame latency of all measured baselines and modes: p50 1.30 s at 10.8 fps on one GPU (teacher 3.24 s / 4.2 fps, student 1.43 s / 10.4 fps); on jar + sandcastle SDVG is slower than the teacher at both τ (3.0–3.7 vs 4.2 fps).

Protocol (all rows): jar + sandcastle × seeds 1234/4321/9999, 161 frames (153 output frames per clip), empty text prompt, one RTX PRO 6000 Blackwell GPU per run, streaming full-VAE decode on a side CUDA stream of the same GPU (decode competes with generation), prepare excluded.

- Profile rows (n = 6): scripts/sdvg_profile.sh. Original stack = bf16 eager (10/5 slide code, run fast0); optimized stack = FP8 W8A8 + SageAttention 2 + torch.compile + ring KV (run fast1); the two ran concurrently on two separate GPUs; no other generation jobs, a VLM judge server (vLLM, Qwen3-VL-32B) ran on another GPU; prepare config means 23–27 s (both models, T5 on CPU).
- Ours rows: scripts/timing_pass.sh, a later separate run on the optimized stack (2 lanes on 2 GPUs, no other generation jobs, VLM judges on other GPUs; facts §15), specwm.res_split_infer --stream-decode 1 --no-device: cloud 14B + streaming decode on the same GPU, no device stage, frames decoded at the cloud resolution (realtime 368–384×704, quality 448–464×848–864, 704 variants 368–384×704, 832 separate 448–464×848–864, 576 304×560; loop.json lo_shape × 8).
- n = 5 for ours realtime, ours quality and 704 separate commit: castle s1234 excluded, because the warm-up covered only jar's latent shape and that clip paid a cold torch.compile (gen 38.8 / 47.5 / 30.7 s vs 14.0–14.6 / 22.1–22.6 / 16.4–17.0 s for the kept clips; timing_pass/sandcastle_wave_impact/<arm>_s1234/loop.json). n = 6 for the other variants (per-task warm-up seed 1).

Definitions (scripts/wm_latency.py): server_ms = frame ready in host memory − start of its chunk (action consumed). Output fps = (frames − 1) / (last ready − first ready) per clip (whole clip incl. the start, not steady fps; facts §13), mean over the clips. First frame = first ready − start of generation, mean over the clips. Miss = share of frames with server_ms > 750 ms (one chunk of playback). Gen s = loop.json gen_s (generation wall time, prepare excluded), mean over the clips.

Bold = best, underlined = second best in each column among the baselines and our operating modes (on the displayed rounding). The other variants (italic, second table) are not operating modes and are not ranked.

| config | clips | output fps ↑ | server_ms p50 ↓ | p90 ↓ | p99 ↓ | miss > 750 ms ↓ | first frame after gen start ↓ | gen s / 10 s clip ↓ |
|---|---|---|---|---|---|---|---|---|
| original: draft only (1.3B) | 6 | 8.2 | 1750 | 2046 | 2076 | <u>99.3 %</u> | 641 ms | 18.8 |
| original: target only (14B 4-step) | 6 | 2.2 | 5931 | 6424 | 6601 | 100 % | 3855 ms | 71.1 |
| original: SDVG τ −0.7 | 6 | 1.8 | 7568 | 7997 | 8064 | 100 % | 3974 ms | 86.8 |
| original: SDVG τ −2.2 | 6 | 2.1 | 7379 | 7922 | 8083 | 100 % | 3971 ms | 75.7 |
| optimized: draft only (= student) | 6 | <u>10.4</u> | <u>1434</u> | <u>1680</u> | <u>1700</u> | **96.7 %** | **439 ms** | <u>14.6</u> |
| optimized: target only (= teacher) | 6 | 4.2 | 3237 | 3728 | 3810 | 100 % | 1714 ms | 37.8 |
| optimized: SDVG τ −0.7 | 6 | 3.0 | 4463 | 4897 | 4946 | 100 % | 1794 ms | 51.2 |
| optimized: SDVG τ −2.2 | 6 | 3.7 | 4156 | 4851 | 4947 | 100 % | 1791 ms | 43.3 |
| ours: realtime mode (704, 2-step, fused + partial commit) | 5 | **10.8** | **1296** | **1492** | **1495** | **96.7 %** | <u>574 ms</u> | **14.3** |
| ours: quality mode (832, 2-step, fused + partial commit) | 5 | 6.9 | 1995 | 2298 | 2310 | 100 % | 852 ms | 22.4 |

Other variants of ours (same timing pass; not operating modes):

| config | clips | output fps ↑ | server_ms p50 ↓ | p90 ↓ | p99 ↓ | miss > 750 ms ↓ | first frame after gen start ↓ | gen s / 10 s clip ↓ |
|---|---|---|---|---|---|---|---|---|
| *704 2-step, fused, no partial commit (c704_fused)* | 6 | 12.0 | 1180 | 1368 | 1373 | 96.7 % | 494 ms | 12.9 |
| *704 2-step, separate commit (c704_sep)* | 5 | 9.3 | 1484 | 1689 | 1697 | 99.3 % | 716 ms | 16.7 |
| *832 2-step, separate commit (u2_832_sep)* | 6 | 6.1 | 2229 | 2577 | 2585 | 100 % | 1052 ms | 25.7 |
| *576 2-step, fused + partial commit (c576_pc1)* | 6 | 17.1 | 818 | 929 | 930 | 60.1 % | 390 ms | 9.1 |

- 704 fused without partial commit is faster than the realtime mode but has 12 % boundary jumps vs 4 % with partial commit (facts §5; data/f02_boundary_counts.json).
- 576 fused + partial has the lowest latency measured (p50 818 ms) but runs at a lower resolution (304×560) and still misses 750 ms on 60.1 % of frames.
- Generation alone (quiet steady, decode excluded; facts §15): ours realtime 15.73 fps, quality 9.74 fps, teacher-parity 6.80 fps, teacher 4.08 fps. For our two modes the output fps above (10.8 / 6.9, whole-clip means) are lower because the full-VAE decode shares the GPU (facts §13). Do not set the teacher's output fps (4.2, a whole-clip mean, 1.095× its steady rate) against its 4.08: on the steady basis the teacher gives 3.80 fps with the streamed decode vs 4.08 fps generation only (data/e_steady_fps.json opt_target; facts §13 / §15). Teacher-parity has no output-fps row.
- SDVG scope: slower than the teacher on jar + sandcastle only. Here τ −2.2 takes 5.2 (optimized) / 3.8 (original) of 13 chunks from the draft (hard prompts; facts §13); on bench-43 it takes 10.1 and is faster than the teacher (6.4 fps whole-clip / 6.7 steady vs 3.9, under load; facts §2), at a quality cost (Δ quality vs teacher −0.84 ± 0.10).
- Teacher-parity mode (cw07_safe) has no latency row: spec_early has no streaming decode, so it records no per-frame ready times (needs --stream-decode support; facts §13). Matrix-Game 2.0 and Waypoint log no per-frame ready times either (whole-clip fps only, facts §15).

Per clip, SDVG vs the teacher of the same stack (same task and seed): τ −0.7 is slower on 6/6 clips in both stacks. τ −2.2 is faster on 1/6 (optimized) and 1/6 (original) clips, only with ≥ 7 of 13 chunks taken from the draft: optimized jar s4321, 10 of 13: 4.65 vs 4.12 fps, p50 2645 vs 3240 ms; original castle s1234, 7 of 13: 2.58 vs 2.28 fps, p50 3766 vs 5784 ms; optimized castle s4321 (7 of 13) has a lower p50 (2765 vs 3181 ms) but lower fps (4.01 vs 4.21).

Sources: lingbot-world-v2/specwm_out/sdvg_profile/latency/<label>/summary.json (profile rows) and lingbot-world-v2/specwm_out/timing_pass/latency/<label>/summary.json (ours rows); both match paper/meeting_1012/facts.md §13 / §15. Bench-43 SDVG contrast: facts §2; teacher steady output fps: paper/meeting_1012/data/e_steady_fps.json. Generated by paper/meeting_1012/scripts/f13_latency.py.
