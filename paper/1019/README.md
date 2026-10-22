# 1019 — side-by-side videos per backbone (frozen baselines)

One video per (backbone scene set, scene, camera program): every method on the same start image and the same camera program, at the backbone's native resolution, resampled to 16 fps, first 10 s, played in sync. `<set>_ALL_<n>clips.mp4` = all clips of that set back to back.

| folder | backbone | resolution | scenes × programs |
|---|---|---|---|
| `mg2/` | Matrix-Game 2.0 | 640x352 | 12 × 7 |
| `wp/` | Waypoint-1.5 1B | 1024x512 | 12 × 7 |
| `abot/` | ABot-World-0 5B | 1280x704 | 12 × 7 |
| `astra/` | Astra | 832x480 | 3 × 7 |

Tiles (when present): Baseline · Speculative Forcing (its own 1.3B setting, Astra set only; on LingBot 14B) · LingBot-World 14B 4-step (base) · LingBot-World 14B 2-step (naive few-step) · Ours w/o cloud (ablation) · **OURS (device screen)** · Ours, cloud-corrected stream.

Under each label: time to generate the 10-s clip on one RTX PRO 6000 (DiT only, decode excluded; hybrid = device GPU) and the Qwen3-VL-32B judge (quality 1–10 ↑ / drift 0–100 ↓). "not scored yet" = scoring was still running when the video was built (10/19 ~13:00 CDT); rebuild with `make_grids.py` after scoring.

Programs: P1 forward · P2 turn right 18°/s · P2p turn right 16°/s · P3 forward + turn 12°/s · P4 S-curve ±18°/s · P4p S-curve ±16°/s · P5 stop-and-go (forward, stop, back up, forward).

Source: `baselines/runs/*` (frozen, see `baselines/PROTOCOL.md`); builder `make_grids.py` (run with `tools/vbench_env/bin/python`).
