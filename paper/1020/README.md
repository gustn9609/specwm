# 1020 — divergence injection: does a small/fast generator follow an injected 14B context?

**Terms (ours, not standard names).** "envelope ratio" = normalized distance to the 14B: |seed-mean path of the arm − seed-mean path of the 14B| ÷ (mean distance of the 14B's own seeds from their mean), i.e. a z-score-like distance in units of the 14B's seed-to-seed spread; < 1 = within the range the 14B itself produces across seeds (distribution level). "follow ratio" = same-seed following: d_own / (d_14B + d_own) with same-seed, same-noise latent distances; 1 = on the 14B's own trajectory, 0 = on the generator's own trajectory (trajectory level; its ceiling is set by rollout chaos, see the `every` arm).

Question (10/20): generate with the 1.3B; at chunk k replace its context with the 14B 4-step latents of the same seed (re-committed through the 1.3B's
own clean-context forward); does the 1.3B then continue like the 14B, or go back to what it would have generated anyway?
Setup = the f20 path figure: key-press stream `tau2_s0` (13 chunks x 3 latent frames, 832x480), 5 scenes x 10 seeds, arms `alone`, `once@k` (k = 1/3/6/9),
`every` (14B context after every chunk = ceiling), `self@3` (re-commit its OWN latents = floor, numerical noise only). Probe: `lingbot-world-v2/specwm/inject_probe.py`;
latents `lingbot-world-v2/specwm_out/inject/<gen>/<scene>/<arm>_s<seed>/lat.pt`; facts §52 in `paper/meeting_1012/facts.md`.

| file | what |
|---|---|
| `figs/f21_inject_clean_1p3b_4step.png` | meeting figure: row 1 seed-mean paths in latent PCA (T black, 1.3B alone orange, inject @6 blue, every green dashed; diamond = injection); row 2 envelope ratio per chunk (< 1 = inside the 14B seed spread); right = follow ratio per arm (`f21_clean.py`) |
| `figs/f21_inject_full_1p3b_4step.png` | the probe's full figure (per-seed lines, same-seed distance curves) |
| `figs/sheet_1p3b_sandcastle_s1234.png` | 1 fps contact sheet, sandcastle seed 1234, 7 rows |
| `videos/inject_1p3b_4step_<scene>_s1234.mp4`, `..._ALL5_s1234.mp4` | 3 x 2 side-by-side at 16 fps: 14B (context source), 1.3B alone, injected @3 / @6 / @9, every. Red border + red bar segments = the shown chunk's context came from the 14B; white box on the bar = current chunk (`make_inject_videos.py`) |
| `data/inject_tables.md` | follow ratio / envelope distance / first-4-chunk tables (English) |
| `data/f21_inject.json`, `data/f21_inject_clean_1p3b_4step.json` | numbers behind the figures |

Findings (1.3B <- 14B, 10 seeds x 5 scenes):
- After injection the 1.3B does NOT return to its own path: envelope ratio alone 1.4–16 -> 0.1–0.5 after injection (inside the 14B seed spread); distance to its own alone path 0.4–6.8 vs 0.01–0.05 for self re-commit.
- It does not reproduce the 14B's same-seed trajectory either: follow ratio once@k 0.49–0.61, every 0.67 (ceiling: the AR rollout is chaotic), self 0.15.
- One injection fades in ~4–6 chunks (3–4.5 s); later injections follow better (once@1 0.49 -> once@9 0.61) because the injected context then already contains the 14B's committed content (e.g. the person on the shore in sandcastle), which the 1.3B keeps; an early context (castle + wave only) is continued with the 1.3B's own prior (sand flats, bushes).
- Divergence point k*: the 1.3B mean path is outside the 14B envelope from chunk 0 on 4/5 scenes (jar: chunk 3) — there is no "similar first, then diverge" phase at the distribution level; the f20 impression came from the shared first frame.
- Control (14B 2-step <- 14B 4-step, same procedure) pending (running 10/20 PM); it tells whether the 0.67 ceiling is model difference or rollout chaos.
