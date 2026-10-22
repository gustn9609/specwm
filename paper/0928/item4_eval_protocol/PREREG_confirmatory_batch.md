# Pre-registration: item 4 confirmatory test (written 2026-09-28 08:42 CDT, before any of these clips existed)

## Question
On the two failure prompts, how often does each arm produce a clip free of the failures Base-70 rarely makes?

## Primary endpoint (binary, per clip)
severe = majority (>= 2 of 3) blind raters mark drift AND list at least one of teleport, color_collapse, texture_collapse, freeze
(with >= 2 votes for that category). Clean = not severe.
- Raters: the unchanged rubric workflow `rater_workflow.js` (sha1 f8d32cc864c4), 3 independent model raters per clip.
- Sheets: `scripts/blind_sheets.py` (sha1 8a437c89028c), START frame + 15 frames, hashed file names, arm hidden.
- Scoring: `scripts/blind_cats.py` (sha1 b52aec3a1357) 'no_severe'.
- The strict any-drift endpoint is reported as secondary only (Base passes it 3/16 in the exploratory batch).

## Design
- Prompts: sand_filling_up_jar, sandcastle_wave_impact (camera command as in all earlier runs).
- Fresh seeds: 101-116 for both prompts (16 x 2 = 32 clips per arm). None of these seeds were used before.
- Arms: A Base-70 (official pipeline, 464x848); D Fast alone (`never`); E1 v2-safe (cr plan + plan6_ck400 render, hint-low plain, start 0);
  E2 v2-fast-CFG (l4c plan + same render). No other arms; no tuning between now and the analysis.

## Hypotheses and tests (alpha 0.05, two-sided, Holm over the 5 tests)
1. E1 vs A, 2. E2 vs A: pooled Boschloo exact test on clean counts; Cochran-Mantel-Haenszel by prompt as a check.
3. E1 vs D, 4. E2 vs D, 5. D vs A: same test.
Exploratory estimate this is powered for: A 0.75 vs E 0.40 (power ~0.8 at n = 32 per arm).

## Validity checks reported regardless of outcome
- Fleiss kappa of the raters; per-prompt tables; by-eye review of every disagreement between the rater majority and the user's own judgement on a 20-clip sample.

## Amendment (2026-09-28 11:00 PM CDT, before any of these clips existed)
Arm added after registration at the user's instruction to include the SDVG baseline in every comparison: **F = SDVG port, τ = −2.2** (`sdvg_<task>:<seed>:-2.2`; Fast draft, ImageReward worst-frame router, first generated chunk forced to Base, rejected chunks Base 16-step RePaint), same seeds 101–116 × jar/castle. Reported as a secondary comparison (F vs A, F vs E2), not part of the five pre-registered tests.
