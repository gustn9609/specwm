# step1_sdvg_selfforcing

Times on dedicated GPUs (10/2 clean re-run), model loading excluded; 832x480 unless noted, 16 fps.

| scene | seed | arm | time ↓ (s, incl. decode) | blocks accepted / rejected |
|---|---|---|---|---|
| jar | 1234 | drafter only | 16.1 | 13 / 0 (pure: no verifier) |
| jar | 1234 | SDVG tau -0.7 | 28.9 | 12 / 1 |
| jar | 1234 | SDVG tau 1.3 | 65.5 | 1 / 12 |
| jar | 1234 | target only | 53.4 | 0 / 13 (pure: drafter never runs) |
| jar | 4321 | drafter only | 16.2 | 13 / 0 (pure: no verifier) |
| jar | 4321 | SDVG tau -0.7 | 29.0 | 12 / 1 |
| jar | 4321 | SDVG tau 1.3 | 55.6 | 4 / 9 |
| jar | 4321 | target only | 53.3 | 0 / 13 (pure: drafter never runs) |
| jar | 9999 | drafter only | 16.2 | 13 / 0 (pure: no verifier) |
| jar | 9999 | SDVG tau -0.7 | 29.5 | 12 / 1 |
| jar | 9999 | SDVG tau 1.3 | 55.4 | 4 / 9 |
| jar | 9999 | target only | 53.2 | 0 / 13 (pure: drafter never runs) |
| castle | 1234 | drafter only | 17.5 | 13 / 0 (pure: no verifier) |
| castle | 1234 | SDVG tau -0.7 | 30.3 | 12 / 1 |
| castle | 1234 | SDVG tau 1.3 | 30.3 | 12 / 1 |
| castle | 1234 | target only | 54.2 | 0 / 13 (pure: drafter never runs) |
| castle | 4321 | drafter only | 17.6 | 13 / 0 (pure: no verifier) |
| castle | 4321 | SDVG tau -0.7 | 30.1 | 12 / 1 |
| castle | 4321 | SDVG tau 1.3 | 56.5 | 4 / 9 |
| castle | 4321 | target only | 53.6 | 0 / 13 (pure: drafter never runs) |
| castle | 9999 | drafter only | 17.4 | 13 / 0 (pure: no verifier) |
| castle | 9999 | SDVG tau -0.7 | 29.3 | 12 / 1 |
| castle | 9999 | SDVG tau 1.3 | 36.9 | 10 / 3 |
| castle | 9999 | target only | 54.0 | 0 / 13 (pure: drafter never runs) |
