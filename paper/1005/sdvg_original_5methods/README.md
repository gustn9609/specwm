# sdvg_original_5methods

Times on dedicated GPUs (10/2 clean re-run), model loading excluded; 832x480 unless noted, 16 fps.

| prompt | seed | drafter only | SDVG tau -0.7 | SDVG tau 1.3 | SDVG tau 1.5 | target only |
|---|---|---|---|---|---|---|
| tokyo_street | 1234 | 19.8 s (pure) | 31.8 s (12 acc) | 49.1 s (7 acc) | 47.6 s (7 acc) | 55.1 s (pure) |
| tokyo_street | 4321 | 17.6 s (pure) | 29.2 s (12 acc) | 29.8 s (12 acc) | 29.9 s (12 acc) | 53.4 s (pure) |
| tokyo_street | 9999 | 17.6 s (pure) | 29.9 s (12 acc) | 43.0 s (8 acc) | 36.5 s (10 acc) | 53.5 s (pure) |
| bigsur_drone | 1234 | 18.3 s (pure) | 30.7 s (12 acc) | 70.2 s (pure) | 69.6 s (pure) | 53.9 s (pure) |
| bigsur_drone | 4321 | 18.1 s (pure) | 29.9 s (12 acc) | 70.2 s (pure) | 70.0 s (pure) | 53.4 s (pure) |
| bigsur_drone | 9999 | 18.0 s (pure) | 30.2 s (12 acc) | 70.0 s (pure) | 70.4 s (pure) | 54.4 s (pure) |
| jar | 1234 | 16.1 s (pure) | 28.9 s (12 acc) | 65.5 s (1 acc) | 69.1 s (pure) | 53.4 s (pure) |
| jar | 4321 | 16.2 s (pure) | 29.0 s (12 acc) | 55.6 s (4 acc) | 55.3 s (4 acc) | 53.3 s (pure) |
| jar | 9999 | 16.2 s (pure) | 29.5 s (12 acc) | 55.4 s (4 acc) | 69.2 s (pure) | 53.2 s (pure) |
| castle | 1234 | 17.5 s (pure) | 30.3 s (12 acc) | 30.3 s (12 acc) | 59.8 s (3 acc) | 54.2 s (pure) |
| castle | 4321 | 17.6 s (pure) | 30.1 s (12 acc) | 56.5 s (4 acc) | 56.4 s (4 acc) | 53.6 s (pure) |
| castle | 9999 | 17.4 s (pure) | 29.3 s (12 acc) | 36.9 s (10 acc) | 69.6 s (pure) | 54.0 s (pure) |
