# target only: original vs optimized 14B

| scene | seed | version | 14B steps | prepare (s) | gen (s) | decode (s) | e2e ↓ (s) |
|---|---|---|---|---|---|---|---|
| jar | 1234 | original (bf16, eager, SDVG loop) | 4 | 11.3 | 64.4 | 8.6 | 84.3 |
| jar | 1234 | optimized S7 (FP8+compile+Sage+ring KV, streamed bf16 decode) | 2 | 6.2 | 43.5 | overlapped | 49.6 |
| jar | 4321 | original (bf16, eager, SDVG loop) | 4 | 11.3 | 64.4 | 8.6 | 84.3 |
| jar | 4321 | optimized S7 (FP8+compile+Sage+ring KV, streamed bf16 decode) | 2 | 2.6 | 25.3 | overlapped | 27.9 |
| jar | 9999 | original (bf16, eager, SDVG loop) | 4 | 11.3 | 64.4 | 8.6 | 84.3 |
| jar | 9999 | optimized S7 (FP8+compile+Sage+ring KV, streamed bf16 decode) | 2 | 2.7 | 25.3 | overlapped | 28.0 |
| castle | 1234 | original (bf16, eager, SDVG loop) | 4 | 11.3 | 62.5 | 8.5 | 82.3 |
| castle | 1234 | optimized S7 (FP8+compile+Sage+ring KV, streamed bf16 decode) | 2 | 6.0 | 47.7 | overlapped | 53.7 |
| castle | 4321 | original (bf16, eager, SDVG loop) | 4 | 11.3 | 62.5 | 8.5 | 82.3 |
| castle | 4321 | optimized S7 (FP8+compile+Sage+ring KV, streamed bf16 decode) | 2 | 2.9 | 24.8 | overlapped | 27.7 |
| castle | 9999 | original (bf16, eager, SDVG loop) | 4 | 11.3 | 62.5 | 8.5 | 82.3 |
| castle | 9999 | optimized S7 (FP8+compile+Sage+ring KV, streamed bf16 decode) | 2 | 3.0 | 24.8 | overlapped | 27.8 |
