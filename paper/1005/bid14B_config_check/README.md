# Bidirectional Wan2.1 T2V 14B: config check (10/2)

All clips: seed 1234, 81 frames, 16 fps, 50 UniPC steps, default negative prompt, no prompt extension, one GPU; 832x480 unless noted.
Panels (left to right): 1.3B official code CFG5/shift5 (what we used) | 1.3B with the README recommendation CFG6/shift8 | 14B SDVG-repo code (the clip shown on 9/30) |
14B official Wan2.1 code, same settings | 14B CFG3 | 14B shift3 | 14B at 1280x720 (its native resolution).

Findings: (1) the SDVG repo's modified wan package (bf16 residual/RoPE) gives the same output as the official code (PSNR 24-26 dB, same colour stats), so the code is not the problem.
(2) The harsh, over-contrasted 14B look comes from CFG 5 at 480p: CFG 3 cuts clipped pixels 65->20 % (bigsur), 55->32 % (tokyo), 29->11 % (castle); 720p does the same (castle 29->8 %) and follows the prompt better ('elaborate sandcastle').
(3) shift 3 (diffusers' 480p hint) does not help.

| prompt | variant | sat mean ↓ | high-sat px % ↓ | clipped px % ↓ | value mean | sat drift first→last ↓ | gen time (s) |
|---|---|---|---|---|---|---|---|
| jar | 1.3B archive | 0.295 | 11.2 | 16.3 | 0.569 | 0.034 | 132.2 |
| jar | 1.3B official | 0.297 | 11.2 | 16.3 | 0.567 | 0.034 | 133.4 |
| jar | 14B archive | 0.269 | 8.1 | 11.7 | 0.531 | 0.040 | 650.2 |
| jar | 14B official | 0.270 | 8.2 | 11.8 | 0.532 | 0.039 | 647.6 |
| jar | 1.3B official guide 6 shift 8 | 0.286 | 11.7 | 20.1 | 0.565 | 0.025 | 133.8 |
| castle | 1.3B archive | 0.366 | 14.8 | 23.3 | 0.738 | 0.003 | 130.1 |
| castle | 1.3B official | 0.366 | 14.9 | 21.4 | 0.733 | 0.001 | 130.4 |
| castle | 14B archive | 0.357 | 9.4 | 29.9 | 0.792 | 0.019 | 645.2 |
| castle | 14B official | 0.351 | 8.9 | 29.0 | 0.791 | 0.020 | 643.2 |
| castle | 14B official 720p | 0.347 | 9.9 | 8.0 | 0.634 | 0.032 | 2242.7 |
| castle | 14B official CFG 3 | 0.343 | 6.0 | 10.8 | 0.768 | 0.004 | 643.2 |
| castle | 14B official shift 3 | 0.333 | 6.9 | 36.4 | 0.813 | 0.031 | 641.0 |
| castle | 1.3B official guide 6 shift 8 | 0.368 | 15.0 | 20.9 | 0.703 | 0.002 | 168.6 |
| tokyo_street | 1.3B archive | 0.599 | 34.5 | 36.1 | 0.578 | 0.013 | 132.2 |
| tokyo_street | 1.3B official | 0.591 | 32.4 | 34.2 | 0.578 | 0.015 | 156.2 |
| tokyo_street | 14B archive | 0.536 | 35.9 | 55.7 | 0.462 | 0.029 | 647.4 |
| tokyo_street | 14B official | 0.535 | 35.6 | 55.3 | 0.461 | 0.029 | 644.5 |
| tokyo_street | 14B official 720p | 0.571 | 37.0 | 50.6 | 0.357 | 0.017 | 2242.4 |
| tokyo_street | 14B official CFG 3 | 0.534 | 25.7 | 32.0 | 0.488 | 0.012 | 643.5 |
| tokyo_street | 14B official shift 3 | 0.492 | 33.4 | 56.5 | 0.443 | 0.013 | 641.4 |
| tokyo_street | 1.3B official guide 6 shift 8 | 0.569 | 33.2 | 38.6 | 0.599 | 0.001 | 130.3 |
| bigsur_drone | 1.3B archive | 0.806 | 68.4 | 51.1 | 0.537 | 0.005 | 130.1 |
| bigsur_drone | 1.3B official | 0.804 | 68.1 | 51.1 | 0.533 | 0.005 | 133.7 |
| bigsur_drone | 14B archive | 0.758 | 58.0 | 66.7 | 0.594 | 0.010 | 645.4 |
| bigsur_drone | 14B official | 0.754 | 57.0 | 64.8 | 0.593 | 0.009 | 641.2 |
| bigsur_drone | 14B official CFG 3 | 0.665 | 38.7 | 20.3 | 0.553 | 0.006 | 646.0 |
| bigsur_drone | 14B official shift 3 | 0.758 | 57.6 | 67.0 | 0.627 | 0.031 | 644.6 |
| bigsur_drone | 1.3B official guide 6 shift 8 | 0.793 | 67.6 | 50.4 | 0.522 | 0.008 | 154.9 |

