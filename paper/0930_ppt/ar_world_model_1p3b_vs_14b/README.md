# Action-conditioned AR world model alone: LingBot v2 1.3B causal-fast vs 14B causal-fast (jar, castle; seeds 1234/4321/9999 shown, 19 seeds rated)

Same start image, same camera-action file, same prompt, same seed; 4 denoising nodes, 480p (832x480), 153 frames, 16 fps, one GPU each; no plan, no verifier, no Base. Files `<scene>__seed<seed>__<arm>__<seconds>s.mp4`, side-by-side = 1.3B | 14B.

| model | time / clip (153 f) | blind severe-clean /38 (3 raters, 63/76 unanimous) | strict-clean /38 | majority drift categories |
|---|---|---|---|---|
| 1.3B causal-fast | 11.4 s (11.2–11.9) | 7 (jar 3, castle 4) | 0 | new_object 35, teleport 29, material_swap 17, texture_collapse 3 |
| 14B causal-fast | 63.4 s (61.3–64.7) | 2 (jar 0, castle 2) | 0 | teleport 36, lighting_change 22, new_object 19 |

Both drift in essentially every clip once the camera leaves the start view (that is why the pipeline needs a plan). The 1.3B is 5.6x faster; its failures are objects morphing / new objects rather than the 14B's scene teleport + lighting change.
For reference (other batches, same seeds): Base-70 28/32, v2 l4 plan + 14B render 30/38.
