# SDVG original code, 30-second clips (61 blocks = 729 frames, 832x480, 16 fps (re-muxed from 24 fps on 9/30; frames unchanged), seed 1234)

Prompts:
- tokyo_street: "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."
- bigsur_drone: "Drone view of waves crashing against the rugged cliffs along Big Sur's garay point beach. The crashing blue waters create white-tipped waves, while the golden light of the setting sun illuminates the rocky shore. A small island with a lighthouse sits in the distance, and green shrubbery covers the cliff's edge."

| prompt | method | time | blocks accepted / rejected (of 61) |
|---|---|---|---|
| tokyo_street | 1_drafter-only_1.3B | 144 s | 61 / 0 |
| tokyo_street | 2_SDVG_paper_tau-0.7 | 183 s | 46 / 15 |
| tokyo_street | 2b_SDVG_tau-1.3 | 256 s | 24 / 37 |
| tokyo_street | 3_target-only_Krea14B | 342 s | 0 / 61 |
| bigsur_drone | 1_drafter-only_1.3B | 140 s | 61 / 0 |
| bigsur_drone | 2_SDVG_paper_tau-0.7 | 274 s | 19 / 42 |
| bigsur_drone | 2b_SDVG_tau-1.3 | 338 s | 0 / 61 |
| bigsur_drone | 3_target-only_Krea14B | 339 s | 0 / 61 |

File names: `<prompt>__seed1234__<method>__<seconds>s.mp4`; `*__side-by-side_*.mp4` = drafter | SDVG -0.7 | SDVG 1.3 | target, labelled.
