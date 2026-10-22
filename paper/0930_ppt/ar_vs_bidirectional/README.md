# AR (SDVG drafter / target) vs their bidirectional teachers, same prompts, seed 1234, 832x480

AR clips: 16 fps (native Wan rate; the first upload was muxed at 24 fps by mistake and re-muxed 9/30 11 AM, frames unchanged) (jar/castle 13 blocks = 150 frames; tokyo/bigsur 61 blocks = 729 frames). Bidirectional Wan2.1 T2V 1.3B and 14B: 81 frames at 16 fps (5 s, their training length; they cannot make 30 s clips), 50 UniPC steps, CFG 5, one GPU. Side-by-side clips show the first 5 s of each, resampled to 16 fps.

| prompt | AR 1.3B Self-Forcing (drafter) | AR 14B Krea (target) | bidirectional 1.3B | bidirectional 14B |
|---|---|---|---|---|
| jar | 28 s | 77 s | 132 s | 650 s |
| castle | 28 s | 70 s | 130 s | 645 s |
| tokyo_street | 144 s | 342 s | 132 s | 647 s |
| bigsur_drone | 140 s | 339 s | 130 s | 645 s |

Times: AR = full clip (150 or 729 frames); bidirectional = 81 frames. Per 81 frames the bidirectional 14B costs ~645 s vs ~40 s for AR 14B (81 frames = 27 blocks).


## 14B only: AR 14B (Krea target) vs bidirectional 14B (Wan2.1 T2V, its teacher)
`<prompt>__seed1234__side-by-side_AR14B-vs-bid14B_first5s.mp4` — left AR 14B (first 5 s of the clip, 24→16 fps), right bidirectional 14B (81 frames, 16 fps), 768x432 each. Per-clip time: AR 14B 70–77 s (150 frames) / 339–342 s (726 frames); bidirectional 14B 645–650 s (81 frames).
