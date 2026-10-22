# 10/26 meeting material (started 10/22)

## videos/ — 60-s side-by-side, Matrix-Game set (640x352, program P6, same seed 1234)
2x2 grid: top-left teacher (4-step, original 14B, 72 GiB) | top-right device-only 32 GB class (mx32: FP8 KV, window 12, 2-step)
bottom-left 32 GB device + cloud (rt2c1hyb32) | bottom-right 24 GB device + cloud (rt2c1hyb24, GPTQ-4). Label = scene-persistence
(fraction of 1-fps frames whose CLIP zero-shot class is "scene" vs sky / abstract / UI / blank) and DINOv2 cut count (scripts/scene_persist.py).
figs/m60_mg2_<scene>_montage.png = the same grid at t = 0 / 15 / 30 / 45 / 59 s.

| clip | what you see (checked frame by frame) |
|---|---|
| m60_mg2_universal_0000 (hay stack, snow) | teacher: haystack -> blue bubble texture at 15 s -> clouds / sky only from 30 s. mx32: bubble texture, sky, fur-like smear at 59 s. Both +cloud: snowy field with a lake, hills and mountains for the whole minute (scene 57 / 61 % vs 31 / 21 %). |
| m60_mg2_mc_forest_00 (Minecraft forest) | teacher: forest -> trees from below -> blue lattice at 30 s -> sky / moon at 45-59 s. mx32: forest -> a crowd / blue grid -> sky -> blurry trees. Both +cloud: Minecraft forest and grass all 60 s; the 24 GB one grows a walking person at 30-45 s (content invention, but the scene stays) — scene 98 / 100 % vs 70 / 64 %. |
| m60_mg2_universal_0003 (houses, street) | teacher keeps the street (97 %). mx32 goes to sky / sun at 30-45 s (56 %). Both +cloud keep houses and street (97 / 100 %). |
| m60_mg2_universal_0001 (football pitch) | the modest case: teacher 87 %, mx32 70 % (a person, then a grey bubble sphere at 59 s), +cloud 93 / 87 % (pitch -> building courtyard; the last 24 GB frame goes grey). |

n = 4-scene means in paper/1020/data/scene_persist_1021.md: device-only 53-69 %, teacher 79 %, +cloud 84-98 %. Per-scene numbers vary
(universal_0006: mx32 87 % > rt2c1hyb24 48 %), so the n = 12 table (pending) is the one to quote.
