"""[10/16 frozen baselines] Collect ABot-World's OWN example images into baselines/scenes/abot/<scene>.{png,txt} (txt = their caption or empty) plus
baselines/scenes/abot/SOURCES.md (provenance) and baselines/scenes/abot/fit1280x704/<scene>.png (the image exactly as ABot's loader sees it:
PIL ImageOps.fit to 1280x704, LANCZOS, centre crop — pipeline/causal_inference.py set_first_frame_latent; feed THIS copy to the other models so
every model sees the same crop). Sources: tools/abot/web_client/datasets/images + scene_presets.yaml (7 images, 2 with captions, 5 with an
explicit empty prompt), and the 8 project-page case posters assets/imgs/ABot-World-Case-{0..7}.png that live in the repo's git history
(commit 38d1575, 'Add Github Page', 2026-04-19; removed from HEAD, still served from the project page's OSS bucket, unreachable from here).
The current assets/gifs/ABot-World-0.gif is a 4x4 mosaic with a logo overlay, not a usable start image (not used).
usage: /home/hk28965/specwm/tools/abot/venv/bin/python -I /home/hk28965/specwm/baselines/abot/collect_scenes.py
"""
import os, sys, hashlib, subprocess, io
from PIL import Image, ImageOps
S = "/home/hk28965/specwm"; ABOT = S + "/tools/abot"; OUT = S + "/baselines/scenes/abot"; FIT = OUT + "/fit1280x704"; os.makedirs(FIT, exist_ok=True)
W, H = 1280, 704
DEFAULT_PROMPT = "A realistic outdoor world scene with a navigable path, natural lighting, detailed ground texture, and stable forward motion."
SUBURB = ("A modest, single-story suburban house with weathered stucco walls, a low-pitched shingled roof, and striped awnings over the windows sits behind a sagging "
          "chain-link fence, flanked by overgrown shrubs and a mature palm tree; distant glass-and-steel skyscrapers pierce the hazy skyline, contrasting sharply with the "
          "sun-bleached, faded earth tones of the foreground. The street is lined with cracked concrete sidewalks, utility poles strung with overhead wires, and scattered "
          "debris, while a rusted sedan rests beside a garage under a row of tall palms. Adjacent commercial structures feature peeling signage—“Beer • Wine” and a vintage "
          "“Busch” billboard—above corrugated metal facades, all bathed in bright, diffused daylight that casts soft shadows and emphasizes textures of aged paint, dry grass, "
          "and worn asphalt. The overall aesthetic evokes a sun-drenched, slightly neglected urban fringe, where mid-century residential architecture meets fading commercial "
          "infrastructure under a pale blue sky.")
# (scene name, source, caption, view, note)
REPO = [
    ("ruins_explorer_3rd", "web_client/datasets/images/example.png", DEFAULT_PROMPT, "third-person", "scene_presets.yaml group example, label 'example'; no 'prompt' key -> inherits default_prompt (= caption here); also HF Space examples/example.png ('Sample scene'); official bench image (scripts/inference.py BENCH_REF_IMAGE)"),
    ("suburban_house_1st", "web_client/datasets/images/example_first.jpg", SUBURB, "first-person", "scene_presets.yaml group example, label '城郊小院' (suburban yard); explicit caption (JPEG decoded once and stored as PNG)"),
    ("desert_valley_3rd", "web_client/datasets/images/84b90ad568b693d2.png", "", "third-person", "scene_presets.yaml showcase label '004_第三人称沙漠河谷行走' (third-person desert river-valley walk), prompt '' ; HF Space examples/desert_valley.png ('Desert valley'); RGBA -> RGB"),
    ("alpine_meadow_path_1st", "web_client/datasets/images/20fee71742eed3c0.png", "", "first-person", "scene_presets.yaml showcase label '007_第一人称高山草甸小径' (first-person alpine meadow trail), prompt ''; the HF Space ships the same file as examples/forest_stream.png ('Forest stream' — their label is swapped with the next one)"),
    ("forest_stream_stones_1st", "web_client/datasets/images/960a962e0e710275.png", "", "first-person", "scene_presets.yaml showcase label '010_第一人称林间溪流踏石' (first-person forest stream stepping stones), prompt ''; HF Space examples/mountain_meadow.png ('Mountain meadow' — swapped label)"),
    ("aurora_ice_3rd", "web_client/datasets/images/fe7d590734555ed0.png", "", "third-person", "scene_presets.yaml showcase label 'memory_3rd_data_Landscape_0055', prompt ''"),
    ("desert_camel_3rd", "web_client/datasets/images/8a0f99ae390546c4.png", "", "third-person", "scene_presets.yaml showcase label 'physics_3rd_data_Landscape_0010', prompt ''"),
]
CASES = ["case0_lake_grassland_1st", "case1_shanghai_night_1st", "case2_great_wall_1st", "case3_sailing_ship_1st", "case4_desert_caravan_3rd", "case5_neon_alley_cat_3rd", "case6_cherry_blossom_park_1st", "case7_cbd_skyline_1st"]
CASE_COMMIT = "38d1575950bcf31b66158d6700d647a4b04cdf42"

def md5(b): return hashlib.md5(b).hexdigest()
rows = []
def put(name, im, src_desc, caption, view, note, raw_md5):
    im = im.convert("RGB"); im.save("%s/%s.png" % (OUT, name)); open("%s/%s.txt" % (OUT, name), "w").write(caption)
    ImageOps.fit(im, (W, H), method=Image.LANCZOS, centering=(0.5, 0.5)).save("%s/%s.png" % (FIT, name))
    rows.append((name, "%dx%d" % im.size, view, src_desc, raw_md5, "yes (%d chars)" % len(caption) if caption else "empty", note)); print(name, im.size, view, "caption" if caption else "no caption")
for name, rel, cap, view, note in REPO:
    b = open(os.path.join(ABOT, rel), "rb").read(); put(name, Image.open(io.BytesIO(b)), "tools/abot/" + rel, cap, view, note, md5(b))
for i, name in enumerate(CASES):
    b = subprocess.run(["git", "-C", ABOT, "show", "%s:assets/imgs/ABot-World-Case-%d.png" % (CASE_COMMIT, i)], capture_output=True, check=True).stdout
    put(name, Image.open(io.BytesIO(b)), "git history: assets/imgs/ABot-World-Case-%d.png @ %s" % (i, CASE_COMMIT[:7]), "", "third-person" if name.endswith("3rd") else "first-person",
        "project-page case %d poster (https://amap-cvlab.github.io/ABot-World/ 'Explore Our Created Worlds' lane, <img alt='ABot-World case %d preview'>, the first frame shown for video ABot-World-Case-%d.mp4); no caption published" % (i, i, i), md5(b))
head = subprocess.run(["git", "-C", ABOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
md = ["# ABot-World scene set (baselines/scenes/abot) — provenance", "",
      "Collected 2026-10-16 by baselines/abot/collect_scenes.py from the ABot-World repo clone at /home/hk28965/specwm/tools/abot (github.com/amap-cvlab/ABot-World, HEAD %s)." % head,
      "`<scene>.png` = the image as shipped (JPEG/RGBA decoded once, saved as PNG); `<scene>.txt` = the caption ABot ships for it (empty when their preset says prompt '' or no caption exists).",
      "ABot's loader (pipeline/causal_inference.py set_first_frame_latent) fits every start image to 1280x704 by scale-to-cover + centre crop (PIL ImageOps.fit, LANCZOS);",
      "`fit1280x704/<scene>.png` is that exact crop — give it to the other models / ours so everybody sees the same pixels (PROTOCOL rule 2).",
      "The web client prepends `| unknown |` to every caption (web_client/app.py _apply_caption_prefix) and sends `| unknown |` alone for an empty caption; baselines/abot/bench_abot.py does the same.",
      "", "| scene | size (as shipped) | view | source | md5 (source bytes) | caption | notes |", "|---|---|---|---|---|---|---|"]
for r in rows: md.append("| " + " | ".join(r) + " |")
md += ["", "Not used: `assets/gifs/ABot-World-0.gif` (HEAD) is a 4x4 mosaic of demo clips with the logo overlaid on every frame — no single start frame; the HF Space acvlab/abot-world-interactive ships the same 4 files as the repo (md5-identical: desert_valley = 84b90ad…, example = example.png, forest_stream = 20fee71…, mountain_meadow = 960a962…);",
       "the 8 case posters are also at https://amap-cvlab.oss-cn-zhangjiakou.aliyuncs.com/github/imgs/ABot-World-Case-{0..7}.png (DNS fails from spi0), identical files in the repo's git history at commit %s." % CASE_COMMIT,
       "Views: 'third-person' = a character/animal is in frame (ABot's A/D then drive the character), 'first-person' = no character. Captions are ABot's own text where it ships one; the case posters have none."]
open(OUT + "/SOURCES.md", "w").write("\n".join(md) + "\n"); print("wrote", OUT + "/SOURCES.md", len(rows), "scenes")
