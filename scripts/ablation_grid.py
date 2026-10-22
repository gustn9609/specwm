"""Ablation comparison videos: for each task, a 3x2 grid of the cumulative stages S0..S5 (same seed), each panel labelled with the stage and its
end-to-end time; plus ablation.json with per-stage timings. usage: python scripts/ablation_grid.py ROOT OUTDIR"""
import sys, os, json, glob, numpy as np, imageio
from PIL import Image, ImageDraw, ImageFont
ROOT, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True); EXT = len(sys.argv) > 3 and sys.argv[3] == "ext"
ST = [("S0_original", "S0 original 14B"), ("S1_patches", "S1 + exact-math patches"), ("S2_compile", "S2 + torch.compile"),
      ("S3_fp8", "S3 + FP8 W8A8"), ("S4_2step", "S4 + 2 sampling steps"), ("S5_pipeline", "S5 + T5/VAE pipeline")]
if EXT: ST = [("S5_pipeline", "S5 2 steps"), ("S6_1step", "S6 1 step"), ("S7_sage", "S7 2 steps + SageAttn")]
try: FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
except Exception: FONT = ImageFont.load_default()
PW = 448
FIRST = 'easy_lake'   # first task of every process: compiled stages include the one-time torch.compile warm-up (and S5 the first full VAE encode)
def label(fr, txt):
    im = Image.fromarray(fr); d = ImageDraw.Draw(im); w = d.textlength(txt, font=FONT) if hasattr(d, "textlength") else 8 * len(txt)
    d.rectangle([0, 0, w + 12, 24], fill=(0, 0, 0)); d.text((6, 4), txt, fill=(255, 255, 255), font=FONT); return np.array(im)
J = {}
for task in sorted({os.path.basename(os.path.dirname(p)) for p in glob.glob(ROOT + "/*/S0_original_s*")}):
    info = []; vids = []
    for arm, name in ST:
        d = glob.glob("%s/%s/%s_s*" % (ROOT, task, arm))[0]; j = json.load(open(d + "/loop.json"))
        e2e = j.get("e2e_s") or (j["prepare_s"] + j["gen_s"] + (j["decode_s"] or 0))
        info.append(dict(stage=name, arm=arm, prepare_s=j["prepare_s"], gen_s=j["gen_s"], denoise_s=j["times"]["denoise"], commit_s=j["times"]["commit"],
                         decode_s=j["decode_s"], e2e_s=round(e2e, 1), steps=len(j["steps"]), streamed=j.get("stream_decode", False)))
        fr = [np.asarray(f) for f in imageio.get_reader(d + "/video.mp4")]; h, w = fr[0].shape[:2]; ph = int(round(h * PW / w / 2)) * 2
        warm = task == FIRST and arm != "S0_original" and arm != "S1_patches"; info[-1]["includes_warmup"] = warm
        vids.append([label(np.array(Image.fromarray(f).resize((PW, ph), Image.BILINEAR)), "%s  |  %.1f s%s" % (name, e2e, " *" if warm else "")) for f in fr])
    T = min(len(v) for v in vids); gap = np.zeros((vids[0][0].shape[0], 4, 3), np.uint8)
    if EXT:
        frames = [np.concatenate([x for k in range(3) for x in ((vids[k][t],) + ((gap,) if k < 2 else ()))], 1) for t in range(T)]
        imageio.mimwrite("%s/%s_ext.mp4" % (OUT, task), frames, fps=16, quality=6, macro_block_size=8); Image.fromarray(frames[T // 2]).save("%s/%s_ext_poster.jpg" % (OUT, task), quality=80)
        J[task] = info; print(task, os.path.getsize("%s/%s_ext.mp4" % (OUT, task)) // 1024, "KB", flush=True); continue
    rows = lambda t, r: np.concatenate([x for k in range(3) for x in ((vids[r * 3 + k][t],) + ((gap,) if k < 2 else ()))], 1)
    frames = [np.concatenate([rows(t, 0), np.zeros((4, rows(t, 0).shape[1], 3), np.uint8), rows(t, 1)], 0) for t in range(T)]
    imageio.mimwrite("%s/%s.mp4" % (OUT, task), frames, fps=16, quality=6, macro_block_size=8)
    Image.fromarray(frames[T // 2]).save("%s/%s_poster.jpg" % (OUT, task), quality=80)
    J[task] = info; print(task, os.path.getsize("%s/%s.mp4" % (OUT, task)) // 1024, "KB", flush=True)
json.dump(J, open(OUT + ("/ablation_ext.json" if EXT else "/ablation.json"), "w"), indent=1)
