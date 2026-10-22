"""Meeting material 03: six-arm comparison on the action-conditioned LingBot-World v2 (jar/castle x seeds 1234/4321/9999):
draft only | SDVG tau -0.7 (paper default) | SDVG tau -1.5 | SDVG tau -2.2 | target only (14B causal-fast) | Bidirectional 14B (official, 70 steps).
Writes paper/1005/meeting/03_six_arms/: 3x2 grid videos per case (label = arm + time), e2e chart (PNG), README.md + numbers.json.
Times: AR arms = latent generation + final VAE decode + prepare (T5 on CPU + VAE encode, 11.3 s measured once); Bidirectional = WanI2V.generate
(text/image encoding + 70 UniPC steps + decode). Model loading excluded everywhere. usage: python scripts/meeting_03_six.py"""
import os, json, glob, numpy as np, imageio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
S = "/home/hk28965/specwm"; L = S + "/lingbot-world-v2/specwm_out/loop1005"; BD = os.environ.get("BID_DIR", S + "/lingbot-world-v2/specwm_out/bid_official")
O = S + "/paper/1005/meeting/03_six_arms"; os.makedirs(O + "/videos", exist_ok=True); PREP = 11.3
ARMS = [("none", "Draft only (1.3B)"), ("ir_-0.7", "SDVG τ −0.7 (default)"), ("ir_-1.5", "SDVG τ −1.5"), ("ir_-2.2", "SDVG τ −2.2"),
        ("target", "Target only (14B AR)"), ("bid", "Bidirectional 14B")]
SC = [("sand_filling_up_jar", "jar"), ("sandcastle_wave_impact", "castle")]; SEEDS = [1234, 4321, 9999]
try: FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
except Exception: FONT = ImageFont.load_default()
def label(fr, txt):
    im = Image.fromarray(fr); d = ImageDraw.Draw(im); w = d.textlength(txt, font=FONT); d.rectangle([0, 0, w + 12, 24], fill=(0, 0, 0)); d.text((6, 4), txt, fill=(255, 255, 255), font=FONT); return np.array(im)
R = {a: [] for a, _ in ARMS}; ACC = {a: [] for a, _ in ARMS}
for task, sn in SC:
    for sd in SEEDS:
        panels = []
        for arm, name in ARMS:
            if arm == "bid":
                j = json.load(open("%s/%s_s%d.json" % (BD, task, sd))); t = j["generate_s"]; vid = "%s/%s_s%d.mp4" % (BD, task, sd); acc = None
            else:
                j = json.load(open("%s/%s/%s_s%d/loop.json" % (L, task, arm, sd))); t = j["gen_s"] + (j["decode_s"] or 0) + PREP; vid = "%s/%s/%s_s%d/video.mp4" % (L, task, arm, sd); acc = j["accepted"]
            R[arm].append(t); ACC[arm].append(acc)
            fr = [np.asarray(f) for f in imageio.get_reader(vid)]; h, w = fr[0].shape[:2]; pw = 448; ph = int(round(h * pw / w / 2)) * 2
            panels.append([label(np.array(Image.fromarray(f).resize((pw, ph), Image.BILINEAR)), "%s | %.0f s" % (name, t)) for f in fr])
        T = min(len(p) for p in panels); gap = np.zeros((panels[0][0].shape[0], 4, 3), np.uint8)
        row = lambda t_, r: np.concatenate([x for k in range(3) for x in ((panels[r * 3 + k][t_],) + ((gap,) if k < 2 else ()))], 1)
        frames = [np.concatenate([row(t_, 0), np.zeros((4, row(t_, 0).shape[1], 3), np.uint8), row(t_, 1)], 0) for t_ in range(T)]
        imageio.mimwrite("%s/videos/%s__seed%d__six_arms.mp4" % (O, sn, sd), frames, fps=16, quality=6, macro_block_size=8)
        Image.fromarray(frames[T // 2]).save("%s/videos/%s__seed%d__six_arms_mid.jpg" % (O, sn, sd), quality=85)
        print(sn, sd, "ok", flush=True)
# chart
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3dd", "axes.axisbelow": True})
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]   # validated categorical slots 1-5, 7 (fixed order)
fig, ax = plt.subplots(figsize=(8.0, 4.0), dpi=200); m = [np.mean(R[a]) for a, _ in ARMS]
bars = ax.bar(range(6), m, 0.7, color=COL)
for b_, v in zip(bars, m): ax.text(b_.get_x() + b_.get_width() / 2, v * 1.08, ("%.0f s" % v) if v < 600 else ("%.0f s (%.0f min)" % (v, v / 60)), ha="center", va="bottom", fontsize=9)
ax.set_xticks(range(6)); ax.set_xticklabels(["Draft only\n(1.3B)", "SDVG τ −0.7\n(default)", "SDVG τ −1.5", "SDVG τ −2.2", "Target only\n(14B AR)", "Bidirectional\n14B (70 steps)"], fontsize=9)
ax.set_yscale("log"); ax.set_ylim(10, 12000); ax.set_ylabel("seconds per clip ↓ (log scale)")
ax.set_title("End-to-end time per 161-frame clip (mean of jar/castle × 3 seeds)", fontsize=11, loc="left", pad=10); ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(O + "/e2e_six_arms.png"); plt.close(fig)
md = ["# 03 Six arms on the action-conditioned world model (jar/castle × seeds 1234/4321/9999)", "",
      "Times per clip, model loading excluded. AR arms: prepare (11.3 s) + latent generation + final VAE decode. Bidirectional: WanI2V.generate (encoding + 70 UniPC steps, guide 5, shift 10 + decode).", "",
      "| Arm | e2e mean ↓ (s) | min–max (s) | accepted chunks (of 13) |", "|---|---|---|---|"]
for a, n in ARMS:
    acc = "–" if ACC[a][0] is None else "%.1f" % np.mean(ACC[a])
    md.append("| %s | %.1f | %.1f–%.1f | %s |" % (n, np.mean(R[a]), min(R[a]), max(R[a]), acc))
md += ["", "![e2e](e2e_six_arms.png)", "", "Videos: `videos/<scene>__seed<k>__six_arms.mp4`, top row: draft | SDVG −0.7 | SDVG −1.5; bottom row: SDVG −2.2 | target | Bidirectional."]
open(O + "/README.md", "w").write("\n".join(md) + "\n"); json.dump(dict(times=R, accepted=ACC, prepare_s=PREP), open(O + "/numbers.json", "w"), indent=1); print("\n".join(md))
