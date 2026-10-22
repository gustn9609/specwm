"""[10/20] Tables + contact sheets for the inject follow-ups: reads data/inject_vlm.json (judge), data/inject_boundary.json, data/sheet_frames/*.npz
-> prints markdown tables, writes figs/boundary_<scene>_k6_s1234.png. usage: python3 inject_followups_report.py"""
import json, os, glob, re, numpy as np
from PIL import Image, ImageDraw, ImageFont
S = "/home/hk28965/specwm"; D = S + "/paper/1020/data"; FG = S + "/paper/1020/figs"
SCENES = ["sand_filling_up_jar", "sandcastle_wave_impact", "easy_lake", "easy_stonehenge_sunset", "pouring_water_into_cup"]
SHORT = {"sand_filling_up_jar": "jar", "sandcastle_wave_impact": "sandcastle", "easy_lake": "lake", "easy_stonehenge_sunset": "stonehenge", "pouring_water_into_cup": "cup"}
ARMS = ["T", "alone", "every", "once@6", "hand@3m1", "hand@3m3", "hand@6m2"]
def mse(x): x = np.asarray(x, float); x = x[~np.isnan(x)]; return x.mean(), (x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0), len(x)
def fmt(m, se, best, hi): s = "%.2f ± %.2f" % (m, se); return "**%s**" % s if (best is not None and abs(m - best) < 1e-9) else s
# ---------------- Test 1: judge
vj = D + "/inject_vlm.json"
if os.path.exists(vj):
    J = json.load(open(vj)); rows = {}
    for p, r in J.items():
        m = re.match(r".*/(T|alone|every|once@6|hand@3m1|hand@3m3|hand@6m2)_(.+)_s(\d+)\.mp4$", p)
        if m: rows[(m.group(1), m.group(2), int(m.group(3)))] = r
    print("\n## Test 1 — Qwen3-VL judge (8 frames/clip): quality 1–10 ↑, drift 0–100 ↓; mean ± s.e. over seeds (n)\n")
    for key, arrow in (("quality", "↑"), ("drift", "↓")):
        print("| %s %s | " % (key, arrow) + " | ".join(ARMS) + " |"); print("|---|" + "---|" * len(ARMS))
        for sc in SCENES + ["pooled"]:
            vals = {a: [r[key] for (aa, s, sd), r in rows.items() if aa == a and (sc == "pooled" or s == sc)] for a in ARMS}
            ms = {a: mse(v) for a, v in vals.items() if v}; best = (max if key == "quality" else min)(m[0] for m in ms.values())
            print("| %s | " % SHORT.get(sc, sc) + " | ".join(fmt(*ms[a][:2], best, None) + (" (n=%d)" % ms[a][2] if sc == "pooled" else "") if a in ms else "–" for a in ARMS) + " |")
        print()
    print("### Paired differences (same scene + seed), mean ± s.e.; (+/−/0) = seeds better / worse / tie in the metric's own direction\n")
    PAIRS = [("every", "alone"), ("once@6", "alone"), ("T", "alone"), ("hand@3m1", "alone"), ("hand@3m3", "alone"), ("hand@6m2", "alone"), ("hand@3m1", "once@6"), ("hand@3m3", "once@6"), ("hand@6m2", "once@6")]
    for key, arrow in (("quality", "↑"), ("drift", "↓")):
        print("| %s %s | " % (key, arrow) + " | ".join("%s − %s" % pr for pr in PAIRS) + " |"); print("|---|" + "---|" * len(PAIRS))
        for sc in SCENES + ["pooled"]:
            cells = []
            for a, ref in PAIRS:
                d = np.array([rows[(a, s, sd)][key] - rows[(ref, s, sd)][key] for (aa, s, sd) in rows if aa == a and (sc == "pooled" or s == sc) and (ref, s, sd) in rows]); d = d[~np.isnan(d)]
                if len(d) == 0: cells.append("–"); continue
                m, se, n = mse(d); good = int((d > 0).sum()) if key == "quality" else int((d < 0).sum()); bad = int((d < 0).sum()) if key == "quality" else int((d > 0).sum())
                cells.append("%+.2f ± %.2f (n=%d; %d/%d/%d)" % (m, se, n, good, bad, n - good - bad))
            print("| %s | " % SHORT.get(sc, sc) + " | ".join(cells) + " |")
        print()
# ---------------- Test 2: boundary
bj = D + "/inject_boundary.json"
if os.path.exists(bj):
    B = json.load(open(bj)); KS = [3, 6, 9]
    print("\n## Test 2 — re-anchor boundary jump (frame-to-frame change at the chunk boundary k|k+1; pixel L1 on 8-bit ↓, 1−SSIM ↓)\n")
    print("Splice = alone chunks 0..k + once@k chunks k+1.. (what a device user sees). Ordinary = the same boundary index k|k+1 in alone and in T (unspliced). Ratio = splice / mean(alone, T) at that boundary. n = seeds.\n")
    for met, name in (("l1_b", "L1"), ("ss_b", "1−SSIM")):
        print("| %s | k | splice | alone (ordinary) | T (ordinary) | once@k own boundary | ratio splice/ordinary | seeds ratio>1.5 | within-chunk alone |" % name); print("|---|---|---|---|---|---|---|---|---|")
        for sc in SCENES + ["pooled"]:
            for k in KS:
                keys = [kk for kk in B if (sc == "pooled" or kk.startswith(sc + "/")) and all(x in B[kk] for x in ("splice@%d" % k, "once@%d" % k, "alone", "T"))]
                if not keys: continue
                sp = np.array([B[kk]["splice@%d" % k][met][k] for kk in keys]); al = np.array([B[kk]["alone"][met][k] for kk in keys]); T = np.array([B[kk]["T"][met][k] for kk in keys])
                ow = np.array([B[kk]["once@%d" % k][met][k] for kk in keys]); wi = np.array([B[kk]["alone"][met[:2] + "_in"] for kk in keys])
                ratio = sp / ((al + T) / 2); flag = " ⚠" if ratio.mean() > 1.5 else ""
                print("| %s | %d | %.2f ± %.2f | %.2f ± %.2f | %.2f ± %.2f | %.2f ± %.2f | **%.2f**%s (med %.2f) | %d/%d | %.2f |" % (SHORT.get(sc, sc), k, *mse(sp)[:2], *mse(al)[:2], *mse(T)[:2], *mse(ow)[:2], ratio.mean(), flag, np.median(ratio), int((ratio > 1.5).sum()), len(ratio), wi.mean()))
        print()
    # all-boundary average reference
    print("Ordinary boundaries averaged over ALL 12 chunk boundaries (pooled over scenes and seeds), to show the boundary itself is not special:\n")
    print("| arm | L1 at boundaries | L1 within chunk | 1−SSIM at boundaries | 1−SSIM within chunk |"); print("|---|---|---|---|---|")
    for a in ("alone", "T", "every", "once@6"):
        V = [v for v in B.values() if a in v]
        print("| %s | %.2f | %.2f | %.3f | %.3f | n=%d |" % (a, np.mean([np.mean(v[a]["l1_b"]) for v in V]), np.mean([v[a]["l1_in"] for v in V]), np.mean([np.mean(v[a]["ss_b"]) for v in V]), np.mean([v[a]["ss_in"] for v in V]), len(V)))
# ---------------- contact sheets
try: FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
except Exception: FONT = ImageFont.load_default()
for f in sorted(glob.glob(D + "/sheet_frames/*_k6_splice@6.npy")):
    base = f[:-len("splice@6.npy")]; b = 12 * 6 + 8; lo = b - 8; sc = os.path.basename(f).split("_s1234")[0]; W, H = 208, 120
    arms = [("device view: alone 0..6 → once@6 7..", "splice@6"), ("1.3B alone", "alone"), ("once@6 (14B ctx 0..6, 1.3B 7..)", "once@6"), ("14B 4-step (T)", "T")]
    rows = [(lab, np.load(base + a + ".npy")) for lab, a in arms if os.path.exists(base + a + ".npy")]
    n = rows[0][1].shape[0]; im = Image.new("RGB", (W * n, (H + 18) * len(rows) + 16), (20, 20, 20)); d = ImageDraw.Draw(im)
    for i in range(n):
        t = lo + i; d.text((i * W + 4, 1), "f%d%s" % (t, "  | chunk 7 starts" if t == b + 1 else ("  chunk 6 ends |" if t == b else "")), font=FONT, fill=(255, 90, 90) if t in (b, b + 1) else (200, 200, 200))
    for r, (lab, fr) in enumerate(rows):
        y0 = 16 + r * (H + 18)
        for i in range(n):
            im.paste(Image.fromarray(fr[i]).resize((W, H), Image.BILINEAR), (i * W, y0 + 18))
        d.text((4, y0 + 2), lab, font=FONT, fill=(240, 240, 240)); x = (b + 1 - lo) * W; d.line([(x, y0 + 18), (x, y0 + 18 + H)], fill=(255, 60, 60), width=3)
    out = "%s/boundary_%s_k6_s1234.png" % (FG, sc); im.save(out); print("wrote", out)
