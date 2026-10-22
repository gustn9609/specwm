"""[10/20] Analysis of the multi-resolution denoising pilot (specwm/multires_pilot.py): reads specwm_out/multires/<arm>/<scene>_s<seed>/run.json
(per-chunk CUDA-synced times, rel-L2 to the 4-step reference T, Laplacian-variance sharpness) and the decoded clips specwm_out/multires/judge/*.mp4,
scores every clip with the Qwen3-VL judge on :30100 (8 evenly spaced frames, the scripts/vlm_clip_score.py prompts; cached in data/multires_vlm.json),
and writes data/multires_pilot.md (tables), figs/multires_rel_l2.png (per-chunk rel-L2 to T, mean +- s.e. over clips) and videos/multires_<scene>_s1234.mp4
(T | full2 | mr4 | mr2 side by side). CPU + HTTP only (no GPU). usage: envrun python /home/hk28965/specwm/paper/1020/multires_analysis.py [--no-judge] [--no-video]"""
import os, sys, json, glob, io, base64, time, argparse, subprocess, numpy as np
S = "/home/hk28965/specwm"; R = S + "/lingbot-world-v2/specwm_out/multires"; D = S + "/paper/1020/data"; FG = S + "/paper/1020/figs"; VD = S + "/paper/1020/videos"
ap = argparse.ArgumentParser(); ap.add_argument("--no-judge", action="store_true"); ap.add_argument("--no-video", action="store_true"); ap.add_argument("--video-scenes", default="sandcastle_wave_impact,sand_filling_up_jar,easy_lake"); a = ap.parse_args()
SCENES = ["sand_filling_up_jar", "sandcastle_wave_impact", "easy_lake", "easy_stonehenge_sunset", "pouring_water_into_cup"]; SEEDS = [1234, 4321, 9999]
SHORT = {"sand_filling_up_jar": "jar", "sandcastle_wave_impact": "sandcastle", "easy_lake": "lake", "easy_stonehenge_sunset": "stonehenge", "pouring_water_into_cup": "cup"}
ARMS = ["T", "full4", "full2", "mr4", "mr4b", "mr2"]; GEN = ["full4", "full2", "mr4", "mr4b", "mr2"]
LABEL = {"T": "T (archived 4-step)", "full4": "full4 (4-step full-res, rerun)", "full2": "full2 (2-step, nodes 0,500)", "mr4": "mr4 (steps 1-2 half)", "mr4b": "mr4b (step 1 half)", "mr2": "mr2 (2-step, step 1 half)"}
COL = {"full4": "#2a78d6", "full2": "#eb6834", "mr4": "#1baf7a", "mr4b": "#eda100", "mr2": "#e87ba4"}   # dataviz default categorical slots 1-5, fixed order
runs = {}
for arm in ARMS:
    for sc in SCENES:
        for sd in SEEDS:
            p = "%s/%s/%s_s%d/run.json" % (R, arm, sc, sd)
            if os.path.exists(p): runs[(arm, sc, sd)] = json.load(open(p))
print("runs:", {arm: sum(1 for k in runs if k[0] == arm) for arm in ARMS})
# ---------------- judge
VJ = D + "/multires_vlm.json"; J = json.load(open(VJ)) if os.path.exists(VJ) else {}
if not a.no_judge:
    import cv2; from PIL import Image; from openai import OpenAI
    client = OpenAI(api_key="local", base_url="http://127.0.0.1:%s/v1" % os.environ.get("JUDGE_PORT", "30100"))
    def frames(p, n=8):
        cap = cv2.VideoCapture(p); fr = []
        while True:
            ok, f = cap.read()
            if not ok: break
            fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        return [fr[int(k * (len(fr) - 1) / (n - 1))] for k in range(n)]
    def b64(im, W=448):
        im = Image.fromarray(im); im = im.resize((W, int(im.height * W / im.width)), Image.BICUBIC); b = io.BytesIO(); im.save(b, format="JPEG", quality=85); return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
    def ask(images, prompt):
        r = client.chat.completions.create(model="qwen3-vl-32b", messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": u}} for u in images] + [{"type": "text", "text": prompt}]}], max_tokens=8, temperature=0)
        txt = r.choices[0].message.content.strip()
        try: return float(next(t for t in txt.replace("/", " ").split() if t.replace(".", "").isdigit())), txt
        except StopIteration: return float("nan"), txt
    DRIFT = ("These 8 frames are evenly spaced from one 10-second AI-generated video that should keep showing the scene of the first frame while the camera moves. "
             "Rate how severely the video DRIFTS away from the first frame's scene: teleporting to a different place, the main objects vanishing or morphing into other things, colours or textures collapsing, or the picture freezing. "
             "Answer with ONE integer 0-100 (0 = same scene throughout, 100 = completely different scene / collapsed). Only the number.")
    QUAL = ("These 8 frames are evenly spaced from one AI-generated video. Rate the VISUAL QUALITY of the frames themselves (sharpness and detail, absence of artifacts such as blur, blocking, smearing or grid noise, and realism of textures). "
            "Ignore whether the scene changes over time. Answer with ONE integer 1-10 (10 = photographic, 1 = unusable). Only the number.")
    n0 = len(J); t0 = time.time()
    for (arm, sc, sd) in sorted(runs):
        p = "%s/judge/%s_%s_s%d.mp4" % (R, arm, sc, sd)
        if p in J or not os.path.exists(p): continue
        ims = [b64(f) for f in frames(p)]; d, rd = ask(ims, DRIFT); q, rq = ask(ims, QUAL); J[p] = dict(label=arm, drift=d, quality=q, raw=[rd, rq]); json.dump(J, open(VJ, "w"), indent=0)
    print("judge: %d new clips in %.0f s (%d total)" % (len(J) - n0, time.time() - t0, len(J)))
def jv(arm, sc, sd, key):
    r = J.get("%s/judge/%s_%s_s%d.mp4" % (R, arm, sc, sd)); return float("nan") if r is None else r[key]
# ---------------- helpers
def mse(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; return (x.mean(), x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0, len(x)) if len(x) else (float("nan"), 0.0, 0)
def rank_fmt(cells, hi_good, fmt="%.2f", se=None):
    """cells: {arm: value}; returns {arm: str} with best bold and second underlined (direction hi_good)"""
    ok = {k: v for k, v in cells.items() if v == v}; order = sorted(ok, key=lambda k: -ok[k] if hi_good else ok[k]); out = {}
    for k, v in cells.items():
        s = (fmt % v) if v == v else "–"
        if se is not None and v == v: s += " ± " + (fmt % se[k])
        if ok and k == order[0]: s = "**%s**" % s
        elif len(order) > 1 and k == order[1]: s = "<u>%s</u>" % s
        out[k] = s
    return out
L = []
def P(*x): L.append(" ".join(str(v) for v in x))
P("# Multi-resolution denoising steps — training-free pilot (10/20, GPU 5, 14B FP8 + Sage + ring KV window 18, 832x480-class grids, 13 chunks x 3 latent frames)\n")
P("Code: `lingbot-world-v2/specwm/multires_pilot.py` (generation, timing, decode, sharpness, rel-L2), `paper/1020/multires_analysis.py` (judge, tables, figure, videos). Latents: `specwm_out/multires/<arm>/<scene>_s<seed>/lat.pt`; clips `specwm_out/multires/judge/<arm>_<scene>_s<seed>.mp4`; judge cache `data/multires_vlm.json`.\n")
P("Arms: **T** = archived 14B 4-step reference (paths2/K0_uniform4P_fx, act_spec2 code path); **full4** = the same 4-step schedule rerun in this code path (= same-config floor for rel-L2 and the judge); **full2** = 2-step full-res, nodes (0,500) = sigma 1 -> 0.833 (NOT the archived `inject/14b_2step` clips, which use nodes (0,250)); **mr4** = 4-step with steps 1-2 at the half grid, 3-4 full; **mr4b** = step 1 only at the half grid; **mr2** = 2-step (0,500) with step 1 at the half grid. Half grid = round(h/4)*2 x round(w/4)*2 (58x106 -> 28x52, 56x108 -> 28x54, 58x104 -> 28x52). The half-res step attends to a separate half-res K/V cache rebuilt from every committed chunk (bilinear-antialias downsample + clean t=0 commit, the 'lo-commit' column); x0 route: the half-res x0 prediction is bilinear-upsampled and re-noised with the same full-res noise draw the reference uses. 5 scenes x 3 seeds = 15 clips per arm.\n")
# ---------------- timing
P("## 1. Per-chunk DiT time (CUDA-synced, s/chunk, mean over chunks 6-12 = full KV window; fps = 12 frames / chunk_s; share = half-res steps / total) — lower s ↓, higher fps ↑\n")
P("| arm | steps (half+full) | s/chunk all 13 ↓ | s/chunk steady ↓ | fps steady ↑ | half-res steps s | full-res steps s | full commit s | half commit s | half share | speed-up vs full4 ↑ | n clips |"); P("|---|---|---|---|---|---|---|---|---|---|---|---|")
tim = {}
for arm in GEN:
    rs = [runs[k] for k in runs if k[0] == arm]
    if not rs: continue
    st = lambda key, lo=6: np.mean([np.mean([c[key] for c in r["chunks"][lo:]]) for r in rs])
    tim[arm] = dict(all=st("total", 0), steady=st("total"), lo=st("lo_steps"), hi=st("hi_steps"), hc=st("hi_commit"), lc=st("lo_commit"), n=len(rs), steps="%d+%d" % (rs[0]["lo_steps"], len(rs[0]["nodes"]) - rs[0]["lo_steps"]))
if tim:
    rk_all = rank_fmt({k: v["all"] for k, v in tim.items()}, False); rk_st = rank_fmt({k: v["steady"] for k, v in tim.items()}, False); rk_fps = rank_fmt({k: 12 / v["steady"] for k, v in tim.items()}, True, "%.1f")
    base = tim.get("full4", {}).get("steady", float("nan")); rk_sp = rank_fmt({k: base / v["steady"] for k, v in tim.items()}, True, "%.2fx")
    for arm, v in tim.items():
        P("| %s | %s | %s | %s | %s | %.3f | %.3f | %.3f | %.3f | %.0f %% | %s | %d |" % (arm, v["steps"], rk_all[arm], rk_st[arm], rk_fps[arm], v["lo"], v["hi"], v["hc"], v["lc"], 100 * v["lo"] / v["steady"], rk_sp[arm], v["n"]))
    P("")
# ---------------- judge tables
def table(key, arrow, hi_good, fmt="%.2f", getter=None, title=""):
    P(title); P("| %s %s | " % (key, arrow) + " | ".join(ARMS) + " |"); P("|---|" + "---|" * len(ARMS))
    for sc in SCENES + ["pooled"]:
        vals = {arm: [getter(arm, s, sd) for (a_, s, sd) in runs if a_ == arm and (sc == "pooled" or s == sc)] for arm in ARMS}
        ms = {arm: mse(v) for arm, v in vals.items()}; cells = rank_fmt({arm: ms[arm][0] for arm in ARMS}, hi_good, fmt, se={arm: ms[arm][1] for arm in ARMS})
        P("| %s | " % SHORT.get(sc, sc) + " | ".join(cells[arm] + (" (n=%d)" % ms[arm][2] if sc == "pooled" else "") for arm in ARMS) + " |")
    P("")
def paired(key, arrow, hi_good, getter, pairs, title):
    P(title); P("| %s %s | " % (key, arrow) + " | ".join("%s − %s" % pr for pr in pairs) + " |"); P("|---|" + "---|" * len(pairs))
    for sc in SCENES + ["pooled"]:
        cells = []
        for arm, ref in pairs:
            d = np.array([getter(arm, s, sd) - getter(ref, s, sd) for (a_, s, sd) in runs if a_ == arm and (sc == "pooled" or s == sc) and (ref, s, sd) in runs]); d = d[~np.isnan(d)]
            if len(d) == 0: cells.append("–"); continue
            m, se, n = mse(d); good = int((d > 0).sum()) if hi_good else int((d < 0).sum()); bad = int((d < 0).sum()) if hi_good else int((d > 0).sum())
            cells.append("%+.2f ± %.2f (n=%d; %d/%d/%d)" % (m, se, n, good, bad, n - good - bad))
        P("| %s | " % SHORT.get(sc, sc) + " | ".join(cells) + " |")
    P("")
if J:
    P("## 2. Qwen3-VL judge (8 frames per clip): visual quality 1–10 ↑, drift 0–100 ↓; mean ± s.e. over clips\n")
    table("quality", "↑", True, getter=lambda arm, s, sd: jv(arm, s, sd, "quality")); table("drift", "↓", False, getter=lambda arm, s, sd: jv(arm, s, sd, "drift"))
    PAIRS = [("full4", "T"), ("full2", "T"), ("mr4", "T"), ("mr4b", "T"), ("mr2", "T"), ("mr4", "full4"), ("mr4b", "full4"), ("mr2", "full2"), ("mr4", "full2")]
    paired("quality", "↑", True, lambda arm, s, sd: jv(arm, s, sd, "quality"), PAIRS, "### Paired differences (same scene + seed), mean ± s.e.; (+/−/0) = clips better / worse / tie in the metric's own direction\n")
    paired("drift", "↓", False, lambda arm, s, sd: jv(arm, s, sd, "drift"), PAIRS, "")
# ---------------- sharpness
P("## 3. Sharpness proxy: mean Laplacian variance of the decoded 8-bit grey frames (↑ sharper; paired within a scene, grids differ across scenes)\n")
table("lapvar", "↑", True, "%.1f", getter=lambda arm, s, sd: runs[(arm, s, sd)].get("lapvar_mean", float("nan")) if (arm, s, sd) in runs else float("nan"))
paired("lapvar", "↑", True, lambda arm, s, sd: runs[(arm, s, sd)].get("lapvar_mean", float("nan")) if (arm, s, sd) in runs else float("nan"), [("full4", "T"), ("full2", "T"), ("mr4", "T"), ("mr4b", "T"), ("mr2", "T"), ("mr4", "full4"), ("mr2", "full2")], "### Paired lapvar differences\n")
# ---------------- rel-L2 to T
P("## 4. Distance to the 4-step reference T (same seed): rel-L2 of each chunk's latent, ↓ (full4 = same-config floor; FP8 + Sage are non-deterministic and same-seed paths are chaotic, so the floor rises along the clip)\n")
P("| rel-L2 to T ↓ | " + " | ".join(GEN) + " |"); P("|---|" + "---|" * len(GEN))
for lab, sel in (("chunk 0", lambda r: r[0]), ("chunk 1", lambda r: r[1]), ("chunks 0-3 mean", lambda r: np.mean(r[:4])), ("chunk 6", lambda r: r[6]), ("chunk 12", lambda r: r[12]), ("all chunks mean", lambda r: np.mean(r))):
    cells = {arm: mse([sel(runs[k]["rel_l2_T"]) for k in runs if k[0] == arm]) for arm in GEN}
    rk = rank_fmt({arm: cells[arm][0] for arm in GEN}, False, "%.3f", se={arm: cells[arm][1] for arm in GEN}); P("| %s | " % lab + " | ".join(rk[arm] for arm in GEN) + " |")
P("")
# per-scene chunk-0 (the only chunk where noise coupling is exact and the context identical)
P("| chunk-0 rel-L2 to T ↓ per scene | " + " | ".join(GEN) + " |"); P("|---|" + "---|" * len(GEN))
for sc in SCENES:
    cells = {arm: mse([runs[k]["rel_l2_T"][0] for k in runs if k[0] == arm and k[1] == sc]) for arm in GEN}; rk = rank_fmt({arm: cells[arm][0] for arm in GEN}, False, "%.3f", se={arm: cells[arm][1] for arm in GEN})
    P("| %s | " % SHORT[sc] + " | ".join(rk[arm] for arm in GEN) + " |")
P("")
# ---------------- figure
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(FG, exist_ok=True); fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=160)
    for arm in GEN:
        M = np.array([runs[k]["rel_l2_T"] for k in runs if k[0] == arm])
        if len(M) == 0: continue
        m = M.mean(0); se = M.std(0, ddof=1) / np.sqrt(len(M)) if len(M) > 1 else 0 * m; x = np.arange(len(m))
        ax.plot(x, m, color=COL[arm], lw=2, marker="o", ms=4, label="%s (n=%d)" % (arm, len(M))); ax.fill_between(x, m - se, m + se, color=COL[arm], alpha=0.15, lw=0)
        ax.annotate(arm, (x[-1] + 0.15, m[-1]), color=COL[arm], fontsize=8, va="center")
    ax.set_xlabel("chunk (3 latent frames = 0.75 s)"); ax.set_ylabel("rel-L2 to the 4-step reference T (same seed)"); ax.set_xticks(range(13)); ax.set_xlim(-0.3, 13.2)
    ax.set_title("Per-chunk latent distance to the 14B 4-step reference (mean ± s.e. over 5 scenes x 3 seeds)", fontsize=9)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    ax.grid(axis="y", color="#e6e6e3", lw=0.8); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=8, loc="upper left"); fig.tight_layout(); fig.savefig(FG + "/multires_rel_l2.png"); P("Figure: `figs/multires_rel_l2.png` (per-chunk rel-L2 to T, mean ± s.e.).\n")
except Exception as e: P("figure failed: %r\n" % e)
# ---------------- side-by-side videos
if not a.no_video:
    import cv2; from PIL import Image, ImageDraw, ImageFont
    FF = S + "/tools/ffmpeg-7.0.2-amd64-static/ffmpeg"; os.makedirs(VD, exist_ok=True); TW, TH, LH = 416, 240, 40
    try: FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15); FONT2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception: FONT = FONT2 = ImageFont.load_default()
    def read(p):
        cap = cv2.VideoCapture(p); fr = []
        while True:
            ok, f = cap.read()
            if not ok: break
            fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        return fr
    made = []
    for sc in a.video_scenes.split(","):
        sd = 1234; tiles = [("T", "14B 4-step reference"), ("full2", "2-step full-res (nodes 0,500)"), ("mr4", "mr4: steps 1-2 at half res"), ("mr2", "mr2: step 1 at half res + step 2 full")]
        paths = ["%s/judge/%s_%s_s%d.mp4" % (R, arm, sc, sd) for arm, _ in tiles]
        if not all(os.path.exists(p) for p in paths): print("video skip", sc, [p for p in paths if not os.path.exists(p)]); continue
        vids = [read(p) for p in paths]; n = min(len(v) for v in vids); out = "%s/multires_%s_s%d.mp4" % (VD, sc, sd)
        sub = {arm: ("%.2f s/chunk, judge q %.0f / drift %.0f, lapvar %.0f" % (np.mean([c["total"] for c in runs[(arm, sc, sd)]["chunks"][6:]]) if arm != "T" else float("nan"), jv(arm, sc, sd, "quality"), jv(arm, sc, sd, "drift"), runs[(arm, sc, sd)].get("lapvar_mean", float("nan")))) if (arm, sc, sd) in runs else "" for arm, _ in tiles}
        p = subprocess.Popen([FF, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (TW * len(tiles), LH + TH), "-r", "16", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
        for f in range(n):
            canvas = Image.new("RGB", (TW * len(tiles), LH + TH), (18, 18, 18)); d = ImageDraw.Draw(canvas)
            for i, ((arm, lab), v) in enumerate(zip(tiles, vids)):
                canvas.paste(Image.fromarray(v[f]).resize((TW, TH), Image.BILINEAR), (i * TW, LH)); d.text((i * TW + 6, 3), lab, font=FONT, fill=(240, 240, 240)); d.text((i * TW + 6, 22), sub[arm].replace("nan s/chunk, ", ""), font=FONT2, fill=(180, 180, 180))
            d.text((TW * len(tiles) - 150, 22), "%s s%d  chunk %d" % (SHORT[sc], sd, min(max(f - 1, 0) // 12, 12)), font=FONT2, fill=(180, 180, 180)); p.stdin.write(np.asarray(canvas).tobytes())
        p.stdin.close(); p.wait(); made.append(out); print("wrote", out)
    if made: P("Videos: " + ", ".join("`videos/%s`" % os.path.basename(m) for m in made) + " (T | full2 | mr4 | mr2, seed 1234).\n")
open(D + "/multires_pilot.md", "w").write("\n".join(L)); print("\n".join(L)); print("[ANALYSIS_DONE]")
