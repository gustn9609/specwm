#!/usr/bin/env python3
"""Build the 'Speculative Rollout Viewer' (single self-contained HTML).
Each case = one rollout of our method (Fast draft + DINO/colour verifier + Base correction) on one task/seed.
Per chunk: committed frames (video.mp4), Fast draft frames and Base candidates when the run was instrumented (--save-drafts -> viz/),
gate score, decision, measured timings; references: Base 70-step and Fast alone on the same seed (frames + measured total time).
usage: build_flow_viz.py OUT.html RUN_DIR [RUN_DIR ...]"""
import sys, os, json, glob, base64, numpy as np, imageio, cv2

def b64jpg(im, w=224, q=62):
    h = int(round(w * im.shape[0] / im.shape[1])); im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q]); return "data:image/jpeg;base64," + base64.b64encode(buf).decode()
def chunk_frame_idx(c, n):
    lo, hi = (0, 9) if c == 0 else (12 * c - 3, 12 * c + 9); return [k for k in range(lo, min(hi, n))]
def video_chunks(path, N, per=None):
    """every 2nd pixel frame of each chunk (played back at 8 fps = real-time 16 fps duration)"""
    if not path or not os.path.exists(path): return None
    V = [np.asarray(f) for f in imageio.get_reader(path)]; out = []
    for c in range(N):
        ids = chunk_frame_idx(c, len(V))[::2]; out.append([b64jpg(V[k]) for k in ids])
    return out
def viz_imgs(run, c, tag):
    fs = sorted(glob.glob(os.path.join(run, "viz", "c%02d_%s_*.jpg" % (c, tag))), key=lambda p: int(p.rsplit("_", 1)[1][:-4]))
    return [b64jpg(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)) for p in fs[::2][:3]] if fs else []
def first(*paths):
    for p in paths:
        if p and os.path.exists(p): return p
    return None

TASK = {"sand_filling_up_jar": ("Sand filling a jar", "jar"), "sandcastle_wave_impact": ("Wave hits a sandcastle", "castle")}
cases = []
for run in sys.argv[2:]:
    j = json.load(open(os.path.join(run, "spec_v0.json"))); task, seed = j["task"], j["seed"]; N = len(j["plan"]); tdir = os.path.dirname(run)
    b70 = os.path.join(tdir, "base70_s%d" % seed); base_s = json.load(open(os.path.join(b70, "revisit.json")))["gen_s"] if os.path.exists(os.path.join(b70, "revisit.json")) else 2829
    nv = first(os.path.join(tdir, "never_s%d" % seed, "spec_v0.json")); fast_s = json.load(open(nv))["gen_s"] if nv else 62.0
    fast_vid = first(os.path.join(tdir, "never_s%d" % seed, "video.mp4"), os.path.join(tdir, "probe2_s%d" % seed, "video.mp4"))
    base_ref = video_chunks(os.path.join(b70, "video.mp4"), N); fast_ref = video_chunks(fast_vid, N); final = video_chunks(os.path.join(run, "video.mp4"), N)
    d360 = os.path.join(tdir, "d360_never_s%d" % seed); f360_ref = video_chunks(os.path.join(d360, "video.mp4"), N)
    fast360_s = json.load(open(os.path.join(d360, "spec_v0.json")))["gen_s"] if os.path.exists(os.path.join(d360, "spec_v0.json")) else None
    instrumented = os.path.isdir(os.path.join(run, "viz"))
    chunks = []
    for l in j["log"]:
        c = l["chunk"]; bases = [b for b in (viz_imgs(run, c, "base%d" % r) for r in range(3)) if b]
        chunks.append(dict(c=c, dec=j["plan"][c], warm=(1 <= c < 1 + j.get("warmup", 0)), score=l.get("verify_dist"), corr=l.get("corrected_scores") or [],
                           t_done=l.get("t_done"), t_draft=l.get("t_draft"), t_base=l.get("t_base"),
                           draft=viz_imgs(run, c, "draft"), bases=bases, final=final[c], base_ref=base_ref[c] if base_ref else [], fast_ref=fast_ref[c] if fast_ref else [], fast360_ref=f360_ref[c] if f360_ref else []))
    name, short = TASK[task]
    cases.append(dict(id="%s_s%d" % (short, seed), task=task, name=name, short=short, seed=seed, plan=j["plan"], thr=float(j["policy"].split(":")[1]), warmup=j.get("warmup", 0),
                      gen_s=j["gen_s"], base_s=base_s, fast_s=fast_s, fast360_s=fast360_s, instrumented=instrumented, chunks=chunks))
    print(run, "instrumented" if instrumented else "no drafts", flush=True)
cases.sort(key=lambda c: (c["short"] != "jar", c["seed"]))
DATA = json.dumps(dict(cases=cases, chunk_s=0.75, n=13, kv_sink=2, kv_win=4))
html = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow_viz_template.html")).read().replace("/*__DATA__*/null", DATA)
open(sys.argv[1], "w").write(html); print("wrote", sys.argv[1], round(len(html) / 1e6, 2), "MB")
