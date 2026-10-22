"""[10/19 EXP 1] Per-segment VLM judge for LONG clips (the 60-s program P6): every video is cut by time into pieces of --seg-s seconds (default 10 ->
6 pieces for 60 s) and every piece gets exactly the whole-clip judge of scripts/vlm_clip_score.py: 8 evenly spaced frames of the piece, the SAME two
prompts (DRIFT 0-100, QUALITY 1-10; read from vlm_clip_score.py itself, so they can never diverge), the same Qwen3-VL-32B server (:JUDGE_PORT, default
30100), max_tokens 8, temperature 0, frames resized to VLM_W (448) wide. Each 10-s piece is therefore scored like one of the frozen 10-s clips and the
per-piece values can be plotted against time (drift over a long rollout).
Piece k = frames [round(k * seg_s * fps), round((k + 1) * seg_s * fps)) with fps = the container fps (the trailing frame of a 961-frame / 16 fps clip
joins the last piece); n_seg = round(n_frames / (seg_s * fps)) (6 for 961 @ 16, 720 @ 12, 3600 @ 60).
Writes OUT.json = {abs video path: {label, seg_s, n_seg, fps, frames, seg_t [[t0, t1] s], seg_frames [[i0, i1)], seg_drift [...], seg_quality [...],
drift (mean over pieces), quality (mean), drift_first, drift_last, quality_first, quality_last, raw [[drift_txt, quality_txt] per piece]}}; append-only:
a path that already has a record under this label is skipped (baselines/score_runs.sh step 5 runs it on the *__P6* clips into <metrics>/vlm_seg.json).
usage: JUDGE_PORT=30100 scripts/envrun.sh python scripts/vlm_seg_score.py OUT.json LABEL=GLOB [LABEL=GLOB ...] [--seg-s 10]"""
import sys, os, ast, glob, json, io, base64, time, argparse, numpy as np, cv2
from PIL import Image
from openai import OpenAI
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("specs", nargs="+"); ap.add_argument("--seg-s", type=float, default=10.0); ap.add_argument("--n-frames", type=int, default=8)
a = ap.parse_args()
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vlm_clip_score.py")
CONST = {}
for node in ast.parse(open(SRC).read()).body:   # the whole-clip judge's prompts, verbatim
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("DRIFT", "QUAL"):
        CONST[node.targets[0].id] = ast.literal_eval(node.value)
assert set(CONST) == {"DRIFT", "QUAL"}, "could not read DRIFT / QUAL from %s" % SRC
DRIFT, QUAL = CONST["DRIFT"], CONST["QUAL"]
client = OpenAI(api_key="local", base_url="http://127.0.0.1:%s/v1" % os.environ.get("JUDGE_PORT", "30100"))
res = json.load(open(a.out)) if os.path.exists(a.out) else {}
def read_all(p):
    cap = cv2.VideoCapture(p); fps = float(cap.get(cv2.CAP_PROP_FPS)) or 16.0; fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release(); return fr, fps
def b64(im, W=int(os.environ.get("VLM_W", "448"))):   # = vlm_clip_score.b64
    im = Image.fromarray(im); im = im.resize((W, int(im.height * W / im.width)), Image.BICUBIC); b = io.BytesIO(); im.save(b, format="JPEG", quality=85); return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
def ask(images, prompt):   # = vlm_clip_score.ask
    r = client.chat.completions.create(model="qwen3-vl-32b", messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": u}} for u in images] + [{"type": "text", "text": prompt}]}], max_tokens=8, temperature=0)
    txt = r.choices[0].message.content.strip()
    try: return float(next(t for t in txt.replace("/", " ").split() if t.replace(".", "").isdigit())), txt
    except StopIteration: return float("nan"), txt
for spec in a.specs:
    lab, pat = spec.split("=", 1); paths = sorted(os.path.abspath(p) for p in glob.glob(pat)); t0 = time.time()
    for p in paths:
        if isinstance(res.get(p), dict) and res[p].get("label") == lab and "seg_drift" in res[p]: continue
        fr, fps = read_all(p); N = len(fr); per = a.seg_s * fps; n_seg = max(1, int(round(N / per)))
        bounds = [int(round(k * per)) for k in range(n_seg)] + [N]
        R = dict(label=lab, seg_s=a.seg_s, n_seg=n_seg, fps=fps, frames=N, seg_t=[], seg_frames=[], seg_drift=[], seg_quality=[], raw=[])
        for k in range(n_seg):
            s_, e_ = bounds[k], bounds[k + 1]; n = a.n_frames
            idx = [s_ + int(j * (e_ - 1 - s_) / (n - 1)) for j in range(n)]
            ims = [b64(fr[i]) for i in idx]; d, rd = ask(ims, DRIFT); q, rq = ask(ims, QUAL)
            R["seg_t"].append([round(s_ / fps, 3), round(e_ / fps, 3)]); R["seg_frames"].append([s_, e_]); R["seg_drift"].append(d); R["seg_quality"].append(q); R["raw"].append([rd, rq])
        R.update(drift=float(np.nanmean(R["seg_drift"])), quality=float(np.nanmean(R["seg_quality"])), drift_first=R["seg_drift"][0], drift_last=R["seg_drift"][-1],
                 quality_first=R["seg_quality"][0], quality_last=R["seg_quality"][-1])
        res[p] = R; json.dump(res, open(a.out, "w"), indent=0)
        print("[vlm_seg] %s %s: %d pieces of %.0f s, drift %s, quality %s" % (lab, os.path.relpath(p, "/home/hk28965/specwm"), n_seg, a.seg_s, R["seg_drift"], R["seg_quality"]), flush=True)
    json.dump(res, open(a.out, "w"), indent=0)
    Rs = [res[p] for p in paths if p in res and res[p].get("label") == lab]
    if Rs:
        k = min(r["n_seg"] for r in Rs)
        print("| %s | n %d | drift per piece %s | quality per piece %s |" % (lab, len(Rs), [round(float(np.nanmean([r["seg_drift"][i] for r in Rs])), 1) for i in range(k)],
                                                                          [round(float(np.nanmean([r["seg_quality"][i] for r in Rs])), 2) for i in range(k)]), flush=True)
print("[VLM_SEG_DONE]")
