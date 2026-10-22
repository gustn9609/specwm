#!/usr/bin/env python3
"""Exp 5: visual-quality pairwise judge (Qwen3-VL via OpenAI-compatible server :30100).
For each task: 8 timesteps, each a side-by-side image (Left | Right) of the two arms at the same time (both resized to the same display size),
judge picks the visually better side + 1-5 scores (sharpness/detail, artifacts, temporal consistency). Both orders (A|B, B|A) are judged to cancel position bias.
usage: vq_judge.py --a <dir_or_glob> --b <dir> --name A_vs_B [--n 85]"""
import argparse, base64, glob, io, json, os, re, numpy as np, imageio
from PIL import Image, ImageDraw
from openai import OpenAI
ap = argparse.ArgumentParser(); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--out", default="/home/hk28965/specwm/lingbot-world-v2/specwm_out/stevo/vq"); ap.add_argument("--n-frames", type=int, default=8); ap.add_argument("--width", type=int, default=640); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--protocol", default="v1", help="v1: single preference; v2: 4 questions (sharpness / camera motion / object consistency / overall)")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
client = OpenAI(api_key="local", base_url="http://127.0.0.1:30100/v1")
def vids(d):
    m = {}
    for p in glob.glob(d + "/*.mp4"): m[os.path.basename(p)[:-4]] = p
    for p in glob.glob(d + "/*/video.mp4"): m[os.path.basename(os.path.dirname(p))] = p
    return m
A, B = vids(a.a), vids(a.b); tasks = sorted(set(A) & set(B)); tasks = tasks[:a.limit] if a.limit else tasks
print("[vq] %s: %d common tasks" % (a.name, len(tasks)), flush=True)
def frames(p, n):
    V = [np.asarray(f) for f in imageio.get_reader(p)]; idx = [int(i * (len(V) - 1) / (n - 1)) for i in range(n)]; return [V[i] for i in idx]
def sbs(fa, fb, W):
    def fit(f): im = Image.fromarray(f); h = int(round(im.height * W / im.width)); return im.resize((W, h), Image.BICUBIC)
    ia, ib = fit(fa), fit(fb); H = max(ia.height, ib.height); can = Image.new("RGB", (2 * W + 16, H + 28), (0, 0, 0)); can.paste(ia, (0, 28)); can.paste(ib, (W + 16, 28))
    d = ImageDraw.Draw(can); d.text((8, 6), "LEFT", fill=(255, 255, 0)); d.text((W + 24, 6), "RIGHT", fill=(255, 255, 0)); buf = io.BytesIO(); can.save(buf, format="JPEG", quality=90); return base64.b64encode(buf.getvalue()).decode()
PROMPT = ("You are judging the VISUAL QUALITY of two generated videos of the same scene and same camera motion. Each image shows the same moment: LEFT = video 1, RIGHT = video 2; images are in temporal order. "
          "Ignore which is more plausible physically; judge only what a viewer sees: sharpness / fine detail, absence of artifacts (blur, smearing, blocking, flicker, warped geometry), and temporal consistency across the frames. "
          "Answer in JSON only: {\"left\": {\"sharpness\": 1-5, \"artifacts\": 1-5 (5 = none), \"consistency\": 1-5}, \"right\": {same}, \"preferred\": \"left\"|\"right\"|\"tie\", \"reason\": \"<=30 words\"}")
PROMPT_V2 = ("Two generated videos of the same scene with the same commanded camera motion. Each image shows the same moment: LEFT = video 1, RIGHT = video 2; images are in temporal order. "
             "Answer FOUR separate questions, each independently, with \"left\", \"right\" or \"tie\":\n"
             "Q1 sharpness: which side is sharper / has more fine detail and fewer blur, blocking or smearing artifacts?\n"
             "Q2 camera: which side's camera motion looks more natural and consistent with a single smooth camera path (no jumps, no scene resets)?\n"
             "Q3 objects: which side keeps objects, layout and identity more consistent across the frames (nothing appearing / disappearing / morphing)?\n"
             "Q4 overall: which video would you rather use?\n"
             "Answer in JSON only: {\"q1_sharpness\": ..., \"q2_camera\": ..., \"q3_objects\": ..., \"q4_overall\": ..., \"reason\": \"<=30 words\"}")
def ask(imgs):
    content = [{"type": "text", "text": PROMPT_V2 if a.protocol == "v2" else PROMPT}] + [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + x, "detail": "high"}} for x in imgs]
    r = client.chat.completions.create(model="qwen3-vl-32b", messages=[{"role": "user", "content": content}], max_completion_tokens=400, temperature=0.0)
    t = r.choices[0].message.content or ""; m = re.search(r"\{.*\}", t, re.S)
    try: return json.loads(m.group(0))
    except Exception: return {"raw": t}
rp = os.path.join(a.out, a.name + ("_v2" if a.protocol == "v2" else "") + ".json"); res = json.load(open(rp)) if os.path.exists(rp) else {}
for t in tasks:
    if t in res and "ab" in res[t] and "ba" in res[t]: continue
    FA, FB = frames(A[t], a.n_frames), frames(B[t], a.n_frames)
    ab = ask([sbs(x, y, a.width) for x, y in zip(FA, FB)]); ba = ask([sbs(y, x, a.width) for x, y in zip(FA, FB)])
    res[t] = dict(ab=ab, ba=ba); json.dump(res, open(rp, "w"), indent=1); print("[%s] ab=%s ba=%s" % (t, ab.get("preferred"), ba.get("preferred")), flush=True)
if a.protocol == "v2":
    QS = ["q1_sharpness", "q2_camera", "q3_objects", "q4_overall"]; summ = dict(name=a.name, n=len(res), A=a.a, B=a.b, protocol="v2")
    def pref2(r, key, q):
        p = r.get(key, {}).get(q)
        if key == "ab": return {"left": "A", "right": "B"}.get(p, "tie")
        return {"left": "B", "right": "A"}.get(p, "tie")
    for q in QS:
        w = {"A": 0, "B": 0, "tie": 0}; cons = 0
        for t, r in res.items():
            pa, pb = pref2(r, "ab", q), pref2(r, "ba", q); cons += (pa == pb)
            v = "A" if (pa == "A" and pb != "B") or (pb == "A" and pa != "B") else ("B" if (pa == "B" and pb != "A") or (pb == "B" and pa != "A") else "tie"); w[v] += 1
        summ[q] = dict(A=w["A"], B=w["B"], tie=w["tie"], order_consistent=cons)
    json.dump(summ, open(os.path.join(a.out, a.name + "_v2_summary.json"), "w"), indent=1); print("[VQ_DONE]", json.dumps(summ)); raise SystemExit
# summary: A preferred if (ab.preferred==left) ; in ba, A is on the right
def pref(r, key):
    p = r.get(key, {}).get("preferred"); 
    if key == "ab": return {"left": "A", "right": "B"}.get(p, "tie")
    return {"left": "B", "right": "A"}.get(p, "tie")
wins = {"A": 0, "B": 0, "tie": 0}; consist = 0; sc = {"A": [], "B": []}
for t, r in res.items():
    pa, pb = pref(r, "ab"), pref(r, "ba"); consist += (pa == pb)
    v = "A" if (pa == "A" and pb != "B") or (pb == "A" and pa != "B") else ("B" if (pa == "B" and pb != "A") or (pb == "B" and pa != "A") else "tie"); wins[v] += 1
    for key, side_of_A in (("ab", "left"), ("ba", "right")):
        d = r.get(key, {}); other = "right" if side_of_A == "left" else "left"
        if isinstance(d.get(side_of_A), dict) and isinstance(d.get(other), dict):
            sc["A"].append([d[side_of_A].get(k) for k in ("sharpness", "artifacts", "consistency")]); sc["B"].append([d[other].get(k) for k in ("sharpness", "artifacts", "consistency")])
n = len(res); mean = lambda L: np.nanmean(np.array(L, dtype=float), 0) if L else [np.nan] * 3
summ = dict(name=a.name, n=n, A=a.a, B=a.b, A_wins=wins["A"], B_wins=wins["B"], ties=wins["tie"], order_consistent=consist, A_scores=[round(float(x), 2) for x in mean(sc["A"])], B_scores=[round(float(x), 2) for x in mean(sc["B"])])
json.dump(summ, open(os.path.join(a.out, a.name + "_summary.json"), "w"), indent=1); print("[VQ_DONE]", json.dumps(summ))
