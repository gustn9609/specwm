"""Absolute (single-image) VLM quality score of every percep_dump strip (the pairwise judge is position-biased: it calls the SECOND image
worse whenever it sees a difference). Score 1-10 per strip, then acceptable(k) = score(step k) >= score(final) - margin.
usage: python scripts/percep_abs.py DIR [--margin 0] [--thr 0.7]"""
import sys, os, glob, json, base64, numpy as np
from openai import OpenAI
d = sys.argv[1]; margin = float(sys.argv[sys.argv.index("--margin") + 1]) if "--margin" in sys.argv else 0.0; thr = float(sys.argv[sys.argv.index("--thr") + 1]) if "--thr" in sys.argv else 0.7
client = OpenAI(api_key="local", base_url="http://127.0.0.1:30100/v1"); SP = d + "/abs_scores.json"; SC = json.load(open(SP)) if os.path.exists(SP) else {}
Q = ("This image shows 4 frames (left to right, spanning 0.75 s) of an AI-generated video. Rate the VISUAL QUALITY of the frames (sharpness and detail, absence of artefacts such as blur, smearing, blocking, ghosting or grid noise, realism of textures, and consistency between the 4 frames). "
     "Answer with ONE integer 1-10 (10 = photographic, 1 = unusable). Only the number.")
def score(p):
    if p in SC: return SC[p]
    r = client.chat.completions.create(model="qwen3-vl-32b", messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(open(p, "rb").read()).decode()}}, {"type": "text", "text": Q}]}], max_tokens=6, temperature=0)
    t = r.choices[0].message.content.strip()
    try: v = float(next(x for x in t.split() if x.replace(".", "").isdigit()))
    except StopIteration: v = float("nan")
    SC[p] = v; json.dump(SC, open(SP, "w")); return v
rows = []
for sp in sorted(glob.glob(d + "/*/stats.json")):
    R = json.load(open(sp)); rd = os.path.dirname(sp)
    for ch in R["chunks"]:
        c = ch["c"]; sf = score("%s/c%02d_k4.jpg" % (rd, c))
        for k in (1, 2, 3): rows.append(dict(k=k, s=score("%s/c%02d_k%d.jpg" % (rd, c, k)), sf=sf, head=ch["head_frac"][k - 1], within=ch["within"][k - 1]))
print("strips scored:", len(SC), "| final strips mean quality %.2f" % np.mean([r["sf"] for r in rows if r["k"] == 1]))
print("| step k | n | mean quality of step-k strip | final | Δ (k − final) | acceptable (Δ ≥ −%g) ↑ | distance rule accepts (head ≥ %.1f) | rule rejects BUT acceptable (room) | rule accepts BUT not acceptable (risk) |" % (margin, thr))
print("|---|---|---|---|---|---|---|---|---|")
for k in (1, 2, 3):
    r = [x for x in rows if x["k"] == k and np.isfinite(x["s"])]; dq = np.array([x["s"] - x["sf"] for x in r]); acc = dq >= -margin; rule = np.array([x["head"] >= thr for x in r])
    print("| %d | %d | %.2f | %.2f | %+.2f ± %.2f | %.0f %% | %.0f %% | %.0f %% | %.0f %% |" % (k, len(r), np.mean([x["s"] for x in r]), np.mean([x["sf"] for x in r]), dq.mean(), dq.std() / np.sqrt(len(dq)), 100 * acc.mean(), 100 * rule.mean(), 100 * (~rule & acc).mean(), 100 * (rule & ~acc).mean()))
print("| head-predicted within-ε | n | mean Δ quality (k − final) | acceptable ↑ |"); print("|---|---|---|---|")
for lo, hi in ((0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)):
    r = [x for x in rows if lo <= x["head"] < hi and np.isfinite(x["s"])]
    if r: dq = np.array([x["s"] - x["sf"] for x in r]); print("| %.2f–%.2f | %d | %+.2f | %.0f %% |" % (lo, hi, len(r), dq.mean(), 100 * (dq >= -margin).mean()))
