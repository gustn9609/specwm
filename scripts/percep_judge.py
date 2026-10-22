"""Pairwise VLM judgement of specwm/percep_dump.py strips: for every chunk and step k in 1..3, the step-k strip and the final (k = 4) strip
are shown as two images in RANDOM order (position bias control) and the judge (Qwen3-VL-32B on :30100) answers whether one is noticeably
worse. Writes <dir>/judge.json {task_s/c/k: {order, answer, worse}} and prints the go/no-go table: perceptual acceptance per step vs the
distance rule (head-predicted within-eps >= 0.7) and the true within-eps. usage: python scripts/percep_judge.py DIR [--thr 0.7]"""
import sys, os, glob, json, base64, random, numpy as np
from openai import OpenAI
d = sys.argv[1]; thr = float(sys.argv[sys.argv.index("--thr") + 1]) if "--thr" in sys.argv else 0.7
client = OpenAI(api_key="local", base_url="http://127.0.0.1:30100/v1"); JP = d + "/judge.json"; J = json.load(open(JP)) if os.path.exists(JP) else {}
Q = ("These two images each show 4 frames (left to right, 0.75 s apart in total) of the SAME moment of the same AI-generated video, produced by two different settings of the generator. "
     "Compare their visual quality and content: sharpness, detail, artefacts (blur, smearing, blocking, ghosting), realism of textures, and whether the content (objects, layout, motion) is the same. "
     "Answer with ONE letter: A if the FIRST image is noticeably worse, B if the SECOND image is noticeably worse, S if they are of the same quality (no noticeable difference). Only the letter.")
def b64(p): return "data:image/jpeg;base64," + base64.b64encode(open(p, "rb").read()).decode()
def ask(p1, p2):
    r = client.chat.completions.create(model="qwen3-vl-32b", messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": b64(p1)}}, {"type": "image_url", "image_url": {"url": b64(p2)}}, {"type": "text", "text": Q}]}], max_tokens=4, temperature=0)
    t = r.choices[0].message.content.strip().upper(); return t[0] if t and t[0] in "ABS" else "?"
rng = random.Random(0); rows = []
for sp in sorted(glob.glob(d + "/*/stats.json")):
    R = json.load(open(sp)); rd = os.path.dirname(sp); key0 = os.path.basename(rd)
    for ch in R["chunks"]:
        c = ch["c"]; fin = "%s/c%02d_k4.jpg" % (rd, c)
        for k in (1, 2, 3):
            key = "%s/c%02d/k%d" % (key0, c, k); cand = "%s/c%02d_k%d.jpg" % (rd, c, k)
            if key not in J:
                order = rng.random() < 0.5; ans = ask(cand, fin) if order else ask(fin, cand)   # order True: candidate first
                worse = (ans == ("A" if order else "B")); better = (ans == ("B" if order else "A")); J[key] = dict(order=order, answer=ans, worse=worse, better=better)
                json.dump(J, open(JP, "w"))
            rows.append(dict(key=key, k=k, worse=J[key]["worse"], better=J[key]["better"], same=J[key]["answer"] == "S", head=ch["head_frac"][k - 1], within=ch["within"][k - 1]))
print("chunks x steps judged:", len(rows))
print("| step k | n | perceptually acceptable (not worse) ↑ | 'same' ↑ | candidate better | distance rule accepts (head ≥ %.1f) | rule accepts AND acceptable | rule rejects BUT acceptable (room) | rule accepts BUT worse (risk) | true within-ε mean |" % thr)
print("|---|---|---|---|---|---|---|---|---|---|")
for k in (1, 2, 3):
    r = [x for x in rows if x["k"] == k]; n = len(r); acc = np.array([not x["worse"] for x in r]); same = np.array([x["same"] for x in r]); bet = np.array([x["better"] for x in r]); rule = np.array([x["head"] >= thr for x in r])
    print("| %d | %d | %.0f %% | %.0f %% | %.0f %% | %.0f %% | %.0f %% | %.0f %% | %.0f %% | %.0f %% |" % (k, n, 100 * acc.mean(), 100 * same.mean(), 100 * bet.mean(), 100 * rule.mean(), 100 * (rule & acc).mean(), 100 * (~rule & acc).mean(), 100 * (rule & ~acc).mean(), 100 * np.mean([x["within"] for x in r])))
# acceptance vs the head statistic, binned
print("| head-predicted within-ε | n | perceptually acceptable ↑ |"); print("|---|---|---|")
for lo, hi in ((0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)):
    r = [x for x in rows if lo <= x["head"] < hi]
    if r: print("| %.2f–%.2f | %d | %.0f %% |" % (lo, hi, len(r), 100 * np.mean([not x["worse"] for x in r])))
