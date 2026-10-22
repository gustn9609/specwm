"""Blind contact sheets (same layout as blind_sheets.py) for an explicit clip list [(task, arm, seed), ...] JSON; videos under the spec_v0 or loop roots.
usage: blind_sheets_list.py OUTDIR LIST.json"""
import sys, os, json, hashlib, numpy as np, imageio, cv2
S = "/home/hk28965/specwm"; V0 = S + "/lingbot-world-v2/specwm_out/spec_v0"; LP = os.environ.get("EXTRA_ROOT", S + "/lingbot-world-v2/specwm_out/loop")
out, lst = sys.argv[1], json.load(open(sys.argv[2])); os.makedirs(out, exist_ok=True)
mp = json.load(open(out + "/mapping.json")) if os.path.exists(out + "/mapping.json") else {}; done = {(v["task"], v["arm"], v["seed"]) for v in mp.values()}; n = 0
for task, arm, seed in lst:
    if (task, arm, seed) in done: continue
    p = next((q for q in ("%s/%s/%s_s%d/video.mp4" % (r, task, arm, seed) for r in (V0, LP)) if os.path.exists(q)), None)
    if p is None: continue
    try: fr = [np.asarray(f) for f in imageio.get_reader(p)]
    except Exception as ex: print("[fail]", p, ex); continue
    L = len(fr)
    if L < 60: continue
    idx = [int(round(x)) for x in np.linspace(10, L - 1, 15)]
    big = cv2.resize(fr[0], (512, 280)); tiles = [cv2.resize(fr[i], (256, 140)) for i in idx]
    grid = np.concatenate([np.concatenate(tiles[r * 5:(r + 1) * 5], 1) for r in range(3)], 0)
    left = np.full((420, 512, 3), 255, np.uint8); left[70:350] = big
    img = np.concatenate([left, np.full((420, 8, 3), 255, np.uint8), grid], 1); cv2.putText(img, "START", (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    hid = hashlib.sha1(("blind-more|" + p).encode()).hexdigest()[:10]
    cv2.imwrite("%s/%s.jpg" % (out, hid), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82]); mp[hid] = dict(task=task, seed=seed, arm=arm, frames=L); n += 1
    if n % 50 == 0: json.dump(mp, open(out + "/mapping.json", "w"), indent=1); print("sheets", n, flush=True)
json.dump(mp, open(out + "/mapping.json", "w"), indent=1); print("new sheets %d, total %d" % (n, len(mp)))
