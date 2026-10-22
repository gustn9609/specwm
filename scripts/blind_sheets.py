"""Anonymised contact sheets for blind drift labelling. For each (task, seed, arm) video that exists: one JPEG with the start frame large on the left
and 15 later frames (every ~10 frames) in a 3x5 grid; file name = random-looking id (sha1 of a salt + path). Writes mapping.json (id -> task/seed/arm).
usage: blind_sheets.py OUTDIR TASKS SEEDS ARMS [--skip-existing]"""
import sys, os, json, hashlib, numpy as np, imageio, cv2
S = "/home/hk28965/specwm"; V0 = os.environ.get("BLIND_ROOT", S + "/lingbot-world-v2/specwm_out/spec_v0")
out, tasks, seeds, arms = sys.argv[1], sys.argv[2].split(","), [int(s) for s in sys.argv[3].split(",")], sys.argv[4].split(",")
os.makedirs(out, exist_ok=True); mp = json.load(open(out + "/mapping.json")) if os.path.exists(out + "/mapping.json") else {}
done = {(v["task"], v["seed"], v["arm"]) for v in mp.values()}; n = 0
for t in tasks:
    for sd in seeds:
        for arm in arms:
            p = "%s/%s/%s_s%d/video.mp4" % (V0, t, arm, sd)
            if not os.path.exists(p) or (t, sd, arm) in done: continue
            fr = [np.asarray(f) for f in imageio.get_reader(p)]; L = len(fr)
            idx = [int(round(x)) for x in np.linspace(10, L - 1, 15)]
            big = cv2.resize(fr[0], (512, 280)); tiles = [cv2.resize(fr[i], (256, 140)) for i in idx]
            grid = np.concatenate([np.concatenate(tiles[r * 5:(r + 1) * 5], 1) for r in range(3)], 0)   # 420 x 1280
            left = np.full((420, 512, 3), 255, np.uint8); left[70:350] = big
            img = np.concatenate([left, np.full((420, 8, 3), 255, np.uint8), grid], 1)
            cv2.putText(img, "START", (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            hid = hashlib.sha1(("blind-0928|" + p).encode()).hexdigest()[:10]
            cv2.imwrite("%s/%s.jpg" % (out, hid), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
            mp[hid] = dict(task=t, seed=sd, arm=arm, frames=L); n += 1
json.dump(mp, open(out + "/mapping.json", "w"), indent=1); print("new sheets %d, total %d" % (n, len(mp)))
