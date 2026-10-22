"""Fidelity to the target: for every run ROOT/<task>/<arm>_s<seed>/lat.pt, rel. L2 to the target-only latents of the same task/seed (clip and per chunk),
plus accepted chunks and time from loop.json. usage: python agree_fidelity.py ROOT [ROOT2 ...] -> markdown table (mean over cases)"""
import sys, json, glob, os, collections, numpy as np, torch
rows = collections.defaultdict(list)
for root in sys.argv[1:]:
    for f in glob.glob(root + "/*/*/lat.pt"):
        d = os.path.dirname(f); task = os.path.basename(os.path.dirname(d)); arm, _, sd = os.path.basename(d).rpartition("_s")
        ref = "%s/%s/target_s%s/lat.pt" % (root, task, sd)
        if not os.path.exists(ref) or arm == "target": continue
        a = torch.load(f).float(); b = torch.load(ref).float(); j = json.load(open(d + "/loop.json"))
        per = [float((a[:, 3*c:3*c+3] - b[:, 3*c:3*c+3]).norm() / b[:, 3*c:3*c+3].norm()) for c in range(a.shape[1] // 3)]
        rows[arm].append(dict(err=float((a - b).norm() / b.norm()), first=per[1], last=per[-1], acc=j["accepted"], n=j["chunks"], gen=j["gen_s"], task=task))
print("| arm | cases | accepted ↑ (of 13) | latent gen ↓ (s) | rel. L2 to target ↓ (clip) | chunk 1 ↓ | last chunk ↓ |")
print("|---|---|---|---|---|---|---|")
for arm, v in sorted(rows.items(), key=lambda kv: np.mean([x["err"] for x in kv[1]])):
    m = lambda k: np.mean([x[k] for x in v])
    print("| %s | %d | %.1f | %.1f | %.3f | %.3f | %.3f |" % (arm, len(v), m("acc"), m("gen"), m("err"), m("first"), m("last")))
