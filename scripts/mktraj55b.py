import os
import numpy as np, os
S = os.environ["SCRATCH"]+""
ref = np.load(S + "/rt2_action_05/poses.npy"); rs = np.linalg.norm(np.diff(ref[:, :3, 3], axis=0), axis=1)
print("LingBot ref (S05 rt2): step median %.3f max %.3f" % (np.median(rs), rs.max()))
def build(sc, idx):
    P = np.load(S + "/poses_%s.npy" % sc) if os.path.exists(S + "/poses_%s.npy" % sc) else np.load(S + "/lingbot-world-v2/examples/%s/poses.npy" % sc)
    Q = P[idx]; step = np.linalg.norm(np.diff(Q[:, :3, 3], axis=0), axis=1); F = Q[:, :3, 2]; ang = np.degrees(np.arccos(np.clip((F[0]*F).sum(1), -1, 1)))
    homes = []
    for i in range(40, len(ang)-1):
        if ang[i] < 8 and ang[i] <= ang[i-1] and ang[i] <= ang[i+1] and (not homes or i - homes[-1] > 40): homes.append(i)
    rt = np.load(S + "/rt2_action_%s/poses.npy" % sc); ok = np.allclose(Q[:161], rt)
    print("S%s: %d frames (%d latents, %d chunks) | step mean %.3f max %.3f | max view angle %.1f | first161==rt2 %s | home views at %s s" % (sc, len(idx), (len(idx)-1)//4+1, ((len(idx)-1)//4+1)//3, step.mean(), step.max(), ang.max(), ok, [round(h/16,1) for h in homes][:6]))
    d = S + "/traj55_action_%s" % sc; os.makedirs(d, exist_ok=True); np.save(d + "/poses.npy", Q.astype(np.float32))
    for f in ["intrinsics.npy", "action.npy", "wasd_action.npy"]:
        a = np.load(S + "/rt2_action_%s/%s" % (sc, f)); np.save(os.path.join(d, f), (np.repeat(a[:1], len(idx), axis=0) if len(a) < len(idx) else a[:len(idx)]).astype(np.float32))
    return ok
# S02: examples[:873] (rt2 = examples[:161])
build("02", np.arange(873))
# S01: rt2 idx (476..516 comb) + 왕복 comb over [440, 560]
i0 = np.load(S + "/revisit_rt_idx_01.npy"); last = int(i0[-1]); segs = []
lo, hi = 440, 560; cur = last + 1; up = True; need = 873 - len(i0)
while need > 0:
    if up: seg = np.arange(cur, hi + 1); nxt = hi - 1; up = False
    else: seg = np.arange(cur, lo - 1, -1); nxt = lo + 1; up = True
    seg = seg[:need]; segs.append(seg); need -= len(seg); cur = nxt
build("01", np.concatenate([i0] + segs))
