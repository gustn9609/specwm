"""[10/14, item 3] OpenAI VPT contractor logs (Minecraft, 20 Hz keyboard state + absolute yaw/pitch) -> our action-stream format + statistics.
Per latent frame (5 ticks = 0.25 s): yaw rate = yaw[t+5] - yaw[t] (deg), pitch rate likewise, speed = (#W - #S) / 5, strafe = (#D - #A) / 5 * 0.5;
ticks with the GUI open are dropped. Statistics on the RAW stream: constant-velocity hit rate per 3-frame chunk under the harness tolerance
(rotation sum <= 3 deg, relative translation <= 0.5), the distribution of per-frame rotation, and the first-order transition prior over the
quantized key state (bins of the synthetic benchmark: yaw {0, +-4, +-8, +-12.65}, pitch {0, +-3}, speed {0, 0.5, 1}, strafe {0, +-0.5}):
top-1 / top-2 mass of P(next | current) = how peaked a real prior is. Then writes N 41-frame segments (quantized state -> relative poses, so
the harness prior machinery applies; rotations in the model's range) to data/actstream/vpt_<i>/ and the prior counts (from the OTHER logs,
split by player) to data/vpt/prior.json.  usage: python -I scripts/vpt_streams.py [--n 8] [--seed 0]"""
import sys, os, json, glob, math, random, numpy as np
sys.path.insert(0, "/home/hk28965/specwm/lingbot-world-v2"); from specwm.act_stream import se3, c2w_from_rel
S = "/home/hk28965/specwm"; a = dict(n=8, seed=0)
for i, x in enumerate(sys.argv):
    if x in ("--n", "--seed"): a[x[2:]] = int(sys.argv[i + 1])
YAW = [0, 4, -4, 8, -8, 12.65, -12.65]; PITCH = [0, 3, -3]; SPEED = [0, 0.5, 1]; STRAFE = [0, 0.5, -0.5]
def q(v, S_): return min(S_, key=lambda s: abs(s - v))
def load(p):
    T = []
    for l in open(p):
        try: j = json.loads(l)
        except Exception: continue
        if j.get("isGuiOpen"): continue
        k = set(j["keyboard"]["keys"]); T.append((float(j["yaw"]), float(j["pitch"]), ("key.keyboard.w" in k) - ("key.keyboard.s" in k), ("key.keyboard.d" in k) - ("key.keyboard.a" in k)))
    return T
def frames(T):
    out = []
    for i in range(0, len(T) - 5, 5):
        w = T[i:i + 5]; dy = (w[-1][0] - w[0][0] + 180) % 360 - 180; dp = w[-1][1] - w[0][1]
        out.append((dy, dp, np.mean([x[2] for x in w]), np.mean([x[3] for x in w]) * 0.5))
    return out
def rot_angle(y1, p1, y2, p2):   # geodesic angle between Ry(y1)Rx(p1) and Ry(y2)Rx(p2)
    A = se3(y1, p1, 0, 0)[:3, :3]; B_ = se3(y2, p2, 0, 0)[:3, :3]; c = (np.trace(A.T @ B_) - 1) / 2; return math.degrees(math.acos(max(-1, min(1, c))))
logs = sorted(glob.glob(S + "/data/vpt/logs/*.jsonl")); rng = random.Random(a["seed"]); players = sorted(set(os.path.basename(p).split("-")[0] for p in logs)); rng.shuffle(players)
test_players = set(players[:max(1, len(players) // 5)]); F = {p: frames(load(p)) for p in logs}
# ---- raw statistics
rot = []; hits = []; hits_r = []; still = 0; tot = 0
for p, fr in F.items():
    for c in range(1, len(fr) // 3):
        prev = fr[3 * c - 1]; chunk = fr[3 * c:3 * c + 3]; rs = sum(rot_angle(prev[0], prev[1], f[0], f[1]) for f in chunk)
        tp = np.array([[prev[3], 0, -prev[2]]] * 3); tt = np.array([[f[3], 0, -f[2]] for f in chunk]); dt = np.linalg.norm(tp - tt, axis=1).sum(); den = max(np.linalg.norm(tp, axis=1).sum(), np.linalg.norm(tt, axis=1).sum(), 1e-6)
        rel_t = dt / den if dt > 1e-6 else 0.0; hits.append(rs <= 3.0 and rel_t <= 0.5); hits_r.append(rs <= 3.0); tot += 1
    rot += [abs(f[0]) + abs(f[1]) for f in fr]; still += sum(1 for f in fr if abs(f[0]) < 0.5 and abs(f[1]) < 0.5 and f[2] == 0 and f[3] == 0)
rot = np.array(rot); print("logs %d (%d players, %d test players), latent frames %d (%.1f h of play), chunks %d" % (len(logs), len(players), len(test_players), len(rot), len(rot) * 0.25 / 3600, tot))
print("per-frame rotation |dyaw|+|dpitch| (deg / 0.25 s): median %.1f, p75 %.1f, p90 %.1f, share < 1 deg %.0f %%, share > 12.65 %.0f %%; still frames (no key, no mouse) %.0f %%" % (np.median(rot), np.percentile(rot, 75), np.percentile(rot, 90), 100 * np.mean(rot < 1), 100 * np.mean(rot > 12.65), 100 * still / len(rot)))
print("constant-velocity hit rate on the RAW stream (rotation <= 3 deg over the chunk AND relative translation <= 0.5): %.1f %% ; rotation criterion alone %.1f %%" % (100 * np.mean(hits), 100 * np.mean(hits_r)))
# ---- quantized key state: transition prior and hit rate at the key level
def qstate(f): return (q(f[0], YAW), q(f[1], PITCH), q(f[2], SPEED), q(f[3], STRAFE))
cnt = {}; same3 = []; qhits = []
for p, fr in F.items():
    st = [qstate(f) for f in fr]; train = os.path.basename(p).split("-")[0] not in test_players
    for t in range(1, len(st)):
        if train: d = cnt.setdefault(str(st[t - 1]), {}); d[str(st[t])] = d.get(str(st[t]), 0) + 1
    for c in range(1, len(st) // 3): same3.append(all(st[3 * c + j] == st[3 * c - 1] for j in range(3)))
top1 = []; top2 = []; wts = []
for k, d in cnt.items():
    n = sum(d.values()); v = sorted(d.values(), reverse=True); top1.append(v[0] / n); top2.append(sum(v[:2]) / n); wts.append(n)
print("quantized key state: 'same state for the next 3 frames' (= exact hit at the key level) %.1f %%; first-order prior (train players): weighted top-1 mass %.1f %%, top-2 %.1f %%, top-3 %.1f %%" % (100 * np.mean(same3), 100 * np.average(top1, weights=wts), 100 * np.average(top2, weights=wts), 100 * np.average([sum(sorted(d.values(), reverse=True)[:3]) / sum(d.values()) for d in cnt.values()], weights=wts)))
# P(same | current) and the most likely change
same_mass = np.average([d.get(k, 0) / sum(d.values()) for k, d in cnt.items()], weights=wts); print("P(next state = current) averaged over states: %.1f %% -> the K = 2 alternative carries the remaining mass's top item" % (100 * same_mass))
json.dump(dict(counts=cnt, train_players=sorted(set(players) - test_players), test_players=sorted(test_players), yaw=YAW, pitch=PITCH, speed=SPEED, strafe=STRAFE), open(S + "/data/vpt/prior.json", "w"))
# ---- segments for the harness (test players), quantized states -> relative poses
K = np.array([[502.9, 503.1, 415.8, 239.8]], dtype=np.float32); segs = []
for p, fr in F.items():
    if os.path.basename(p).split("-")[0] not in test_players or len(fr) < 45: continue
    for s0 in range(0, len(fr) - 41, 41):
        seg = [qstate(f) for f in fr[s0:s0 + 41]]
        if sum(1 for x in seg if x == (0, 0, 0, 0)) > 20: continue   # skip mostly idle segments
        segs.append((p, s0, seg))
rng.shuffle(segs); written = []
for i, (p, s0, seg) in enumerate(segs[:a["n"]]):
    rel = np.stack([np.eye(4)] + [se3(*v) for v in seg[1:]]).astype(np.float32); d = S + "/data/actstream/vpt_%d" % i; os.makedirs(d, exist_ok=True)
    np.save(d + "/rel.npy", rel); np.save(d + "/intrinsics.npy", np.repeat(K, 161, 0)); np.save(d + "/poses.npy", c2w_from_rel(rel).astype(np.float32))
    ch = [t for t in range(1, 41) if seg[t] != seg[t - 1]]; chunks3 = sorted(set(c // 3 for c in ch if c >= 3))
    json.dump(dict(source=os.path.basename(p), start_frame=s0, frames=41, vel=[list(map(float, v)) for v in seg], change_frames=ch, change_chunks3=chunks3, params=dict(yaw=",".join("%g" % y for y in YAW), pitch=",".join("%g" % y for y in PITCH), speed=",".join("%g" % y for y in SPEED), strafe=",".join("%g" % y for y in STRAFE), tau=0, frames=41, still=0, single=0, release=0, jitter=0.0), prior="vpt"), open(d + "/stream.json", "w"), indent=1)
    written.append("vpt_%d: %s +%d, %d change frames -> chunks with a change %d/14" % (i, os.path.basename(p)[:20], s0, len(ch), len(chunks3)))
print("\n".join(written))
