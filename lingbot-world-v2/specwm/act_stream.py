"""[10/13, proposal A benchmark] Synthetic camera-action streams with change points, for action-speculative generation.
A stream is one relative SE(3) per LATENT frame (0.25 s), in the normalized units the Fast model sees after compute_relative_poses
(StEvo: pressing S = translation z -1 per latent frame (camera +z = forward, so 'speed' here is along -z; symmetric for the benchmark), 'right-30' = yaw +12.65 deg per latent frame = 3 deg per video frame).
Key-press model: the velocity (yaw rate, pitch rate, forward speed, strafe) is piecewise constant; segment lengths ~ 1 + Poisson(4 tau - 1)
latent frames (--tau = mean segment length in seconds = the change-point density axis); at a change point ONE component changes with
probability --single (with probability --release a held key is released, i.e. the component goes to 0), otherwise all four are redrawn. --jitter adds per-frame Gaussian noise (deg) to the
yaw / pitch rates (a hand on a mouse instead of a key), which makes 'same action' hits inexact.
Output <out>/<name>/{rel.npy [F,4,4], intrinsics.npy, stream.json, poses.npy [4F-3,4,4] (c2w at video frame rate, for prepare()/other tools)}.
make_stream(params, seed) is importable (specwm/act_spec.py fits its action prior on fresh streams of the same family).
usage: python -m specwm.act_stream --out DIR --tau 2 --seeds 0,1,2 [--frames 41] [--name tau2] [--jitter 1.0]"""
import argparse, os, json, math, numpy as np
DEF = dict(tau=2.0, frames=41, yaw="0,0,4,-4,8,-8,12.65,-12.65", pitch="0,0,0,0,3,-3", speed="0,0.5,1,1", strafe="0,0,0,0.5,-0.5", still=0.15, single=0.75, release=0.5, jitter=0.0)
def rot(axis, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg)); R = np.eye(4)
    if axis == "y": R[0, 0] = c; R[0, 2] = s; R[2, 0] = -s; R[2, 2] = c      # yaw: positive = StEvo 'right'
    else: R[1, 1] = c; R[1, 2] = -s; R[2, 1] = s; R[2, 2] = c             # pitch about x
    return R
def se3(yaw, pitch, speed, strafe):
    T = rot("y", yaw) @ rot("x", pitch); T[:3, 3] = [strafe, 0.0, -speed]; return T
def make_stream(P, seed):
    """returns rel [F,4,4] (frame 0 = identity), per-frame velocity tuples (F, the 'key state'), segments"""
    rng = np.random.default_rng(seed); F = int(P["frames"]); vel = []; segs = []; f = 0
    sets = [[float(x) for x in str(P[k]).split(",")] for k in ("yaw", "pitch", "speed", "strafe")]
    def draw(): return tuple(float(rng.choice(s)) for s in sets)
    while f < F:
        n = 1 + rng.poisson(max(P["tau"] * 4 - 1, 0)); n = int(min(n, F - f))
        if not segs: v = (0.0, 0.0, 0.0, 0.0) if rng.random() < P["still"] else draw()
        elif rng.random() < P["still"] and segs[-1]["v"] != (0.0, 0.0, 0.0, 0.0): v = (0.0, 0.0, 0.0, 0.0)
        elif rng.random() < P["single"]:   # one key: change exactly one component
            v = list(segs[-1]["v"]); i = int(rng.integers(4)); cand = [x for x in sets[i] if x != v[i]]
            v[i] = 0.0 if (v[i] != 0 and rng.random() < P["release"]) else float(rng.choice(cand)); v = tuple(v)   # release the held key, or press another
        else:
            v = draw()
            while v == segs[-1]["v"]: v = draw()
        segs.append(dict(start=f, len=n, v=v)); vel += [v] * n; f += n
    rel = [np.eye(4)]
    for i in range(1, F):
        y, p, s, st = vel[i]
        if P["jitter"] > 0: y += float(rng.normal(0, P["jitter"])); p += float(rng.normal(0, P["jitter"]))
        rel.append(se3(y, p, s, st))
    return np.stack(rel), vel, segs
def c2w_from_rel(rel):
    """absolute c2w per latent frame (chain), then 4 video frames per latent step (translation interpolated, rotation stepped) for prepare()"""
    C = [np.eye(4)]
    for r in rel[1:]: C.append(C[-1] @ r)
    out = []
    for i in range(len(C) - 1):
        for k in range(4):
            t = k / 4.0; A, Bm = C[i], C[i + 1]; M = A.copy(); M[:3, 3] = (1 - t) * A[:3, 3] + t * Bm[:3, 3]; M[:3, :3] = A[:3, :3] if k < 2 else Bm[:3, :3]; out.append(M)
    out.append(C[-1]); return np.stack(out)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--seeds", default="0"); ap.add_argument("--name", default="")
    for k, v in DEF.items(): ap.add_argument("--" + k, type=type(v), default=v)
    a = ap.parse_args(); P = {k: getattr(a, k) for k in DEF}; os.makedirs(a.out, exist_ok=True); K = np.array([[502.9, 503.1, 415.8, 239.8]], dtype=np.float32)
    for seed in [int(s) for s in a.seeds.split(",")]:
        rel, vel, segs = make_stream(P, seed); name = "%s_s%d" % (a.name or ("tau%g" % a.tau), seed); d = os.path.join(a.out, name); os.makedirs(d, exist_ok=True)
        np.save(d + "/rel.npy", rel.astype(np.float32)); np.save(d + "/intrinsics.npy", np.repeat(K, 4 * len(rel) - 3, 0)); np.save(d + "/poses.npy", c2w_from_rel(rel).astype(np.float32))
        ch = [s["start"] for s in segs[1:]]; chunks3 = sorted(set(c // 3 for c in ch if c >= 3))   # 3-frame chunks (frames 3c..3c+2) containing a change point = unpredictable by the constant-velocity predictor
        json.dump(dict(seed=seed, params=P, frames=len(rel), segments=segs, vel=vel, change_frames=ch, change_chunks3=chunks3, units="relative SE(3) per latent frame, normalized translation (1 = StEvo straight), yaw deg/frame positive = right"), open(d + "/stream.json", "w"), indent=1)
        print("[AS] %s: %d segments, change frames %s -> 3-frame chunks with a change %d/%d" % (name, len(segs), ch, len(chunks3), (len(rel) - 1) // 3 + 1))
