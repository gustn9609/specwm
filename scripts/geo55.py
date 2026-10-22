#!/usr/bin/env python3
"""카메라-기하 일관성 probe (geo_metric) 를 이 서버 데이터 형식에 맞춰 실행.
poses: action_dir/poses.npy 의 프레임 i 자세 (생성 프레임 i 에 그대로 대응), K: action_dir/intrinsics.npy[0] (832x480 기준).
frames_all.npy 는 4fps (프레임 인덱스 i*4).
지표: abs inlier(전체 쌍 평균), in = |dt|<=4.5s, out = |dt|>4.5s, retention = out/in, dt-bin 곡선,
      revisit 쌍(자세각 <5°, 위치거리 < 2% extent, dt>4.5s) 의 inlier = '같은 자리로 돌아왔을 때' 일관성.
사용: python geo55.py <out.json> tag:frames_all.npy:action_dir [...]   (또는 --scan 으로 base55/fast55/oracle/floor 자동)
"""
import os, sys, json, glob, re
import numpy as np
S = os.environ.get("SCRATCH", "/home/hk28965/specwm"); R = S + "/lingbot-world-v2/specwm_out"
sys.path.insert(0, S + "/scripts")
from geo_metric import PatchMatcher, fundamental, sym_epi_dist


def geo_consistency(V, poses, K, matcher, thr_px=8.0, min_baseline=1e-3, max_pairs=400, seed=0):
    """geo_metric.geo_consistency 와 동일하되 (i, j) 인덱스를 함께 반환 (return-pair 지표용)."""
    T = len(V); feats, gh, gw = matcher.patches(V); H, W = V.shape[1], V.shape[2]
    fwd = poses[:, :3, :3][:, :, 2]; rng = np.random.default_rng(seed)
    pairs = [(i, j) for i in range(T) for j in range(i + 1, T)]
    if len(pairs) > max_pairs: pairs = [pairs[k] for k in rng.choice(len(pairs), max_pairs, replace=False)]
    rec = []
    for i, j in pairs:
        F, base = fundamental(poses[i], poses[j], K)
        if base < min_baseline: continue
        pi, pj, _ = matcher.mutual_nn(feats[i], feats[j], gh, gw, H, W)
        ang = float(np.degrees(np.arccos(np.clip((fwd[i] * fwd[j]).sum(), -1, 1))))
        if len(pi) < 10: rec.append(((j - i) / 4.0, ang, 0.0, len(pi), i, j)); continue
        d = sym_epi_dist(F, pi, pj); rec.append(((j - i) / 4.0, ang, float((d < thr_px).mean()), len(pi), i, j))
    return np.array(rec)

ACT = {"05": S + "/rt2_action_05", "01": S + "/rt2_action_01", "00": S + "/rt2_action_00", "02": S + "/rt2_action_02", "hands": S + "/hands_action"}
ACT55 = {"05": S + "/traj55_action_05", "01": S + "/traj55_action_01", "02": S + "/traj55_action_02"}


def scan():
    items = []
    for p in glob.glob(R + "/base55/*/frames_all.npy") + glob.glob(R + "/fast55/*/frames_all.npy") + glob.glob(R + "/oracle/*/*/frames_all.npy") + glob.glob(R + "/posegram/*/*/frames_all.npy"):
        tag = os.path.relpath(p, R).replace("/frames_all.npy", ""); mm = re.search(r"S(\d\d)", tag); sc = mm.group(1) if mm else "05"
        mr = re.search(r"R(\d\d)r(\d+)", tag)                      # 재방문 통제 궤적: posegram/R05r4_W18/...
        items.append((tag, p, S + "/data/rev_action_%s_r%s" % (mr.group(1), mr.group(2)) if mr else ACT55[sc]))
    for p in glob.glob(R + "/floor/base/S*/frames_all.npy") + glob.glob(R + "/floor/fast/S*/W18_sink6/fast_alone_s*/frames_all.npy"):
        tag = os.path.relpath(p, R).replace("/frames_all.npy", ""); mm = re.search(r"/S(\d\d|hands)", p); sc = mm.group(1)
        items.append((tag, p, ACT[sc]))
    return items


def analyze(rec, win_s=4.5, poses=None):
    if len(rec) == 0: return {}
    dt, ang, inl, n = rec[:, 0], rec[:, 1], rec[:, 2], rec[:, 3]
    ok = n >= 10; dt, ang, inl = dt[ok], ang[ok], inl[ok]
    o = dict(pairs=int(ok.sum()), abs_inlier=round(float(inl.mean()), 4))
    i_in, i_out = inl[dt <= win_s], inl[dt > win_s]
    o["in_4.5s"] = round(float(i_in.mean()), 4) if len(i_in) else None; o["out_4.5s"] = round(float(i_out.mean()), 4) if len(i_out) else None
    o["retention"] = round(o["out_4.5s"] / o["in_4.5s"], 4) if o["in_4.5s"] and o["out_4.5s"] is not None else None
    bins = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 40), (40, 100)]
    o["curve"] = {"%d-%ds" % b: (round(float(inl[(dt > b[0]) & (dt <= b[1])].mean()), 4) if ((dt > b[0]) & (dt <= b[1])).any() else None) for b in bins}
    rv = (dt > win_s) & (ang < 5.0)
    o["revisit_pairs"] = int(rv.sum()); o["revisit_inlier"] = round(float(inl[rv].mean()), 4) if rv.any() else None
    if poses is not None and rec.shape[1] >= 6:                      # true-return 쌍: 같은 자리(위치<5% extent, 각<10°) 로 15% 이상 이탈 후 복귀
        pos = poses[:, :3, 3]; ext = float(np.linalg.norm(pos.max(0) - pos.min(0))) + 1e-9; ii = rec[ok, 4].astype(int); jj = rec[ok, 5].astype(int)
        ret = np.zeros(len(ii), bool)
        for q in range(len(ii)):
            i, j = ii[q], jj[q]
            if dt[q] > win_s and ang[q] < 10 and np.linalg.norm(pos[i] - pos[j]) / ext < 0.05:
                ret[q] = np.linalg.norm(pos[i:j] - pos[j], axis=1).max() / ext > 0.15
        o["return_pairs"] = int(ret.sum()); o["return_inlier"] = round(float(inl[ret].mean()), 4) if ret.any() else None
    return o


def main():
    out_path = sys.argv[1]; args = sys.argv[2:]
    items = scan() if args == ["--scan"] else [tuple(a.split(":")) for a in args]
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    mt = PatchMatcher(hub=S + "/.torch_hub")
    for tag, p, act in items:
        if tag in res: continue
        V = np.load(p); P = np.load(os.path.join(act, "poses.npy")); K4 = np.load(os.path.join(act, "intrinsics.npy"))[0]
        fi = np.arange(len(V)) * 4; fi = fi[fi < len(P)]; V = V[:len(fi)]; Ps = P[fi]
        H, W = V.shape[1], V.shape[2]; fx, fy, cx, cy = K4; sx, sy = W / 832.0, H / 480.0
        K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]])
        rec = geo_consistency(V, Ps, K, mt, thr_px=8.0, max_pairs=(3000 if len(V) > 60 else 100000))
        res[tag] = analyze(rec, poses=Ps); res[tag]["frames"] = int(len(V))
        print("  %-55s abs %.4f in %s out %s ret %s revisit %s(n=%s) return %s(n=%s)" % (tag, res[tag]["abs_inlier"], res[tag]["in_4.5s"], res[tag]["out_4.5s"], res[tag]["retention"], res[tag]["revisit_inlier"], res[tag]["revisit_pairs"], res[tag].get("return_inlier"), res[tag].get("return_pairs")), flush=True)
        json.dump(res, open(out_path, "w"), indent=1)
    print("[GEO55_DONE] %d items -> %s" % (len(res), out_path), flush=True)


if __name__ == "__main__":
    main()
