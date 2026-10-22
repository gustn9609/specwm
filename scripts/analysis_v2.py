import os
"""J3 분석 (gh-dev, specwm-venv). 입력: resync_v2/*/frames_all.npy, base_frames.npy, E1 Base 시드 영상, 기존 resync_s05 arm.
출력: analysis_v2.json
  - dino_to_base[arm]: 청크별 DINO 거리 (Base seed1234 프레임 대비)
  - E1 floor: Base16 시드 간 청크별 DINO 거리 (Base-vs-Base) 및 Base70(1234) vs Base16(시드) 거리
  - E2 local epipolar consistency per chunk (arm 자체 기하 일관성, ±1 청크 쌍) + 재접지 후 재성장 시상수 τ (DINO 기준)
  - E4 rollback 오프라인: snap_L = |fast_alone[chunk k+L 직전 프레임] - resync_k3[chunk k+L 첫 프레임]|, drift after k+L
  - snap for keep arms
"""
import numpy as np, torch, os, json, glob, sys
sys.path.insert(0, os.environ["SCRATCH"]+"")
S = os.environ["SCRATCH"]+""; R = S + "/lingbot-world-v2/specwm_out"; RV = R + "/resync_v2"
torch.hub.set_dir(S + "/.torch_hub")
m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval().cuda()
mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1).cuda(); std = torch.tensor([0.229,0.224,0.225]).view(3,1,1).cuda()
def feats(V):
    X = torch.from_numpy(V).permute(0,3,1,2).float().cuda()/255
    X = torch.nn.functional.interpolate(X, size=(518,518), mode="bilinear"); X = (X-mean)/std
    o = []
    with torch.no_grad():
        for i in range(0, len(X), 8): o.append(torch.nn.functional.normalize(m(X[i:i+8]).flatten(1), dim=-1).cpu())
    return torch.cat(o)
def per_chunk(dist, n=13): return [float(dist[c*3:(c+1)*3].mean()) for c in range(min(n, len(dist)//3))]
def tau_fit(d):
    """재접지 직후 3~4 점으로 d(t)=d_inf-(d_inf-d0)exp(-t/tau) 를 격자탐색 (t 단위 청크)."""
    d = np.asarray(d, float)
    if len(d) < 3: return None
    best = None
    for dinf in np.linspace(d.max(), d.max()*1.5+0.05, 30):
        for tau in np.linspace(0.3, 8, 78):
            t = np.arange(len(d)); pred = dinf - (dinf - d[0]) * np.exp(-t / tau); err = ((pred - d)**2).sum()
            if best is None or err < best[0]: best = (err, tau, dinf)
    return dict(tau_chunks=round(best[1], 2), tau_s=round(best[1]*0.75, 2), d_inf=round(best[2], 3))

out = {}
B = np.load(RV + "/base_frames.npy"); FB = feats(B); n = len(B)
# ---- E1 noise floor ----
e1 = {}
B16 = {}
for tag, p in [("b70_s1234", R + "/revisit/base_official_S05/frames_all.npy"), ("b16_s1234", R + "/revisit/base_official_S05_s16/frames_all.npy"),
               ("b16_s4321", R + "/revisit/base_official_S05_s16_seed4321/frames_all.npy"), ("b16_s9999", R + "/revisit/base_official_S05_s16_seed9999/frames_all.npy")]:
    if os.path.exists(p): B16[tag] = feats(np.load(p)[:n])
keys = list(B16)
for i in range(len(keys)):
    for j in range(i+1, len(keys)):
        a, b = keys[i], keys[j]; nn_ = min(len(B16[a]), len(B16[b]))
        e1["%s_vs_%s" % (a, b)] = per_chunk(1 - (B16[a][:nn_] * B16[b][:nn_]).sum(-1).numpy())
out["E1_floor"] = e1
print("E1 pairs:", {k: round(float(np.mean(v[6:])), 3) for k, v in e1.items()}, flush=True)

# ---- local epipolar consistency (E2) ----
from geo_metric import PatchMatcher, geo_consistency
K4 = np.load(S + "/lingbot-world-v2/examples/00/intrinsics.npy")[0]
idx = np.load(S + "/revisit_rt_idx_05.npy"); P = np.load(S + "/poses_05.npy")
mt = PatchMatcher(hub=S + "/.torch_hub")
def local_epi(V):
    """청크별 국소 기하 일관성: 청크 c 의 저장프레임(3장) ± 인접 청크 프레임까지 창(최대 9장)에서 에피폴라 inlier 평균."""
    fi_all = np.arange(len(V)) * 4; fi_all = fi_all[fi_all < len(idx)]; V = V[:len(fi_all)]
    H, W = V.shape[1], V.shape[2]; fx, fy, cx, cy = K4; sx, sy = W/832.0, H/480.0
    K = np.array([[fx*sx,0,cx*sx],[0,fy*sy,cy*sy],[0,0,1]])
    vals = []
    nC = len(V) // 3
    for c in range(nC):
        lo, hi = max(0, 3*c - 3), min(len(V), 3*c + 6)
        Vw = V[lo:hi]; Pw = P[idx[fi_all[lo:hi]]]
        r = geo_consistency(Vw, Pw, K, mt, thr_px=8.0, max_pairs=100000)
        vals.append(float(r[:,2].mean()) if len(r) else float("nan"))
    return vals
# ---- arms ----
arms = {}
for d in sorted(glob.glob(RV + "/W*/*/frames_all.npy")):
    key = d.split("/resync_v2/")[1].rsplit("/", 1)[0]; V = np.load(d); nn_ = min(len(V), n)
    F = feats(V[:nn_]); dist = 1 - (F * FB[:nn_]).sum(-1).numpy(); pc = per_chunk(dist)
    v = V[:nn_].astype(np.float32)/255; dif = np.abs(v[1:]-v[:-1]).mean(axis=(1,2,3))
    plan = json.load(open(RV + "/resync_v2.json")).get(key, {}).get("plan", "")
    rec = dict(plan=plan, dist_per_chunk=pc, med_diff=float(np.median(dif)))
    # 재접지 후 τ (첫 생성 청크부터 4청크)
    if plan and "G" in plan and plan[0] != "G":
        k = plan.index("G") if plan.startswith("B") else (len(plan) - plan[::-1].index("F") if "F" in plan else plan.index("G"))
        if plan.startswith("B") and "F" in plan: k = len(plan) - plan[::-1].index("F")   # keep arms: 마지막 F 다음
        seg = pc[k:k+5]; rec["regrow_from_chunk"] = k; rec["tau"] = tau_fit(seg)
        rec["post3"] = float(np.mean(pc[k:k+3])) if k+3 <= len(pc) else None
    try:
        rec["epi_per_chunk"] = local_epi(V[:nn_])
        if rec.get("regrow_from_chunk") is not None:
            k = rec["regrow_from_chunk"]; e = rec["epi_per_chunk"]
            # 기하 지표는 '높을수록 좋음' 이라 감소 곡선; τ 는 (1-e) 로 맞춘다
            rec["tau_epi"] = tau_fit([1 - x for x in e[k:k+5]])
    except Exception as ex: rec["epi_err"] = str(ex)[:80]
    arms[key] = rec; print("  %-36s post3=%s tau=%s" % (key, rec.get("post3"), rec.get("tau")), flush=True)
out["arms"] = arms
# ---- E4 rollback 오프라인 (W18_sink6, seed 1234): resync_k3 = rollback 상태. snap_L = |fast_alone[3(k+L)-1] - resync_k3[3(k+L)]|
try:
    FA = np.load(RV + "/W18_sink6/fast_alone_s1234/frames_all.npy").astype(np.float32)/255
    RK3 = np.load(RV + "/W18_sink6/resync_k3_s1234/frames_all.npy").astype(np.float32)/255
    med = np.median(np.abs(FA[1:]-FA[:-1]).mean(axis=(1,2,3))); e4 = {}
    for L in (2, 4, 8):
        j = 3 * (3 + L); snap = float(np.abs(FA[j-1] - RK3[j]).mean()) / med
        e4["rollback_L%d" % L] = dict(snap_ratio=round(snap, 2), drift_after=float(np.mean(arms["W18_sink6/resync_k3_s1234"]["dist_per_chunk"][3+L:])))
    for L in (2, 4, 8):
        kk = "W18_sink6/keep_k3_L%d_s1234" % L
        if kk in arms:
            KV = np.load(RV + "/" + kk + "/frames_all.npy").astype(np.float32)/255; j = 3*(3+L)
            e4["keep_L%d" % L] = dict(snap_ratio=round(float(np.abs(KV[j-1] - KV[j]).mean())/med, 2), drift_after=float(np.mean(arms[kk]["dist_per_chunk"][3+L:])))
    e4["fast_alone_after"] = {("L%d" % L): float(np.mean(arms["W18_sink6/fast_alone_s1234"]["dist_per_chunk"][3+L:])) for L in (2,4,8)}
    out["E4"] = e4
except Exception as e: out["E4"] = {"error": str(e)[:200]}
json.dump(out, open(RV + "/analysis_v2.json", "w"), indent=1); print("[ANALYSIS_V2_DONE]", flush=True)
