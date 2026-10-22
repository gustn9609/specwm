import os
"""3라운드 분석 (gh-dev). 출력 analysis_v3.json
  cond arms: (fast_alone - cond)/(fast_alone - resync_k6) 이득 유지율, prefix 원천별
  E3-sep / E3' Base 영상 자체: prefix 이후 구간의 Base(seed1234 독립) 대비 DINO 거리 + prefix 경계 seam(픽셀 점프/중앙값)
  xfade: snap = |fast_alone[3(k+L)-1] - xfade[3(k+L)]| / median ; drift after
  20 s: fast_alone / resync_k6 / W24_sink18 의 청크별 DINO-to-Base20(seed1234) ; floor20 = Base20 seed1234 vs 4321 ;
        excess = dist - floor ; tau_acc: excess(t) = e_inf - (e_inf - e0) exp(-t/tau) 피팅 (재접지 후)
"""
import numpy as np, torch, os, json, glob
S = os.environ["SCRATCH"]+""; R = S + "/lingbot-world-v2/specwm_out"; RV = R + "/resync_v2"; R20 = R + "/resync20"
torch.hub.set_dir(S + "/.torch_hub")
m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval().cuda()
mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1).cuda(); std = torch.tensor([0.229,0.224,0.225]).view(3,1,1).cuda()
def feats(V):
    X = torch.from_numpy(V).permute(0,3,1,2).float().cuda()/255; X = torch.nn.functional.interpolate(X, size=(518,518), mode="bilinear"); X = (X-mean)/std
    o = []
    with torch.no_grad():
        for i in range(0, len(X), 8): o.append(torch.nn.functional.normalize(m(X[i:i+8]).flatten(1), dim=-1).cpu())
    return torch.cat(o)
def dist(FA, FB):
    n = min(len(FA), len(FB)); return (1 - (FA[:n] * FB[:n]).sum(-1)).numpy()
def per_chunk(d): return [float(d[c*3:(c+1)*3].mean()) for c in range(len(d)//3)]
def fit(y, lo=0.3, hi=40):
    y = np.asarray(y, float); best = None
    for yinf in np.linspace(y.min(), max(y.max()*1.6, y.min()+0.01), 40):
        for tau in np.linspace(lo, hi, 120):
            t = np.arange(len(y)); pred = yinf - (yinf - y[0]) * np.exp(-t / tau); err = ((pred - y)**2).sum()
            if best is None or err < best[0]: best = (err, tau, yinf)
    return dict(tau_chunks=round(best[1], 2), tau_s=round(best[1]*0.75, 2), y_inf=round(best[2], 3))
out = {}
# ---------- 10 s: Base ref + cond/xfade arms ----------
B = np.load(RV + "/base_frames.npy"); FB = feats(B)
FAf = feats(np.load(RV + "/W18_sink6/fast_alone_s1234/frames_all.npy")); fa = per_chunk(dist(FAf, FB))
ub = per_chunk(dist(feats(np.load(RV + "/W18_sink6/resync_k6_s1234/frames_all.npy")), FB))
cond = {}
RK3 = np.load(RV + "/W18_sink6/resync_k3_s1234/frames_all.npy").astype(np.float32)/255
for d in sorted(glob.glob(RV + "/W18_sink6/cond_*_s1234/frames_all.npy")) + sorted(glob.glob(RV + "/W18_sink6/xfade_*_s1234/frames_all.npy")) + sorted(glob.glob(R + "/resync_v2_xf3/W18_sink6/xfade_*_s1234/frames_all.npy")):
    key = d.split("/W18_sink6/")[1].split("/")[0] + ("_node3" if "_xf3/" in d else ""); pc = per_chunk(dist(feats(np.load(d)), FB))
    rec = dict(dist_per_chunk=pc)
    if key.startswith("cond"):
        g = (np.mean(fa[6:]) - np.mean(pc[6:])) / (np.mean(fa[6:]) - np.mean(ub[6:])); rec["gain_retention"] = round(float(g), 3); rec["ctx_chunks_3_5_dist"] = round(float(np.mean(pc[3:6])), 3)
    if key.startswith("xfade"):
        L = int(key.split("L")[-1].split("_")[0]); j = 3*(3+L)
        A = np.load(RV + "/W18_sink6/fast_alone_s1234/frames_all.npy").astype(np.float32)/255; X = np.load(d).astype(np.float32)/255
        med = np.median(np.abs(A[1:]-A[:-1]).mean(axis=(1,2,3)))
        rec["snap_ratio"] = round(float(np.abs(A[j-1] - X[j]).mean() / med), 2); rec["snap_rollback"] = round(float(np.abs(A[j-1] - RK3[j]).mean() / med), 2); rec["drift_after"] = round(float(np.mean(pc[3+L:])), 3); rec["drift_after_rollback"] = round(float(np.mean(per_chunk(dist(feats((RK3*255).astype(np.uint8)), FB))[3+L:])), 3); rec["drift_after_fast"] = round(float(np.mean(fa[3+L:])), 3)
    cond[key] = rec; print("  ", key, {k: v for k, v in rec.items() if k != "dist_per_chunk"}, flush=True)
out["fast_alone_pc"] = fa; out["resync_k6_pc"] = ub; out["cond_xfade"] = cond
# ---------- E3 Base 영상 자체 평가 ----------
e3 = {}
for tag in ["e3sep_self_mask", "e3sep_otherseed_mask", "e3a_repaint_fast", "e3a_repaint_self", "e3b_causal12_repaint_fast", "e3c_repaint_laundered", "base_prefix_S05_L9"]:
    p = R + "/revisit/%s/frames_all.npy" % tag
    if not os.path.exists(p): continue
    V = np.load(p); pc = per_chunk(dist(feats(V), FB)); v = V.astype(np.float32)/255; dif = np.abs(v[1:]-v[:-1]).mean(axis=(1,2,3))
    e3[tag] = dict(dist_per_chunk=pc, after_prefix_dist=round(float(np.mean(pc[3:])), 3), seam_ratio=round(float(dif[8] / (np.median(dif) + 1e-9)), 2))   # 저장프레임 8->9 = 픽셀 32->36 경계
    print("  ", tag, "after-prefix dist %.3f seam %.2f" % (e3[tag]["after_prefix_dist"], e3[tag]["seam_ratio"]), flush=True)
out["e3_base"] = e3
# ---------- 20 s ----------
t20 = {}
ref20 = R + "/revisit/base20_S05_s16_seed4321/frames_all.npy"; b15 = {sd: R + "/revisit/base15_S05_s16_seed%d/frames_all.npy" % sd for sd in (1234, 9999)}
if os.path.exists(ref20) and all(os.path.exists(p) for p in b15.values()):
    F1 = feats(np.load(ref20)); Fa = feats(np.load(b15[1234])); Fb = feats(np.load(b15[9999]))
    fl = per_chunk(dist(Fa, Fb)); fl2 = per_chunk(dist(Fa, F1)); fl3 = per_chunk(dist(Fb, F1))      # 3 쌍 평균 (15 s 구간)
    floor15 = [float(np.mean(v)) for v in zip(fl, fl2, fl3)]; n20 = 27
    floor = floor15 + [float(np.mean(floor15[-5:]))] * (n20 - len(floor15)); t20["floor20_pc"] = floor; t20["floor_note"] = "chunks 0-19 measured (3 seed pairs), 20-26 extended flat"
    t20["ref_vs_fast10s_check"] = per_chunk(dist(F1, FB))[:13]
    for d in sorted(glob.glob(R20 + "/W*/*/frames_all.npy")):
        key = d.split("/resync20/")[1].rsplit("/", 1)[0]; pc = per_chunk(dist(feats(np.load(d)), F1)); ex = [a - b for a, b in zip(pc, floor)]
        rec = dict(dist_per_chunk=pc, excess=ex)
        if "resync_k6" in key: rec["tau_acc"] = fit(ex[6:]); rec["excess_6_8"] = round(float(np.mean(ex[6:9])), 3); rec["excess_last6"] = round(float(np.mean(ex[-6:])), 3)
        if "fast_alone" in key: rec["excess_mean"] = round(float(np.mean(ex[6:])), 3)
        t20[key] = rec; print("  ", key, {k: v for k, v in rec.items() if k not in ("dist_per_chunk", "excess")}, flush=True)
else:
    print("  base20 not ready:", b20, flush=True)
out["t20"] = t20
json.dump(out, open(R + "/analysis_v3.json", "w"), indent=1); print("[ANALYSIS_V3_DONE]", flush=True)
