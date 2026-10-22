import os
"""E3 재계산 (4fps 정렬) + cond arm lock-on 지표. 출력 analysis_v3b.json"""
import numpy as np, torch, os, json, glob
S = os.environ["SCRATCH"]+""; R = S + "/lingbot-world-v2/specwm_out"; RV = R + "/resync_v2"
torch.hub.set_dir(S + "/.torch_hub"); m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval().cuda()
mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1).cuda(); std = torch.tensor([0.229,0.224,0.225]).view(3,1,1).cuda()
def feats(V):
    X = torch.from_numpy(np.ascontiguousarray(V)).permute(0,3,1,2).float().cuda()/255; X = torch.nn.functional.interpolate(X, size=(518,518), mode="bilinear"); X = (X-mean)/std; o = []
    with torch.no_grad():
        for i in range(0, len(X), 8): o.append(torch.nn.functional.normalize(m(X[i:i+8]).flatten(1), dim=-1).cpu())
    return torch.cat(o)
def dist(FA, FB): n = min(len(FA), len(FB)); return (1 - (FA[:n] * FB[:n]).sum(-1)).numpy()
def pc(d): return [float(d[c*3:(c+1)*3].mean()) for c in range(len(d)//3)]
def load4(p):
    V = np.load(p); return V[::4] if len(V) > 100 else V          # 16fps 161 -> 4fps 41 ; 이미 4fps 면 그대로
B = load4(RV + "/base_frames.npy"); FB = feats(B); print("base ref frames", B.shape, flush=True)
FA_np = load4(RV + "/W18_sink6/fast_alone_s1234/frames_all.npy"); FAf = feats(FA_np); fa = pc(dist(FAf, FB)); ub = pc(dist(feats(load4(RV + "/W18_sink6/resync_k6_s1234/frames_all.npy")), FB))
out = {"fast_alone_pc": fa, "resync_k6_pc": ub, "floor_10s": 0.284}
# E3 Base 영상 (정렬 수정)
e3 = {}; e3feats = {}
for tag in ["e3sep_self_mask", "e3sep_otherseed_mask", "base_prefix_S05_L9", "e3a_repaint_fast", "e3a_repaint_self", "e3b_causal12_repaint_fast", "e3c_repaint_laundered"]:
    p = R + "/revisit/%s/frames_all.npy" % tag
    if not os.path.exists(p): continue
    V = load4(p); F = feats(V); e3feats[tag] = F; d = pc(dist(F, FB)); dfa = pc(dist(F, FAf))
    v = V.astype(np.float32)/255; dif = np.abs(v[1:]-v[:-1]).mean(axis=(1,2,3))
    e3[tag] = dict(dist_to_base_pc=d, after_prefix_to_base=round(float(np.mean(d[3:13])), 3), after_prefix_to_fast=round(float(np.mean(dfa[3:13])), 3), ctx_3_5_to_base=round(float(np.mean(d[3:6])), 3),
                   seam_ratio=round(float(dif[8] / (np.median(dif) + 1e-9)), 2), prefix_check_to_fast=round(float(np.mean(dfa[:3])), 3), prefix_check_to_base=round(float(np.mean(d[:3])), 3))
    print("  %-28s toBase(after) %.3f  toFast(after) %.3f  ctx3-5 %.3f  seam %.2f | prefix->fast %.3f base %.3f" % (tag, e3[tag]["after_prefix_to_base"], e3[tag]["after_prefix_to_fast"], e3[tag]["ctx_3_5_to_base"], e3[tag]["seam_ratio"], e3[tag]["prefix_check_to_fast"], e3[tag]["prefix_check_to_base"]), flush=True)
out["e3_base"] = e3
# cond arms: (a) Base-1234 기준 이득 유지율 (기존), (b) lock-on: 자기 context 제공자 영상의 청크 6-12 와의 거리
prov = {"cond_sep_self": "e3sep_self_mask", "cond_sep_other": "e3sep_otherseed_mask", "cond_repaint_fast": "e3a_repaint_fast", "cond_repaint_self": "e3a_repaint_self", "cond_causal12": "e3b_causal12_repaint_fast", "cond_laundered": "e3c_repaint_laundered", "cond_k6_W3": "base_prefix_S05_L9"}
cond = {}
for key, tag in prov.items():
    p = RV + "/W18_sink6/%s_s1234/frames_all.npy" % key
    if not os.path.exists(p) or tag not in e3feats: continue
    F = feats(load4(p)); d = pc(dist(F, FB)); dl = pc(dist(F, e3feats[tag]))
    g = (np.mean(fa[6:]) - np.mean(d[6:])) / (np.mean(fa[6:]) - np.mean(ub[6:]))
    cond[key] = dict(gain_retention_vs_base1234=round(float(g), 3), dist_to_base1234_6_12=round(float(np.mean(d[6:])), 3), lockon_to_provider_6_12=round(float(np.mean(dl[6:])), 3), lockon_pc=dl)
    print("  %-18s retention %.3f  toBase1234 %.3f  lock-on(provider) %.3f" % (key, g, np.mean(d[6:]), np.mean(dl[6:])), flush=True)
# 참조: resync_k6 의 lock-on (= 0.287, provider = Base1234), fast_alone 의 provider-less 거리 = 0.449
cond["_ref"] = dict(resync_k6_lockon=round(float(np.mean(ub[6:])), 3), fast_alone_to_base=round(float(np.mean(fa[6:])), 3))
out["cond"] = cond
json.dump(out, open(R + "/analysis_v3b.json", "w"), indent=1); print("[ANALYSIS_V3B_DONE]", flush=True)
