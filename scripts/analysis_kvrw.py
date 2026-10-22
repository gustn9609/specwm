import os
"""KV rewrite 실험: 표시 프레임은 Fast 그대로, KV(context 0-4.5 s)만 provider 영상으로 전부 commit.
   arm resync_k6 (chunks 0-5 <- provider) / resync_k3 (0-2 <- provider) 의 chunks 6-12 거리: to Base1234, to provider(lock-on), to Fast alone."""
import numpy as np, torch, os, json
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
FB = feats(np.load(RV + "/base_frames.npy")); FAf = feats(np.load(RV + "/W18_sink6/fast_alone_s1234/frames_all.npy"))
out = {"ref": {"fast_alone_to_base_6_12": float(np.mean(pc(dist(FAf, FB))[6:])), "resync_k6_base1234_to_base_6_12": float(np.mean(pc(dist(feats(np.load(RV + "/W18_sink6/resync_k6_s1234/frames_all.npy")), FB))[6:]))}}
for prov in ["e3c_repaint_laundered", "e3a_repaint_fast", "e3sep_self_mask"]:
    d0 = R + "/kvrw_%s/W18_sink6" % prov
    if not os.path.isdir(d0): continue
    FP = feats(np.load(R + "/revisit/%s/frames_all.npy" % prov)); rec = {}
    for arm in ["resync_k6", "resync_k3", "every3", "resync_k9"]:
        p = d0 + "/%s_s1234/frames_all.npy" % arm
        if not os.path.exists(p): continue
        F = feats(np.load(p)); rec[arm] = dict(to_base1234_6_12=round(float(np.mean(pc(dist(F, FB))[6:])), 3), lockon_provider_6_12=round(float(np.mean(pc(dist(F, FP))[6:])), 3), to_fast_alone_6_12=round(float(np.mean(pc(dist(F, FAf))[6:])), 3), to_base_pc=pc(dist(F, FB)))
        print("  %-24s %-10s toBase1234 %.3f  lock-on %.3f  toFast %.3f" % (prov, arm, rec[arm]["to_base1234_6_12"], rec[arm]["lockon_provider_6_12"], rec[arm]["to_fast_alone_6_12"]), flush=True)
    # 표시 프레임(Fast 0-2.25 s) 과 provider 0-2.25 s 의 차이 (KV 만 바꿨을 때 화면과 KV 의 괴리)
    rec["display_vs_kv_prefix_0_2"] = round(float(np.mean(pc(dist(FAf, FP))[:3])), 3); out[prov] = rec
json.dump(out, open(R + "/analysis_kvrw.json", "w"), indent=1); print("[KVRW_ANALYSIS_DONE]", flush=True)
