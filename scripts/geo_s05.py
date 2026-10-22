import os
"""프롬프트가 이미지와 맞는 scene 05 로 Base vs Fast 를 재측정한다.
scene 00 과 05 는 이미지·궤적이 동일하고 프롬프트만 다르다(00=만리장성 불일치, 05=드래곤 일치)."""
import numpy as np, os, sys, glob
sys.path.insert(0, os.environ["SCRATCH"]+"")
from geo_metric import PatchMatcher, geo_consistency
S = os.environ["SCRATCH"]+""; R = S + "/lingbot-world-v2/specwm_out"
K4 = np.load(S + "/lingbot-world-v2/examples/00/intrinsics.npy")[0]
IDX = {"00": np.load(S+"/rt2_idx.npy"), "05": np.load(S+"/revisit_rt_idx_05.npy")}
POS = {k: np.load(S+"/poses_%s.npy"%k) for k in ["00","05"]}
mt = PatchMatcher(hub=S+"/.torch_hub"); out = {}
targets = [("BASE_S05", "05", R+"/revisit/base_S05/frames_all.npy"),
           ("BASE_S00", "00", R+"/revisit/base_rt2/frames_all.npy")]
for tag in ["S05_W18_bf16_s1234","S05_W18_kv4_s1234","S05_W36_kv4_s1234","S05_W72_kv4_s1234"]:
    targets.append(("FAST_"+tag, "05", R+"/multi_scene/%s/frames_all.npy" % tag))
for tag, scene, p in targets:
    if not os.path.exists(p): print("  없음:", tag, p, flush=True); continue
    V = np.load(p); idx = IDX[scene]
    fi = np.arange(len(V))*4; fi = fi[fi < len(idx)]; V = V[:len(fi)]
    Ps = POS[scene][idx[fi]]; H, W = V.shape[1], V.shape[2]
    fx, fy, cx, cy = K4; sx, sy = W/832.0, H/480.0
    K = np.array([[fx*sx,0,cx*sx],[0,fy*sy,cy*sy],[0,0,1]])
    r = geo_consistency(V, Ps, K, mt, thr_px=8.0, max_pairs=1200)
    out[tag] = r
    print("  %-22s %3d프레임 %4d쌍 inlier %.4f" % (tag, len(V), len(r), r[:,2].mean()), flush=True)
np.savez(S+"/geo_s05.npz", **out); print("저장 %d" % len(out), flush=True)
