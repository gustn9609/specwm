import os
"""4라운드 분석 (gh-dev GPU). 출력 $R/analysis_r4.json
 repro: 장면 3 × 시드 3: fast_alone / kvrw_ideal / kvrw_laundered 의 chunks 6-12 거리
        - to Base16 ref (같은 시드), to provider(target_s{seed}) lock-on, to fast_alone
        - retention(lock-on) = (d(fast,prov) - d(arm,prov)) / (d(fast,prov) - d(ideal,base16))
 stale: 55 s arms: 홈뷰 프레임(각 excursion 의 최소 시야각 프레임)에서 Base 홈 프레임(t=0 입력/ Base16 첫 프레임)과의 DINO 거리; 청크별 to-Base16(0-10 s 구간) ; ctrl 비트동일
 lockon: S05 s1234: provider l15..l80,raw: display/KV 격차 = d(fast_alone 0-5, provider 0-5); gain = d(fast,base16)-d(arm,base16) over 6-12; boundary snap 5->6
 targetcost: revisit.json 의 gen_s/peak/hw + VBench 는 별도
"""
import numpy as np, torch, os, json, glob
S = os.environ["SCRATCH"]+""; R = S + "/lingbot-world-v2/specwm_out"
torch.hub.set_dir(S + "/.torch_hub"); m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval().cuda()
mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1).cuda(); std = torch.tensor([0.229,0.224,0.225]).view(3,1,1).cuda()
def feats(V):
    X = torch.from_numpy(np.ascontiguousarray(V)).permute(0,3,1,2).float().cuda()/255; X = torch.nn.functional.interpolate(X, size=(518,518), mode="bilinear"); X = (X-mean)/std; o = []
    with torch.no_grad():
        for i in range(0, len(X), 8): o.append(torch.nn.functional.normalize(m(X[i:i+8]).flatten(1), dim=-1).cpu())
    return torch.cat(o)
def dist(FA, FB): n = min(len(FA), len(FB)); return (1 - (FA[:n] * FB[:n]).sum(-1)).numpy()
def pc(d): return [float(d[c*3:(c+1)*3].mean()) for c in range(len(d)//3)]
def load4(p): V = np.load(p); return V[::4] if len(V) > 100 else V
def ex(p): return os.path.exists(p)
out = {}
# ---------------- repro ----------------
rep = {}
for sc in ["05", "01", "02"]:
    d0 = R + "/repro/S%s" % sc
    for sd in (1234, 4321, 9999):
        fa_p = d0 + "/W18_sink6/fast_alone_s%d/frames_all.npy" % sd; b16_p = d0 + "/base16_s%d/frames_all.npy" % sd; tg_p = d0 + "/target_s%d/frames_all.npy" % sd
        if sc == "05" and sd == 1234:
            b16_p = b16_p if ex(b16_p) else R + "/revisit/base_official_S05_s16/frames_all.npy"; tg_p = tg_p if ex(tg_p) else R + "/revisit/e3c_repaint_laundered/frames_all.npy"
        arms = {"kvrw_laundered": d0 + "/W18_sink6/kvrw_laundered_s%d/frames_all.npy" % sd, "kvrw_ideal": d0 + "/W18_sink6/kvrw_ideal_s%d/frames_all.npy" % sd}
        if sc == "05" and sd == 1234:
            if not ex(arms["kvrw_laundered"]): arms["kvrw_laundered"] = R + "/kvrw_e3c_repaint_laundered/W18_sink6/resync_k6_s1234/frames_all.npy"
            if not ex(arms["kvrw_ideal"]): arms["kvrw_ideal"] = R + "/resync_v2/W18_sink6/resync_k6_ctx16_s1234/frames_all.npy"
        if not (ex(fa_p) and ex(b16_p) and ex(tg_p)): print("repro 미완:", sc, sd, [ex(fa_p), ex(b16_p), ex(tg_p)], flush=True); continue
        FA = feats(load4(fa_p)); FB = feats(load4(b16_p)); FT = feats(load4(tg_p)); r = {}
        d_fa_b = float(np.mean(pc(dist(FA, FB))[6:])); d_fa_t = float(np.mean(pc(dist(FA, FT))[6:])); r["fast_alone"] = dict(to_base16=round(d_fa_b, 3), to_target=round(d_fa_t, 3))
        r["target_after_prefix_to_base16"] = round(float(np.mean(pc(dist(FT, FB))[3:])), 3)
        ideal = None
        for n, p in arms.items():
            if not ex(p): continue
            F = feats(load4(p)); db = float(np.mean(pc(dist(F, FB))[6:])); dt = float(np.mean(pc(dist(F, FT))[6:])); dfa = float(np.mean(pc(dist(F, FA))[6:]))
            r[n] = dict(to_base16=round(db, 3), to_target=round(dt, 3), to_fast=round(dfa, 3))
            if n == "kvrw_ideal": ideal = db
        if ideal is not None and "kvrw_laundered" in r:
            r["retention_base16"] = round((d_fa_b - r["kvrw_laundered"]["to_base16"]) / (d_fa_b - ideal + 1e-9), 3)
            r["retention_lockon"] = round((d_fa_t - r["kvrw_laundered"]["to_target"]) / (d_fa_t - ideal + 1e-9), 3)
        rep["S%s_s%d" % (sc, sd)] = r; print("  repro", sc, sd, {k: v for k, v in r.items() if k.startswith("retention")}, "fast", r["fast_alone"], "laund", r.get("kvrw_laundered"), "ideal", r.get("kvrw_ideal"), flush=True)
out["repro"] = rep
# ---------------- stale (55 s) ----------------
st = {}; d55 = R + "/resync55"
if os.path.isdir(d55):
    P = np.load(S + "/traj55_action_05/poses.npy"); F0 = P[:, :3, 2]; ang = np.degrees(np.arccos(np.clip((F0[0] * F0).sum(1), -1, 1)))
    # 홈뷰: 4fps 프레임 인덱스 = 픽셀/4 ; 후보: 각 구간(10-25 s, 25-40 s, 40-55 s)에서 최소 각도 프레임
    homes = []
    for lo, hi in [(10, 25), (25, 40), (40, 54.5)]:
        seg = np.arange(int(lo*16), int(hi*16)); j = int(seg[np.argmin(ang[seg])]); homes.append((round(j/16, 2), j//4, round(float(ang[j]), 1)))
    st["home_views"] = homes
    B16 = load4(R + "/revisit/base_official_S05_s16/frames_all.npy"); FB = feats(B16); home_ref = FB[0:1]
    for p in sorted(glob.glob(d55 + "/W*/*/frames_all.npy")):
        key = p.split("/resync55/")[1].rsplit("/", 1)[0]; V = np.load(p); F = feats(V)
        home_d = [round(float(1 - (F[min(h[1], len(F) - 1)] @ home_ref[0])), 3) for h in homes]
        first10 = pc(dist(F, FB)); rec = dict(home_dist=home_d, to_base16_chunks6_12=round(float(np.mean(first10[6:13])), 3), gen_s=None)
        # 색 편향 (자홍) 구간별
        v = V.astype(np.float32); mag = ((v[..., 0] + v[..., 2]) / 2 - v[..., 1]).mean(axis=(1, 2)); rec["magenta_bias_per_10s"] = [round(float(mag[i*40:(i+1)*40].mean()), 1) for i in range(6) if len(mag[i*40:(i+1)*40])]
        st[key] = rec; print("  stale", key, rec, flush=True)
    # ctrl 비트동일
    fa = d55 + "/W24_sink18/fast_alone_s1234/frames_all.npy"; ct = d55 + "/W24_sink18/ctrl_stale_k20_s1234/frames_all.npy"
    if ex(fa) and ex(ct): st["ctrl_max_abs_diff_255"] = int(np.abs(np.load(fa).astype(int) - np.load(ct).astype(int)).max()); print("  ctrl diff", st["ctrl_max_abs_diff_255"], flush=True)
out["stale"] = st
# ---------------- lock-on curve ----------------
lk = {}; dl = R + "/lockon"
if os.path.isdir(dl):
    FA = feats(load4(R + "/resync_v2/W18_sink6/fast_alone_s1234/frames_all.npy")); FB = feats(load4(R + "/revisit/base_official_S05_s16/frames_all.npy"))
    A = load4(R + "/resync_v2/W18_sink6/fast_alone_s1234/frames_all.npy").astype(np.float32); med = np.median(np.abs(A[1:] - A[:-1]).mean(axis=(1, 2, 3)))
    provs = {"l15": R + "/lockon/launder/REFINE_0.15_0.07_0.03", "l25": R + "/lockon/launder/REFINE_0.25_0.12_0.05", "l40": R + "/hybrid/s05_launder/REFINE_0.40_0.20_0.08", "l60": R + "/lockon/launder/REFINE_0.60_0.30_0.12", "l80": R + "/lockon/launder/REFINE_0.80_0.40_0.16", "raw": R + "/resync_v2/W18_sink6/fast_alone_s1234", "maskref": R + "/revisit/e3d_mask_refined", "repaintref": R + "/revisit/e3c_repaint_laundered"}
    d_fa_b = float(np.mean(pc(dist(FA, FB))[6:])); lk["fast_alone_to_base16_6_12"] = round(d_fa_b, 3)
    for n, pd in provs.items():
        pp = pd + "/frames_all.npy"; ap_ = dl + "/W18_sink6/kvrw_%s_s1234/frames_all.npy" % n
        if not (ex(pp) and ex(ap_)): continue
        FP = feats(load4(pp)); F = feats(load4(ap_)); X = load4(ap_).astype(np.float32)
        gap = float(np.mean(pc(dist(FA, FP))[:6])); db = float(np.mean(pc(dist(F, FB))[6:])); dfa = float(np.mean(pc(dist(F, FA))[6:]))
        snap = float(np.abs(A[17] - X[18]).mean() / med)     # 4fps: 청크5 마지막(17) -> 청크6 첫(18)
        lk[n] = dict(display_kv_gap=round(gap, 3), to_base16_6_12=round(db, 3), gain=round(d_fa_b - db, 3), to_fast_6_12=round(dfa, 3), boundary_snap=round(snap, 2), provider_to_base16_0_5=round(float(np.mean(pc(dist(FP, FB))[:6])), 3))
        print("  lockon", n, lk[n], flush=True)
out["lockon"] = lk
# ---------------- target cost ----------------
tc = {}
for d in sorted(glob.glob(R + "/targetcost/area*_s*")):
    j = d + "/revisit.json"
    if ex(j): r = json.load(open(j)); tc[os.path.basename(d)] = {k: r.get(k) for k in ("hw", "steps", "gen_s", "peak_alloc_GiB", "frames")}
out["targetcost"] = tc; print("  targetcost", tc, flush=True)

# ---------------- targetcost_kv: 저해상/8step 타깃을 provider 로 ----------------
tk = {}; dtk = R + "/targetcost_kv/W18_sink6"
if os.path.isdir(dtk):
    FA = feats(load4(R + "/resync_v2/W18_sink6/fast_alone_s1234/frames_all.npy")); FB = feats(load4(R + "/revisit/base_official_S05_s16/frames_all.npy"))
    ref_p = dtk + "/kvrw_tc386048_s16_s1234/frames_all.npy"; FR = feats(load4(ref_p)) if ex(ref_p) else None
    d_fa_b = float(np.mean(pc(dist(FA, FB))[6:])); tk["fast_alone_to_base16"] = round(d_fa_b, 3)
    for d in sorted(glob.glob(dtk + "/kvrw_*_s1234/frames_all.npy")):
        n = d.split("/kvrw_")[1].split("_s1234")[0]; F = feats(load4(d)); db = float(np.mean(pc(dist(F, FB))[6:]))
        tk[n] = dict(to_base16_6_12=round(db, 3), gain=round(d_fa_b - db, 3), to_fullres_target_arm=round(float(np.mean(pc(dist(F, FR))[6:])), 3) if FR is not None else None); print("  targetcost_kv", n, tk[n], flush=True)
out["targetcost_kv"] = tk
json.dump(out, open(R + "/analysis_r4.json", "w"), indent=1); print("[ANALYSIS_R4_DONE]", flush=True)
