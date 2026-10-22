"""비교군별 side-by-side 영상. 각 타일에 라벨, 길이는 최장에 맞춰 마지막 프레임 hold.
출력: $SCRATCH/compare_videos/<group>.mp4"""
import os; os.environ["OMP_NUM_THREADS"] = "1"
import cv2; cv2.setNumThreads(1)
import numpy as np, imageio, json, glob
S = os.environ["SCRATCH"]+""; R = S + "/lingbot-world-v2/specwm_out"; OUT = S + "/compare_videos"; os.makedirs(OUT, exist_ok=True)
def read(p, max_frames=None):
    c = cv2.VideoCapture(p); fr = []
    while True:
        ok, f = c.read()
        if not ok or (max_frames and len(fr) >= max_frames): break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return np.stack(fr) if fr else None
def label(img, txt, scale=0.7):
    img = img.copy(); (w, h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(img, (0, 0), (w + 16, h + 16), (0, 0, 0), -1); cv2.putText(img, txt, (8, h + 8), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2); return img
def tile(vids, cols, tile_h):
    """vids: [(label, frames)], 격자 배치. 빈 칸은 검정."""
    n = len(vids); rows = (n + cols - 1) // cols; T = max(len(v) for _, v in vids)
    H0, W0 = vids[0][1].shape[1:3]; th = tile_h; tw = int(W0 * th / H0)
    out = []
    for t in range(T):
        rowimgs = []
        for r in range(rows):
            cells = []
            for c in range(cols):
                i = r * cols + c
                if i < n:
                    lab, v = vids[i]; f = v[min(t, len(v) - 1)]
                    f = cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA); cells.append(label(f, lab, 0.55 if th < 300 else 0.7))
                else: cells.append(np.zeros((th, tw, 3), np.uint8))
            rowimgs.append(np.concatenate(cells, axis=1))
        out.append(np.concatenate(rowimgs, axis=0))
    return out
def composite_rollback(fast, regen, k, L):
    """사용자가 보는 스트림: Fast 청크 0..k+L-1 그대로, 그 뒤는 재생성(resync_k) 프레임"""
    j = 12 * (k + L); return np.concatenate([fast[:j], regen[j:]], axis=0)
def p(*a): return os.path.join(*a)
RV = R + "/resync_v2/W18_sink6"; BASE = R + "/revisit/base_official_S05/video.mp4"
G = []
G.append(("01_base_sampler_bug_vs_official_S05", [("Base BUGGY sampler (re-noise)", R + "/revisit/base_S05/video.mp4"), ("Base OFFICIAL (UniPC 70)", BASE), ("Fast 4-step", RV + "/fast_alone_s1234/video.mp4")], 3, 300))
G.append(("02_base_sampler_bug_vs_official_hands", [("Base BUGGY", R + "/hands/base_p1/video.mp4"), ("Base OFFICIAL", R + "/hands/base_official_p1/video.mp4"), ("Fast", R + "/hands/fast_p1/video.mp4")], 3, 300))
G.append(("03_base_steps_S05", [("Base 70 steps", BASE), ("Base 16 steps", R + "/revisit/base_official_S05_s16/video.mp4"), ("Base 8 steps", R + "/revisit/base_official_S05_s8/video.mp4"), ("Base 4 steps", R + "/revisit/base_official_S05_s4/video.mp4"), ("Fast 4 steps (DMD)", RV + "/fast_alone_s1234/video.mp4")], 3, 260))
G.append(("04_base_steps_hands", [("Base 70", R + "/hands/base_official_p1/video.mp4"), ("Base 16", R + "/hands/base_official_p1_s16/video.mp4"), ("Base 4", R + "/hands/base_official_p1_s4/video.mp4"), ("Fast 4 (DMD)", R + "/hands/fast_p1/video.mp4")], 2, 300))
G.append(("05_solarwm_stage1_vs_stage2", [("SolarWM Stage1 50-step (non-distilled AR)", S + "/solar_stage1_nfe50.mp4"), ("SolarWM Stage1 4-step", S + "/solar_stage1_nfe4.mp4"), ("SolarWM Stage2 4-step (DMD)", S + "/solar_stage2_nfe4.mp4"), ("LingBot Fast 4-step (14B)", RV + "/fast_alone_s1234/video.mp4"), ("LingBot Base 70 (14B)", BASE)], 3, 260))
G.append(("06_resync_k3_k6_k9_every3", [("Base (teacher)", BASE), ("Fast alone", RV + "/fast_alone_s1234/video.mp4"), ("re-grounded @2.25s (k3)", RV + "/resync_k3_s1234/video.mp4"), ("re-grounded @4.5s (k6)", RV + "/resync_k6_s1234/video.mp4"), ("re-grounded @6.75s (k9)", RV + "/resync_k9_s1234/video.mp4"), ("re-grounded every 2.25s", RV + "/every3_s1234/video.mp4")], 3, 260))
G.append(("07_plateau_window_sink", [("W18 sink6 (baseline)", RV + "/resync_k6_s1234/video.mp4"), ("W36 sink6", R + "/resync_v2/W36_sink6/resync_k6_s1234/video.mp4"), ("W18 sink0", R + "/resync_v2/W18_sink0/resync_k6_s1234/video.mp4"), ("W24 sink6", R + "/resync_v2/W24_sink6/resync_k6_s1234/video.mp4"), ("W24 sink18 (pinned anchors)", R + "/resync_v2/W24_sink18/resync_k6_s1234/video.mp4"), ("ctx = Base 16-step", RV + "/resync_k6_ctx16_s1234/video.mp4")], 3, 260))
G.append(("08_e6_hands_causal_horizon", [("Base16 causal N=3 (0.75s)", R + "/hands/base_causal_N3/video.mp4"), ("N=6 (1.5s)", R + "/hands/base_causal_N6/video.mp4"), ("N=12 (3s)", R + "/hands/base_causal_N12/video.mp4"), ("N=all (bidirectional)", R + "/hands/base_official_p1_s16/video.mp4"), ("Fast 4-step (N=18, DMD)", R + "/hands/fast_p1/video.mp4")], 3, 260))
G.append(("09_e7_seeds", [("Fast alone s1234", RV + "/fast_alone_s1234/video.mp4"), ("Fast alone s4321", RV + "/fast_alone_s4321/video.mp4"), ("Fast alone s9999", RV + "/fast_alone_s9999/video.mp4"), ("re-grounded k6 s1234", RV + "/resync_k6_s1234/video.mp4"), ("re-grounded k6 s4321", RV + "/resync_k6_s4321/video.mp4"), ("re-grounded k6 s9999", RV + "/resync_k6_s9999/video.mp4")], 3, 260))
G.append(("10_serving_fast_vs_ours_hands", [("Fast bf16 (58 GiB)", R + "/ours_same/caseB_hands/FAST_bf16/video.mp4"), ("Ours int6+KV4+tiled (23 GiB)", R + "/ours_same/caseB_hands/OURS_gptq6_kv4_tiled/video.mp4"), ("Base official", R + "/hands/base_official_p1/video.mp4")], 3, 300))
G.append(("11_physics_hybrid_refine", [("Fast", R + "/hybrid/hands/FAST_asis/video.mp4"), ("Fast + Base low-noise refine", R + "/hybrid/hands/REFINE_0.40_0.20_0.08/video.mp4"), ("Base official", R + "/hands/base_official_p1/video.mp4")], 3, 300))
G.append(("12_window_sweep_55s", [("Fast window 4.5s", R + "/win60/W18/video.mp4"), ("window 9s", R + "/win60/W36/video.mp4"), ("window 18s", R + "/win60/W72/video.mp4")], 3, 300))
# E3 (J-A 결과가 있으면 포함)
e3 = [("Fast alone (prefix source)", RV + "/fast_alone_s1234/video.mp4"), ("Base | mask prefix=Fast (E3)", R + "/revisit/base_prefix_S05_L9/video.mp4"), ("Base independent", BASE)]
for tag, lab in [("e3sep_self_mask", "Base | mask prefix=own seed"), ("e3sep_otherseed_mask", "Base | mask prefix=other seed"), ("e3a_repaint_fast", "Base | RePaint prefix=Fast"), ("e3a_repaint_self", "Base | RePaint prefix=own"), ("e3b_causal12_repaint_fast", "Base-causal N=12 | RePaint prefix=Fast"), ("e3c_repaint_laundered", "Base | RePaint prefix=laundered Fast")]:
    if os.path.exists(R + "/revisit/%s/video.mp4" % tag): e3.append((lab, R + "/revisit/%s/video.mp4" % tag))
G.append(("13_e3_prefix_conditioning", e3, 3, 240))
la = glob.glob(R + "/hybrid/s05_launder/REFINE*/video.mp4")
if la: G.append(("14_laundering_fast_prefix", [("Fast prefix (original)", RV + "/fast_alone_s1234/video.mp4"), ("laundered (low-noise expert 3 steps)", la[0]), ("Base independent", BASE)], 3, 300))
done = []
G = [g for g in G if g[0][:2] in ('13', '14')]
for name, items, cols, th in G:
    vids = []
    for lab, path in items:
        if os.path.exists(path):
            v = read(path); 
            if v is not None: vids.append((lab, v))
        else: print("  없음:", name, lab, path, flush=True)
    if len(vids) < 2: print("  건너뜀:", name, flush=True); continue
    frames = tile(vids, cols, th)
    imageio.mimwrite(p(OUT, name + ".mp4"), frames, fps=16, quality=7, codec="libx264", pixelformat="yuv420p"); done.append(name); print("  저장:", name, frames[0].shape, len(frames), flush=True)
# E4: rollback 표시 스트림 합성 + keep + xfade
fast = read(RV + "/fast_alone_s1234/video.mp4"); rk3 = read(RV + "/resync_k3_s1234/video.mp4")
if fast is not None and rk3 is not None:
    vids = [("Fast alone (what the user sees)", fast)]
    for L in (2, 4, 8):
        vids.append(("rollback L=%d (switch @%.2fs)" % (L, (3 + L) * 0.75), composite_rollback(fast, rk3, 3, L)))
        kp = RV + "/keep_k3_L%d_s1234/video.mp4" % L
        if os.path.exists(kp): vids.append(("keep L=%d (no rollback)" % L, read(kp)))
        xf = RV + "/xfade_k3_L%d_s1234/video.mp4" % L
        if os.path.exists(xf): vids.append(("+xfade s=0.94 L=%d" % L, composite_rollback(fast, read(xf), 3, L)))
        xf3 = R + "/resync_v2_xf3/W18_sink6/xfade_k3_L%d_s1234/video.mp4" % L
        if os.path.exists(xf3): vids.append(("+xfade s=0.63 L=%d" % L, composite_rollback(fast, read(xf3), 3, L)))
    frames = tile(vids, 4 if len(vids) <= 8 else 5, 230); imageio.mimwrite(p(OUT, "15_e4_delayed_rollback_keep_xfade.mp4"), frames, fps=16, quality=7, codec="libx264", pixelformat="yuv420p"); done.append("15_e4"); print("  저장: 15_e4", frames[0].shape, flush=True)

# 20 s
t20 = [("Base16 20 s (seed 4321, teacher)", R + "/revisit/base20_S05_s16_seed4321/video.mp4"), ("Fast alone 20 s", R + "/resync20/W18_sink6/fast_alone_s1234/video.mp4"), ("re-grounded @4.5s (k6)", R + "/resync20/W18_sink6/resync_k6_s1234/video.mp4"), ("re-grounded every 2.25s (to 6.75s)", R + "/resync20/W18_sink6/every3_s1234/video.mp4"), ("W24 sink18: Fast alone", R + "/resync20/W24_sink18/fast_alone_s1234/video.mp4"), ("W24 sink18: re-grounded k6 (pinned)", R + "/resync20/W24_sink18/resync_k6_s1234/video.mp4")]
vids = [(l, read(p)) for l, p in t20 if os.path.exists(p)]
if len(vids) >= 2:
    frames = tile(vids, 3, 260); imageio.mimwrite(p(OUT, "16_t20_long_horizon.mp4"), frames, fps=16, quality=7, codec="libx264", pixelformat="yuv420p"); print("  저장: 16_t20", frames[0].shape, len(frames), flush=True)
json.dump(done, open(p(OUT, "done.json"), "w")); print("[CONCAT_DONE]", len(done), flush=True)
