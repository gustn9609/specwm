"""마스터 릴: 그룹별 [타이틀 카드 2.5 s] + [나란히 비교 클립] 을 순서대로 이어붙여 한 파일로.
   캔버스 1600x900 (letterbox), 16 fps. 챕터 타임스탬프 chapters.json 저장. 각 그룹 대표 프레임 storyboard/*.jpg."""
import os; os.environ["OMP_NUM_THREADS"] = "2"
import cv2; cv2.setNumThreads(2)
import numpy as np, imageio, json
S = os.environ["SCRATCH"]+""; R = S + "/lingbot-world-v2/specwm_out"; CV = S + "/compare_videos"; SB = CV + "/storyboard"; os.makedirs(SB, exist_ok=True)
W, H, FPS = 1600, 900, 16
def read(p):
    c = cv2.VideoCapture(p); fr = []; fps = c.get(cv2.CAP_PROP_FPS) or 16
    while True:
        ok, f = c.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return fr, fps
def fit(img):
    h, w = img.shape[:2]; s = min(W / w, (H - 60) / h); nw, nh = int(w * s) // 2 * 2, int(h * s) // 2 * 2
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA); can = np.zeros((H, W, 3), np.uint8); y = 60 + (H - 60 - nh) // 2; x = (W - nw) // 2; can[y:y+nh, x:x+nw] = im; return can
def text(img, s, y, scale, color=(255, 255, 255), thick=2, x=None):
    (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick); x = (W - tw) // 2 if x is None else x
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA); return y + th + 18
def card(idx, n, title, lines, hold=2.5):
    im = np.zeros((H, W, 3), np.uint8); im[:] = (18, 18, 22)
    y = text(im, "%d / %d" % (idx, n), 150, 0.9, (160, 160, 170), 2); y = text(im, title, y + 30, 1.5, (255, 255, 255), 3); y += 30
    for l in lines: y = text(im, l, y, 0.85, (200, 220, 240), 2)
    return [im] * int(hold * FPS)
def header(can, idx, n, title):
    cv2.rectangle(can, (0, 0), (W, 60), (18, 18, 22), -1); cv2.putText(can, "%d/%d  %s" % (idx, n, title), (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA); return can
def composite(fast_p, arm_p, k):
    """KV-rewrite 표시 스트림: 0..k 청크는 Fast 화면, 이후 arm 프레임"""
    A, _ = read(fast_p); B, _ = read(arm_p); j = 12 * k; return A[:j] + B[j:]
# ---- KV-rewrite 그룹 (17) 생성 ----
RV = R + "/resync_v2/W18_sink6"; FA = RV + "/fast_alone_s1234/video.mp4"
def lab(img, t, sc=0.55):
    img = img.copy(); (w, h), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, sc, 2); cv2.rectangle(img, (0, 0), (w + 16, h + 16), (0, 0, 0), -1); cv2.putText(img, t, (8, h + 8), cv2.FONT_HERSHEY_SIMPLEX, sc, (255, 255, 255), 2); return img
def tile(vids, cols, th):
    n = len(vids); rows = (n + cols - 1) // cols; T = max(len(v) for _, v in vids); H0, W0 = vids[0][1][0].shape[:2]; tw = int(W0 * th / H0); out = []
    for t in range(T):
        rws = []
        for r in range(rows):
            cells = []
            for c in range(cols):
                i = r * cols + c
                cells.append(lab(cv2.resize(vids[i][1][min(t, len(vids[i][1]) - 1)], (tw, th), interpolation=cv2.INTER_AREA), vids[i][0]) if i < n else np.zeros((th, tw, 3), np.uint8))
            rws.append(np.concatenate(cells, 1))
        out.append(np.concatenate(rws, 0))
    return out
if not os.path.exists(CV + "/17_kv_rewrite.mp4"):
    v17 = [("Base (teacher)", read(R + "/revisit/base_official_S05/video.mp4")[0]), ("Fast alone", read(FA)[0]), ("ideal: KV = Base 0-4.5s", read(RV + "/resync_k6_s1234/video.mp4")[0]),
           ("mixed ctx: Fast 0-2.25 + laundered target", read(RV + "/cond_laundered_s1234/video.mp4")[0]),
           ("KV-rewrite: display=Fast, KV=laundered+RePaint", composite(FA, R + "/kvrw_e3c_repaint_laundered/W18_sink6/resync_k6_s1234/video.mp4", 6)),
           ("KV-rewrite: display=Fast, KV=Base-like", composite(FA, R + "/kvrw_e3sep_self_mask/W18_sink6/resync_k6_s1234/video.mp4", 6))]
    fr = tile(v17, 3, 260); imageio.mimwrite(CV + "/17_kv_rewrite.mp4", fr, fps=16, quality=7, codec="libx264", pixelformat="yuv420p"); print("17 저장", flush=True)
# ---- 릴 구성 (스토리 순서) ----
G = [
 ("01_base_sampler_bug_vs_official_S05", "Base sampler bug -> official UniPC", ["My custom Base sampler re-noised every step: collapse on rotating camera.", "Official UniPC (70 steps) is the corrected Base reference (visual 8/10)."]),
 ("02_base_sampler_bug_vs_official_hands", "Same fix, static hands scene", ["Static scenes hid the bug; only rotating trajectories exposed it."]),
 ("03_base_steps_S05", "Few-step vs distillation (LingBot)", ["Base 16 steps ~= 70 (VBench subj 0.777 vs 0.781). Base 4 steps: foggy (2/10).", "Fast 4-step (DMD) 6/10 -> distillation RECOVERS few-step loss."]),
 ("04_base_steps_hands", "Same on hands", ["Base 70 / 16 / 4 vs Fast 4 (DMD)."]),
 ("05_solarwm_stage1_vs_stage2", "SolarWM Stage1 vs Stage2", ["Stage1 (teacher-forced, non-distilled) collapses at ANY step count (exposure bias).", "Stage2 (self-forcing/DMD) survives rollout. Same conclusion as LingBot."]),
 ("08_e6_hands_causal_horizon", "E6: causal horizon (Base16, chunk-causal mask)", ["Failure onset scales with visible horizon: N=3 ~1s, N=6 ~1.5-2s, N=12 ~3-4s, all: never.", "VBench subj 0.732 -> 0.801 -> 0.870 -> 0.948. Physics collapse = horizon effect, not distillation."]),
 ("11_physics_hybrid_refine", "Hybrid refine: Fast + Base low-noise expert", ["3 SDEdit steps (0.40/0.20/0.08) with Base low-noise expert restore texture (5 -> 7/10)."]),
 ("10_serving_fast_vs_ours_hands", "Serving: 14B Fast on 24 GB", ["GPTQ int6 + KV 4-bit + VAE tiling: 58.4 -> 23.3 GiB peak.", "TTFF 3.6 s, 2.45 fps (Fast bf16: 3.4 s, 3.8 fps)."]),
 ("12_window_sweep_55s", "55 s clip: KV window 4.5 / 9 / 18 s (4 fps)", ["Window gain collapses to +0.001 at 55 s. Earlier +0.227 was a prompt-image mismatch artifact."]),
 ("06_resync_k3_k6_k9_every3", "Re-grounding (cloud-grounded streaming)", ["Commit Base frames into Fast's KV at k. Drift-to-teacher 0.449 -> 0.287 (= seed floor 0.284).", "k=6 (4.5 s) best. Plateau IS the measurement floor."]),
 ("09_e7_seeds", "E7: reproducibility across 3 seeds", ["Seed std 0.006-0.026 vs effect size -0.12."]),
 ("07_plateau_window_sink", "E5 / plateau: window, pinned anchors, cheaper context", ["Pinned Base anchors (sink 18) extend tau 1.8 -> 4.1 s; Base16 context = Base70 context.", "Imaging quality stuck at 0.72-0.73 regardless: Fast's own 4-step texture."]),
 ("15_e4_delayed_rollback_keep_xfade", "E4: delayed correction - rollback / keep / crossfade", ["Rollback snap 1.27/1.68/1.91x (L=2/4/8); keep: no snap, no fix.", "Latent crossfade (sigma 0.94): snap 0.82/1.25/0.95x, keeps 70-90% of the correction."]),
 ("13_e3_prefix_conditioning", "E3: target conditioned on Fast's drifted prefix", ["Cause = Fast's distribution shift (mask works on Base's own prefix: 0.232).", "Fast prefix + mask 0.392 -> RePaint 0.324 -> laundered + RePaint 0.310 (floor 0.30)."]),
 ("14_laundering_fast_prefix", "Laundering the Fast prefix", ["Low-noise expert, 3 steps: sharper wings/foliage, no new artifacts (5 -> 7/10)."]),
 ("17_kv_rewrite", "KV-rewrite: display stays Fast, KV becomes the target", ["Mixed context caps lock-on at 0.39 even with an oracle target.", "Rewriting KV 0-4.5 s: laundered target 63% retention; Base-like target 104% (0.285)."]),
 ("16_t20_long_horizon", "20 s horizon: does drift re-accumulate?", ["One re-grounding at 4.5 s: excess ~0 for the next 15 s (identity self-propagates).", "Color reverts to Fast's palette after eviction (~9 s) unless anchors are pinned."]),
]
n = len(G); frames_out = []; chapters = []; t = 0.0
wr = imageio.get_writer(CV + "/00_MASTER_REEL.mp4", fps=FPS, quality=6, codec="libx264", pixelformat="yuv420p", macro_block_size=None)
for i, (name, title, lines) in enumerate(G, 1):
    p = CV + "/%s.mp4" % name
    if not os.path.exists(p): print("없음", name); continue
    fr, fps = read(p); rep = max(1, int(round(FPS / fps)))
    chapters.append(dict(idx=i, group=name, title=title, start_s=round(t, 1)))
    for c in card(i, n, title, lines): wr.append_data(c); t += 1 / FPS
    for k, f in enumerate(fr):
        can = header(fit(f), i, n, title)
        if k == len(fr) // 2: cv2.imwrite(SB + "/%02d.jpg" % i, cv2.cvtColor(can, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        for _ in range(rep): wr.append_data(can); t += 1 / FPS
    print("  %2d %-45s %5.1fs (누적 %5.1fs)" % (i, name, len(fr) * rep / FPS, t), flush=True)
wr.close(); json.dump(chapters, open(CV + "/chapters.json", "w"), indent=1); print("[REEL_DONE] total %.1f s" % t, flush=True)
