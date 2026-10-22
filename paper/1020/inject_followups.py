"""[10/20] Follow-ups to the divergence-injection probe (paper/1020/README.md). GPU 5 only. Phased (fastest-first, meeting imminent):
 phase 1: T / alone / every for the first 5 seeds -> judge mp4s;  phase 2: once@6 (all seeds) + splice@6 boundary stats + sheet frames;
 phase 3: T / alone / every for the remaining seeds;  phase 4: once@3 / once@9 + splice@3 / @9.
 mp4: specwm_out/inject/judge/<arm>_<scene>_s<seed>.mp4 (crf 23, 16 fps). Boundary stats (data/inject_boundary.json): per clip, frame-to-frame
 pixel L1 (8-bit) and 1-SSIM (grey, 11x11 gaussian) at each chunk boundary c|c+1 (pixel frames 12c+8 | 12c+9: latent frame 0 -> 1 pixel frame,
 then 4 per latent frame, 3 latent frames per chunk) and the within-chunk mean. splice@k = alone chunks 0..k + once@k chunks k+1.. decoded jointly.
usage: CUDA_VISIBLE_DEVICES=5 envrun python /home/hk28965/specwm/paper/1020/inject_followups.py"""
import os, sys, json, subprocess, time, threading, numpy as np, torch, torch.nn.functional as F
S = "/home/hk28965/specwm"; R = S + "/lingbot-world-v2/specwm_out/inject/1p3b_4step"; TD = S + "/lingbot-world-v2/specwm_out/paths2"
JD = S + "/lingbot-world-v2/specwm_out/inject/judge"; OUTJ = S + "/paper/1020/data/inject_boundary.json"; SH = S + "/paper/1020/data/sheet_frames"
FF = S + "/tools/ffmpeg-7.0.2-amd64-static/ffmpeg"; FPC = 3; FPS = 16
SCENES = ["sand_filling_up_jar", "sandcastle_wave_impact", "easy_lake", "easy_stonehenge_sunset", "pouring_water_into_cup"]
SEEDS = [11, 22, 33, 44, 55, 66, 77, 1234, 4321, 9999]; JUDGE_ARMS = {"T", "alone", "every", "once@6"}
SHEET = {("sandcastle_wave_impact", 1234), ("easy_stonehenge_sunset", 1234)}
PHASES = [(SEEDS[:5], ["T", "alone", "every"]), (SEEDS, ["once@6", "splice@6"]), (SEEDS[5:], ["T", "alone", "every"]), (SEEDS, ["once@3", "splice@3", "once@9", "splice@9"])]
sys.path.insert(0, S + "/lingbot-world-v2"); from wan.modules.vae2_1 import Wan2_1_VAE
vae = Wan2_1_VAE(vae_pth=S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast/Wan2.1_VAE.pth", device="cuda:0")
@torch.no_grad()
def decode(lat):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16): im = vae.decode([lat.float().cuda()])[0].float().clamp(-1, 1)
    return ((im + 1) / 2 * 255).permute(1, 2, 3, 0).round().to(torch.uint8)   # (F, H, W, 3) uint8 on GPU
def _enc(fr, out):
    Fn, H, W, _ = fr.shape
    p = subprocess.Popen([FF, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "fast", "-threads", "4",
                          "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out + ".tmp.mp4"], stdin=subprocess.PIPE)
    p.stdin.write(fr.tobytes()); p.stdin.close(); p.wait(); os.replace(out + ".tmp.mp4", out)
THREADS = []
def write_mp4(fr, out): t = threading.Thread(target=_enc, args=(fr.cpu().numpy(), out)); t.start(); THREADS.append(t)
_g = torch.exp(-(torch.arange(11).float() - 5) ** 2 / (2 * 1.5 ** 2)); _g = (_g / _g.sum()); GW = (_g[:, None] * _g[None, :])[None, None].cuda()
@torch.no_grad()
def ssim_pairs(fr):
    g = (fr.float() @ torch.tensor([0.299, 0.587, 0.114], device=fr.device))[:, None]; C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2; x, y = g[:-1], g[1:]
    mx, my = F.conv2d(x, GW), F.conv2d(y, GW); sxx = F.conv2d(x * x, GW) - mx ** 2; syy = F.conv2d(y * y, GW) - my ** 2; sxy = F.conv2d(x * y, GW) - mx * my
    s = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2)); return (1 - s.mean((1, 2, 3))).cpu().numpy()
@torch.no_grad()
def l1_pairs(fr): return (fr[1:].float() - fr[:-1].float()).abs().mean((1, 2, 3)).cpu().numpy()
def bidx(c): return 12 * c + 8
def stats(fr, nch):
    l1, ss = l1_pairs(fr), ssim_pairs(fr); B = [bidx(c) for c in range(nch - 1)]; mask = np.ones(len(l1), bool); mask[B] = False
    return dict(l1_b=[float(l1[t]) for t in B], ss_b=[float(ss[t]) for t in B], l1_in=float(l1[mask].mean()), ss_in=float(ss[mask].mean()), l1_all=[float(v) for v in l1])
res = json.load(open(OUTJ)) if os.path.exists(OUTJ) else {}; t00 = time.time()
def lat_path(a, sc, sd): return "%s/%s/tau2_s0/K0_uniform4P_fx_s%d/lat.pt" % (TD, sc, sd) if a == "T" else "%s/%s/%s_s%d/lat.pt" % (R, sc, a, sd)
for pi, (seeds, arms) in enumerate(PHASES):
    for sc in SCENES:
        for sd in seeds:
            key = "%s/s%d" % (sc, sd); r = res.setdefault(key, {}); done = []
            for a in arms:
                mp4 = "%s/%s_%s_s%d.mp4" % (JD, a, sc, sd)
                if a in r and (a not in JUDGE_ARMS or os.path.exists(mp4)): continue
                if a.startswith("splice@"):
                    k = int(a[7:]); pa, po = lat_path("alone", sc, sd), lat_path("once@%d" % k, sc, sd)
                    if not (os.path.exists(pa) and os.path.exists(po)): print("skip", key, a, flush=True); continue
                    la, lo = torch.load(pa, map_location="cpu"), torch.load(po, map_location="cpu"); lat = torch.cat([la[:, :FPC * (k + 1)], lo[:, FPC * (k + 1):]], 1)
                else:
                    p = lat_path(a, sc, sd)
                    if not os.path.exists(p): print("skip", key, a, flush=True); continue
                    lat = torch.load(p, map_location="cpu")
                nch = lat.shape[1] // FPC; fr = decode(lat); r[a] = stats(fr, nch)
                if a in JUDGE_ARMS and not os.path.exists(mp4): write_mp4(fr, mp4)
                if (sc, sd) in SHEET and a in ("alone", "T", "once@6", "splice@6"):
                    os.makedirs(SH, exist_ok=True); lo_, hi_ = bidx(6) - 8, bidx(6) + 9; np.save("%s/%s_s%d_k6_%s.npy" % (SH, sc, sd, a), fr[lo_:hi_].cpu().numpy())
                del fr; done.append(a)
            if done:
                json.dump(res, open(OUTJ, "w")); torch.cuda.empty_cache()
                print("phase %d %s %s done, %.0f s total" % (pi + 1, key, done, time.time() - t00) + ("; splice@6 L1@6 %.2f vs alone %.2f / T %.2f" % (r["splice@6"]["l1_b"][6], r["alone"]["l1_b"][6], r["T"]["l1_b"][6]) if "splice@6" in r and "alone" in r and "T" in r else ""), flush=True)
    for t in THREADS: t.join()
    THREADS.clear(); print("[PHASE %d DONE] %.0f s" % (pi + 1, time.time() - t00), flush=True)
print("[FOLLOWUPS_DONE]")
