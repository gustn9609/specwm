"""[10/20] Soft re-anchor sweep, GPU part (VAE decode only; GPU 5 under flock): for the same-model pair (device = 14B 2-step, cloud = 14B 4-step T),
k = 6, arms alone (alpha 0) / soft@6_a0.25 / a0.5 / a0.75 / once@6 (alpha 1) / prog@6 (0.25 -> 0.5 -> 1.0 over chunks 6, 7, 8):
 (a) DEVICE VIEW per arm = alone chunks 0..6 + the arm's chunks 7.. decoded jointly (what the viewer sees); frame-to-frame pixel L1 / 1-SSIM at every chunk
     boundary (inject_followups.py convention: boundary c|c+1 = pixel frames 12c+8 | 12c+9) + within-chunk mean -> data/soft_boundary.json
     (also the un-spliced clips: alone, T, once@6 own boundary, soft own boundary = the blended context decoded as stored);
 (b) judge mp4s of the device views (crf 23, 16 fps) -> specwm_out/inject/judge_soft/<arm>_<scene>_s<seed>.mp4 (+ alone full clip, T reused from judge/);
 (c) side-by-side video per scene at seed 1234: 2 rows x 4: top = device view for alpha 0 | 0.25 | 0.5 | 1.0, bottom = the KV content (blended context as
     stored in lat.pt, then the same continuation) -> paper/1020/videos/soft_reanchor_<scene>_s1234.mp4.
usage: CUDA_VISIBLE_DEVICES=5 envrun python /home/hk28965/specwm/paper/1020/soft_decode.py"""
import os, sys, json, subprocess, time, threading, numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
S = "/home/hk28965/specwm"; R = S + "/lingbot-world-v2/specwm_out/inject/14b_2step"; TD = S + "/lingbot-world-v2/specwm_out/paths2"
JD = S + "/lingbot-world-v2/specwm_out/inject/judge_soft"; OUTJ = S + "/paper/1020/data/soft_boundary.json"; VD = S + "/paper/1020/videos"; os.makedirs(JD, exist_ok=True); os.makedirs(VD, exist_ok=True)
FF = S + "/tools/ffmpeg-7.0.2-amd64-static/ffmpeg"; FPC = 3; FPS = 16; K = 6
SCENES = ["sandcastle_wave_impact", "easy_stonehenge_sunset", "sand_filling_up_jar"]; SEEDS = [1234, 4321, 9999, 11, 22, 33, 44, 55, 66, 77]
ARMS = ["soft@6_a0.25", "soft@6_a0.5", "soft@6_a0.75", "once@6", "prog@6"]; JUDGE = {"alone", "soft@6_a0.25", "soft@6_a0.5", "soft@6_a0.75", "once@6", "prog@6"}
sys.path.insert(0, S + "/lingbot-world-v2"); from wan.modules.vae2_1 import Wan2_1_VAE
vae = Wan2_1_VAE(vae_pth=S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast/Wan2.1_VAE.pth", device="cuda:0")
@torch.no_grad()
def decode(lat):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16): im = vae.decode([lat.float().cuda()])[0].float().clamp(-1, 1)
    return ((im + 1) / 2 * 255).permute(1, 2, 3, 0).round().to(torch.uint8)   # (F, H, W, 3) uint8 on GPU
def _enc(fr, out, crf=23, preset="fast"):
    Fn, H, W, _ = fr.shape
    p = subprocess.Popen([FF, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", preset, "-threads", "4",
                          "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", out + ".tmp.mp4"], stdin=subprocess.PIPE)
    p.stdin.write(fr.tobytes()); p.stdin.close(); p.wait(); os.replace(out + ".tmp.mp4", out)
THREADS = []
def write_mp4(fr, out, **kw): t = threading.Thread(target=_enc, args=(fr.cpu().numpy() if torch.is_tensor(fr) else fr, out), kwargs=kw); t.start(); THREADS.append(t)
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
    l1, ss = l1_pairs(fr), ssim_pairs(fr); Bn = [bidx(c) for c in range(nch - 1)]; mask = np.ones(len(l1), bool); mask[Bn] = False
    return dict(l1_b=[float(l1[t]) for t in Bn], ss_b=[float(ss[t]) for t in Bn], l1_in=float(l1[mask].mean()), ss_in=float(ss[mask].mean()), l1_all=[float(v) for v in l1])
def lat_path(a, sc, sd): return "%s/%s/tau2_s0/K0_uniform4P_fx_s%d/lat.pt" % (TD, sc, sd) if a == "T" else "%s/%s/%s_s%d/lat.pt" % (R, sc, a, sd)
def load(a, sc, sd): p = lat_path(a, sc, sd); return torch.load(p, map_location="cpu") if os.path.exists(p) else None
res = json.load(open(OUTJ)) if os.path.exists(OUTJ) else {}; t00 = time.time()
# ---------------- (a) + (b): boundary stats of device views + judge mp4s
for sc in SCENES:
    for sd in SEEDS:
        key = "%s/s%d" % (sc, sd); r = res.setdefault(key, {}); la = load("alone", sc, sd); lT = load("T", sc, sd)
        if la is None or lT is None: print("skip", key, "no alone/T", flush=True); continue
        nch = la.shape[1] // FPC; done = []
        for a in ["alone", "T"]:
            mp4 = "%s/%s_%s_s%d.mp4" % (JD, a, sc, sd)
            if a in r and (a != "alone" or os.path.exists(mp4)): continue
            fr = decode(la if a == "alone" else lT); r[a] = stats(fr, nch)
            if a == "alone" and not os.path.exists(mp4): write_mp4(fr, mp4)
            del fr; done.append(a)
        for a in ARMS:
            lo = load(a, sc, sd)
            if lo is None: print("skip", key, a, flush=True); continue
            mp4 = "%s/%s_%s_s%d.mp4" % (JD, a, sc, sd); dv = "dev:" + a
            if dv not in r or not os.path.exists(mp4):
                lat = torch.cat([la[:, :FPC * (K + 1)], lo[:, FPC * (K + 1):]], 1); fr = decode(lat); r[dv] = stats(fr, nch)
                if not os.path.exists(mp4): write_mp4(fr, mp4)
                del fr; done.append(dv)
            if a not in r: fr = decode(lo); r[a] = stats(fr, nch); del fr; done.append(a)   # un-spliced: blended context as stored + continuation (own boundary)
        if done:
            json.dump(res, open(OUTJ, "w")); torch.cuda.empty_cache()
            msg = "; L1@6 dev-view: " + " ".join("%s %.1f" % (a.replace("soft@6_", ""), r["dev:" + a]["l1_b"][K]) for a in ARMS if "dev:" + a in r) + " | alone %.1f T %.1f" % (r["alone"]["l1_b"][K], r["T"]["l1_b"][K])
            print("%s %s done, %.0f s total%s" % (key, done, time.time() - t00, msg), flush=True)
for t in THREADS: t.join()
THREADS.clear(); print("[BOUNDARY DONE] %.0f s" % (time.time() - t00), flush=True)
# ---------------- (c): side-by-side video per scene, seed 1234
try: FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15); FONT2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
except Exception: FONT = FONT2 = ImageFont.load_default()
TW, TH, LH, BAR = 416, 240, 44, 10; COLS = 4
def tile(frame, label, sub, f, nchunks, k, red):
    c = min(f // (FPC * 4), nchunks - 1) if f > 0 else 0; inj = red and c <= k
    im = Image.new("RGB", (TW, LH + TH + BAR + 4), (18, 18, 18)); fr = Image.fromarray(frame).resize((TW, TH), Image.BILINEAR); im.paste(fr, (0, LH))
    d = ImageDraw.Draw(im); d.text((6, 4), label, font=FONT, fill=(240, 240, 240)); d.text((6, 24), "chunk %d / %d   %s" % (c, nchunks - 1, sub), font=FONT2, fill=(255, 90, 90) if inj else (180, 180, 180))
    if inj: d.rectangle([0, LH, TW - 1, LH + TH - 1], outline=(255, 60, 60), width=4)
    w = TW / nchunks
    for i in range(nchunks):
        on = red and i <= k
        d.rectangle([i * w + 1, LH + TH + 2, (i + 1) * w - 1, LH + TH + BAR], fill=(220, 50, 50) if on else (70, 70, 70))
        if i == c: d.rectangle([i * w + 1, LH + TH + 2, (i + 1) * w - 1, LH + TH + BAR], outline=(255, 255, 255), width=2)
    return np.asarray(im)
def write_grid(tiles, out, nch):
    """tiles: list of (frames uint8 (F,H,W,3) numpy, label, sub, k, red) in row-major order, COLS per row"""
    Fn = min(len(t[0]) for t in tiles); rows = (len(tiles) + COLS - 1) // COLS; th = LH + TH + BAR + 4
    p = subprocess.Popen([FF, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (TW * COLS, th * rows), "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    for f in range(Fn):
        canvas = np.zeros((th * rows, TW * COLS, 3), np.uint8)
        for i, (frames, label, sub, k, red) in enumerate(tiles):
            rr, cc = divmod(i, COLS); canvas[rr * th:(rr + 1) * th, cc * TW:(cc + 1) * TW] = tile(frames[f], label, sub, f, nch, k, red)
        p.stdin.write(canvas.tobytes())
    p.stdin.close(); p.wait()
SD = 1234; VARMS = [("alone", "alpha = 0 (device alone, 14B 2-step)"), ("soft@6_a0.25", "alpha = 0.25"), ("soft@6_a0.5", "alpha = 0.5"), ("once@6", "alpha = 1 (hard re-anchor = once@6)")]
for sc in SCENES:
    out = "%s/soft_reanchor_%s_s%d.mp4" % (VD, sc, SD)
    if os.path.exists(out): print("exists", out); continue
    la = load("alone", sc, SD); L = {a: load(a, sc, SD) for a, _ in VARMS}
    if any(v is None for v in L.values()): print("skip video", sc, [a for a, v in L.items() if v is None]); continue
    nch = la.shape[1] // FPC; top = []; bot = []
    for a, lab in VARMS:
        lo = L[a]; dvl = torch.cat([la[:, :FPC * (K + 1)], lo[:, FPC * (K + 1):]], 1); fd = decode(dvl).cpu().numpy(); fk = decode(lo).cpu().numpy() if a != "alone" else fd
        top.append((fd, "DEVICE VIEW  " + lab, "shown: own chunks 0-6, then re-anchored continuation" if a != "alone" else "own rollout throughout", K, a != "alone"))
        bot.append((fk, "KV CONTENT  " + lab, "context 0-6 = own + alpha (cloud - own), decoded" if a != "alone" else "(same as above)", K, a != "alone"))
    write_grid(top + bot, out, nch); print("wrote", out, flush=True); torch.cuda.empty_cache()
print("[SOFT_DECODE_DONE]")
