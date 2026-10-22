"""[10/20] Side-by-side videos of the divergence-injection probe (specwm/inject_probe.py, facts §52): for one scene + seed, decode the saved
latents of T (14B 4-step), G alone, inject once @3 / @6 / @9 and inject every chunk, and tile them 3 x 2 at 16 fps with labels. While a tile is
showing frames whose context came from the 14B (chunks <= k for once@k; all chunks for every), its border is drawn in red and a bar under the
frame marks the 14B-context chunks. Output: paper/1020/videos/inject_<gen>_<scene>_s<seed>.mp4 (+ all scenes concatenated).
usage: CUDA_VISIBLE_DEVICES=5 envrun python /home/hk28965/specwm/paper/1020/make_inject_videos.py [GEN=1p3b_4step] [SEED=1234]"""
import os, sys, glob, subprocess, numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
S = "/home/hk28965/specwm"; GEN = sys.argv[1] if len(sys.argv) > 1 else "1p3b_4step"; SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1234
R = "%s/lingbot-world-v2/specwm_out/inject/%s" % (S, GEN); TD = S + "/lingbot-world-v2/specwm_out/paths2"; OUT = S + "/paper/1020/videos"; os.makedirs(OUT, exist_ok=True)
FF = S + "/tools/ffmpeg-7.0.2-amd64-static/ffmpeg"; FPC = 3; FPS = 16
GLABEL = {"1p3b_4step": "1.3B 4-step", "14b_2step": "14B 2-step"}[GEN]
TILES = [("T", "14B 4-step (the injected context)", None), ("alone", "%s alone" % GLABEL, -1), ("once@3", "%s, 14B latents injected @ chunk 3" % GLABEL, 3),
         ("once@6", "%s, injected @ chunk 6" % GLABEL, 6), ("once@9", "%s, injected @ chunk 9" % GLABEL, 9), ("every", "%s, 14B latents every chunk" % GLABEL, 99)]
SCENES = ["sand_filling_up_jar", "sandcastle_wave_impact", "easy_lake", "easy_stonehenge_sunset", "pouring_water_into_cup"]
sys.path.insert(0, S + "/lingbot-world-v2"); from wan.modules.vae2_1 import Wan2_1_VAE
vae = Wan2_1_VAE(vae_pth=S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast/Wan2.1_VAE.pth", device="cuda:0")
@torch.no_grad()
def decode(lat):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16): im = vae.decode([lat.float().cuda()])[0].float().clamp(-1, 1).cpu()
    return ((im + 1) / 2 * 255).permute(1, 2, 3, 0).numpy().astype("uint8")   # (F, H, W, 3)
try: FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15); FONT2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
except Exception: FONT = FONT2 = ImageFont.load_default()
TW, TH, LH, BAR = 416, 240, 44, 10; COLS = 3
def tile(frame, label, k, f, nchunks, C):
    """one tile: frame resized to TW x TH, red border while the shown chunk's context is 14B's, label strip above, chunk bar below"""
    c = min(f // (FPC * 4), nchunks - 1) if f > 0 else 0   # pixel frame f -> chunk (chunk 0 = 1 + 8 px frames, then 12 per chunk; approx. by 12)
    inj = (k is None) or (k == 99) or (k >= 0 and c <= k)
    im = Image.new("RGB", (TW, LH + TH + BAR + 4), (18, 18, 18)); fr = Image.fromarray(frame).resize((TW, TH), Image.BILINEAR); im.paste(fr, (0, LH))
    d = ImageDraw.Draw(im); d.text((6, 4), label, font=FONT, fill=(240, 240, 240))
    sub = "chunk %d / %d" % (c, nchunks - 1) + ("   context: 14B" if inj and k is not None else ("   context: own" if k is not None else ""))
    d.text((6, 24), sub, font=FONT2, fill=(255, 90, 90) if (inj and k is not None) else (180, 180, 180))
    if k is not None and inj: d.rectangle([0, LH, TW - 1, LH + TH - 1], outline=(255, 60, 60), width=4)
    w = TW / nchunks
    for i in range(nchunks):
        on = (k is None) or (k == 99) or (0 <= k and i <= k)
        d.rectangle([i * w + 1, LH + TH + 2, (i + 1) * w - 1, LH + TH + BAR], fill=(220, 50, 50) if (on and k is not None) else (70, 70, 70) if k is not None else (200, 200, 200))
        if i == c: d.rectangle([i * w + 1, LH + TH + 2, (i + 1) * w - 1, LH + TH + BAR], outline=(255, 255, 255), width=2)
    return np.asarray(im)
def write(frames_by_tile, labels, out):
    F = min(len(v) for v in frames_by_tile); rows = (len(frames_by_tile) + COLS - 1) // COLS; th = LH + TH + BAR + 4
    p = subprocess.Popen([FF, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (TW * COLS, th * rows), "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    for f in range(F):
        canvas = np.zeros((th * rows, TW * COLS, 3), np.uint8)
        for i, (frames, (label, k)) in enumerate(zip(frames_by_tile, labels)):
            r, c = divmod(i, COLS); canvas[r * th:(r + 1) * th, c * TW:(c + 1) * TW] = tile(frames[f], label, k, f, NCH, None)
        p.stdin.write(canvas.tobytes())
    p.stdin.close(); p.wait()
made = []
for sc in SCENES:
    paths = {"T": "%s/%s/tau2_s0/K0_uniform4P_fx_s%d/lat.pt" % (TD, sc, SEED)}
    for a, _, _ in TILES[1:]: paths[a] = "%s/%s/%s_s%d/lat.pt" % (R, sc, a, SEED)
    if not all(os.path.exists(p) for p in paths.values()): print("skip", sc, [a for a, p in paths.items() if not os.path.exists(p)]); continue
    lat = {a: torch.load(p, map_location="cpu") for a, p in paths.items()}; NCH = lat["T"].shape[1] // FPC
    frames = [decode(lat[a]) for a, _, _ in TILES]; out = "%s/inject_%s_%s_s%d.mp4" % (OUT, GEN, sc, SEED)
    write(frames, [(l, k) for _, l, k in TILES], out); made.append(out); print("wrote", out, frames[0].shape, flush=True)
if len(made) > 1:
    lst = OUT + "/.concat_%s_s%d.txt" % (GEN, SEED); open(lst, "w").write("".join("file '%s'\n" % m for m in made))
    subprocess.run([FF, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "%s/inject_%s_ALL5_s%d.mp4" % (OUT, GEN, SEED)], check=True); print("wrote ALL5")
