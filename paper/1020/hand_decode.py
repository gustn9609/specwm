"""[10/20] decode the HANDOFF arms (specwm_out/inject/handoff/<scene>/hand@{3m1,3m3,6m2}_s<seed>/lat.pt) to the judge dir (crf 23, 16 fps) for the Qwen3-VL judge."""
import os, sys, subprocess, threading, torch
S = "/home/hk28965/specwm"; H = S + "/lingbot-world-v2/specwm_out/inject/handoff"; JD = S + "/lingbot-world-v2/specwm_out/inject/judge"; FF = S + "/tools/ffmpeg-7.0.2-amd64-static/ffmpeg"
sys.path.insert(0, S + "/lingbot-world-v2"); from wan.modules.vae2_1 import Wan2_1_VAE
vae = Wan2_1_VAE(vae_pth=S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast/Wan2.1_VAE.pth", device="cuda:0")
@torch.no_grad()
def decode(lat):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16): im = vae.decode([lat.float().cuda()])[0].float().clamp(-1, 1)
    return ((im + 1) / 2 * 255).permute(1, 2, 3, 0).round().to(torch.uint8).cpu().numpy()
def enc(fr, out):
    Fn, Hh, W, _ = fr.shape
    p = subprocess.Popen([FF, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, Hh), "-r", "16", "-i", "-", "-c:v", "libx264", "-preset", "fast", "-threads", "4", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out + ".tmp.mp4"], stdin=subprocess.PIPE)
    p.stdin.write(fr.tobytes()); p.stdin.close(); p.wait(); os.replace(out + ".tmp.mp4", out)
TH = []
for sc in ["sandcastle_wave_impact", "easy_stonehenge_sunset"]:
    for a in ["hand@3m1", "hand@3m3", "hand@6m2"]:
        for sd in [11, 22, 33, 44, 55, 66, 77, 1234, 4321, 9999]:
            p = "%s/%s/%s_s%d/lat.pt" % (H, sc, a, sd); out = "%s/%s_%s_s%d.mp4" % (JD, a, sc, sd)
            if not os.path.exists(p) or os.path.exists(out): continue
            lat = torch.load(p, map_location="cpu"); print(sc, a, sd, tuple(lat.shape), flush=True); fr = decode(lat)
            t = threading.Thread(target=enc, args=(fr, out)); t.start(); TH.append(t)
for t in TH: t.join()
print("[HAND_DONE]")
