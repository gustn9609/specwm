"""Is a VLM (Qwen3-VL-32B via :30100) a scene-general drift verifier? Human-labelled clips (exp/drift_lat, 'severe' = teleport / colour or
texture collapse / freeze by rater majority): all non-jar/castle scenes (32 scenes) + a jar/castle sample. Latents are decoded with the VAE.
Two protocols: clip = 8 frames of the whole clip, one score; chunk = online use: frame 0 (the given scene) + 2 frames of chunk c, score per
chunk, clip score = max over chunks, latency per call recorded. AUROC vs the human 'severe' label, by scene group.
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.vlm_drift_judge --out DIR [--n-other 320 --n-jc 80]"""
import argparse, os, sys, json, glob, io, base64, time, random, numpy as np, torch
from PIL import Image
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--n-other", type=int, default=320); ap.add_argument("--n-jc", type=int, default=80); ap.add_argument("--chunks", default="2,4,6,8,10,12"); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from wan.modules.vae2_1 import Wan2_1_VAE
from openai import OpenAI
from sklearn.metrics import roc_auc_score
vae = Wan2_1_VAE(vae_pth=S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast/Wan2.1_VAE.pth", device="cuda"); client = OpenAI(api_key="local", base_url="http://127.0.0.1:30100/v1"); os.makedirs(a.out, exist_ok=True)
F = [f for f in sorted(glob.glob(S + "/exp/drift_lat/*.pt")) if "/cam_" not in f]; random.seed(0)
meta = [(f, torch.load(f, map_location="cpu", mmap=True)) for f in F]; JC = ("sand_filling_up_jar", "sandcastle_wave_impact")
other = [m for m in meta if m[1]["task"] not in JC]; jc = [m for m in meta if m[1]["task"] in JC]; random.shuffle(other); random.shuffle(jc)
sel = other[:a.n_other] + jc[:a.n_jc]; print("[vlm] %d clips (%d other scenes, %d jar/castle)" % (len(sel), min(len(other), a.n_other), min(len(jc), a.n_jc)), flush=True)
def b64(im, W=448):
    im = Image.fromarray(im); im = im.resize((W, int(im.height * W / im.width)), Image.BICUBIC); b = io.BytesIO(); im.save(b, format="JPEG", quality=85); return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
def ask(images, prompt, max_tokens=8):
    t = time.time(); r = client.chat.completions.create(model="qwen3-vl-32b", messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": u}} for u in images] + [{"type": "text", "text": prompt}]}], max_tokens=max_tokens, temperature=0)
    txt = r.choices[0].message.content.strip(); dt = time.time() - t
    try: return float(next(tok for tok in txt.replace("/", " ").split() if tok.replace(".", "").isdigit())), dt, txt
    except StopIteration: return 50.0, dt, txt
CLIP_P = ("These 8 frames are evenly spaced from one 10-second AI-generated video that should keep showing the scene of the first frame while the camera moves. "
          "Rate how severely the video DRIFTS away from the first frame's scene: teleporting to a different place, the main objects vanishing or morphing into other things, colours or textures collapsing, or the picture freezing. "
          "Answer with ONE integer 0-100 (0 = same scene throughout, 100 = completely different scene / collapsed). Only the number.")
CHUNK_P = ("Image 1 is the starting scene of an AI-generated video. Images 2 and 3 are two frames from a later moment of the same video (the camera may have moved). "
           "Rate how severely the later frames have DRIFTED from the starting scene: a different place, main objects vanished or turned into other things, colours/textures collapsed, or frozen. "
           "Answer with ONE integer 0-100 (0 = clearly the same scene, 100 = a different scene / collapsed). Only the number.")
recs = []; CH = [int(c) for c in a.chunks.split(",")]
for i, (f, d) in enumerate(sel):
    with torch.no_grad(): v = vae.decode([d["lat"].float().cuda()])[0].clamp(-1, 1)
    fr = ((v + 1) * 127.5).permute(1, 2, 3, 0).byte().cpu().numpy(); T = len(fr)
    idx = [int(k * (T - 1) / 7) for k in range(8)]; s_clip, t_clip, raw = ask([b64(fr[k]) for k in idx], CLIP_P)
    ch = {}; lat = []
    for c in CH:
        f0, f1 = min(12 * c - 8, T - 1), min(12 * c - 2, T - 1); s, dt, _ = ask([b64(fr[0]), b64(fr[f0]), b64(fr[f1])], CHUNK_P); ch[c] = s; lat.append(dt)
    recs.append(dict(task=d["task"], arm=d["arm"], seed=int(d["seed"]), severe=bool(d["severe"]), drift=bool(d["drift"]), group="jc" if d["task"] in JC else "other", clip=s_clip, clip_s=t_clip, chunk=ch, chunk_max=max(ch.values()), chunk_s=float(np.mean(lat)), raw=raw))
    if i % 20 == 0: print("[vlm] %d/%d %s %s severe=%s clip %.0f chunkmax %.0f (%.2f s/call)" % (i, len(sel), d["task"], d["arm"], d["severe"], s_clip, max(ch.values()), np.mean(lat)), flush=True); json.dump(recs, open(a.out + "/records.json", "w"))
json.dump(recs, open(a.out + "/records.json", "w"))
print("| scene group | n | severe rate | AUROC clip-level (8 frames) ↑ | AUROC max over chunk scores ↑ | AUROC vs any-drift (chunk max) | s per chunk call | s per clip call |"); print("|---|---|---|---|---|---|---|---|")
for g, lab in (("other", "32 scenes never seen by the latent critic"), ("jc", "jar / castle"), ("all", "all")):
    R = [r for r in recs if g == "all" or r["group"] == g]; y = [r["severe"] for r in R]; yd = [r["drift"] for r in R]
    au = lambda yy, s: roc_auc_score(yy, s) if 0 < sum(yy) < len(yy) else float("nan")
    print("| %s | %d | %.0f %% | %.2f | %.2f | %.2f | %.2f | %.2f |" % (lab, len(R), 100 * np.mean(y), au(y, [r["clip"] for r in R]), au(y, [r["chunk_max"] for r in R]), au(yd, [r["chunk_max"] for r in R]), np.mean([r["chunk_s"] for r in R]), np.mean([r["clip_s"] for r in R])))
print("[VLM_DONE]", flush=True)
