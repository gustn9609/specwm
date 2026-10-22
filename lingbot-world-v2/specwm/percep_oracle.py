"""[10/14, proposal 5 ceiling test] Chunk-wise early exit with a PERCEPTUAL ORACLE verifier: at every chunk all 4 steps are run, the step-1/2/3
estimates and the final chunk are decoded (same VAE cache state, snapshot/restore) into 4-frame strips and scored by the VLM (absolute
quality 1-10, scripts/percep_abs.py prompt); the chunk is committed at the earliest step k whose score >= final score - margin (policy
oracle<margin>), or always at step 4 (policy teacher). The committed chunk becomes the history (the rollout follows the accepted chunks).
Output: <out>/<task>/<policy>_s<seed>/{video.mp4, loop.json (mean_steps, per-chunk scores)}; the clips are then judged with
scripts/vlm_clip_score.py paired with the teacher policy, as in facts §25. Decode for the final video is one full-clip VAE decode.
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.percep_oracle --out DIR --policy oracle0 [--tasks a,b] [--seeds 7]"""
import argparse, os, sys, json, time, base64, numpy as np, torch, cv2, imageio
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--policy", default="oracle0", help="oracle<margin> | teacher"); ap.add_argument("--seeds", default="7"); ap.add_argument("--shard", default="0/1")
ap.add_argument("--tasks", default=open(S + "/lingbot-world-v2/specwm_out/distill_heldout12.txt").read().strip()); ap.add_argument("--frames", type=int, default=161); ap.add_argument("--judge", default="http://127.0.0.1:30100/v1"); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
from openai import OpenAI
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"; dev = "cuda:0"
si, sn = (int(x) for x in a.shard.split("/")); tasks = a.tasks.split(",")[si::sn]; seeds = [int(s) for s in a.seeds.split(",")]; MARGIN = None if a.policy == "teacher" else float(a.policy[6:])
B = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True); FP.fp8_swap(B.target_model, True, keep=False)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 512; _dyn.config.accumulated_recompile_limit = 8192
for blk in B.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
B.pipe.text_encoder.model.to(dev); _te = B.pipe.text_encoder
class _TE:
    def __init__(s, te): s.te = te; s.model = te.model
    def __call__(s, texts, device): return s.te(texts, dev)
B.pipe.text_encoder = _TE(_te); sig = [float(s) for s in B.node_sigmas()]; VAE = B.pipe.vae; VM = VAE.model; client = OpenAI(api_key="local", base_url=a.judge)
Q = ("This image shows 4 frames (left to right, spanning 0.75 s) of an AI-generated video. Rate the VISUAL QUALITY of the frames (sharpness and detail, absence of artefacts such as blur, smearing, blocking, ghosting or grid noise, realism of textures, and consistency between the 4 frames). "
     "Answer with ONE integer 1-10 (10 = photographic, 1 = unusable). Only the number.")
def score(img):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    r = client.chat.completions.create(model="qwen3-vl-32b", messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()}}, {"type": "text", "text": Q}]}], max_tokens=6, temperature=0)
    t = r.choices[0].message.content.strip()
    try: return float(next(x for x in t.split() if x.replace(".", "").isdigit()))
    except StopIteration: return float("nan")
def gen(seed, c, k=0): return torch.Generator().manual_seed(seed * 1009 + c * 7 + k)
def vae_snapshot(): return [t.clone() if torch.is_tensor(t) else t for t in VM._feat_map]
def vae_restore(snap): VM._feat_map = [t.clone() if torch.is_tensor(t) else t for t in snap]
@torch.no_grad()
def decode(z):
    with torch.amp.autocast("cuda", dtype=VAE.dtype):
        sc = VAE.scale; zz = z.unsqueeze(0).to(VAE.dtype) / sc[1].view(1, -1, 1, 1, 1) + sc[0].view(1, -1, 1, 1, 1); x = VM.conv2(zz); outs = []
        for i in range(x.shape[2]):
            VM._conv_idx = [0]; o = VM.decoder(x[:, :, i:i + 1], feat_cache=VM._feat_map, feat_idx=VM._conv_idx).float().clamp_(-1, 1); outs.append(((o[0] + 1) * 127.5).to(torch.uint8).permute(1, 2, 3, 0))
    return torch.cat(outs, 0).cpu().numpy()
def strip(fr):
    idx = [0, min(4, len(fr) - 1), min(8, len(fr) - 1), len(fr) - 1]; return np.concatenate([fr[i] for i in idx], 1)
for task in tasks:
    for seed in seeds:
        od = "%s/%s/%s_s%d" % (a.out, task, a.policy, seed)
        if os.path.exists(od + "/loop.json"): continue
        T = ALL[task]; os.makedirs(od, exist_ok=True); t0 = time.time(); rows = []; chunks = []
        with torch.no_grad():
            cond = B.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=a.frames, max_area=480 * 832); B.reset_caches(); VM.clear_cache(); N = cond["num_chunks"]
            for c in range(N):
                z = B.chunk_noise(c, gen(seed, c)); g = gen(seed, c, 1); xs = []
                for i in range(4):
                    x = B.target_x0(z, i, c).float(); xs.append(x)
                    if i < 3: z = (1 - sig[i + 1]) * x + sig[i + 1] * B.randn_like(x, g)
                kstar = 4; scores = []
                if MARGIN is not None:
                    snap = vae_snapshot(); fr4 = decode(xs[3]); vae_restore(snap); s4 = score(strip(fr4)); scores = [None, None, None, s4]
                    for k in range(3):
                        fr = decode(xs[k]); vae_restore(snap); sk = score(strip(fr)); scores[k] = sk
                        if np.isfinite(sk) and np.isfinite(s4) and sk >= s4 - MARGIN: kstar = k + 1; break
                xc = xs[kstar - 1]; decode(xc)   # advance the VAE cache with the committed chunk (keeps the strip scoring consistent)
                B.commit(xc, c); chunks.append(xc); rows.append(dict(c=c, k=kstar, scores=scores)); print("[PO] %s c%02d k* %d scores %s" % (task, c, kstar, scores), flush=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16): im = B.pipe.vae.decode([torch.cat(chunks, 1).to(dev)])[0].float().clamp(-1, 1).cpu()
        v = ((im + 1) / 2).permute(1, 2, 3, 0).numpy(); imageio.mimwrite(od + "/video.mp4", list((v * 255).astype("uint8")), fps=16, quality=8)
        json.dump(dict(task=task, seed=seed, policy=a.policy, chunks=N, mean_steps=float(np.mean([r["k"] for r in rows])), rows=rows, total_s=round(time.time() - t0, 1)), open(od + "/loop.json", "w"), indent=1)
        print("[PO_DONE] %s %s mean steps %.2f (%.0f s)" % (task, a.policy, np.mean([r["k"] for r in rows]), time.time() - t0), flush=True); torch.cuda.empty_cache()
print("[ALL_DONE]", flush=True)
