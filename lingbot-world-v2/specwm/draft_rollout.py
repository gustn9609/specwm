"""RL rollouts for the 1.3B causal-fast draft (optionally with a merged LoRA): for each (task, seed) generate the 13-chunk action-conditioned clip,
then score it: drift = drift-critic LSE over chunk logits (latent, human-label critic; higher = more drift), cam_err = mean VGGT relative-rotation error
vs the commanded poses (deg), reward = -drift/4 - cam_err/30. Saves <out>/<task>__s<seed>.pt {lat fp16 [16,39,h,w], drift, cam_err, reward, chunk logits}
(+ video.mp4 for the first --save-video clips). usage: CUDA_VISIBLE_DEVICES=g python -m specwm.draft_rollout --tasks @f --seeds 1,2,3,4 --out DIR [--lora L.pt]"""
import argparse, os, sys, json, time, math, numpy as np, torch, torch.nn.functional as F, imageio
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--tasks", required=True); ap.add_argument("--seeds", required=True); ap.add_argument("--out", required=True); ap.add_argument("--lora", default="")
ap.add_argument("--critic", default=S + "/lingbot-world-v2/specwm_out/drift/noprev_noplanrc/critic.pt"); ap.add_argument("--save-video", type=int, default=4); ap.add_argument("--lam-cam", type=float, default=1.0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from specwm.binding_v2 import V2Binding; from specwm.critic_model import CriticScorer
FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"; dev = torch.device("cuda:0")
D = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), cfg_name="i2v-1.3B", assets_dir=FAST)
if a.lora:
    st = torch.load(a.lora); i = 0
    with torch.no_grad():
        for blk in D.target_model.blocks:
            for m in [getattr(blk.self_attn, n) for n in ("q", "k", "v", "o")] + ([blk.ffn[0], blk.ffn[2]] if st.get("ffn", 1) else []):
                m.weight.add_((st["B"][i].float() @ st["A"][i].float()).to(m.weight.device, m.weight.dtype)); i += 1
    print("[rollout] merged LoRA %s (%d deltas, step %s)" % (a.lora, i, st.get("step")), flush=True)
crit = CriticScorer(a.critic, device="cuda:0"); temp = torch.load(a.critic, map_location="cpu").get("temp", 2.0)
sys.path.insert(0, S + "/b1/vggt")
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
vggt = VGGT.from_pretrained(S + "/hf/VGGT-1B").to(dev).eval()
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}
tl = open(a.tasks[1:]).read().split() if a.tasks.startswith("@") else a.tasks.split(","); seeds = [int(x) for x in a.seeds.split(",")]
def rot_angle(R): return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
@torch.no_grad()
def cam_error(frames, task):   # frames uint8 [T,H,W,3]
    idx = list(range(0, len(frames), 4))[:39]; V = torch.from_numpy(frames[idx]).float().div(255).permute(0, 3, 1, 2).to(dev); H, W = V.shape[2:]
    nh = int(round(H * 518 / W / 14)) * 14; img = F.interpolate(V, size=(nh, 518), mode="bilinear", align_corners=False, antialias=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16): pred = vggt(img[None])
    E, _ = pose_encoding_to_extri_intri(pred["pose_enc"], img.shape[-2:]); R = E[0, :, :, :3].float().cpu().numpy()
    P = np.load(os.path.join(ALL[task]["action"], "poses.npy")); P = P.reshape(len(P), 4, 4) if P.ndim == 2 else P; Rc = [P[min(k, len(P) - 1)][:3, :3] for k in idx]
    return float(np.mean([rot_angle((R[k] @ R[0].T) @ (Rc[k].T @ Rc[0]).T) for k in range(len(idx))]))
nvid = 0
for seed in seeds:
    for task in tl:
        out = "%s/%s__s%d.pt" % (a.out, task, seed)
        if os.path.exists(out): continue
        T = ALL[task]; t0 = time.time()
        cond = D.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); N = cond["num_chunks"]; sig = [float(s) for s in D.node_sigmas()]; D.reset_caches()
        g = torch.Generator().manual_seed(seed); chunks = []
        with torch.no_grad():
            for ci in range(N):
                z = D.chunk_noise(ci, g); x0 = None
                for i in range(len(sig)):
                    x0 = D.target_x0(z, i, ci)
                    if i < len(sig) - 1: s = sig[i + 1]; z = (1 - s) * x0 + s * D.randn_like(x0, g)
                D.commit(x0, ci); chunks.append(x0.float())
            lat = torch.cat(chunks, 1); logits = [crit(lat[:, 3 * k:3 * k + 3], None, D._emb_chunks[k].squeeze(0))[0] for k in range(N)]
            L = torch.tensor(logits); drift = float(temp * torch.logsumexp(L / temp, 0) - temp * math.log(len(L)))
            im = D.pipe.vae.decode([lat.to(dev)])[0].float(); fr = ((im.clamp(-1, 1) + 1) * 127.5).permute(1, 2, 3, 0).byte().cpu().numpy()
            ce = cam_error(fr[:153], task)
        reward = -drift / 4.0 - a.lam_cam * ce / 30.0
        torch.save(dict(task=task, seed=seed, lat=lat.half().cpu(), drift=drift, cam_err=ce, reward=reward, logits=[round(x, 3) for x in logits]), out)
        if nvid < a.save_video: imageio.mimwrite(out[:-3] + ".mp4", list(fr), fps=16, quality=8); nvid += 1
        print("[rollout] %s s%d drift %+.2f cam_err %.1f reward %+.3f (%.0fs)" % (task[:24], seed, drift, ce, reward, time.time() - t0), flush=True)
print("[ROLLOUT_DONE]", flush=True)
