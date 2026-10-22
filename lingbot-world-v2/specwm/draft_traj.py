"""Re-generate labelled 1.3B draft rollouts (same loop/noise as draft_rollout.py) and save the whole 4-step trajectory: x0-hat at every draft node.
Out: <out>/<arm>__<task>__s<seed>.pt {task, seed, arm, traj fp16 [4,16,39,h,w] (node 0..3 x0-hat per chunk, node 3 = committed latent), match = rel. L2 vs the saved rollout latent}
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.draft_traj --jobs JOBS.json --shard i/n --out DIR   (JOBS = [[arm, task, seed, src_pt, lora or ""], ...])"""
import argparse, os, sys, json, time, torch
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--jobs", required=True); ap.add_argument("--shard", default="0/1"); ap.add_argument("--out", required=True); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); si, sn = (int(x) for x in a.shard.split("/"))
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from specwm.binding_v2 import V2Binding
FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
jobs = [j for j in json.load(open(a.jobs)) if not os.path.exists("%s/%s__%s__s%d.pt" % (a.out, j[0], j[1], j[2]))]
jobs = sorted(jobs, key=lambda j: (j[4], j[1], j[2]))[si::sn]; jobs = sorted(jobs, key=lambda j: j[4])   # group by LoRA
D = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), cfg_name="i2v-1.3B", assets_dir=FAST)
W0 = {}; cur = None
def set_lora(path):
    global cur
    if path == cur: return
    mods = [m for blk in D.target_model.blocks for m in [getattr(blk.self_attn, n) for n in ("q", "k", "v", "o")] + [blk.ffn[0], blk.ffn[2]]]
    with torch.no_grad():
        if not W0: W0.update({id(m): m.weight.detach().clone() for m in mods})
        for m in mods: m.weight.copy_(W0[id(m)])
        if path:
            st = torch.load(path); ms = [m for blk in D.target_model.blocks for m in [getattr(blk.self_attn, n) for n in ("q", "k", "v", "o")] + ([blk.ffn[0], blk.ffn[2]] if st.get("ffn", 1) else [])]
            for i, m in enumerate(ms): m.weight.add_((st["B"][i].float() @ st["A"][i].float()).to(m.weight.device, m.weight.dtype))
    cur = path; print("[traj] LoRA ->", path or "none", flush=True)
TT = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}
for arm, task, seed, src, lora in jobs:
    set_lora(lora); T = TT[task]; t0 = time.time()
    cond = D.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); N = cond["num_chunks"]; sig = [float(s) for s in D.node_sigmas()]; D.reset_caches()
    g = torch.Generator().manual_seed(seed); traj = []
    with torch.no_grad():
        for ci in range(N):
            z = D.chunk_noise(ci, g); xs = []
            for i in range(len(sig)):
                x0 = D.target_x0(z, i, ci); xs.append(x0.half().cpu())
                if i < len(sig) - 1: s = sig[i + 1]; z = (1 - s) * x0 + s * D.randn_like(x0, g)
            D.commit(x0, ci); traj.append(torch.stack(xs))
    traj = torch.cat(traj, 2)   # [4,16,39,h,w]
    ref = torch.load(src, map_location="cpu")["lat"].float(); match = float((traj[3].float() - ref).norm() / ref.norm())
    torch.save(dict(task=task, seed=seed, arm=arm, traj=traj, match=match, sig=sig), "%s/%s__%s__s%d.pt" % (a.out, arm, task, seed))
    print("[traj] %s %s s%d match %.4f (%.0fs)" % (arm, task[:20], seed, match, time.time() - t0), flush=True)
print("[TRAJ_DONE] shard", a.shard, flush=True)
