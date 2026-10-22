"""Shadow data for the latent verifier: draft = LingBot v2 1.3B causal-fast, target = LingBot v2 14B causal-fast (both action-conditioned AR,
same latent space, same chunking). For EVERY chunk both models generate the chunk from the SAME context and the SAME noise draws; one of the two is
committed to both KV caches (coin with P(target) = --p-target), so the stored pairs are (draft chunk, target chunk | identical context).
Per chunk we also store the ③ one-step errors of both models on both candidates at node --e3-node, and the camera (Plucker) embedding of the chunk.
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.spec_ar_shadow --task sand_filling_up_jar --seed 1234 --out DIR [--p-target 0.5] [--e3-node 1]"""
import argparse, os, sys, json, time, numpy as np, torch, imageio
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--task", default=""); ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--out", required=True, help="output dir (single task) or ROOT dir (with --tasks: ROOT/<task>_s<seed>/)")
ap.add_argument("--tasks", default="", help="comma list or @file of task ids: models are loaded once and all (task, seed) pairs are processed in this process"); ap.add_argument("--seeds", default="1234")
ap.add_argument("--frames", type=int, default=161); ap.add_argument("--max-area", type=int, default=480 * 832); ap.add_argument("--p-target", type=float, default=0.5)
ap.add_argument("--e3-node", type=int, default=1, help="node index at which the one-step (③) errors are measured; -1 = skip"); ap.add_argument("--no-video", action="store_true")
a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from specwm.binding_v2 import V2Binding
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}
dev = torch.device("cuda:0"); FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
D = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), cfg_name="i2v-1.3B", assets_dir=FAST)
Tg = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
def gen_chunk(B, ci, g, sig):
    """4-node sampling of chunk ci by binding B (draft or target) from generator g; the FIRST draw is the chunk noise, then one re-noise per node."""
    z = B.chunk_noise(ci, g); x0 = None
    for i in range(len(sig)):
        x0 = B.target_x0(z, i, ci)
        if i < len(sig) - 1: s = sig[i + 1]; z = (1 - s) * x0 + s * B.randn_like(x0, g)
    return x0.float()
@torch.no_grad()
def e3(x, ci, eps, sig):
    """one-step errors of both models on candidate chunk x at node a.e3_node: noise to sigma s, predict x0, squared error per latent frame; also the flow difference."""
    i = a.e3_node; s = sig[i]; z = (1 - s) * x + s * eps
    xd = D.target_x0(z, i, ci); xt = Tg.target_x0(z, i, ci)
    ed = ((xd - x) ** 2).mean(dim=(0, 2, 3)); et = ((xt - x) ** 2).mean(dim=(0, 2, 3)); fd = (((xd - xt) / s) ** 2).mean(dim=(0, 2, 3))
    return dict(e_draft=ed.cpu().tolist(), e_target=et.cpu().tolist(), flow_diff=fd.cpu().tolist())
def run_clip(task, seed, out):
    os.makedirs(out, exist_ok=True); T = ALL[task]; img, act, prompt = T["image"], T["action"], T.get("camera_WM", "")
    cond = D.prepare(img, prompt, act, frame_num=a.frames, max_area=a.max_area); N = cond["num_chunks"]; sig = [float(s) for s in D.node_sigmas()]
    Tg.prepare(img, prompt, act, frame_num=a.frames, max_area=a.max_area); D.reset_caches(); Tg.reset_caches()
    print("[shadow] %s s%d: %d chunks, chunk %s, sigmas %s, prompt %r" % (task, seed, N, tuple(D._chunk_shape), ["%.3f" % s for s in sig], prompt[:60]), flush=True)
    g = torch.Generator().manual_seed(seed); coin = torch.Generator().manual_seed(seed + 7919); ge = torch.Generator().manual_seed(seed + 104729)
    rows = []; committed = []; who = ""; t0 = time.time(); td = tt = te = 0.0; prev = None
    with torch.no_grad():
        for ci in range(N):
            st = g.get_state()
            t1 = time.time(); x_d = gen_chunk(D, ci, g, sig); td += time.time() - t1; st_after = g.get_state()
            g.set_state(st); t1 = time.time(); x_t = gen_chunk(Tg, ci, g, sig); tt += time.time() - t1; g.set_state(st_after)
            feat = {}
            if a.e3_node >= 0:
                t1 = time.time(); eps = D.randn_like(x_d, ge); feat = dict(draft=e3(x_d, ci, eps, sig), target=e3(x_t, ci, eps, sig)); te += time.time() - t1
            use_t = bool(torch.rand((), generator=coin) < a.p_target); x_c = x_t if use_t else x_d; who += "T" if use_t else "D"
            D.commit(x_c, ci); Tg.commit(x_c, ci)
            rows.append(dict(chunk=ci, committed="target" if use_t else "draft", x_draft=x_d.half().cpu(), x_target=x_t.half().cpu(), x_prev=None,
                             cam=torch.nn.functional.adaptive_avg_pool2d(D._emb_chunks[ci].squeeze(0).float().permute(1, 0, 2, 3), ((D._chunk_shape[2] + 4) // 4, (D._chunk_shape[3] + 4) // 4)).permute(1, 0, 2, 3).half().cpu(),
                             e3=feat, dist_dt=float((x_d - x_t).norm() / (x_t.norm() + 1e-6))))
            prev = x_c; committed.append(x_c.cpu())
            print("[shadow] chunk %2d/%d %s  |d-t| %.3f  e3 draft: eD %.4f eT %.4f | target: eD %.4f eT %.4f  (%.0fs)" % (ci + 1, N, who[-1], rows[-1]["dist_dt"],
                  np.mean(feat["draft"]["e_draft"]) if feat else -1, np.mean(feat["draft"]["e_target"]) if feat else -1, np.mean(feat["target"]["e_draft"]) if feat else -1, np.mean(feat["target"]["e_target"]) if feat else -1, time.time() - t0), flush=True)
    gen_s = time.time() - t0
    torch.save(dict(task=task, seed=seed, prompt=prompt, sigmas=sig, e3_node=a.e3_node, chunk_shape=list(D._chunk_shape), plan=who, rows=rows), os.path.join(out, "shadow.pt"))
    if not a.no_video:
        lat = torch.cat(committed, dim=1)
        with torch.no_grad(): im = Tg.pipe.vae.decode([lat.to(dev)])[0].detach().float().cpu()
        v = ((im.clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy(); imageio.mimwrite(os.path.join(out, "video.mp4"), list((v * 255).astype("uint8")), fps=16, quality=8)
    json.dump(dict(task=task, seed=seed, plan=who, chunks=N, gen_s=round(gen_s, 1), draft_s=round(td, 1), target_s=round(tt, 1), e3_s=round(te, 1), p_target=a.p_target,
                   dist_dt=[round(r["dist_dt"], 4) for r in rows]), open(os.path.join(out, "shadow.json"), "w"), indent=1)
    print("[SHADOW_DONE] %s s%d plan=%s %.0fs (draft %.0f, target %.0f, e3 %.0f)" % (task, seed, who, gen_s, td, tt, te), flush=True)
if a.tasks:
    tl = open(a.tasks[1:]).read().split() if a.tasks.startswith("@") else a.tasks.split(",")
    for seed in [int(x) for x in a.seeds.split(",")]:
        for task in tl:
            out = "%s/%s_s%d" % (a.out, task, seed)
            if os.path.exists(out + "/shadow.pt"): print("[skip]", out, flush=True); continue
            try: run_clip(task, seed, out)
            except Exception as ex: print("[SHADOW_FAIL] %s s%d: %r" % (task, seed, ex), flush=True)
else: run_clip(a.task, a.seed, a.out)
