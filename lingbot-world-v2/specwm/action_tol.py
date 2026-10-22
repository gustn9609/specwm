"""[10/13, proposal A measurement (2)] Action tolerance of the Fast 14B: how far does a chunk move (token gap vs the chunk generated with the
TRUE action, same z0 / same noise / same committed history) when the chunk's camera action is perturbed? Perturbations act on the framewise
relative poses (already normalized, so no renormalization confound) of the chunk's 3 latent frames only:
  extrap  : constant velocity = the last relative transform of the previous chunk repeated (= the predictor of action-speculative branching)
  stop    : identity (no motion)      speed0.5 / speed2 : translation scaled      yaw10 / yaw30 : extra yaw of 10 / 30 deg spread over the chunk
  noise   : TRUE action, different z0 (the sample-to-sample floor of facts §21, 29 %)
Stock 14B 4-step, deployed stack, teacher-forced history (true-action rollout), chunks --chunks. Output <out>/<task>_s<seed>/stats.json.
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.action_tol --out DIR [--tasks a,b] [--seeds 7] [--chunks 2,4,6,8,10] [--shard i/n]"""
import argparse, os, sys, json, time, math, numpy as np, torch
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--seeds", default="7"); ap.add_argument("--chunks", default="2,4,6,8,10"); ap.add_argument("--shard", default="0/1")
ap.add_argument("--tasks", default=open(S + "/lingbot-world-v2/specwm_out/distill_heldout12.txt").read().strip()); ap.add_argument("--eps", type=float, default=0.24); ap.add_argument("--frames", type=int, default=161); ap.add_argument("--fine", type=int, default=0); ap.add_argument("--warp", type=int, default=0, help="10/14 proposal 1: warp reconciliation — the branch chunk (hypothesis action) is warped by the per-frame rotation difference (homography in latent space), re-noised to node J and finished with the true action; variants stop/yaw10/yaw30"); ap.add_argument("--late", type=int, default=0, help="10/14 proposal A: late branching — steps < j with the hypothesis action, steps >= j with the true action (4-step j = 1,2,3; 2-step j = 1); variants extrap/stop/yaw10/yaw30/transfwd"); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
from wan.utils.cam_utils import get_Ks_transformed, interpolate_camera_poses, compute_relative_poses, get_plucker_embeddings
from einops import rearrange
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
si, sn = (int(x) for x in a.shard.split("/")); tasks = a.tasks.split(",")[si::sn]; seeds = [int(s) for s in a.seeds.split(",")]; CH = [int(c) for c in a.chunks.split(",")]; dev = "cuda:0"
B = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True); FP.fp8_swap(B.target_model, True, keep=False)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 512; _dyn.config.accumulated_recompile_limit = 8192
for blk in B.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
B.pipe.text_encoder.model.to(dev); _te = B.pipe.text_encoder
class _TE:
    def __init__(s, te): s.te = te; s.model = te.model
    def __call__(s, texts, device): return s.te(texts, dev)
B.pipe.text_encoder = _TE(_te); os.makedirs(a.out, exist_ok=True)
def patch(x):
    c, f, h, w = x.shape; return x.view(c, f, 1, h // 2, 2, w // 2, 2).permute(1, 3, 5, 2, 4, 6, 0).reshape(f * (h // 2) * (w // 2), 64)
def gen(seed, c, k=0): return torch.Generator().manual_seed(seed * 1009 + c * 7 + k)
def rel_poses(T, frame_num, lat_f, h, w):   # exactly prepare()'s camera path, returning the normalized framewise relative poses [lat_f, 4, 4] and Ks
    c2ws = np.load(os.path.join(T["action"], "poses.npy")); len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1; frame_num = min(((frame_num - 1) // 4) * 4 + 1, len_c2ws); c2ws = c2ws[:frame_num]
    Ks = torch.from_numpy(np.load(os.path.join(T["action"], "intrinsics.npy"))).float(); Ks = get_Ks_transformed(Ks, height_org=480, width_org=832, height_resize=h, width_resize=w, height_final=h, width_final=w)[0]
    L = len(c2ws); Lp = int((L - 1) // 4) + 1; Lp = int(Lp - (Lp % 3)); assert Lp == lat_f, (Lp, lat_f)
    c2 = interpolate_camera_poses(src_indices=np.linspace(0, L - 1, L), src_rot_mat=c2ws[:, :3, :3], src_trans_vec=c2ws[:, :3, 3], tgt_indices=np.linspace(0, L - 1, Lp))
    rel = compute_relative_poses(c2, framewise=True).to(dev); Ks = Ks.repeat(len(rel), 1).to(dev); return rel, Ks
def emb_chunk(rel3, Ks3, h, w, lat_h, lat_w):   # plucker embedding of 3 latent frames -> [1, C, 3, lat_h, lat_w] (prepare()'s rearranges)
    e = get_plucker_embeddings(rel3, Ks3, h, w); e = rearrange(e, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)', c1=int(h // lat_h), c2=int(w // lat_w))[None, ...]
    return rearrange(e, 'b (f h w) c -> b c f h w', f=3, h=lat_h, w=lat_w).to(B.param_dtype)
def yaw(theta):
    c, s = math.cos(theta), math.sin(theta); R = torch.eye(4, device=dev); R[0, 0] = c; R[0, 2] = s; R[2, 0] = -s; R[2, 2] = c; return R
def variants(rel, c):
    r = rel[3 * c:3 * c + 3].clone(); V = {}
    V["extrap"] = rel[3 * c - 1:3 * c].expand(3, 4, 4).clone()
    V["stop"] = torch.eye(4, device=dev).expand(3, 4, 4).clone()
    for s in (0.5, 2.0): x = r.clone(); x[:, :3, 3] *= s; V["speed%g" % s] = x
    for deg in (10, 30): x = r.clone(); V["yaw%d" % deg] = torch.stack([x[i] @ yaw(math.radians(deg / 3)) for i in range(3)])
    if a.warp: return {"stop": V["stop"], "yaw10": V["yaw10"], "yaw30": V["yaw30"]}
    if a.late:
        V = {"extrap": V["extrap"], "stop": V["stop"], "yaw10": V["yaw10"], "yaw30": V["yaw30"]}
        x = r.clone(); x[:, :3, 3] = 0; x[:, 2, 3] = -1.0; V["transfwd"] = x; return V
    if a.fine:   # 10/13 evening: finer rotation steps (tolerance calibration for action speculation) + translation on/off with the rotation kept
        for deg in (2, 5): x = r.clone(); V["yaw%d" % deg] = torch.stack([x[i] @ yaw(math.radians(deg / 3)) for i in range(3)])
        x = r.clone(); x[:, :3, 3] = 0; V["trans0"] = x
        x = r.clone(); x[:, :3, 3] = 0; x[:, 2, 3] = -1.0; V["transfwd"] = x   # forward at full normalized speed (StEvo 'straight' = z -1 per latent frame)
    return V
def run4(z, c, g):
    sg = [float(s) for s in B.node_sigmas()]
    for i in range(len(sg)):
        x = B.target_x0(z, i, c).float()
        if i < len(sg) - 1: z = (1 - sg[i + 1]) * x + sg[i + 1] * B.randn_like(x, g)
    return x
def run_switch(z, c, g, E_hyp, E_true, j, K=4):
    """steps i < j see the hypothesis embedding, steps i >= j the true one (same noise path); K = number of nodes used (4 or 2)"""
    sg = [float(s) for s in B.node_sigmas()][:K]
    for i in range(K):
        B._emb_chunks = E_hyp if i < j else E_true
        x = B.target_x0(z, i, c).float()
        if i < K - 1: z = (1 - sg[i + 1]) * x + sg[i + 1] * B.randn_like(x, g)
    return x
def warp_latent(xH, R_h, R_t, Klat, sign=1):
    """xH [16, 3, H, W] branch latent; R_h / R_t [3,3,3] cumulative rotations (chunk-start frame) of the hypothesis / true camera per latent frame;
    returns the branch re-rendered from the true camera (pure-rotation homography, bilinear, holes keep the branch value) and the valid mask"""
    import torch.nn.functional as Fn
    C, Fr, H, W = xH.shape; Ki = torch.inverse(Klat); out = xH.clone(); masks = []
    ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32) + 0.5, torch.arange(W, device=dev, dtype=torch.float32) + 0.5, indexing="ij")
    p = torch.stack([xs, ys, torch.ones_like(xs)], -1).reshape(-1, 3)   # true-frame pixel centres
    for j in range(Fr):
        R = R_h[j].T @ R_t[j] if sign == 1 else R_t[j].T @ R_h[j]       # true-camera ray -> hypothesis-camera ray
        q = (Klat @ (R @ (Ki @ p.T))).T; q = q[:, :2] / q[:, 2:3].clamp_min(1e-6)
        g = torch.stack([q[:, 0] / W * 2 - 1, q[:, 1] / H * 2 - 1], -1).view(1, H, W, 2)
        w = Fn.grid_sample(xH[:, j][None], g, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
        valid = Fn.grid_sample(torch.ones(1, 1, H, W, device=dev), g, mode="bilinear", padding_mode="zeros", align_corners=False)[0, 0] > 0.99
        out[:, j] = torch.where(valid[None], w, xH[:, j]); masks.append(float(valid.float().mean()))
    return out, float(np.mean(masks))
def cum_rot(rel3):
    R = torch.eye(3, device=dev); out = []
    for j in range(rel3.shape[0]): R = R @ rel3[j, :3, :3]; out.append(R.clone())
    return torch.stack(out)
def finish(xW, c, g, E_true, J, K=4):
    """re-noise the warped latent to node J with the SAME re-noise draws the true run used, then run nodes J..K-1 with the true action"""
    sg = [float(s) for s in B.node_sigmas()][:K]; z = None
    for i in range(1, K):
        eps = B.randn_like(xW, g)   # consume the generator exactly as run4 / run_switch do
        if i == J: z = (1 - sg[i]) * xW + sg[i] * eps
    B._emb_chunks = E_true; x = None
    for i in range(J, K):
        x = B.target_x0(z, i, c).float()
        if i < K - 1: z = (1 - sg[i + 1]) * x + sg[i + 1] * B.randn_like(x, g)
    return x
def stat(x, tT, sg): gp = (patch(x) - tT).norm(dim=1) / sg; return dict(mean=round(float(gp.mean()), 4), frac=round(float((gp <= a.eps).float().mean()), 4))
for task in tasks:
    for seed in seeds:
        od = "%s/%s_s%d" % (a.out, task, seed)
        if os.path.exists(od + "/stats.json"): continue
        T = ALL[task]; t0 = time.time(); os.makedirs(od, exist_ok=True)
        try:
            with torch.no_grad():
                cond = B.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=a.frames, max_area=480 * 832); B.reset_caches(); N = cond["num_chunks"]
                lat_h, lat_w = B._chunk_shape[2], B._chunk_shape[3]; h, w = lat_h * 8, lat_w * 8; rel, Ks = rel_poses(T, a.frames, N * 3, h, w)
                e0 = emb_chunk(rel[0:3], Ks[0:3], h, w, lat_h, lat_w); chk = float((e0.float() - B._emb_chunks[0].float()).abs().max()); assert chk < 1e-3, ("emb mismatch", chk)
                E = list(B._emb_chunks); R = dict(task=task, seed=seed, eps=a.eps, chunks=[])
                for c in range(N):
                    z0 = B.chunk_noise(c, gen(seed, c)); xT = run4(z0, c, gen(seed, c))
                    if c in CH:
                        tT = patch(xT); sg = float(tT.norm(dim=1).median()); r = dict(c=c, sigma=round(sg, 4), motion=round(float(rel[3 * c:3 * c + 3, :3, 3].norm(dim=1).mean()), 4))
                        r["noise"] = stat(run4(B.chunk_noise(c, gen(seed, c, 2)), c, gen(seed, c, 3)), tT, sg)
                        if a.late or a.warp:   # 2-step references (nodes 0,1 of the 4-node schedule = the realtime / quality modes)
                            xT2 = run_switch(z0, c, gen(seed, c), E, E, 0, K=2); tT2 = patch(xT2); sg2 = float(tT2.norm(dim=1).median())
                            r["noise_2s"] = stat(run_switch(B.chunk_noise(c, gen(seed, c, 2)), c, gen(seed, c, 3), E, E, 0, K=2), tT2, sg2)
                        for nm, rv in variants(rel, c).items():
                            Ev = E[:c] + [emb_chunk(rv, Ks[3 * c:3 * c + 3], h, w, lat_h, lat_w)] + E[c + 1:]; B._emb_chunks = Ev
                            r[nm] = stat(run4(z0, c, gen(seed, c)), tT, sg)
                            if a.warp:
                                Klat = torch.tensor([[float(Ks[0, 0]) / 8, 0, float(Ks[0, 2]) / 8], [0, float(Ks[0, 1]) / 8, float(Ks[0, 3]) / 8], [0, 0, 1.0]], device=dev)
                                xH = run4(z0, c, gen(seed, c)); Rh = cum_rot(rv); Rt = cum_rot(rel[3 * c:3 * c + 3])
                                for sgn in (1, -1):
                                    xW, vis = warp_latent(xH, Rh, Rt, Klat, sgn); tag = "%s_w%s" % (nm, "p" if sgn == 1 else "n"); r[tag] = dict(stat(xW, tT, sg), visible=round(vis, 3))
                                    for J in (3, 2): r["%s_J%d" % (tag, J)] = stat(finish(xW, c, gen(seed, c), E, J, K=4), tT, sg)
                                    xH2 = run_switch(z0, c, gen(seed, c), Ev, Ev, 0, K=2); xW2, _ = warp_latent(xH2, Rh, Rt, Klat, sgn); r["%s_2s" % tag] = stat(finish(xW2, c, gen(seed, c), E, 1, K=2), tT2, sg2)
                                r["%s_2s" % nm] = stat(run_switch(z0, c, gen(seed, c), Ev, Ev, 0, K=2), tT2, sg2); r["%s_late3" % nm] = stat(run_switch(z0, c, gen(seed, c), Ev, E, 3, K=4), tT, sg)
                            if a.late:
                                for j in (1, 2, 3): r["%s_late%d" % (nm, j)] = stat(run_switch(z0, c, gen(seed, c), Ev, E, j, K=4), tT, sg)
                                r["%s_2s" % nm] = stat(run_switch(z0, c, gen(seed, c), Ev, Ev, 0, K=2), tT2, sg2); r["%s_2s_late1" % nm] = stat(run_switch(z0, c, gen(seed, c), Ev, E, 1, K=2), tT2, sg2)
                        B._emb_chunks = E; R["chunks"].append(r)
                        print("[AT] %s c%02d motion %.3f | noise %.0f%% | %s" % (task, c, r["motion"], 100 * r["noise"]["frac"], " ".join("%s %.0f%%" % (k, 100 * r[k]["frac"]) for k in r if isinstance(r[k], dict) and k != "noise")), flush=True)
                    B.commit(xT, c)
                R["total_s"] = round(time.time() - t0, 1); json.dump(R, open(od + "/stats.json", "w"))
            print("[AT] %s s%d %.0fs" % (task, seed, time.time() - t0), flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); print("[AT_FAIL] %s s%d %r" % (task, seed, e), flush=True)
        finally:
            torch.cuda.empty_cache()
print("[AT_DONE]", flush=True)
