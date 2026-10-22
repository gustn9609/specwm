"""[10/13, proposals A + B] Frame-level action-speculative generation with ADAPTIVE chunk length, on the synthetic action streams of
specwm/act_stream.py. Same speculation as specwm/act_spec.py (K hypotheses of the not-yet-known action, branch = denoise + commit +
K/V slot snapshot + speculative decode; hit test in action space) but the generation unit is m latent frames with m chosen online:
  --adaptive 0 : fixed m = --cs (3) → the A arms, re-run under this code for paired comparison
  --adaptive 1 : after a miss on a 3-frame unit the chunk is regenerated as ONE frame (latency / 3) and the stream continues in 1-frame
                 units (speculated too) until --recover consecutive hits, then back to 3-frame units.
Needs the frame-granular K/V ring (fast_patch FLAGS ring_frame; identical layout to ring_kv for fixed 3-frame chunks) and the model
called per frame range (current_start = f0 * frame_seqlen). Steps: uniform only (--mode uniform:4 | uniform:2). Timing and outputs as in
act_spec.py (loop.json: chunk_start = unit start = the moment its action is consumed; frame_ready per decoded frame; rows per unit).
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.act_spec2 --out DIR --streams tau2_s0 --K 1 --adaptive 1 [--mode uniform:4] [--area 270336]"""
import argparse, os, sys, json, time, math, numpy as np, torch, imageio
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--streams", required=True); ap.add_argument("--K", default="0,1"); ap.add_argument("--seed", type=int, default=7); ap.add_argument("--seeds", default="", help="comma list: run every case for each seed (overrides --seed)")
ap.add_argument("--tasks", default=open(S + "/lingbot-world-v2/specwm_out/distill_heldout12.txt").read().strip()); ap.add_argument("--mode", default="uniform:4"); ap.add_argument("--area", type=int, default=480 * 832)
ap.add_argument("--cs", type=int, default=3); ap.add_argument("--adaptive", type=int, default=0); ap.add_argument("--recover", type=int, default=3, help="consecutive sub-unit hits before returning to 3-frame units"); ap.add_argument("--sub", type=int, default=1, help="unit length (frames) after a miss (B); 10/14: 2 = §23b no-loss sub-chunk")
ap.add_argument("--commit", default="separate", choices=["separate", "partial1"], help="partial1 = fused commit: keep the K/V of the last denoising forward, re-commit only the unit's last frame clean (the realtime mode of §13)")
ap.add_argument("--rt", type=int, default=0, help="1: real-time action arrival — the action of latent frame i is available at t0 + max(0, i - 2) * 0.25 s; the generator waits for it (speculation runs during the wait)")
ap.add_argument("--tol-rot", type=float, default=3.0, help="hit if the sum of per-frame rotation errors over a 3-frame unit <= this (deg); scaled by m/3"); ap.add_argument("--tol-trans", type=float, default=0.5, help="relative translation error (sum |dt| / max(sum |t|)); on/off = 1")
ap.add_argument("--prior-file", default="", help="json with first-order transition counts over the quantized key state (scripts/vpt_streams.py data/vpt/prior.json) used for the K-1 alternative hypotheses instead of the synthetic-family prior"); ap.add_argument("--hedge", type=int, default=0, help="10/14 proposal B: tolerance-aware hypothesis — set the translation magnitude of the constant-velocity hypothesis to 0.707 (covers speeds 0.5..1.4 under the x0.5..x2 tolerance)"); ap.add_argument("--oracle", type=int, default=1); ap.add_argument("--spec-decode", type=int, default=1); ap.add_argument("--eps", type=float, default=0.24); ap.add_argument("--frames", type=int, default=161); ap.add_argument("--shard", default="0/1")
ap.add_argument("--no-video", action="store_true"); ap.add_argument("--warm", type=int, default=1); ap.add_argument("--save-lat", type=int, default=0); ap.add_argument("--stream-mode", default="rel", choices=["rel", "prepare"], help="prepare: relative poses computed from the stream's poses.npy exactly as prepare() does (same actions as spec_early arms)")
ap.add_argument("--edit", default="", help="[10/14 item 2] action-edit refiner LoRA (specwm/edit_train.py checkpoint): on a miss, re-noise the closest branch's chunk to the LoRA's node and run ONE forward with the true action instead of regenerating"); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
from specwm.act_stream import make_stream, se3
from wan.utils.cam_utils import get_Ks_transformed, get_plucker_embeddings
from einops import rearrange
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"; dev = "cuda:0"; sync = torch.cuda.synchronize
B = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True, ring_frame=True); FP.fp8_swap(B.target_model, True, keep=False)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 512; _dyn.config.accumulated_recompile_limit = 8192
EGATE = torch.zeros((), device=dev); EDIT_NODE = None
if a.edit:   # wrap the (FP8) linears with the gated LoRA, same module order as edit_train.py
    import torch.nn as nn; ck = torch.load(a.edit, map_location="cpu"); EDIT_NODE = int(ck["node"]); _i = [0]
    class _LoRA(nn.Module):
        def __init__(self, base, A_, B_): super().__init__(); self.base = base; self.A = A_.to(dev); self.Bm = B_.to(dev)
        def forward(self, x): return self.base(x) + (EGATE * ((x.float() @ self.A.t()) @ self.Bm.t())).to(x.dtype)
    def _wrap(base): l = _LoRA(base, ck["A"][_i[0]], ck["B"][_i[0]]); _i[0] += 1; return l
    for blk in B.target_model.blocks:
        for nm in ("q", "k", "v", "o"): setattr(blk.self_attn, nm, _wrap(getattr(blk.self_attn, nm)))
        if ck.get("ffn", 1):
            blk.ffn[0] = _wrap(blk.ffn[0]); blk.ffn[2] = _wrap(blk.ffn[2])
    assert _i[0] == len(ck["A"]), (_i[0], len(ck["A"])); print("[edit] LoRA %s: %d linears, node %d, step %s" % (a.edit, _i[0], EDIT_NODE, ck.get("step")), flush=True)
for blk in B.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=True)   # dynamic: unit length and ring position vary per unit
B.pipe.text_encoder.model.to(dev); _te = B.pipe.text_encoder
class _TE:
    def __init__(s, te): s.te = te; s.model = te.model
    def __call__(s, texts, device): return s.te(texts, dev)
B.pipe.text_encoder = _TE(_te); os.makedirs(a.out, exist_ok=True); sig = [float(s) for s in B.node_sigmas()]; CS = a.cs
MODE, MVAL = a.mode.split(":"); assert MODE in ("uniform", "cw"); K_UNI = int(MVAL) if MODE == "uniform" else 4; CW_F = float(MVAL) if MODE == "cw" else 0.0; VAE = B.pipe.vae; VM = VAE.model; PAUSED = [0.0]
EE = None
if MODE == "cw":
    from specwm.early_exit import EarlyExit; EE = EarlyExit(B, eps=a.eps, smooth=True)   # chunk-wise rule of the teacher-parity mode: stop at step k if >= CW_F of the tokens are predicted within eps
def clock(): return time.time() - PAUSED[0]
def patch(x):
    c, f, h, w = x.shape; return x.view(c, f, 1, h // 2, 2, w // 2, 2).permute(1, 3, 5, 2, 4, 6, 0).reshape(f * (h // 2) * (w // 2), 64)
SEED = [a.seed]
def gen(seed, f0, k=0, m=None): return torch.Generator().manual_seed(seed * 1009 + ((f0 // CS) if (m == CS and f0 % CS == 0) else 1000 + f0) * 7 + k)   # fixed 3-frame units reproduce act_spec.py's noise (chunk-keyed)
def emb_frames(rel, Ks, h, w, lat_h, lat_w):   # [m,4,4] -> [1, C, m, lat_h, lat_w]
    f = rel.shape[0]; e = get_plucker_embeddings(rel, Ks, h, w); e = rearrange(e, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)', c1=int(h // lat_h), c2=int(w // lat_w))[None, ...]
    return rearrange(e, 'b (f h w) c -> b c f h w', f=f, h=lat_h, w=lat_w).to(B.param_dtype)
def rot_angle(Ra, Rb):
    cosv = ((Ra.transpose(-1, -2) @ Rb).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2; return torch.rad2deg(torch.acos(cosv.clamp(-1, 1)))
PRIOR = {}
def fit_prior(P, n=300):
    key = json.dumps(P, sort_keys=True)
    if key in PRIOR: return PRIOR[key]
    if a.prior_file:
        import ast; raw = json.load(open(a.prior_file))["counts"]; cnt = {ast.literal_eval(k): {ast.literal_eval(k2): v for k2, v in d.items()} for k, d in raw.items()}; PRIOR[key] = cnt; return cnt
    Pp = dict(P); Pp["jitter"] = 0.0; cnt = {}
    for s_ in range(n):
        _, vel, _ = make_stream(Pp, 10_000 + s_)
        for t in range(1, len(vel)): d = cnt.setdefault(vel[t - 1], {}); d[vel[t]] = d.get(vel[t], 0) + 1
    PRIOR[key] = cnt; return cnt
def quantize(T, P):
    R = T[:3, :3]; yaw = math.degrees(math.atan2(float(R[0, 2]), float(R[2, 2]))); pitch = math.degrees(math.atan2(-float(R[1, 2]), float(R[1, 1]))); sp = -float(T[2, 3]); st = float(T[0, 3])
    sets = [[float(x) for x in str(P[k]).split(",")] for k in ("yaw", "pitch", "speed", "strafe")]
    return tuple(min(S_, key=lambda v: abs(v - x)) for S_, x in zip(sets, (yaw, pitch, sp, st)))
def hypotheses(rel, f0, m, K, P):
    last = rel[f0 - 1]; H = [("same", last.expand(m, 4, 4).clone())]
    if a.hedge:
        n = float(last[:3, 3].norm())
        if n > 1e-6: H[0][1][:, :3, 3] *= 0.707 / n
    if K >= 2 and P:
        v = quantize(last, P); ranked = sorted(fit_prior(P).get(v, {}).items(), key=lambda kv: -kv[1])
        for v2, _ in ranked:
            if v2 == v or len(H) >= K: continue
            H.append(("prior:%s" % ",".join("%g" % x for x in v2), torch.from_numpy(se3(*v2)).float().to(dev).expand(m, 4, 4).clone()))
    return H
def act_err(h, tr):
    dt = float((h[:, :3, 3] - tr[:, :3, 3]).norm(dim=1).sum()); den = max(float(h[:, :3, 3].norm(dim=1).sum()), float(tr[:, :3, 3].norm(dim=1).sum()), 1e-6)
    return float(rot_angle(h[:, :3, :3], tr[:, :3, :3]).sum()), (dt / den if dt > 1e-6 else 0.0)
CTX = {}
def call(z, t_val, f0, m):
    emb = torch.cat(CTX["E"][f0:f0 + m], 2); fs = B._frame_seqlen; C = B._caches["target"]
    kw = dict(context=[B._ctx[0]], seq_len=m * fs, y=[CTX["Y"][:, f0:f0 + m]], dit_cond_dict={"c2ws_plucker_emb": emb.chunk(1, dim=0)}, kv_cache=C["kv"], crossattn_cache=C["cross"], current_start=f0 * fs, max_attention_size=B._kv_size, frame_seqlen=fs)
    with torch.amp.autocast("cuda", dtype=B.param_dtype): out = B.target_model(x=[z.to(B.param_dtype)], t=torch.full((1,), float(t_val), device=dev), cross_attn_first_call=not B._cross_init["target"], **kw)[0]
    B._cross_init["target"] = True; return out.float()
def denoise(z, f0, m, g):
    """returns (x0, stop step); cw mode stops early by the head rule"""
    for i in range(K_UNI):
        x = z - sig[i] * call(z, B._timesteps[i], f0, m)
        if EE is not None and i < K_UNI - 1 and float((EE.predict(m, B._chunk_shape[2] // 2, B._chunk_shape[3] // 2).flatten() <= a.eps).float().mean()) >= CW_F: return x, i + 1
        if i < K_UNI - 1: z = (1 - sig[i + 1]) * x + sig[i + 1] * B.randn_like(x, g)
    return x, K_UNI
def edit(xH, f0, m, g):
    """action-edit refiner: re-noise the wrong-action chunk to the LoRA's node and run one forward with the true action (CTX['E'] must already hold it)"""
    s = sig[EDIT_NODE]; z = (1 - s) * xH + s * B.randn_like(xH, g); EGATE.fill_(1.0)
    try: out = call(z, B._timesteps[EDIT_NODE], f0, m)
    finally: EGATE.fill_(0.0)
    return z - s * out
def commit(x0, f0, m, stop=4):
    sep = (a.commit == "separate") if EE is None else (stop <= 2)   # cw: safe commit (fused only for units that ran >= 3 steps)
    if sep or m == 1: call(x0, 0.0, f0, m); return
    fs = B._frame_seqlen; L = m * fs; FP.ACTIVE["a"] = dict(idx=torch.arange(L - fs, L, device=dev), n=fs, L=L)
    try: call(x0, 0.0, f0, m)
    finally: FP.ACTIVE["a"] = None
def noise(f0, m): return torch.randn((16, m, B._chunk_shape[2], B._chunk_shape[3]), generator=gen(SEED[0], f0, 0, m)).to(dev)
def kv_pos(f0, m):
    kv = B._caches["target"]["kv"]; return FP.ring_pos(f0, m, B._frame_seqlen, kv[0]["k"].shape[1], 6, dev)[0]
def kv_snapshot(f0, m):
    p = kv_pos(f0, m); return [(l["k"][:, p].clone(), l["v"][:, p].clone()) for l in B._caches["target"]["kv"]]
def kv_restore(f0, m, snap):
    p = kv_pos(f0, m)
    for l, (k, v) in zip(B._caches["target"]["kv"], snap): l["k"][:, p] = k; l["v"][:, p] = v
def vae_snapshot(): return [t.clone() if torch.is_tensor(t) else t for t in VM._feat_map]
def vae_restore(snap): VM._feat_map = [t.clone() if torch.is_tensor(t) else t for t in snap]
@torch.no_grad()
def decode(z):
    with torch.amp.autocast("cuda", dtype=VAE.dtype):
        sc = VAE.scale; zz = z.unsqueeze(0).to(VAE.dtype) / sc[1].view(1, -1, 1, 1, 1) + sc[0].view(1, -1, 1, 1, 1); x = VM.conv2(zz); outs = []
        for i in range(x.shape[2]):
            VM._conv_idx = [0]; o = VM.decoder(x[:, :, i:i + 1], feat_cache=VM._feat_map, feat_idx=VM._conv_idx).float().clamp_(-1, 1); outs.append(((o[0] + 1) * 127.5).to(torch.uint8).permute(1, 2, 3, 0))
    fr = torch.cat(outs, 0).cpu(); sync(); return fr
@torch.no_grad()
def run(task, stream, K, out, adaptive):
    T = ALL[task]; st = np.load(S + "/data/actstream/%s/rel.npy" % stream); meta = json.load(open(S + "/data/actstream/%s/stream.json" % stream)); P = meta.get("params"); changes = set(meta.get("change_frames", []))
    cond = B.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=a.frames, max_area=a.area, chunk_size=CS); B.reset_caches(); LF = cond["num_chunks"] * CS
    lat_h, lat_w = B._chunk_shape[2], B._chunk_shape[3]; h, w = lat_h * 8, lat_w * 8; rel = torch.from_numpy(st).float().to(dev)[:LF]
    if a.stream_mode == "prepare":   # the same relative poses prepare() derives from the stream's poses.npy (so spec_early arms on --action-dir see identical actions)
        from wan.utils.cam_utils import interpolate_camera_poses, compute_relative_poses
        c2ws = np.load(S + "/data/actstream/%s/poses.npy" % stream)[:a.frames]; L_ = len(c2ws); Lp = (L_ - 1) // 4 + 1; Lp = Lp - (Lp % CS)
        c2 = interpolate_camera_poses(src_indices=np.linspace(0, L_ - 1, L_), src_rot_mat=c2ws[:, :3, :3], src_trans_vec=c2ws[:, :3, 3], tgt_indices=np.linspace(0, L_ - 1, Lp))
        rel = compute_relative_poses(c2, framewise=True).float().to(dev)[:LF]
    Ks = torch.from_numpy(np.load(S + "/data/actstream/%s/intrinsics.npy" % stream)).float(); Ks = get_Ks_transformed(Ks, height_org=480, width_org=832, height_resize=h, width_resize=w, height_final=h, width_final=w)[0].repeat(LF, 1).to(dev)
    E_true = [emb_frames(rel[f:f + 1], Ks[f:f + 1], h, w, lat_h, lat_w) for f in range(LF)]; CTX["E"] = list(E_true); CTX["Y"] = torch.cat(list(B._y_chunks), 1)
    VM.clear_cache(); PAUSED[0] = 0.0; sync(); t_gen = clock(); unit_start = []; ready = []; frames = []; nf = 0; rows = []; tm = dict(branch=0.0, gen=0.0, decode=0.0, copy=0.0, oracle=0.0)
    f = 0; mode3 = True; streak = 0; u = 0; LATS = []
    while f < LF:
        m = min(CS if mode3 else a.sub, LF - f); row = dict(c=u, f0=f, m=m, mode=3 if mode3 else a.sub, change=any(f <= x < f + m for x in changes) if f > 0 else None); branches = []; pre = None
        if K >= 1 and f >= 1:
            t1 = clock(); vsnap = vae_snapshot() if a.spec_decode else None; pre = kv_snapshot(f, m) if ((adaptive and m > a.sub) or a.edit) else None; z0 = noise(f, m)
            for nm, hyp in hypotheses(rel, f, m, K, P):
                for j in range(m): CTX["E"][f + j] = emb_frames(hyp[j:j + 1], Ks[f + j:f + j + 1], h, w, lat_h, lat_w)
                tb = clock(); x0, stp = denoise(z0.clone(), f, m, gen(SEED[0], f, 1, m)); commit(x0, f, m, stp); kv = kv_snapshot(f, m); fr = None; vm = None
                if a.spec_decode: fr = decode(x0); vm = VM._feat_map; vae_restore(vsnap)
                sync(); branches.append(dict(name=nm, hyp=hyp, x0=x0, kv=kv, frames=fr, vae=vm, t=clock() - tb, steps=stp))
            sync(); tm["branch"] += clock() - t1
        if a.rt:   # real-time arrival of the unit's last action
            t_arr = t_gen + max(0, f + m - 3) * 0.25
            while clock() < t_arr: time.sleep(0.001)
        unit_start.append(clock()); tr = rel[f:f + m]; hit = None   # <- the unit's action is consumed here
        if branches:
            errs = [act_err(b["hyp"], tr) for b in branches]; ok = [i for i, (r, t) in enumerate(errs) if r <= a.tol_rot * m / 3.0 and t <= a.tol_trans]
            row["branches"] = [dict(name=b["name"], rot_err=round(e[0], 2), trans_err=round(e[1], 3), t=round(b["t"], 3)) for b, e in zip(branches, errs)]
            if ok: hit = min(ok, key=lambda i: errs[i][0])
        if hit is not None:
            b = branches[hit]; t1 = clock(); kv_restore(f, m, b["kv"]); x0 = b["x0"]
            if a.spec_decode: VM._feat_map = b["vae"]; fr = b["frames"]
            else: fr = decode(x0)
            sync(); t_r = clock(); tm["copy"] += t_r - t1; row.update(hit=b["name"], steps=b["steps"]); streak += 1
            if not mode3 and streak >= a.recover: mode3 = True
            if a.oracle:
                sync(); tp = time.time(); snap = kv_snapshot(f, m); CTX["E"][f:f + m] = E_true[f:f + m]; xo, _ = denoise(z0.clone(), f, m, gen(SEED[0], f, 1, m)); kv_restore(f, m, snap)
                tT = patch(xo); sg = float(tT.norm(dim=1).median()); gp = (patch(x0) - tT).norm(dim=1) / sg; row["oracle"] = dict(gap=round(float(gp.mean()), 4), within=round(float((gp <= a.eps).float().mean()), 4)); sync(); PAUSED[0] += time.time() - tp; tm["oracle"] += time.time() - tp
        elif a.edit and branches:   # item 2: edit the closest branch's chunk with the true action (one forward) instead of regenerating
            kv_restore(f, m, pre); src = min(range(len(branches)), key=lambda i: errs[i][0]); CTX["E"][f:f + m] = E_true[f:f + m]; t1 = clock()
            x0 = edit(branches[src]["x0"], f, m, gen(SEED[0], f, 2, m)); commit(x0, f, m, K_UNI); sync(); t2 = clock(); fr = decode(x0); t_r = clock(); tm["gen"] += t2 - t1; tm["decode"] += t_r - t2; row.update(hit=None, steps=1, edit=branches[src]["name"]); streak = 0
            if a.oracle:   # within-eps of the edited chunk vs the true-action chunk from the same noise (the edit_train validation measure), excluded from the clock
                sync(); tp = time.time(); snap = kv_snapshot(f, m); xo, _ = denoise(z0.clone(), f, m, gen(SEED[0], f, 1, m)); kv_restore(f, m, snap)
                tT = patch(xo); sg = float(tT.norm(dim=1).median()); gp = (patch(x0) - tT).norm(dim=1) / sg; gh = (patch(branches[src]["x0"]) - tT).norm(dim=1) / sg
                row["oracle"] = dict(gap=round(float(gp.mean()), 4), within=round(float((gp <= a.eps).float().mean()), 4), within_asis=round(float((gh <= a.eps).float().mean()), 4)); sync(); PAUSED[0] += time.time() - tp; tm["oracle"] += time.time() - tp
        else:
            if adaptive and m > a.sub and f > 0:   # B: regenerate only --sub frames with the true action and continue in sub-units
                if pre is not None: kv_restore(f, m, pre)   # the branches overwrote the ring slots of frames f..f+m-1 (= the oldest frames in the window); put them back
                m = min(a.sub, LF - f); mode3 = False; row["m"] = m; row["mode"] = 1
            CTX["E"][f:f + m] = E_true[f:f + m]; t1 = clock(); x0, stp = denoise(noise(f, m), f, m, gen(SEED[0], f, 1, m)); commit(x0, f, m, stp); sync(); t2 = clock(); fr = decode(x0); t_r = clock(); tm["gen"] += t2 - t1; tm["decode"] += t_r - t2; row.update(hit=None, steps=stp); streak = 0
        CTX["E"][f:f + m] = E_true[f:f + m]; ready += [(nf + i, u, t_r) for i in range(fr.shape[0])]; nf += fr.shape[0]; frames.append(fr); rows.append(row); LATS.append(x0.half().cpu()); del branches
        print("[AS2] %s %s K%d f%02d m%d %s %s" % (task, stream, K, f, m, ("HIT " + row["hit"]) if hit is not None else (("EDIT " + row["edit"]) if "edit" in row else ("miss" if K and f else "gen")), ("oracle within %.0f%%" % (100 * row["oracle"]["within"])) if "oracle" in row else ""), flush=True)
        f += m; u += 1
    sync(); gen_s = clock() - t_gen; os.makedirs(out, exist_ok=True); vid = torch.cat(frames, 0).numpy()
    if not a.no_video: imageio.mimwrite(out + "/video.mp4", list(vid), fps=16, quality=8)
    if a.save_lat: torch.save(torch.cat(LATS, 1), out + "/lat.pt")
    sp = [r for r in rows if r["f0"] >= 1]; hits = sum(r["hit"] is not None for r in sp); lat = [1000 * (t - unit_start[c]) for _, c, t in ready]; f1 = sum(r["m"] for r in rows if r["mode"] != 3)
    R = dict(task=task, stream=stream, K=K, mode=a.mode, area=a.area, chunk_size=CS, adaptive=adaptive, recover=a.recover, sub=a.sub, commit=a.commit, rt=a.rt, seed=SEED[0], chunks=u, lat_frames=LF, tol_rot=a.tol_rot, tol_trans=a.tol_trans, gen_s=round(gen_s, 2), times={k: round(v, 2) for k, v in tm.items()},
             chunk_start=unit_start, t_gen_start=t_gen, frame_ready=ready, hits=hits, spec_chunks=len(sp), hit_rate=hits / max(len(sp), 1), frames_1 = f1, rows=rows, mean_steps=float(np.mean([r.get("steps", K_UNI) for r in rows])), p50_ms=float(np.percentile(lat, 50)), p90_ms=float(np.percentile(lat, 90)),
             fps=(len(ready) - 1) / (ready[-1][2] - ready[0][2]), prepare_s=None, stream_decode=0)
    json.dump(R, open(out + "/loop.json", "w"), indent=1)
    print("[AS2_DONE] %s %s K%d adaptive %d: units %d, hits %d/%d, 1-frame units cover %d/%d frames, p50 %.0f ms, fps %.1f, gen %.1f s (branch %.1f gen %.1f decode %.1f oracle %.1f)" % (task, stream, K, adaptive, u, hits, len(sp), f1, LF, R["p50_ms"], R["fps"], gen_s, tm["branch"], tm["gen"], tm["decode"], tm["oracle"]), flush=True)
arm = ("%s_%s%s%s%s" % (MODE + MVAL + ("P" if a.stream_mode == "prepare" else ""), ("a%d_" % a.area if a.area != 480 * 832 else ""), ("p1_" if a.commit == "partial1" else ""), ("rt_" if a.rt else ""), (("ad%d" % a.recover + ("s%d" % a.sub if a.sub != 1 else "")) if a.adaptive else "fx") + ("_hg" if a.hedge else "") + ("_vp" if a.prior_file else "") + (("_ed" + os.path.basename(os.path.dirname(a.edit)) + ("" if os.path.basename(a.edit) == "lora.pt" else os.path.basename(a.edit)[4:-3])) if a.edit else ""))).rstrip("_")
si, sn = (int(x) for x in a.shard.split("/")); SEEDS = [int(x) for x in a.seeds.split(",")] if a.seeds else [a.seed]
cases = [(t, s, int(k), sd) for t in a.tasks.split(",") for s in a.streams.split(",") for k in a.K.split(",") for sd in SEEDS][si::sn]
if a.warm: SEED[0] = cases[0][3]; run(cases[0][0], cases[0][1], 1, "%s/_warm/%s" % (a.out, arm), 1)
for t, s, k, sd in cases:
    SEED[0] = sd; od = "%s/%s/%s/K%d_%s_s%d" % (a.out, t, s, k, arm, sd)
    if os.path.exists(od + "/loop.json"): print("[skip]", od, flush=True); continue
    try: run(t, s, k, od, a.adaptive)
    except Exception as e:
        import traceback; traceback.print_exc(); print("[AS2_FAIL] %s %s K%d %r" % (t, s, k, e), flush=True)
    finally: torch.cuda.empty_cache()
print("[ALL_DONE]", flush=True)
