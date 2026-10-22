"""[10/13, proposal A] Action-speculative chunk generation (input-side speculation) on an action stream with change points.
While the user has not yet produced the next chunk's action, the generator computes the next chunk under K hypotheses of that action
(constant velocity = last relative pose repeated; K >= 2 adds the K-1 most likely other key states under a first-order prior fitted on fresh streams of the same family, specwm/act_stream.py), commits each one and keeps
the chunk's K/V slot (ring cache) and, with --spec-decode, the decoded frames + VAE cache per branch. When the true action is read
(request time), a branch whose action is within --tol-rot degrees (sum over the chunk's frames of the geodesic rotation error) and
--tol-trans (relative translation error, 0.5 = scale x0.5..x2 passes, on/off = 1 fails) is emitted at once (K/V slot copied back, frames already decoded); on a
miss the chunk is generated with the true action as usual. K = 0 is the baseline (same stream, no speculation, same noise, same decode).
Timing: wall clock on one GPU, decode synchronous on the same GPU (both arms), loop.json in the scripts/wm_latency.py format
(chunk_start = the moment the chunk's action is consumed; frame_ready = decoded frame in host memory). --oracle 1 also generates the
true-action chunk from the same noise for every hit (token gap of the emitted branch to it, facts §27 measure); its time is excluded.
Streams: data/actstream/<name>/rel.npy (specwm/act_stream.py). Modes: uniform:4 (teacher), uniform:2, cw:0.7 (teacher-parity chunk-wise
head rule + safe commit, as in the bench-43 modes).
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.act_spec --out DIR --streams tau2_s0,tau2_s1 --K 0,1,2,4 [--mode uniform:4] [--area 399360]"""
import argparse, os, sys, json, time, math, numpy as np, torch, imageio
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--streams", required=True); ap.add_argument("--K", default="0,1,2,4"); ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--tasks", default=open(S + "/lingbot-world-v2/specwm_out/distill_heldout12.txt").read().strip()); ap.add_argument("--mode", default="uniform:4"); ap.add_argument("--area", type=int, default=480 * 832)
ap.add_argument("--chunk-size", type=int, default=3); ap.add_argument("--tol-rot", type=float, default=3.0, help="hit if sum of per-frame rotation errors over the chunk <= this (deg)"); ap.add_argument("--tol-trans", type=float, default=0.5, help="hit if relative translation error (sum |dt| / max(sum |t|)) <= this; 1 = motion switched on/off")
ap.add_argument("--oracle", type=int, default=1); ap.add_argument("--spec-decode", type=int, default=1)
ap.add_argument("--eps", type=float, default=0.24); ap.add_argument("--frames", type=int, default=161); ap.add_argument("--shard", default="0/1"); ap.add_argument("--no-video", action="store_true"); ap.add_argument("--warm", type=int, default=1); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
from wan.utils.cam_utils import get_Ks_transformed, get_plucker_embeddings
from einops import rearrange
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"; dev = "cuda:0"; sync = torch.cuda.synchronize
B = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True); FP.fp8_swap(B.target_model, True, keep=False)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 512; _dyn.config.accumulated_recompile_limit = 8192
for blk in B.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
B.pipe.text_encoder.model.to(dev); _te = B.pipe.text_encoder
class _TE:
    def __init__(s, te): s.te = te; s.model = te.model
    def __call__(s, texts, device): return s.te(texts, dev)
B.pipe.text_encoder = _TE(_te); os.makedirs(a.out, exist_ok=True); sig = [float(s) for s in B.node_sigmas()]; cs = a.chunk_size
MODE, MVAL = a.mode.split(":"); K_UNI = int(MVAL) if MODE == "uniform" else 4; EE = None
if MODE == "cw":
    from specwm.early_exit import EarlyExit; EE = EarlyExit(B, eps=a.eps, bucket=128, smooth=True, chunk_frac=float(MVAL))
VAE = B.pipe.vae; VM = VAE.model
PAUSED = [0.0]
def clock(): return time.time() - PAUSED[0]
def patch(x):
    c, f, h, w = x.shape; return x.view(c, f, 1, h // 2, 2, w // 2, 2).permute(1, 3, 5, 2, 4, 6, 0).reshape(f * (h // 2) * (w // 2), 64)
def gen(seed, c, k=0): return torch.Generator().manual_seed(seed * 1009 + c * 7 + k)
def emb_chunk(rel, Ks, h, w, lat_h, lat_w):
    f = rel.shape[0]; e = get_plucker_embeddings(rel, Ks, h, w); e = rearrange(e, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)', c1=int(h // lat_h), c2=int(w // lat_w))[None, ...]
    return rearrange(e, 'b (f h w) c -> b c f h w', f=f, h=lat_h, w=lat_w).to(B.param_dtype)
def yawm(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg)); R = torch.eye(4, device=dev); R[0, 0] = c; R[0, 2] = s; R[2, 0] = -s; R[2, 2] = c; return R
def rot_angle(Ra, Rb):   # geodesic angle (deg) between two rotation matrices
    cosv = ((Ra.transpose(-1, -2) @ Rb).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2; return torch.rad2deg(torch.acos(cosv.clamp(-1, 1)))
from specwm.act_stream import make_stream, se3
PRIOR = {}
def fit_prior(P, n=300):
    """first-order transition counts of the key state, frame level, from fresh streams of the same family (never the test stream)"""
    key = json.dumps(P, sort_keys=True)
    if key in PRIOR: return PRIOR[key]
    Pp = dict(P); Pp["jitter"] = 0.0; cnt = {}
    for s_ in range(n):
        _, vel, _ = make_stream(Pp, 10_000 + s_)
        for t in range(1, len(vel)): d = cnt.setdefault(vel[t - 1], {}); d[vel[t]] = d.get(vel[t], 0) + 1
    PRIOR[key] = cnt; return cnt
def quantize(T, P):
    """nearest key state (yaw, pitch, speed, strafe) of a raw relative pose (R = Ry(yaw) Rx(pitch), t = (strafe, 0, -speed))"""
    R = T[:3, :3]; yaw = math.degrees(math.atan2(float(R[0, 2]), float(R[2, 2]))); pitch = math.degrees(math.atan2(-float(R[1, 2]), float(R[1, 1]))); sp = -float(T[2, 3]); st = float(T[0, 3])
    sets = [[float(x) for x in str(P[k]).split(",")] for k in ("yaw", "pitch", "speed", "strafe")]
    return tuple(min(S_, key=lambda v: abs(v - x)) for S_, x in zip(sets, (yaw, pitch, sp, st)))
def hypotheses(rel, c, K, P):
    """K candidate actions for chunk c (frames cs*c .. cs*c+cs-1) from the stream up to frame cs*c-1: 'same' = last raw pose repeated
    (constant velocity), then the (K-1) most likely other key states under the family prior given the quantized last state"""
    last = rel[cs * c - 1]; H = [("same", last.expand(cs, 4, 4).clone())]
    if K >= 2:
        v = quantize(last, P); cnt = fit_prior(P); ranked = sorted(cnt.get(v, {}).items(), key=lambda kv: -kv[1])
        for v2, n_ in ranked:
            if v2 == v or len(H) >= K: continue
            H.append(("prior:%s" % ",".join("%g" % x for x in v2), torch.from_numpy(se3(*v2)).float().to(dev).expand(cs, 4, 4).clone()))
    return H
def act_err(h, tr):
    """(sum of per-frame rotation errors in deg, RELATIVE translation error = sum |dt| / max(sum |t_h|, sum |t_true|)): action_tol2 (10/13): translation
    scale x0.5 / x2 stays at the sample-to-sample floor but switching motion on/off does not, so the translation test is relative (1 = on/off)"""
    dt = float((h[:, :3, 3] - tr[:, :3, 3]).norm(dim=1).sum()); den = max(float(h[:, :3, 3].norm(dim=1).sum()), float(tr[:, :3, 3].norm(dim=1).sum()), 1e-6)
    return float(rot_angle(h[:, :3, :3], tr[:, :3, :3]).sum()), (dt / den if dt > 1e-6 else 0.0)
def denoise(c, z, g):
    """returns x0, stop step (1..4), per-step hist; the K/V slot of chunk c holds the state of the last forward"""
    if EE is not None:
        x0, ms, hist = EE.gen_chunk(c, z, g, sparse=False); stop = 1 + max(range(len(hist)), key=lambda k: hist[k]); return x0, stop, hist
    for i in range(K_UNI):
        x = B.target_x0(z, i, c).float()
        if i < K_UNI - 1: z = (1 - sig[i + 1]) * x + sig[i + 1] * B.randn_like(x, g)
    return x, K_UNI, [0] * (K_UNI - 1) + [1]
def commit(x0, c, stop):
    if EE is None or stop <= 2: B.commit(x0, c); return
    fs = B._frame_seqlen; L = cs * fs; FP.ACTIVE["a"] = dict(idx=torch.arange(L - fs, L, device=dev), n=fs, L=L)   # safe commit: fused for chunks that ran >= 3 steps
    try: B.commit(x0, c)
    finally: FP.ACTIVE["a"] = None
def slot(c):
    S_ = cs * B._frame_seqlen; kv = B._caches["target"]["kv"]; size = kv[0]["k"].shape[1]; sink = 6 * B._frame_seqlen; sink_c = sink // S_; slots = (size - sink) // S_
    off = c * S_ if c < sink_c + slots else sink + ((c - sink_c) % slots) * S_; return off, off + S_
def kv_snapshot(c):
    s, e = slot(c); return [(l["k"][:, s:e].clone(), l["v"][:, s:e].clone()) for l in B._caches["target"]["kv"]]
def kv_restore(c, snap):
    s, e = slot(c)
    for l, (k, v) in zip(B._caches["target"]["kv"], snap): l["k"][:, s:e].copy_(k); l["v"][:, s:e].copy_(v)
def vae_snapshot(): return [t.clone() if torch.is_tensor(t) else t for t in VM._feat_map]
def vae_restore(snap): VM._feat_map = [t.clone() if torch.is_tensor(t) else t for t in snap]
@torch.no_grad()
def decode(z):
    """decode a chunk's latents [16, F, h, w] with the running VAE cache; returns uint8 frames [T, H, W, 3] on the host (synchronous)"""
    with torch.amp.autocast("cuda", dtype=VAE.dtype):
        sc = VAE.scale; zz = z.unsqueeze(0).to(VAE.dtype) / sc[1].view(1, -1, 1, 1, 1) + sc[0].view(1, -1, 1, 1, 1); x = VM.conv2(zz); outs = []
        for i in range(x.shape[2]):
            VM._conv_idx = [0]; o = VM.decoder(x[:, :, i:i + 1], feat_cache=VM._feat_map, feat_idx=VM._conv_idx).float().clamp_(-1, 1); outs.append(((o[0] + 1) * 127.5).to(torch.uint8).permute(1, 2, 3, 0))
    fr = torch.cat(outs, 0).cpu(); sync(); return fr
@torch.no_grad()
def run(task, stream, K, out):
    T = ALL[task]; st = np.load(S + "/data/actstream/%s/rel.npy" % stream); meta = json.load(open(S + "/data/actstream/%s/stream.json" % stream))
    cond = B.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=a.frames, max_area=a.area, chunk_size=cs); B.reset_caches(); N = cond["num_chunks"]
    lat_h, lat_w = B._chunk_shape[2], B._chunk_shape[3]; h, w = lat_h * 8, lat_w * 8; rel = torch.from_numpy(st).float().to(dev)[:N * cs]
    Ks = torch.from_numpy(np.load(S + "/data/actstream/%s/intrinsics.npy" % stream)).float(); Ks = get_Ks_transformed(Ks, height_org=480, width_org=832, height_resize=h, width_resize=w, height_final=h, width_final=w)[0].repeat(N * cs, 1).to(dev)
    E_true = [emb_chunk(rel[cs * c:cs * c + cs], Ks[cs * c:cs * c + cs], h, w, lat_h, lat_w) for c in range(N)]; E = list(E_true); B._emb_chunks = E
    VM.clear_cache(); PAUSED[0] = 0.0; sync(); t_gen = clock(); chunk_start = [None] * N; ready = []; frames = []; nf = 0; rows = []; tm = dict(branch=0.0, gen=0.0, decode=0.0, copy=0.0, oracle=0.0)
    for c in range(N):
        row = dict(c=c, change=c in meta.get("change_chunks3", []) if cs == 3 else None); z0 = B.chunk_noise(c, gen(a.seed, c)); branches = []
        if K >= 1 and c >= 1:   # speculate before the action is known
            t1 = clock(); vsnap = vae_snapshot() if a.spec_decode else None
            for nm, hyp in hypotheses(rel, c, K, meta["params"]):
                E[c] = emb_chunk(hyp, Ks[cs * c:cs * c + cs], h, w, lat_h, lat_w); B._emb_chunks = E; tb = clock()
                x0, stop, hist = denoise(c, z0.clone(), gen(a.seed, c, 1)); commit(x0, c, stop); kv = kv_snapshot(c); fr = None; vm = None
                if a.spec_decode: fr = decode(x0); vm = VM._feat_map; vae_restore(vsnap)
                sync(); branches.append(dict(name=nm, hyp=hyp, x0=x0, stop=stop, kv=kv, frames=fr, vae=vm, t=clock() - tb))
            sync(); tm["branch"] += clock() - t1
        chunk_start[c] = clock(); tr = rel[cs * c:cs * c + cs]; hit = None   # <- the chunk's action is consumed here
        if branches:
            errs = [act_err(b["hyp"], tr) for b in branches]; ok = [i for i, (r, t) in enumerate(errs) if r <= a.tol_rot and t <= a.tol_trans]
            row["branches"] = [dict(name=b["name"], rot_err=round(e[0], 2), trans_err=round(e[1], 3), steps=b["stop"], t=round(b["t"], 3)) for b, e in zip(branches, errs)]
            if ok: hit = min(ok, key=lambda i: errs[i][0])
        if hit is not None:
            b = branches[hit]; t1 = clock(); kv_restore(c, b["kv"]); x0 = b["x0"]
            if a.spec_decode: VM._feat_map = b["vae"]; fr = b["frames"]
            else: fr = decode(x0)
            sync(); t_r = clock(); tm["copy"] += t_r - t1; row.update(hit=b["name"], steps=b["stop"])
            if a.oracle:   # true-action chunk from the same noise, for the token gap of the emitted branch (not part of the timing)
                sync(); tp = time.time(); snap = kv_snapshot(c); B._emb_chunks = E_true; xo, _, _ = denoise(c, z0.clone(), gen(a.seed, c, 1)); kv_restore(c, snap)
                tT = patch(xo); sg = float(tT.norm(dim=1).median()); gp = (patch(x0) - tT).norm(dim=1) / sg; row["oracle"] = dict(gap=round(float(gp.mean()), 4), within=round(float((gp <= a.eps).float().mean()), 4)); sync(); PAUSED[0] += time.time() - tp; tm["oracle"] += time.time() - tp
        else:
            E[c] = E_true[c]; B._emb_chunks = E; t1 = clock(); x0, stop, hist = denoise(c, z0.clone(), gen(a.seed, c, 1)); commit(x0, c, stop); sync(); t2 = clock(); fr = decode(x0); t_r = clock(); tm["gen"] += t2 - t1; tm["decode"] += t_r - t2; row.update(hit=None, steps=stop)
        E[c] = E_true[c]; ready += [(nf + i, c, t_r) for i in range(fr.shape[0])]; nf += fr.shape[0]; frames.append(fr); rows.append(row); del branches
        print("[AS] %s %s K%d c%02d %s %s" % (task, stream, K, c, ("HIT " + row["hit"]) if hit is not None else ("miss" if K and c else "gen"), ("oracle within %.0f%%" % (100 * row["oracle"]["within"])) if "oracle" in row else ""), flush=True)
    sync(); gen_s = clock() - t_gen; os.makedirs(out, exist_ok=True); vid = torch.cat(frames, 0).numpy()
    if not a.no_video: imageio.mimwrite(out + "/video.mp4", list(vid), fps=16, quality=8)
    sp = [r for r in rows if r["c"] >= 1]; hits = sum(r["hit"] is not None for r in sp); lat = [1000 * (t - chunk_start[c]) for _, c, t in ready]
    R = dict(task=task, stream=stream, K=K, mode=a.mode, area=a.area, chunk_size=cs, seed=a.seed, chunks=N, tol_rot=a.tol_rot, tol_trans=a.tol_trans, gen_s=round(gen_s, 2), times={k: round(v, 2) for k, v in tm.items()},
             chunk_start=chunk_start, t_gen_start=t_gen, frame_ready=ready, hits=hits, spec_chunks=len(sp), hit_rate=hits / max(len(sp), 1), change_chunks=meta.get("change_chunks3"), rows=rows, mean_steps=float(np.mean([r["steps"] for r in rows])),
             p50_ms=float(np.percentile(lat, 50)), p90_ms=float(np.percentile(lat, 90)), fps=(len(ready) - 1) / (ready[-1][2] - ready[0][2]), prepare_s=None, stream_decode=0)
    json.dump(R, open(out + "/loop.json", "w"), indent=1)
    print("[AS_DONE] %s %s K%d: hits %d/%d, p50 %.0f ms, fps %.1f, gen %.1f s (branch %.1f gen %.1f decode %.1f oracle %.1f)" % (task, stream, K, hits, len(sp), R["p50_ms"], R["fps"], gen_s, tm["branch"], tm["gen"], tm["decode"], tm["oracle"]), flush=True)
arm = ("%s_%s%s" % (MODE + MVAL, ("a%d_" % a.area if a.area != 480 * 832 else ""), ("cs%d" % cs if cs != 3 else ""))).rstrip("_")
si, sn = (int(x) for x in a.shard.split("/")); cases = [(t, s, int(k)) for t in a.tasks.split(",") for s in a.streams.split(",") for k in a.K.split(",")][si::sn]
if a.warm: run(cases[0][0], cases[0][1], 1, "%s/_warm/%s" % (a.out, arm))
for t, s, k in cases:
    od = "%s/%s/%s/K%d_%s_s%d" % (a.out, t, s, k, arm, a.seed)
    if os.path.exists(od + "/loop.json"): print("[skip]", od, flush=True); continue
    try: run(t, s, k, od)
    except Exception as e:
        import traceback; traceback.print_exc(); print("[AS_FAIL] %s %s K%d %r" % (t, s, k, e), flush=True)
    finally: torch.cuda.empty_cache()
print("[ALL_DONE]", flush=True)
