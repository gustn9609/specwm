"""[10/16 stages 2+3] LoRA trainers on the causal-fast 14B with teacher-forced context (one chunk per step, history = the dumped clip's latents).
--mode kd   : depth-pruned STUDENT (blocks in --skip bypassed) + LoRA on the remaining blocks, active everywhere (its own commits too);
              data = specwm/target_dump.py 2-node dumps (x0.pt {0,1}, lat.pt, meta.json): loss = sum over nodes 0,1 of ||x0_S - x0_T||^2/||x0_T||^2
              with the teacher's exact inputs (node 0: chunk noise; node 1: x0_T,0 re-noised with the teacher's draw) = deterministic targets.
--mode corr : CORRECTOR LoRA on the full 14B, active only at node 3 (sigma 0.625 = the cloud's one-step node); data = specwm/corr_dump.py
              ({pairs: [{c, xd, eps}], lat.pt of the 4-node dump}): input z = (1-s3) xd + s3 eps, target = the teacher's final chunk; context =
              teacher chunks committed by the stock model (gate 0), as the cloud commits its outputs with the stock weights.
Saves <out>/lora_s<step>.pt and <out>/lora.pt (best val) via specwm.lora_util.state (+ skip list, mode, nodes).
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.lora14_train --mode kd --data specwm_out/kd_data/n2_704 --skip 2,14,.. --out DIR [--rank 128 --lr 3e-5 --epochs 2]"""
import argparse, os, sys, glob, json, time, random, numpy as np, torch
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True, choices=["kd", "corr"]); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--skip", default="", help="kd: block indices bypassed"); ap.add_argument("--rank", type=int, default=128); ap.add_argument("--lr", type=float, default=3e-5); ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--ffn", type=int, default=1); ap.add_argument("--val-clips", type=int, default=6); ap.add_argument("--val-every", type=int, default=400); ap.add_argument("--max-steps", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--area", type=int, default=270336); ap.add_argument("--corr-pairs", default="", help="corr: dir with <clip>/pairs.pt from corr_dump.py")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); random.seed(a.seed); torch.manual_seed(a.seed); dev = torch.device("cuda:0")
os.environ.setdefault("SPECWM_NO_FLASH", "1"); sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from specwm.binding_v2 import V2Binding
from specwm import lora_util as LU
FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"; NODES = (0, 500) if a.mode == "kd" else (0, 250, 500, 750)
b = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=NODES)
_te = b.pipe.text_encoder
class _TE:
    def __init__(s, te): s.te = te; s.model = te.model
    def __call__(s, texts, device): return s.te(texts, dev)
b.pipe.text_encoder = _TE(_te); m = b.target_model.eval(); sig = [float(s) for s in b.node_sigmas()]
for p in m.parameters(): p.requires_grad_(False)
SKIP = [int(v) for v in a.skip.split(",") if v]
for i in SKIP: m.blocks[i].forward = (lambda x, **kw: x)
LOR, GATE = LU.attach(m, a.rank, a.ffn, dev)
if SKIP:   # no LoRA params on bypassed blocks
    per = 6 if a.ffn else 4; keep = [l for bi, blk in enumerate(m.blocks) for l in LOR[bi * per:(bi + 1) * per] if bi not in SKIP]; LOR_T = keep
else: LOR_T = LOR
params = [p for l in LOR_T for p in (l.A, l.B)]
from torch.utils.checkpoint import checkpoint
for bi, blk in enumerate(m.blocks):
    if bi in SKIP: continue
    _f = blk.forward
    def _ck(*args, _f=_f, **kw): return checkpoint(_f, *args, use_reentrant=False, **kw) if torch.is_grad_enabled() else _f(*args, **kw)
    blk.forward = _ck
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
TT = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}
clips = sorted(d for d in glob.glob(a.data + "/*_s*") if os.path.exists(d + "/x0.pt") and (a.mode == "kd" or os.path.exists("%s/%s/pairs.pt" % (a.corr_pairs, os.path.basename(d)))))
random.Random(1).shuffle(clips); VAL, TRAIN = clips[:a.val_clips], clips[a.val_clips:]
def log(*x):
    msg = " ".join(str(v) for v in x); print("[%s %s]" % (a.mode, time.strftime("%H:%M:%S")), msg, flush=True); open(a.out + "/log.txt", "a").write(msg + "\n")
log("clips train %d val %d, LoRA %.1fM params on %d blocks (skip %s), nodes %s sigmas %s" % (len(TRAIN), len(VAL), sum(p.numel() for p in params) / 1e6, len(m.blocks) - len(SKIP), SKIP, NODES, [round(s, 3) for s in sig]))
def gen(seed, c, path): return torch.Generator().manual_seed(seed * 1009 + c * 7 + {"node0": 0, "D": 1, "T": 2}[path])
def pred(z, i, ci, grad):
    with torch.set_grad_enabled(grad), torch.amp.autocast("cuda", dtype=b.param_dtype):
        out = m(x=[z.to(b.param_dtype)], t=torch.full((1,), float(b._timesteps[i]), device=dev), cross_attn_first_call=not b._cross_init["target"], **b._kwargs("target", ci))[0]
    b._cross_init["target"] = True; return z - sig[i] * out.float()
def err(xh, x0): return ((xh - x0) ** 2).mean() / (x0 ** 2).mean().clamp_min(1e-6)
def clip_iter(d):
    """yields (c, [(z, node_i, target)]) per chunk after committing the history; commit gate: kd = student (1), corr = stock (0)"""
    meta = json.load(open(d + "/meta.json")); T = TT[meta["task"]]; seed = meta["seed"]; L = torch.load(d + "/lat.pt").float().to(dev); X0 = torch.load(d + "/x0.pt"); N = meta["chunks"]
    P = None
    if a.mode == "corr": P = {p["c"]: p for p in torch.load("%s/%s/pairs.pt" % (a.corr_pairs, os.path.basename(d)))["pairs"]}
    _te.model.to(dev)
    with torch.no_grad(): b.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=a.area); b.reset_caches()
    _te.model.to("cpu"); torch.cuda.empty_cache(); assert tuple(L.shape[2:]) == tuple(b._chunk_shape[2:]), (L.shape, b._chunk_shape)
    GATE.fill_(1.0 if a.mode == "kd" else 0.0)
    with torch.no_grad(): b.commit(L[:, :3], 0)
    for c in range(1, N):
        items = []
        if a.mode == "kd":
            z0 = b.chunk_noise(c, gen(seed, c, "node0")); items.append((z0, 0, X0[0][c].float().to(dev)))
            x0 = X0[0][c].float().to(dev); g = gen(seed, c, "T"); z1 = (1 - sig[1]) * x0 + sig[1] * b.randn_like(x0, g); items.append((z1, 1, X0[1][c].float().to(dev)))
        else:
            p = P[c]; xd = p["xd"].float().to(dev); eps = p["eps"].float().to(dev); z3 = (1 - sig[3]) * xd + sig[3] * eps; items.append((z3, 3, L[:, 3 * c:3 * c + 3]))
        yield c, items
        GATE.fill_(1.0 if a.mode == "kd" else 0.0)
        with torch.no_grad(): b.commit(L[:, 3 * c:3 * c + 3], c)
@torch.no_grad()
def validate():
    e = {}
    for d in VAL:
        for c, items in clip_iter(d):
            for z, i, tgt in items: GATE.fill_(1.0); e.setdefault(i, []).append(float(err(pred(z, i, c, False), tgt)))
    return {i: float(np.mean(v)) for i, v in e.items()}
def base_err():   # reference: untrained student / raw draft
    for l in LOR: l.gate = torch.zeros((), device=dev)
    v = validate()
    for l in LOR: l.gate = GATE
    return v
step = 0; t0 = time.time(); hist = []
v0 = base_err(); log("step 0 val (LoRA off; kd = untrained pruned student, corr = stock node-3 step) rel-mse", {k: round(x, 4) for k, x in v0.items()})
v = validate(); log("step 0 val rel-mse", {k: round(x, 4) for k, x in v.items()}); best = sum(v.values())
def save(tag): torch.save(LU.state(LOR, rank=a.rank, ffn=a.ffn, skip=SKIP, mode=a.mode, nodes=list(NODES), step=step), a.out + "/lora%s.pt" % tag)
done = False; val_due = False
for ep in range(a.epochs):
    random.shuffle(TRAIN)
    for d in TRAIN:
        for c, items in clip_iter(d):
            GATE.fill_(1.0); loss = sum(err(pred(z, i, c, True), tgt) for z, i, tgt in items) / len(items)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); step += 1; hist.append(float(loss.detach()))
            if step % 50 == 0: log("step %d ep %d train rel-mse %.4f  %.0f min  %.1f GiB" % (step, ep, np.mean(hist[-50:]), (time.time() - t0) / 60, torch.cuda.max_memory_allocated() / 2**30))
            if step % a.val_every == 0: val_due = True
            if a.max_steps and step >= a.max_steps: done = True; break
        if val_due:   # validation re-prepares the binding (own KV context), so it runs only between clips
            val_due = False; v = validate(); log("VAL step %d rel-mse" % step, {k: round(x, 4) for k, x in v.items()}, "(LoRA-off ref %s)" % {k: round(x, 4) for k, x in v0.items()}); save("_s%d" % step)
            if sum(v.values()) < best: best = sum(v.values()); save("")
        if done: break
    if done: break
v = validate(); log("FINAL step %d val rel-mse" % step, {k: round(x, 4) for k, x in v.items()}); save("_last")
if sum(v.values()) < best: save("")
print("[LORA14_DONE] %s steps %d, %.0f min" % (a.mode, step, (time.time() - t0) / 60), flush=True)
