"""KV translator pieces (professor's idea: Base -> Fast KV-cache transfer at LOW resolution):
- base_plan_kv(): Base low-noise expert runs once on the low-res plan latents (t = --kv-t) and we capture, for the chosen blocks, the
  post-norm pre-RoPE keys and the values ([1, M, n, d]) — the whole clip (41 latent frames) as ~28k tokens.
- PlanRope: Fast's 3D RoPE evaluated at the plan tokens' positions mapped onto Fast's grid (frame index kept, spatial scaled and rounded).
- Translator: per block, K_fast = RoPE(W_k K_base + b_k), V_fast = W_v V_base + b_v (full 5120x5120 linears, identity init; the two models share
  the Wan architecture, ridge probes gave cos 0.6-0.9 for a linear map). Fast reads them as extra cache tokens (WanSelfAttention.mem)."""
import torch, torch.nn as nn, math
def capture_base_kv(low_model, lat, c, t_val, layers, dev):
    """one Base forward at t=t_val on lat [16,f,h,w]; returns {l: (k [1,M,n,d] post-norm pre-RoPE, v [1,M,n,d])} on dev (bf16)"""
    out = {}; hooks = []
    for l in layers:
        sa = low_model.blocks[l].self_attn
        def hk(mod, inp, res, l=l, sa=sa):
            b_, s_ = res.shape[:2]; out.setdefault(l, [None, None])[1] = res.view(b_, s_, sa.num_heads, sa.head_dim).detach()
        def hk_k(mod, inp, res, l=l, sa=sa):
            b_, s_ = res.shape[:2]; out.setdefault(l, [None, None])[0] = res.view(b_, s_, sa.num_heads, sa.head_dim).detach()
        hooks.append(sa.v.register_forward_hook(hk)); hooks.append(sa.norm_k.register_forward_hook(hk_k))
    tt = torch.full((1,), float(t_val), device=dev)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        low_model([lat.to(dev, torch.bfloat16)], t=tt, context=[c["ctx"][0]], seq_len=c["seq_len"], y=[c["y"]], dit_cond_dict=c["dit_cond"])
    for h in hooks: h.remove()
    return {l: (kv[0], kv[1]) for l, kv in out.items()}
def subsample_frames(kv, grid_lo, stride):
    """keep every `stride`-th latent frame of the plan tokens: {l: (k, v)} [1, f*h*w, n, d] -> [1, f'*h*w, n, d]; returns (kv, new grid)"""
    f, h, w = grid_lo; keep = list(range(0, f, stride)); idx = torch.cat([torch.arange(fi * h * w, (fi + 1) * h * w) for fi in keep])
    return {l: (k[:, idx.to(k.device)], v[:, idx.to(v.device)]) for l, (k, v) in kv.items()}, (len(keep), h, w)
def plan_rope(freqs, grid_lo, grid_hi, frame_ids=None):
    """complex RoPE multipliers [M, 1, d/2] for plan tokens on grid_lo=(f,hl,wl) placed on Fast's grid_hi=(_,h,w): frame index kept, y/x scaled+rounded"""
    c = freqs.shape[1]; fr = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    f, hl, wl = grid_lo; _, h, w = grid_hi
    ys = torch.tensor([min(h - 1, int(round(y * h / hl))) for y in range(hl)]); xs = torch.tensor([min(w - 1, int(round(x * w / wl))) for x in range(wl)])
    fids = torch.arange(f) if frame_ids is None else torch.tensor(frame_ids); f = len(fids)
    m = torch.cat([fr[0][fids].view(f, 1, 1, -1).expand(f, hl, wl, -1), fr[1][ys].view(1, hl, 1, -1).expand(f, hl, wl, -1), fr[2][xs].view(1, 1, wl, -1).expand(f, hl, wl, -1)], dim=-1)
    return m.reshape(f * hl * wl, 1, -1)
def rope_apply_mem(x, mult):
    """x [1, M, n, d] -> RoPE'd (same maths as causal_rope_apply)"""
    M, n, d = x.shape[1:]; xc = torch.view_as_complex(x[0].to(torch.float64).reshape(M, n, -1, 2))
    return torch.view_as_real(xc * mult.to(xc.device)).flatten(2).unsqueeze(0).to(x.dtype)
class Translator(nn.Module):
    def __init__(self, layers, dim=5120, dev="cuda"):
        super().__init__(); self.layers = list(layers)
        self.k = nn.ModuleDict({str(l): nn.Linear(dim, dim) for l in layers}); self.v = nn.ModuleDict({str(l): nn.Linear(dim, dim) for l in layers})
        for lin in self.k.values(): nn.init.eye_(lin.weight); nn.init.zeros_(lin.bias)   # keys: identity (Base/Fast share the architecture)
        for lin in self.v.values(): nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)   # values: zero -> the memory is silent at init (no disturbance), learns what to contribute
        self.to(dev)
    def forward(self, kv, mult):
        """kv {l: (k [1,M,n,d], v)} (Base) -> {l: (roped k, v)} in Fast's space"""
        out = {}
        for l in self.layers:
            k, v = kv[l]; M, n, d = k.shape[1:]
            kf = self.k[str(l)](k.reshape(1, M, n * d).float()).view(1, M, n, d); vf = self.v[str(l)](v.reshape(1, M, n * d).float()).view(1, M, n, d)
            out[l] = (rope_apply_mem(kf, mult).to(torch.bfloat16), vf.to(torch.bfloat16))   # bf16 memory tokens (the cache dtype); grads flow through the cast
        return out
def set_mem(fast_model, mem):
    """attach (or clear with None) the translated plan memory to Fast's attention blocks"""
    for l, blk in enumerate(fast_model.blocks): blk.self_attn.mem = mem.get(l) if mem else None

class Translator2(nn.Module):
    """kv4: per-head linear (K identity / V zero init) + low-rank cross-head mixing (rank r, zero-init output) for EVERY given block."""
    def __init__(self, layers, dim=5120, heads=40, rank=256, dev="cuda"):
        super().__init__(); self.layers = list(layers); self.n = heads; self.d = dim // heads; self.r = rank
        L = len(self.layers); d = self.d
        self.kw = nn.Parameter(torch.eye(d).repeat(L, heads, 1, 1)); self.kb = nn.Parameter(torch.zeros(L, heads, d))
        self.vw = nn.Parameter(torch.zeros(L, heads, d, d)); self.vb = nn.Parameter(torch.zeros(L, heads, d))
        self.kA = nn.Parameter(torch.randn(L, dim, rank) / math.sqrt(dim)); self.kB = nn.Parameter(torch.zeros(L, rank, dim))
        self.vA = nn.Parameter(torch.randn(L, dim, rank) / math.sqrt(dim)); self.vB = nn.Parameter(torch.zeros(L, rank, dim))
        self.idx = {l: i for i, l in enumerate(self.layers)}; self.to(dev)
    def one(self, x, w, b, A, B):
        M, n, d = x.shape[1:]; xb = x[0].to(torch.bfloat16)   # [M, n, d]; bf16 maths (fp32 master weights) keeps the per-chunk intermediates small
        y = torch.einsum("mnd,nde->mne", xb, w.to(torch.bfloat16)) + b.to(torch.bfloat16); y = y + ((xb.reshape(M, n * d) @ A.to(torch.bfloat16)) @ B.to(torch.bfloat16)).view(M, n, d)
        return y.unsqueeze(0)
    def forward(self, kv, mult):
        out = {}
        for l in self.layers:
            if l not in kv: continue
            i = self.idx[l]; k, v = kv[l]
            kf = self.one(k, self.kw[i], self.kb[i], self.kA[i], self.kB[i]); vf = self.one(v, self.vw[i], self.vb[i], self.vA[i], self.vB[i])
            out[l] = (rope_apply_mem(kf, mult).to(torch.bfloat16), vf)
        return out
def window_tokens(kv, grid_lo, f_lo, f_hi, dev=None, stride=1):
    """plan tokens of latent frames [f_lo, f_hi) (every `stride`-th): {l: (k, v)} sliced (frame-major token order), moved to dev; returns (kv_win, frame_ids)"""
    f, h, w = grid_lo; f_lo = max(0, f_lo); f_hi = min(f, f_hi); fids = list(range(f_lo, f_hi, stride))
    if stride == 1:
        a, b = f_lo * h * w, f_hi * h * w
        return {l: (k[:, a:b].to(dev) if dev else k[:, a:b], v[:, a:b].to(dev) if dev else v[:, a:b]) for l, (k, v) in kv.items()}, fids
    idx = torch.cat([torch.arange(fi * h * w, (fi + 1) * h * w) for fi in fids])
    out = {}
    for l, (k, v) in kv.items():
        ii = idx.to(k.device); kk, vv = k[:, ii], v[:, ii]
        out[l] = (kk.to(dev) if dev else kk, vv.to(dev) if dev else vv)
    return out, fids
