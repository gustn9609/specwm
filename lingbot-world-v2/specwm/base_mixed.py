"""Mixed-resolution forward of the (bidirectional) Base WanModel.

The sequence is a list of segments of latent frames; each segment has its own spatial stride k:
k = 1 is the normal latent grid, k = 2 / 4 means the segment's latents were downsampled k x (so k^2 fewer tokens)
and its tokens take RoPE positions k * i on the full grid (they cover the same image area).
All tokens attend to each other (full bidirectional attention, as in Base); everything else is the model's own
per-token code. Used for Base RePaint where the OLD part of the prefix is given at low resolution (the probe showed
Base barely uses its fine detail) so the correction cost stops growing with the prefix length.
"""
import torch, torch.nn.functional as Fn


def _rope_freqs(grid, freqs, f0, k, c):
    """complex RoPE factors for one segment: frames from f0, rows/cols at stride k -> [f*h*w, 1, c]"""
    fr = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    f, h, w = grid
    hi = torch.arange(h, device=freqs.device) * k; wi = torch.arange(w, device=freqs.device) * k
    return torch.cat([fr[0][f0:f0 + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                      fr[1][hi].view(1, h, 1, -1).expand(f, h, w, -1),
                      fr[2][wi].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1).reshape(f * h * w, 1, -1)


def _rope(x, fi):
    """x: [1, L, n, d] with precomputed factors fi [L, 1, d/2] (all segments concatenated)"""
    L, n = x.size(1), x.size(2)
    xi = torch.view_as_complex(x[0].to(torch.float64).reshape(L, n, -1, 2))
    return torch.view_as_real(xi * fi).flatten(2).unsqueeze(0).float()


@torch.no_grad()
def mixed_forward(m, segs, t_val, context, need):
    """m: WanModel. segs: list of dict(x=[16,f,h,w], y=[20,f,h,w], f0=int, k=int, cam=[1,C,f,h,w] or None) at that segment's resolution.
    t_val: timestep (same for all tokens). context: list with one text embedding. need: indices of segments whose flow to return.
    Returns list of flow tensors [16,f,h,w] for the `need` segments."""
    dev = m.patch_embedding.weight.device; dt = torch.bfloat16
    if m.freqs.device != dev: m.freqs = m.freqs.to(dev)
    from wan.modules.model import sinusoidal_embedding_1d
    from einops import rearrange
    with torch.amp.autocast("cuda", dtype=dt):
        toks, grids, cams = [], [], []
        for s in segs:
            u = m.patch_embedding(torch.cat([s["x"], s["y"]], 0).unsqueeze(0).to(dev, dt)); sb = s.get("sub", 1)
            if sb > 1: u = u[:, :, :, ::sb, ::sb]   # token subsample: real full-res patches at their true positions (RoPE stride = sb)
            grids.append(tuple(u.shape[2:])); toks.append(u.flatten(2).transpose(1, 2))
            if s.get("cam") is not None:
                cm = rearrange(s["cam"].to(dev), '1 c (f c1) (h c2) (w c3) -> 1 f h w (c c1 c2 c3)', c1=m.patch_size[0], c2=m.patch_size[1], c3=m.patch_size[2])
                cams.append(cm[:, :, ::sb, ::sb].flatten(1, 3))
        h = torch.cat(toks, 1); Ls = [t.size(1) for t in toks]; L = h.size(1)
        if any(s.get("lora") for s in segs):   # mixed-memory LoRA on the merged segment(s)
            msk = torch.zeros(1, L, 1, device=dev); o1 = 0
            for s, l in zip(segs, Ls):
                if s.get("lora"): msk[:, o1:o1 + l] = 1
                o1 += l
            LMASK[0] = msk
        with torch.amp.autocast("cuda", dtype=torch.float32):
            t = torch.full((1, L), float(t_val), device=dev)
            e = m.time_embedding(sinusoidal_embedding_1d(m.freq_dim, t.flatten()).unflatten(0, (1, L)).float())
            e0 = m.time_projection(e).unflatten(2, (6, m.dim))
        ctx = m.text_embedding(torch.stack([torch.cat([cc, cc.new_zeros(m.text_len - cc.size(0), cc.size(1))]) for cc in context]))
        FI = torch.cat([_rope_freqs(g, m.freqs, s["f0"], s["k"], m.blocks[0].self_attn.head_dim // 2) for s, g in zip(segs, grids)], 0)
        dcd = None
        if cams:
            pe = m.patch_embedding_wancamctrl(torch.cat(cams, 1).to(dt)); dcd = pe + m.c2ws_hidden_states_layer2(Fn.silu(m.c2ws_hidden_states_layer1(pe)))
        for blk in m.blocks:
            with torch.amp.autocast("cuda", dtype=torch.float32):
                ee = (blk.modulation.unsqueeze(0) + e0).chunk(6, dim=2)
            sa = blk.self_attn; n, d = sa.num_heads, sa.head_dim
            xin = (blk.norm1(h).float() * (1 + ee[1].squeeze(2)) + ee[0].squeeze(2)).to(dt)
            q = sa.norm_q(sa.q(xin)).view(1, L, n, d); k_ = sa.norm_k(sa.k(xin)).view(1, L, n, d); v = sa.v(xin).view(1, L, n, d)
            q = _rope(q, FI).to(dt); k_ = _rope(k_, FI).to(dt)
            o = Fn.scaled_dot_product_attention(q.transpose(1, 2), k_.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2).reshape(1, L, n * d)
            with torch.amp.autocast("cuda", dtype=torch.float32):
                h = h + sa.o(o) * ee[2].squeeze(2)
            if dcd is not None:
                ch = blk.cam_injector_layer2(Fn.silu(blk.cam_injector_layer1(dcd))) + dcd
                h = (1.0 + blk.cam_scale_layer(ch)) * h + blk.cam_shift_layer(ch)
            h = h + blk.cross_attn(blk.norm3(h), ctx, None)
            yf = blk.ffn((blk.norm2(h).float() * (1 + ee[4].squeeze(2)) + ee[3].squeeze(2)))
            with torch.amp.autocast("cuda", dtype=torch.float32):
                h = h + yf * ee[5].squeeze(2)
        out = m.head(h, e)
    LMASK[0] = None
    res, o0 = [], 0; offs = [sum(Ls[:i]) for i in range(len(Ls))]
    for i in need:
        res.append(m.unpatchify(out[:, offs[i]:offs[i] + Ls[i]], torch.tensor([grids[i]]))[0].float())
    return res


def down(x, k):
    """[C,f,h,w] -> [C,f,h/k,w/k] (bilinear, antialiased)"""
    if k == 1: return x
    C, f, hh, ww = x.shape   # target size kept even (2x2 patches)
    return Fn.interpolate(x.permute(1, 0, 2, 3).float(), size=(hh // k // 2 * 2, ww // k // 2 * 2), mode="bilinear", antialias=True).permute(1, 0, 2, 3)


def down_cam(cam, k):
    """[1,C,f,h,w] -> [1,C,f,h/k,w/k]"""
    if k == 1: return cam
    _, C, f, hh, ww = cam.shape
    return Fn.interpolate(cam[0].permute(1, 0, 2, 3).float(), size=(hh // k // 2 * 2, ww // k // 2 * 2), mode="bilinear", antialias=True).permute(1, 0, 2, 3).unsqueeze(0).to(cam.dtype)


# ---------------- mixed-memory LoRA (trained by mixlora_train.py): active only on merged-token positions ----------------
import math
LMASK = [None]


class MaskedLoRA(torch.nn.Module):
    def __init__(self, base, A, B):
        super().__init__(); self.base = base
        self.register_buffer("A", A.to(base.weight.device, torch.float32), persistent=False); self.register_buffer("B", B.to(base.weight.device, torch.float32), persistent=False)

    def forward(self, x):
        y = self.base(x)
        if LMASK[0] is None: return y
        return y + ((x.float() @ self.A.t()) @ self.B.t()).to(y.dtype) * LMASK[0].to(y.dtype)


def attach_lora(m, state):
    """wrap m's self-attn q/k/v/o (+ FFN linears if trained with them) with MaskedLoRA; returns the module list (same order as training)."""
    mods, i = [], 0
    for blk in m.blocks:
        sa = blk.self_attn
        for nm in ("q", "k", "v", "o"):
            l = MaskedLoRA(getattr(sa, nm), state["A"][i], state["B"][i]); setattr(sa, nm, l); mods.append(l); i += 1
        if state.get("ffn", 1):
            for j in (0, 2):
                l = MaskedLoRA(blk.ffn[j], state["A"][i], state["B"][i]); blk.ffn[j] = l; mods.append(l); i += 1
    return mods


def load_lora(mods, state):
    for i, l in enumerate(mods): l.A.copy_(state["A"][i]); l.B.copy_(state["B"][i])
