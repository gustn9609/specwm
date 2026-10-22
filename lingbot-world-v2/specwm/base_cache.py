"""Block-causal Base (LingBot v1 WanModel) with a prefix KV cache — the "target KV" of the speculative loop.

The bidirectional Base normally re-reads the whole prefix for every check (22 s at 480p with CFG).
Here committed chunks are fed once, clean (per-token timestep t = 0), attending only to themselves and
earlier cached chunks; their self-attention K/V are stored per layer. A check then runs only the new
chunk's tokens (at the check noise level) against the cached K/V. Everything except self-attention
(time modulation, camera injection, cross-attention, FFN, head) is the model's own code, applied to the
new tokens only.

Off-distribution on purpose: Base was trained bidirectionally with one timestep for all tokens. Whether
the resulting check signal is still useful is what the experiment measures.

Cache = sink (latent frame 0, the conditioning image) + the last `win` latent frames (rolling, absolute
RoPE positions kept), one cache per text context (cond / uncond for CFG).
"""
import math, torch, torch.nn.functional as Fn


def _rope_offset(x, grid, freqs, f0):
    """wan rope_apply for one sample with the frame axis starting at absolute latent frame f0. x: [1, L, n, d]"""
    n, c = x.size(2), x.size(3) // 2
    fr = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    f, h, w = grid
    xi = torch.view_as_complex(x[0, :f * h * w].to(torch.float64).reshape(f * h * w, n, -1, 2))
    fi = torch.cat([fr[0][f0:f0 + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    fr[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    fr[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1).reshape(f * h * w, 1, -1)
    return torch.view_as_real(xi * fi).flatten(2).unsqueeze(0).float()


class BaseKV:
    def __init__(self, get_model, win=9):
        """get_model(): returns the WanModel to use (the caller makes sure the right expert is loaded)."""
        self.get_model = get_model; self.win = win; self.caches = {}   # key -> dict(frames=[f idx], k=[layer][list per frame], v=...)

    def reset(self):
        self.caches = {}

    def n_frames(self, key):
        c = self.caches.get(key); return 0 if c is None else len(c["frames"])

    @torch.no_grad()
    def forward(self, key, x, y, f0, t_val, context, cam=None, write=False):
        """x: [16, f, h, w] latent (noisy at t_val, or clean with t_val=0 for a commit); y: [20, f, h, w] (mask+image cond);
        f0: absolute latent frame index of x; cam: plucker emb [1, C, f, h, w] (already sliced) or None.
        write: store this chunk's K/V (commit). Returns the model's flow prediction [16, f, h, w]."""
        m = self.get_model(); dev = m.patch_embedding.weight.device; dt = torch.bfloat16
        if m.freqs.device != dev: m.freqs = m.freqs.to(dev)
        cache = self.caches.setdefault(key, dict(frames=[], k=[[] for _ in m.blocks], v=[[] for _ in m.blocks]))
        with torch.amp.autocast("cuda", dtype=dt):
            u = m.patch_embedding(torch.cat([x, y], 0).unsqueeze(0).to(dev, dt))
            grid = tuple(u.shape[2:]); tok = u.flatten(2).transpose(1, 2); L = tok.size(1); per = grid[1] * grid[2]
            with torch.amp.autocast("cuda", dtype=torch.float32):
                t = torch.full((1, L), float(t_val), device=dev)
                from wan.modules.model import sinusoidal_embedding_1d
                e = m.time_embedding(sinusoidal_embedding_1d(m.freq_dim, t.flatten()).unflatten(0, (1, L)).float())
                e0 = m.time_projection(e).unflatten(2, (6, m.dim))
            ctx = m.text_embedding(torch.stack([torch.cat([cc, cc.new_zeros(m.text_len - cc.size(0), cc.size(1))]) for cc in context]))
            dcd = None
            if cam is not None:
                from einops import rearrange
                pe = rearrange(cam.to(dev), '1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)', c1=m.patch_size[0], c2=m.patch_size[1], c3=m.patch_size[2])
                pe = m.patch_embedding_wancamctrl(pe.to(dt)); pe = pe + m.c2ws_hidden_states_layer2(Fn.silu(m.c2ws_hidden_states_layer1(pe)))
                dcd = pe
            h = tok
            for li, blk in enumerate(m.blocks):
                with torch.amp.autocast("cuda", dtype=torch.float32):
                    ee = (blk.modulation.unsqueeze(0) + e0).chunk(6, dim=2)
                sa = blk.self_attn; n, d = sa.num_heads, sa.head_dim
                xin = blk.norm1(h).float() * (1 + ee[1].squeeze(2)) + ee[0].squeeze(2)
                xin = xin.to(dt)
                q = sa.norm_q(sa.q(xin)).view(1, L, n, d); k = sa.norm_k(sa.k(xin)).view(1, L, n, d); v = sa.v(xin).view(1, L, n, d)
                q = _rope_offset(q, grid, m.freqs, f0).to(dt); k = _rope_offset(k, grid, m.freqs, f0).to(dt)
                K = torch.cat(cache["k"][li] + [k], 1) if cache["k"][li] else k
                V = torch.cat(cache["v"][li] + [v], 1) if cache["v"][li] else v
                o = Fn.scaled_dot_product_attention(q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)).transpose(1, 2).reshape(1, L, n * d)
                y_ = sa.o(o)
                with torch.amp.autocast("cuda", dtype=torch.float32):
                    h = h + y_ * ee[2].squeeze(2)
                if dcd is not None:
                    ch = blk.cam_injector_layer2(Fn.silu(blk.cam_injector_layer1(dcd))) + dcd
                    h = (1.0 + blk.cam_scale_layer(ch)) * h + blk.cam_shift_layer(ch)
                h = h + blk.cross_attn(blk.norm3(h), ctx, None)
                yf = blk.ffn(blk.norm2(h).float() * (1 + ee[4].squeeze(2)) + ee[3].squeeze(2))
                with torch.amp.autocast("cuda", dtype=torch.float32):
                    h = h + yf * ee[5].squeeze(2)
                if write:
                    cache["k"][li].append(k); cache["v"][li].append(v)
            out = m.head(h, e)
        if write:
            cache["frames"].append((f0, grid[0]))
            # evict: keep the sink (first entry) + entries covering the last `win` latent frames
            while len(cache["frames"]) > 2 and sum(fr[1] for fr in cache["frames"][1:]) > self.win:
                del cache["frames"][1]
                for li in range(len(m.blocks)): del cache["k"][li][1]; del cache["v"][li][1]
        return m.unpatchify(out, torch.tensor([grid]))[0].float()
