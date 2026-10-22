"""Ulysses sequence parallelism for the patched causal-fast DiT (fast_patch: dedup time embedding, FP8, SageAttention, ring KV), after
wan/distributed/sequence_parallel.py. Every rank runs the same driver with the same noise; inside the model the chunk's tokens are split across
ranks, self-attention all-to-all's to (all tokens, num_heads / P heads), writes its local heads into a chunk-aligned ring KV cache, attends, and
all-to-all's back; the output is all-gathered so every rank holds the full x0. Cross-attention and the FFN run on the local tokens with all heads.
Call apply_sp() after fast_patch.apply(); call local_head_caches(binding) after every binding.prepare() (self-attn KV keeps num_heads / P heads)."""
import math, os, torch, torch.distributed as dist
FUSED = os.environ.get("SP_FUSED_QKV", "1") == "1"      # one all-to-all for q, k, v (same bytes, 2 NCCL calls per layer instead of 4)
COMM_FP8 = os.environ.get("SP_COMM_FP8", "0") == "1"    # send q/k/v (and the attention output) as e4m3 with a per-token-head scale


GROUP = None   # SP process group (None = the whole world); set_group() when extra ranks (e.g. a decode worker) exist
def set_group(g):
    global GROUP; GROUP = g
def P(): return dist.get_world_size(GROUP)
def R(): return dist.get_rank(GROUP)


def a2a(x, scatter_dim, gather_dim):
    p = P()
    inputs = [u.contiguous() for u in x.chunk(p, dim=scatter_dim)]; outputs = [torch.empty_like(u) for u in inputs]
    dist.all_to_all(outputs, inputs, group=GROUP); return torch.cat(outputs, dim=gather_dim).contiguous()


def a2a_fp8(x, scatter_dim, gather_dim):
    """quantize to e4m3 with one scale per (token, head) vector, all-to-all the bytes and the scales, dequantize to the input dtype"""
    sc = (x.abs().amax(dim=-1, keepdim=True).float() / 448.0).clamp_min(1e-12)
    q8 = (x.float() / sc).to(torch.float8_e4m3fn).view(torch.uint8)
    q8 = a2a(q8, scatter_dim, gather_dim); sc = a2a(sc, scatter_dim, gather_dim)
    return (q8.view(torch.float8_e4m3fn).float() * sc).to(x.dtype)


def gather_seq(x, dim=1):
    p = P(); out = [torch.empty_like(x) for _ in range(p)]; dist.all_gather(out, x.contiguous(), group=GROUP); return torch.cat(out, dim=dim)


def local_head_caches(B):
    """re-create the binding's caches with num_heads / P heads in the self-attention KV (cross-attention cache keeps all heads)"""
    p = P(); m = B.target_model.config; hd = m.dim // m.num_heads; pipe = B.pipe
    skv = [1, B._kv_size, m.num_heads // p, hd]; ckv = [1, 512, m.num_heads, hd]
    B._mk_cache = lambda: dict(kv=pipe._initialize_self_kv_cache(num_layers=m.num_layers, shape=skv, dtype=B.param_dtype, device=B.device),
                               cross=pipe._initialize_crossattn_cache(num_layers=m.num_layers, shape=ckv, dtype=B.param_dtype, device=B.device))
    B.reset_caches()


def apply_sp():
    import wan.modules.model_fast as M
    from wan.modules.model import sinusoidal_embedding_1d
    from einops import rearrange
    import torch.nn.functional as F

    def sa_forward_sp(self, x, seq_lens, grid_sizes, freqs, kv_cache=None, current_start=0, max_attention_size=1_000_000, frame_seqlen=None, seq_lens_int=None):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim; p = P()
        q = self.norm_q(self.q(x)).view(b, s, n, d); k = self.norm_k(self.k(x)).view(b, s, n, d); v = self.v(x).view(b, s, n, d)
        comm = a2a_fp8 if COMM_FP8 else a2a
        if FUSED:   # stack on a new axis after the heads so the head scatter splits each of q, k, v
            qkv = comm(torch.stack([q, k, v], dim=3), 2, 1); q, k, v = qkv.unbind(3)   # [b, s*p, n/p, d] each
        else: q, k, v = comm(q, 2, 1), comm(k, 2, 1), comm(v, 2, 1)
        L = seq_lens_int; csf = current_start // frame_seqlen
        rq = M.causal_rope_apply(q, grid_sizes, freqs, start_frame=csf).type_as(v)[:, :L]
        rk = M.causal_rope_apply(k, grid_sizes, freqs, start_frame=csf).type_as(v)[:, :L]; vv = v[:, :L]
        size = kv_cache["k"].shape[1]; sink = self.sink_size * frame_seqlen; c = current_start // L; sink_c = sink // L; slots = (size - sink) // L
        off = c * L if c < sink_c + slots else sink + ((c - sink_c) % slots) * L
        kv_cache["k"][:, off:off + L] = rk.detach(); kv_cache["v"][:, off:off + L] = vv.detach(); valid = min(size, (c + 1) * L)
        o = M.attention(rq, kv_cache["k"][:, :valid], kv_cache["v"][:, :valid])          # [b, L, n/p, d]
        if s * p > L: o = torch.cat([o, o.new_zeros(b, s * p - L, o.size(2), d)], 1)
        o = (a2a_fp8 if COMM_FP8 else a2a)(o, 1, 2)                                     # [b, s, n, d]
        return self.o(o.flatten(2))
    M.CausalWanSelfAttention.forward = sa_forward_sp

    def wm_forward_sp(self, x, t, context, seq_len, y=None, dit_cond_dict=None, kv_cache=None, crossattn_cache=None, current_start=0,
                      max_attention_size=1_000_000, frame_seqlen=None, cross_attn_first_call=None):
        p, r = P(), R(); device = self.patch_embedding.weight.device
        if self.freqs.device != device: self.freqs = self.freqs.to(device)
        if y is not None: x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = torch.cat([u.flatten(2).transpose(1, 2) for u in x]); L = x.size(1); Lp = ((L + p - 1) // p) * p
        if Lp > L: x = torch.cat([x, x.new_zeros(x.size(0), Lp - L, x.size(2))], 1)
        tt = t.view(-1, 1) if t.dim() == 1 else t[:, :1]
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, tt.flatten()).unflatten(0, (tt.size(0), 1)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        context = self.text_embedding(torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]))
        if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
            ce = [rearrange(i, '1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)', c1=self.patch_size[0], c2=self.patch_size[1], c3=self.patch_size[2])
                  for i in dit_cond_dict["c2ws_plucker_emb"]]
            ce = self.patch_embedding_wancamctrl(torch.cat(ce, dim=1)); ce = ce + self.c2ws_hidden_states_layer2(F.silu(self.c2ws_hidden_states_layer1(ce)))
            if ce.size(1) < Lp: ce = torch.cat([ce, ce.new_zeros(ce.size(0), Lp - ce.size(1), ce.size(2))], 1)
            dit_cond_dict = dict(dit_cond_dict); dit_cond_dict["c2ws_plucker_emb"] = ce[:, :Lp].chunk(p, dim=1)[r]
        x = x.chunk(p, dim=1)[r]
        kwargs = dict(e=e0, seq_lens=torch.tensor([L]), grid_sizes=grid_sizes, freqs=self.freqs, context=context, context_lens=None, dit_cond_dict=dit_cond_dict,
                      max_attention_size=max_attention_size, frame_seqlen=frame_seqlen, cross_attn_first_call=cross_attn_first_call, seq_lens_int=L)
        for bi, block in enumerate(self.blocks):
            kwargs.update(kv_cache=kv_cache[bi], crossattn_cache=crossattn_cache[bi], current_start=current_start)
            x = block(x, **kwargs)
        x = self.head(x, e)
        x = gather_seq(x, 1)[:, :L]
        return [u.float() for u in self.unpatchify(x, grid_sizes)]
    M.WanModelFast.forward = wm_forward_sp
    print("[fast_sp] Ulysses SP patches applied, SP size %d" % P(), flush=True)
