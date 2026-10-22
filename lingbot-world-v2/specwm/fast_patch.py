import os
"""Runtime-toggled speed patches for wan.modules.model_fast (the LingBot v2 causal-fast DiT). Call apply() once after sdpa_patch.apply();
then flip FLAGS between runs (no reload needed).
  dedup_t    : timestep embedding computed once per forward and broadcast (was: per token, fp32 [L, 6, dim]) -> same values
  no_kv_cat  : attend directly on the KV-cache slice that already holds the new keys (was: torch.cat of the whole window per layer) -> same values
  rope_cache : RoPE frequency table per (start_frame, f, h, w) cached (still float64) -> same values
  rope_fp32  : RoPE in complex64 instead of complex128 -> ~1e-7 relative change
  attn       : 'flash' (default FA2 via SDPA) | 'cudnn' | 'efficient' | 'sage' -> backend for self (and cross) attention
  ring_kv    : the local part of the KV window is a chunk-aligned ring buffer (no shift copy on eviction) and the write position is a Python int
               (no .item() host syncs). Attention over a key set does not depend on key order -> same math (summation order differs)."""
import math, torch
FLAGS = dict(dedup_t=False, no_kv_cat=False, rope_cache=False, rope_fp32=False, attn="flash", ring_kv=False, ring_frame=False)   # ring_frame (10/13): frame-granular ring slots (variable chunk length); identical layout to ring_kv for fixed 3-frame chunks
# ACTIVE: token-wise early exit (Speculative Forcing). None = dense forward. Else dict(idx=LongTensor[La_pad] token indices of the chunk (padded by
# repeating the last index to a bucket size), n=La real count, L=full chunk token count). Only these tokens go through the blocks and the head; their
# K/V overwrite their own slots in the ring cache, committed tokens keep the K/V of their commit step; the head output is scattered into a zero
# full-size tensor (callers overwrite committed tokens with their frozen x0). Requires dedup_t + no_kv_cat + ring_kv.
ACTIVE = {"a": None}
_ROPE = {}


def _freqs_i(freqs, start_frame, f, h, w, c, dtype):
    key = (start_frame, f, h, w, dtype, freqs.data_ptr())
    if FLAGS["rope_cache"] and key in _ROPE: return _ROPE[key]
    fs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    fi = torch.cat([fs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    fs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    fs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1).reshape(f * h * w, 1, -1).to(dtype)
    if FLAGS["rope_cache"]: _ROPE[key] = fi
    return fi


def ring_pos(f0, m, fs, size, sink_f, device):
    """token slots of latent frames f0..f0+m-1 in the frame-granular ring (FLAGS ring_frame): [m*fs] long tensor, and the number of valid slots"""
    slots_f = (size - sink_f * fs) // fs; fr = torch.arange(f0, f0 + m, device=device)
    offs = torch.where(fr < sink_f + slots_f, fr * fs, sink_f * fs + ((fr - sink_f) % slots_f) * fs)
    return (offs[:, None] + torch.arange(fs, device=device)[None]).reshape(-1), min(size, (f0 + m) * fs)


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2
    fdt, cdt = (torch.float32, torch.complex64) if FLAGS["rope_fp32"] else (torch.float64, torch.complex128)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(fdt).reshape(seq_len, n, -1, 2))
        x_i = torch.view_as_real(x_i * _freqs_i(freqs, start_frame, f, h, w, c, cdt)).flatten(2)
        output.append(torch.cat([x_i, x[i, seq_len:]]) if seq_len < x.size(1) else x_i)
    return torch.stack(output).type_as(x)


def _sdpa(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None, q_scale=None, causal=False,
          window_size=(-1, -1), deterministic=False, dtype=torch.bfloat16, version=None, fa_version=None):
    from torch.nn.attention import sdpa_kernel, SDPBackend
    half = lambda x: x if x.dtype in (torch.float16, torch.bfloat16) else x.to(dtype)
    q, k, v = half(q), half(k), half(v)
    if q_scale is not None: q = q * q_scale
    be = {"flash": [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
          "cudnn": [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION], "efficient": [SDPBackend.EFFICIENT_ATTENTION],
          "sage": [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]}[FLAGS["attn"]]
    # SageAttention gives 3-7 % different outputs when the query set is a strict subset of the keys (qo_len != kv_len; 10/7 check: SDPA flash 1e-4),
    # so active-token (early-exit) forwards use flash attention.
    if FLAGS["attn"] == "sage" and k.shape[1] > 2048 and (ACTIVE["a"] is None or os.environ.get("SPECWM_SAGE_SPARSE") == "1"):   # self-attention over the KV window (cross-attention to 512 text tokens stays on SDPA)
        from sageattention import sageattn
        return sageattn(q.contiguous(), k.contiguous(), v.contiguous(), tensor_layout="NHD", is_causal=causal, sm_scale=softmax_scale)
    with sdpa_kernel(be):
        out = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                                               is_causal=causal, dropout_p=dropout_p, scale=softmax_scale)
    return out.transpose(1, 2).contiguous()


def apply():
    import sys
    import wan.modules.model_fast as M
    import wan.modules.attention as A
    M.causal_rope_apply = causal_rope_apply
    A.flash_attention = _sdpa
    for name, mod in list(sys.modules.items()):
        if name.startswith("wan") and mod is not None and getattr(mod, "flash_attention", None) is not None and mod is not A:
            mod.flash_attention = _sdpa
    if hasattr(A, "attention"):
        _orig_attention = A.attention
        def attention(q, k, v, *a, **kw): return _sdpa(q, k, v, *a, **kw)
        A.attention = attention; M.attention = attention

    SA = M.CausalWanSelfAttention; _sa_fwd = SA.forward
    def sa_forward(self, x, seq_lens, grid_sizes, freqs, kv_cache=None, current_start=0, max_attention_size=1_000_000, frame_seqlen=None, seq_lens_int=None):
        if not FLAGS["no_kv_cat"] or getattr(self, "mem", None) is not None or self.local_attn_size == -1:
            return _sa_fwd(self, x, seq_lens, grid_sizes, freqs, kv_cache, current_start, max_attention_size, frame_seqlen, seq_lens_int)
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d); k = self.norm_k(self.k(x)).view(b, s, n, d); v = self.v(x).view(b, s, n, d)
        if frame_seqlen is None: frame_seqlen = math.prod(grid_sizes[0][1:]).item()
        if FLAGS["ring_kv"]:
            csf = current_start // frame_seqlen; act = seq_lens_int if isinstance(seq_lens_int, dict) else None   # active-token subset is passed through seq_lens_int (a traced input, not a global)
            if act is not None:   # subset of the chunk's tokens: RoPE rows and cache slots are gathered by token index
                f_, h_, w_ = grid_sizes[0].tolist(); cdt = torch.complex64 if FLAGS["rope_fp32"] else torch.complex128; fdt = torch.float32 if FLAGS["rope_fp32"] else torch.float64
                fr = _freqs_i(freqs, csf, f_, h_, w_, d // 2, cdt)[act["idx"]]   # [La, 1, d/2]
                rope = lambda t_: torch.view_as_real(torch.view_as_complex(t_[0].to(fdt).reshape(s, n, -1, 2)) * fr).flatten(2).unsqueeze(0).type_as(v)
                rq, rk = rope(q), rope(k); S = act["L"]
            else:
                rq = M.causal_rope_apply(q, grid_sizes, freqs, start_frame=csf).type_as(v); rk = M.causal_rope_apply(k, grid_sizes, freqs, start_frame=csf).type_as(v); S = s
            size = kv_cache["k"].shape[1]; sink = self.sink_size * frame_seqlen
            if FLAGS["ring_frame"]:   # slot per latent frame: frame f -> f*fs while f < sink_f + slots_f, else sink + ((f - sink_f) % slots_f) * fs
                pos, valid = ring_pos(current_start // frame_seqlen, S // frame_seqlen, frame_seqlen, size, sink // frame_seqlen, x.device)
                if act is not None: kv_cache["k"][:, pos[act["idx"]]] = rk.detach(); kv_cache["v"][:, pos[act["idx"]]] = v.detach()
                else: kv_cache["k"][:, pos] = rk.detach(); kv_cache["v"][:, pos] = v.detach()
                out = M.attention(rq, kv_cache["k"][:, :valid], kv_cache["v"][:, :valid]); return self.o(out.flatten(2))
            c = current_start // S; sink_c = sink // S; slots = (size - sink) // S
            assert size % S == 0 and sink % S == 0 and current_start % S == 0, "ring_kv needs chunk-aligned window/sink"
            off = c * S if c < sink_c + slots else sink + ((c - sink_c) % slots) * S; valid = min(size, (c + 1) * S)
            if act is not None: kv_cache["k"][:, off + act["idx"]] = rk.detach(); kv_cache["v"][:, off + act["idx"]] = v.detach()
            else: kv_cache["k"][:, off:off + s] = rk.detach(); kv_cache["v"][:, off:off + s] = v.detach()
            out = M.attention(rq, kv_cache["k"][:, :valid], kv_cache["v"][:, :valid])
            return self.o(out.flatten(2))
        csf = current_start // frame_seqlen
        rq = M.causal_rope_apply(q, grid_sizes, freqs, start_frame=csf).type_as(v); rk = M.causal_rope_apply(k, grid_sizes, freqs, start_frame=csf).type_as(v)
        current_end = current_start + rq.shape[1]; sink_tokens = self.sink_size * frame_seqlen; kv_cache_size = kv_cache["k"].shape[1]; nn_ = rq.shape[1]
        ge, le = kv_cache["global_end_index"].item(), kv_cache["local_end_index"].item()
        if current_end > ge and nn_ + le > kv_cache_size:
            ne = nn_ + le - kv_cache_size; nr = le - ne - sink_tokens
            kv_cache["k"][:, sink_tokens:sink_tokens + nr] = kv_cache["k"][:, sink_tokens + ne:sink_tokens + ne + nr].clone()
            kv_cache["v"][:, sink_tokens:sink_tokens + nr] = kv_cache["v"][:, sink_tokens + ne:sink_tokens + ne + nr].clone()
            lei = le + current_end - ge - ne
        else:
            lei = le + current_end - ge
        lsi = lei - nn_
        kv_cache["k"][:, lsi:lei] = rk.detach(); kv_cache["v"][:, lsi:lei] = v.detach()
        st = max(0, lei - max_attention_size)
        out = M.attention(rq, kv_cache["k"][:, st:lei], kv_cache["v"][:, st:lei])
        kv_cache["global_end_index"].fill_(current_end); kv_cache["local_end_index"].fill_(lei)
        return self.o(out.flatten(2))
    SA.forward = sa_forward

    WM = M.WanModelFast; _wm_fwd = WM.forward
    def wm_forward(self, x, t, context, seq_len, **kw):
        if FLAGS["dedup_t"] and t.dim() == 1:
            t = t.view(-1, 1)   # [B, 1]: one timestep row, broadcast over tokens inside the blocks and head
            return _wm_fwd_dedup(self, x, t, context, seq_len, **kw)
        return _wm_fwd(self, x, t, context, seq_len, **kw)
    WM.forward = wm_forward

    from wan.modules.model import sinusoidal_embedding_1d
    from einops import rearrange
    import torch.nn.functional as F
    def _wm_fwd_dedup(self, x, t, context, seq_len, y=None, dit_cond_dict=None, kv_cache=None, crossattn_cache=None, current_start=0,
                      max_attention_size=1_000_000, frame_seqlen=None, cross_attn_first_call=None):
        """copy of WanModelFast.forward with t of shape [B, 1] (time embedding for one token, broadcast)."""
        device = self.patch_embedding.weight.device
        if self.freqs.device != device: self.freqs = self.freqs.to(device)
        if y is not None: x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long); assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).unflatten(0, (bt, 1)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        context = self.text_embedding(torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]))
        if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
            ce = [rearrange(i, '1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)', c1=self.patch_size[0], c2=self.patch_size[1], c3=self.patch_size[2])
                  for i in dit_cond_dict["c2ws_plucker_emb"]]
            ce = self.patch_embedding_wancamctrl(torch.cat(ce, dim=1))
            ch = self.c2ws_hidden_states_layer2(F.silu(self.c2ws_hidden_states_layer1(ce)))
            dit_cond_dict = dict(dit_cond_dict); dit_cond_dict["c2ws_plucker_emb"] = ce + ch
        act = ACTIVE["a"]; L_full = x.size(1)
        if act is not None:   # only the active tokens are forwarded
            x = x[:, act["idx"]]
            if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict: dit_cond_dict["c2ws_plucker_emb"] = dit_cond_dict["c2ws_plucker_emb"][:, act["idx"]]
        kwargs = dict(e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=self.freqs, context=context, context_lens=None, dit_cond_dict=dit_cond_dict,
                      max_attention_size=max_attention_size, frame_seqlen=frame_seqlen, cross_attn_first_call=cross_attn_first_call, seq_lens_int=act)
        for bi, block in enumerate(self.blocks):
            kwargs.update(kv_cache=kv_cache[bi], crossattn_cache=crossattn_cache[bi], current_start=current_start)
            x = block(x, **kwargs)
        x = self.head(x, e)
        if act is not None:
            full = x.new_zeros(x.size(0), L_full, x.size(2)); full[:, act["idx"][:act["n"]]] = x[:, :act["n"]]; x = full
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]
    print("[fast_patch] applied (flags off by default)", flush=True)


class FP8Linear(torch.nn.Module):
    """W8A8 e4m3 linear: per-output-channel weight scales (static), per-token activation scales (dynamic), torch._scaled_mm, bf16 out.
    Keeps a reference to the original bf16 Linear so it can be swapped back (FLAGS-free; use fp8_swap(model, on))."""
    def __init__(self, lin):
        super().__init__(); w = lin.weight.data.to(torch.bfloat16)
        sw = (w.abs().amax(dim=1, keepdim=True).float() / 448.0).clamp_min(1e-12)
        self.w8 = (w.float() / sw).to(torch.float8_e4m3fn); self.sw = sw.t().contiguous()   # [1, out]
        self.bias = None if lin.bias is None else lin.bias.data.to(torch.bfloat16); self.inf, self.outf = lin.in_features, lin.out_features
    def forward(self, x):
        sh = x.shape; x2 = x.reshape(-1, sh[-1]).to(torch.bfloat16)
        sa = (x2.abs().amax(dim=1, keepdim=True).float() / 448.0).clamp_min(1e-12)
        x8 = (x2.float() / sa).to(torch.float8_e4m3fn)
        y = torch._scaled_mm(x8, self.w8.t(), scale_a=sa, scale_b=self.sw, out_dtype=torch.bfloat16)
        if self.bias is not None: y = y + self.bias
        return y.reshape(*sh[:-1], self.outf)


def fp8_swap(model, on, keep=True):
    """replace every nn.Linear inside model.blocks (attention, FFN, camera injectors) by FP8Linear (on=True) or restore the bf16 originals (on=False)."""
    if not hasattr(model, "_fp8_store"): model._fp8_store = {}
    for name, mod in list(model.blocks.named_modules()):
        for cn, ch in list(mod.named_children()):
            full = "%s.%s" % (name, cn) if name else cn
            if on and isinstance(ch, torch.nn.Linear) and ch.in_features % 16 == 0 and ch.out_features % 16 == 0:
                if keep: model._fp8_store[full] = ch
                setattr(mod, cn, FP8Linear(ch))
            elif not on and isinstance(ch, FP8Linear) and full in model._fp8_store:
                setattr(mod, cn, model._fp8_store.pop(full))
    if on and not keep: import gc; gc.collect(); torch.cuda.empty_cache()
    return model
