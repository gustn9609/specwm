"""Really-compressed KV cache storage (the `store` mode kvquant.py left open).

WHY THIS IS THE BLOCKING ITEM. With INT4 weights the measured generation peak
is 33.2 GiB, of which the KV cache is 21.3 GiB -- 9 GiB over a 24 GB card. Our
alpha-KV work so far was FAKE quantization: it quantized and immediately
dequantized in place, so it measured the quality cost of low-bit KV while
saving exactly zero bytes.

THE DESIGN POINT THAT MAKES THIS WORK. A naive "store compressed, dequantize
before attention" implementation saves nothing, because expanding the whole
window materialises the very bf16 tensor we were trying to avoid. But the cache
is PER LAYER and a forward touches one layer at a time, so we keep all 40
layers packed and expand only the layer in use:

    persistent : 40 layers packed      21.3 GiB -> ~5.5 GiB (4-bit)
    transient  : one expanded layer    ~0.53 GiB

Quantization is symmetric per (batch, token, head) with the scale taken over
the head dimension -- the standard KV granularity, one bf16 scale per D values.
"""
from __future__ import annotations
import torch


class QuantKVStore:
    """Packed low-bit store for one layer's K or V: logical shape [B, T, H, D]."""

    def __init__(self, B, T, H, D, bits=4, device="cuda"):
        assert bits in (4, 8) and D % 2 == 0
        self.B, self.T, self.H, self.D, self.bits = B, T, H, D, bits
        self.per_row = D if bits == 8 else D // 2
        self.packed = torch.zeros(B, T, H, self.per_row, dtype=torch.uint8, device=device)
        self.scale = torch.zeros(B, T, H, 1, dtype=torch.bfloat16, device=device)
        self.qmax = 127 if bits == 8 else 7

    def nbytes(self):
        return (self.packed.numel() * self.packed.element_size()
                + self.scale.numel() * self.scale.element_size())

    @torch.no_grad()
    def write(self, s, e, x):
        xf = x.float()
        sc = (xf.abs().amax(-1, keepdim=True) / self.qmax).clamp(min=1e-8)
        q = torch.clamp((xf / sc).round(), -self.qmax, self.qmax).to(torch.int16)
        if self.bits == 8:
            self.packed[:, s:e] = (q + 128).to(torch.uint8)
        else:
            q = (q + 8).to(torch.uint8)
            self.packed[:, s:e] = (q[..., 0::2] | (q[..., 1::2] << 4))
        self.scale[:, s:e] = sc.to(torch.bfloat16)

    @torch.no_grad()
    def read(self, s, e, dtype=torch.bfloat16):
        p = self.packed[:, s:e]
        if self.bits == 8:
            q = p.to(torch.int16) - 128
        else:
            lo = (p & 0x0F).to(torch.int16) - 8
            hi = ((p >> 4) & 0x0F).to(torch.int16) - 8
            q = torch.stack((lo, hi), dim=-1).reshape(*p.shape[:-1], self.D)
        return q.to(dtype) * self.scale[:, s:e].to(dtype)

    @torch.no_grad()
    def roll(self, dst, src, n):
        if n <= 0:
            return
        self.packed[:, dst:dst + n] = self.packed[:, src:src + n].clone()
        self.scale[:, dst:dst + n] = self.scale[:, src:src + n].clone()


def install(model, bits=4, verbose=True):
    """Route every attention layer's KV through a packed store."""
    import sys, math

    NEED = ("q", "k", "v", "o", "norm_q", "norm_k",
            "local_attn_size", "sink_size", "num_heads", "head_dim")
    n = 0
    for m in model.modules():
        if not all(hasattr(m, x) for x in NEED):
            continue
        mod = sys.modules[type(m).__module__]
        if not (hasattr(mod, "causal_rope_apply") and hasattr(mod, "attention")):
            continue
        m._kvbits = bits
        m._store = {}
        n += 1

        def fwd(self, x, seq_lens, grid_sizes, freqs, kv_cache=None,
                current_start=0, max_attention_size=1_000_000,
                frame_seqlen=None, seq_lens_int=None,
                _rope=mod.causal_rope_apply, _attn=mod.attention):
            del seq_lens_int
            b, s, nh, d = *x.shape[:2], self.num_heads, self.head_dim
            q = self.norm_q(self.q(x)).view(b, s, nh, d)
            k = self.norm_k(self.k(x)).view(b, s, nh, d)
            v = self.v(x).view(b, s, nh, d)
            if frame_seqlen is None:
                frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            csf = current_start // frame_seqlen
            rq = _rope(q, grid_sizes, freqs, start_frame=csf).type_as(v)
            rk = _rope(k, grid_sizes, freqs, start_frame=csf).type_as(v)

            st = self._store
            # NOTE: `is not` would compare int OBJECT identity here; id() returns a
            # large int, so equal values are distinct objects and the test was
            # always true -> re-init every forward -> read of an emptied tensor.
            if st.get("_id") != id(kv_cache):
                st.clear(); st["_id"] = id(kv_cache)
                T = kv_cache["k"].shape[1]
                st["size"] = T; st["gei"] = 0; st["lei"] = 0
                st["k"] = QuantKVStore(b, T, nh, d, bits=self._kvbits, device=rk.device)
                st["v"] = QuantKVStore(b, T, nh, d, bits=self._kvbits, device=rk.device)
                kv_cache["k"] = torch.empty(0, device=rk.device, dtype=rk.dtype)
                kv_cache["v"] = torch.empty(0, device=rk.device, dtype=rk.dtype)
            K, V = st["k"], st["v"]
            gei, lei, csize = st["gei"], st["lei"], st["size"]

            current_end = current_start + rq.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            num_new = rq.shape[1]

            if self.local_attn_size == -1:
                lo_i, hi_i = current_start, current_end
            elif (current_end > gei) and (num_new + lei > csize):
                evicted = num_new + lei - csize
                rolled = lei - evicted - sink_tokens
                K.roll(sink_tokens, sink_tokens + evicted, rolled)
                V.roll(sink_tokens, sink_tokens + evicted, rolled)
                hi_i = lei + current_end - gei - evicted
                lo_i = hi_i - num_new
            else:
                hi_i = lei + current_end - gei
                lo_i = hi_i - num_new
            K.write(lo_i, hi_i, rk)
            V.write(lo_i, hi_i, v)

            lo = max(0, hi_i - max_attention_size)
            out = _attn(rq, K.read(lo, hi_i, rq.dtype), V.read(lo, hi_i, rq.dtype))
            st["gei"], st["lei"] = current_end, hi_i
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(hi_i)
            return self.o(out.flatten(2))

        m.forward = fwd.__get__(m, type(m))
    if verbose:
        print(f"[kvstore] installed {bits}-bit packed KV on {n} attention layers",
              flush=True)
    return n


def total_bytes(model):
    t = 0
    for m in model.modules():
        st = getattr(m, "_store", None)
        if st:  # None after uninstall
            for key in ("k", "v"):
                if key in st:
                    t += st[key].nbytes()
    return t


def uninstall(model, verbose=True):
    """Restore the stock attention forward and drop the packed stores.

    Needed for a one-at-a-time ablation: weight quantization is destructive and
    cannot be undone in process, so the KV-only arm has to run first and then be
    reverted before the weight-only arm. Deleting the per-instance `forward`
    attribute re-exposes the class method; the caller must also call
    reset_caches() so fresh bf16 tensors are allocated.
    """
    n = 0
    for m in model.modules():
        if getattr(m, "_store", None) is not None:
            try:
                del m.forward
            except AttributeError:
                pass
            m._store = None
            if hasattr(m, "_kvbits"):
                del m._kvbits
            n += 1
    if verbose:
        print(f"[kvstore] uninstalled from {n} layers", flush=True)
    return n
