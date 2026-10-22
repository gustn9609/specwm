"""Better 4-bit KV storage: the fixes the naive store was missing.

The naive store (kvstore.py) scales over the head dimension for BOTH K and V,
i.e. per-token symmetric quantization. Visually that arm degrades noticeably,
and there are three known reasons:

1. K AND V HAVE DIFFERENT OUTLIER STRUCTURE. Key caches carry strong
   PER-CHANNEL outliers; a per-token scale is set by whichever channel spikes,
   so the remaining channels use only a few of the 15 levels. Values are much
   more uniform and per-token is fine for them. (This is the central
   observation of KIVI / KVQuant.)  ->  scale K along the token axis within a
   block (per channel), keep V per-token.
   Bonus: per-channel-per-block scales are also CHEAPER to store than
   per-token ones here (6*40*128 vs 28080*40 values).

2. ALL TOKENS ARE TREATED EQUALLY. This model keeps `sink_size` frames plus a
   rolling window, and attention mass concentrates on the sinks and the most
   recent frames, so those deserve more bits than the bulk.

3. SYMMETRIC QUANTIZATION ASSUMES A ZERO-CENTRED DISTRIBUTION. KV activations
   are not; an asymmetric (zero-point) grid uses the range far better.

Each fix is switchable so the ablation can attribute the gain.
"""
from __future__ import annotations
import torch


class KVStore2:
    """Packed low-bit K or V with selectable quantization strategy.

    axis="token"   : scale over head_dim, one per (b,t,h)      [naive]
    axis="channel" : scale over tokens within a block, one per (b,blk,h,d)
    asym=True      : store zero-point as well as scale
    hp_tail        : keep this many most-recent tokens at 8 bit
    """

    def __init__(self, B, T, H, D, bits=4, axis="token", asym=False,
                 block=4680, hp_tail=0, group=0, device="cuda"):
        assert bits in (4, 8) and D % 2 == 0
        self.B, self.T, self.H, self.D = B, T, H, D
        self.bits, self.axis, self.asym = bits, axis, asym
        self.block, self.hp_tail = max(block, 1), hp_tail
        # group>0 splits head_dim into sub-groups, so a token gets several
        # scales instead of one. Scales still travel WITH the token, so unlike
        # the per-block channel layout this survives sliding-window eviction
        # without a requantize.
        self.group = group if group and D % group == 0 else 0
        self.ng = (D // self.group) if self.group else 1
        self.per_row = D if bits == 8 else D // 2
        self.packed = torch.zeros(B, T, H, self.per_row, dtype=torch.uint8, device=device)
        nb = (T + self.block - 1) // self.block
        if axis == "channel":
            self.scale = torch.zeros(B, nb, H, D, dtype=torch.bfloat16, device=device)
            self.zero = torch.zeros(B, nb, H, D, dtype=torch.bfloat16, device=device) if asym else None
        elif axis == "channel_frozen":
            # one scale per channel for the WHOLE cache, calibrated on the first
            # write and then frozen: nothing depends on token position, so
            # eviction is a pure byte move.
            self.scale = torch.zeros(B, 1, H, D, dtype=torch.bfloat16, device=device)
            self.zero = torch.zeros(B, 1, H, D, dtype=torch.bfloat16, device=device) if asym else None
            self.frozen = False
        else:
            self.scale = torch.zeros(B, T, H, self.ng, dtype=torch.bfloat16, device=device)
            self.zero = torch.zeros(B, T, H, self.ng, dtype=torch.bfloat16, device=device) if asym else None
        # high-precision tail kept in bf16
        self.tail = (torch.zeros(B, hp_tail, H, D, dtype=torch.bfloat16, device=device)
                     if hp_tail > 0 else None)
        self.tail_lo = -1
        self.qmax = (2 ** (bits - 1)) - 1
        self.qlev = (2 ** bits) - 1

    def nbytes(self):
        t = self.packed.numel() * self.packed.element_size()
        t += self.scale.numel() * self.scale.element_size()
        if self.zero is not None:
            t += self.zero.numel() * self.zero.element_size()
        if self.tail is not None:
            t += self.tail.numel() * self.tail.element_size()
        return t

    # ---------------- helpers ----------------
    def _qparams(self, xf, dim, keepdim=True):
        if self.asym:
            mx = xf.amax(dim, keepdim=keepdim); mn = xf.amin(dim, keepdim=keepdim)
            sc = ((mx - mn) / self.qlev).clamp(min=1e-8)
            zp = mn
            return sc, zp
        sc = (xf.abs().amax(dim, keepdim=keepdim) / self.qmax).clamp(min=1e-8)
        return sc, None

    def _pack(self, q):
        q = q.to(torch.uint8)
        if self.bits == 8:
            return q
        return q[..., 0::2] | (q[..., 1::2] << 4)

    def _unpack(self, p):
        if self.bits == 8:
            return p.to(torch.int16)
        lo = (p & 0x0F).to(torch.int16)
        hi = ((p >> 4) & 0x0F).to(torch.int16)
        return torch.stack((lo, hi), dim=-1).reshape(*p.shape[:-1], self.D)

    # ---------------- write ----------------
    @torch.no_grad()
    def write(self, s, e, x):
        xf = x.float()
        if self.axis == "channel":
            blk = s // self.block
            sc, zp = self._qparams(xf, dim=1)            # over tokens
            if self.asym:
                q = ((xf - zp) / sc).round().clamp(0, self.qlev)
            else:
                q = (xf / sc).round().clamp(-self.qmax, self.qmax) + (self.qmax + 1)
            self.scale[:, blk] = sc.squeeze(1).to(torch.bfloat16)
            if self.asym:
                self.zero[:, blk] = zp.squeeze(1).to(torch.bfloat16)
        elif self.axis == "channel_frozen":
            if not self.frozen:
                sc, zp = self._qparams(xf, dim=1)        # calibrate on first block
                self.scale[:, 0] = sc.squeeze(1).to(torch.bfloat16)
                if self.asym:
                    self.zero[:, 0] = zp.squeeze(1).to(torch.bfloat16)
                self.frozen = True
            sc = self.scale[:, 0].unsqueeze(1).float()
            if self.asym:
                zp = self.zero[:, 0].unsqueeze(1).float()
                q = ((xf - zp) / sc).round().clamp(0, self.qlev)
            else:
                q = (xf / sc).round().clamp(-self.qmax, self.qmax) + (self.qmax + 1)
        else:
            xg = xf.reshape(*xf.shape[:-1], self.ng, -1) if self.group else xf.unsqueeze(-2)
            sc, zp = self._qparams(xg, dim=-1)           # over (sub-)head_dim
            if self.asym:
                q = ((xg - zp) / sc).round().clamp(0, self.qlev)
            else:
                q = (xg / sc).round().clamp(-self.qmax, self.qmax) + (self.qmax + 1)
            q = q.reshape(*xf.shape)
            self.scale[:, s:e] = sc.squeeze(-1).to(torch.bfloat16)
            if self.asym:
                self.zero[:, s:e] = zp.squeeze(-1).to(torch.bfloat16)
        self.packed[:, s:e] = self._pack(q)
        if self.tail is not None:
            n = min(self.hp_tail, e - s)
            self.tail[:, :n] = x[:, e - s - n:].to(torch.bfloat16)
            self.tail_lo = e - n

    # ---------------- read ----------------
    @torch.no_grad()
    def read(self, s, e, dtype=torch.bfloat16):
        q = self._unpack(self.packed[:, s:e]).to(torch.float32)
        if self.axis == "channel":
            blk = s // self.block
            sc = self.scale[:, blk].unsqueeze(1).float()
            if self.asym:
                out = q * sc + self.zero[:, blk].unsqueeze(1).float()
            else:
                out = (q - (self.qmax + 1)) * sc
        elif self.axis == "channel_frozen":
            sc = self.scale[:, 0].unsqueeze(1).float()
            if self.asym:
                out = q * sc + self.zero[:, 0].unsqueeze(1).float()
            else:
                out = (q - (self.qmax + 1)) * sc
        else:
            qg = q.reshape(*q.shape[:-1], self.ng, -1) if self.group else q.unsqueeze(-2)
            sc = self.scale[:, s:e].float().unsqueeze(-1)
            if self.asym:
                out = (qg * sc + self.zero[:, s:e].float().unsqueeze(-1))
            else:
                out = (qg - (self.qmax + 1)) * sc
            out = out.reshape(*q.shape)
        out = out.to(dtype)
        if self.tail is not None and self.tail_lo >= 0:
            lo = max(s, self.tail_lo)
            if lo < e:
                out[:, lo - s:e - s] = self.tail[:, :e - lo].to(dtype)
        return out

    @torch.no_grad()
    def roll(self, dst, src, n):
        """Sliding-window eviction.

        BUG THIS FIXES. Per-token scales travel with their token, so moving the
        packed bytes is enough. Per-CHANNEL scales are stored per BLOCK of
        token positions, so once tokens slide the token->block mapping shifts
        and the moved data gets dequantized with a different block's scales.
        Measured cost of getting this wrong: PSNR 9.98, i.e. below the 13.73
        noise floor, versus 15.72 for the naive store -- it looked like the
        per-channel idea had failed when in fact the scales were mismatched.

        Correct move for the channel layout is therefore a dequantize at the
        source followed by a requantize at the destination, which re-derives
        the scales for the block the data now lives in. It costs one extra
        pass over the window per eviction (once per chunk), not per step.
        """
        if n <= 0:
            return
        if self.axis == "channel_frozen":
            self.packed[:, dst:dst + n] = self.packed[:, src:src + n].clone()
            return
        if self.axis != "channel":
            self.packed[:, dst:dst + n] = self.packed[:, src:src + n].clone()
            self.scale[:, dst:dst + n] = self.scale[:, src:src + n].clone()
            if self.zero is not None:
                self.zero[:, dst:dst + n] = self.zero[:, src:src + n].clone()
            return
        moved = self.read(src, src + n, dtype=torch.float32).clone()
        step = self.block
        off = 0
        while off < n:
            m = min(step - ((dst + off) % step), n - off)
            self.write(dst + off, dst + off + m, moved[:, off:off + m])
            off += m


def install(model, bits=4, hp_tail=0, k_axis="channel", v_axis="token",
            asym=True, group=0, verbose=True):
    """Install KVStore2 with DIFFERENT strategies for K and V.

    Measured on synthetic data with the channel-outlier structure real Key
    caches show (relative reconstruction error, 4-bit):

        Key    per-token sym 0.286 -> per-channel asym 0.141   (-51%)
        Value  per-token sym 0.117 -> per-token   asym 0.100   (-15%)
                                      per-channel asym 0.142   (WORSE)

    so K and V must not share a strategy -- which the first store did.
    """
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
        m._cfg2 = dict(bits=bits, hp_tail=hp_tail, k_axis=k_axis,
                       v_axis=v_axis, asym=asym, group=group)
        m._store2 = {}
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

            st, cfg = self._store2, self._cfg2
            if st.get("_id") != id(kv_cache):
                st.clear(); st["_id"] = id(kv_cache)
                T = (kv_cache["k"].shape[1] if kv_cache["k"].numel()
                     else kv_cache["kv_shape"][1])   # lowbit_alloc 지연할당 대응
                st["size"] = T; st["gei"] = 0; st["lei"] = 0
                blk = max(rq.shape[1], 1)
                st["k"] = KVStore2(b, T, nh, d, bits=cfg["bits"], axis=cfg["k_axis"],
                                   asym=cfg["asym"], block=blk, group=cfg["group"],
                                   hp_tail=cfg["hp_tail"], device=rk.device)
                st["v"] = KVStore2(b, T, nh, d, bits=cfg["bits"], axis=cfg["v_axis"],
                                   asym=cfg["asym"], block=blk, group=cfg["group"],
                                   hp_tail=cfg["hp_tail"], device=rk.device)
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
                ev = num_new + lei - csize
                rolled = lei - ev - sink_tokens
                K.roll(sink_tokens, sink_tokens + ev, rolled)
                V.roll(sink_tokens, sink_tokens + ev, rolled)
                hi_i = lei + current_end - gei - ev
                lo_i = hi_i - num_new
            else:
                hi_i = lei + current_end - gei
                lo_i = hi_i - num_new
            K.write(lo_i, hi_i, rk); V.write(lo_i, hi_i, v)
            lo = max(0, hi_i - max_attention_size)
            out = _attn(rq, K.read(lo, hi_i, rq.dtype), V.read(lo, hi_i, rq.dtype))
            st["gei"], st["lei"] = current_end, hi_i
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(hi_i)
            return self.o(out.flatten(2))

        m.forward = fwd.__get__(m, type(m))
    if verbose:
        print(f"[kvstore2] {n} layers | {bits}bit K={k_axis} V={v_axis} "
              f"asym={asym} group={group} hp_tail={hp_tail}", flush=True)
    return n


def uninstall(model, verbose=True):
    n = 0
    for m in model.modules():
        if getattr(m, "_store2", None) is not None:
            try: del m.forward
            except AttributeError: pass
            m._store2 = None
            n += 1
    if verbose:
        print(f"[kvstore2] uninstalled from {n} layers", flush=True)
    return n


def total_bytes(model):
    t = 0
    for m in model.modules():
        st = getattr(m, "_store2", None)
        if st:
            for key in ("k", "v"):
                if key in st:
                    t += st[key].nbytes()
    return t
