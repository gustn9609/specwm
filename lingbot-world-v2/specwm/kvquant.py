"""alpha-guided KV cache quantization.

Motivation (measured this session): for a world model the KV cache is as large
as the weights (23 GiB vs 35 GiB at 480x832 / local_attn 18) AND it grows with
the interaction horizon -- so it is THE world-model-specific bottleneck.

Existing work (QVG, ICML'26) quantizes the KV uniformly.  Here we allocate
precision using the same closed-form acceptance quantity we use for
speculative verification:

    alpha(d, sigma) = 2 * Phi( -d / (2 sigma) )

Reading it backwards gives an error budget per unit:

    d_max = 2 * sigma * Phi^{-1}(alpha_target / 2)        (per tile)
    per-dim budget  =  d_max / sqrt(D_tile)

so the number of bits needed scales like  log2( sqrt(D)/sigma ).  Concretely
that means: coarse (high-sigma) nodes and OLD frames tolerate few bits, while
recent frames and low-sigma nodes need more.

Two modes:
  * mode="fake"  -- quantize->dequantize in place. No memory saving; used to
                    MEASURE the quality/alpha impact of a bit allocation.
  * mode="store" -- keep an int8/int4 payload + scales and rematerialize a
                    single layer at a time (peak overhead = one layer, not the
                    whole cache).  Real saving; needs the model to read through
                    `KVStore.view(layer)`.
"""
from __future__ import annotations
import math
import torch


def bits_for(alpha_target: float, sigma: float, dim: int,
             floor_bits: int = 2, ceil_bits: int = 8) -> int:
    """Bits needed so that quantization noise keeps per-tile acceptance >= alpha."""
    a = min(max(alpha_target, 1e-6), 1 - 1e-6)
    # inverse of 2*Phi(-d/2s)  ->  d = -2 s * Phi^{-1}(a/2)
    from statistics import NormalDist
    d_max = -2.0 * sigma * NormalDist().inv_cdf(a / 2.0)
    per_dim = d_max / max(math.sqrt(dim), 1.0)
    # uniform quantization noise for range R and b bits ~ R / (2^b * sqrt(12))
    # solve for b with R normalised to 1 (scales are per-group, so R~1)
    if per_dim <= 0:
        return ceil_bits
    b = math.log2(1.0 / (per_dim * math.sqrt(12.0) + 1e-12))
    return int(min(ceil_bits, max(floor_bits, math.ceil(b))))


@torch.no_grad()
def fake_quant_tensor(t: torch.Tensor, bits: int, group: int = 128) -> torch.Tensor:
    """Symmetric per-group quantize->dequantize along the last dim."""
    if bits >= 16:
        return t
    shp = t.shape
    flat = t.reshape(-1, shp[-1]).float()
    g = min(group, shp[-1])
    if shp[-1] % g:
        g = shp[-1]
    x = flat.reshape(flat.shape[0], -1, g)
    qmax = 2 ** (bits - 1) - 1
    s = (x.abs().amax(-1, keepdim=True) / qmax).clamp(min=1e-12)
    x = torch.clamp((x / s).round(), -qmax, qmax) * s
    return x.reshape(flat.shape).reshape(shp).to(t.dtype)


class AlphaKVPolicy:
    """Maps (frame age, node sigma) -> bit-width.

    NOTE (measured): with D ~ 8e6 per frame the closed-form budget saturates at
    ceil_bits for every sigma, i.e. it is a CONSERVATIVE PRIOR, not a usable
    allocation on its own.  The usable signal is (a) age decay and (b) an
    empirically calibrated table alpha(bits) obtained by actually quantizing the
    cache and measuring acceptance -- use `calibrate()` for that.
    """

    def __init__(self, alpha_target=0.9, frame_dim=1560 * 40 * 128,
                 recent_frames=3, floor_bits=2, ceil_bits=8):
        self.alpha_target = alpha_target
        self.frame_dim = frame_dim
        self.recent_frames = recent_frames
        self.floor_bits, self.ceil_bits = floor_bits, ceil_bits

    def bits(self, age_frames: int, sigma: float) -> int:
        """age_frames = how many latent frames ago this entry was committed."""
        b = bits_for(self.alpha_target, max(sigma, 1e-3), self.frame_dim,
                     self.floor_bits, self.ceil_bits)
        if age_frames <= self.recent_frames:      # protect the recent window
            return self.ceil_bits
        decay = min(3, age_frames // max(self.recent_frames, 1))
        return int(max(self.floor_bits, b - decay))

    def calibrate(self, table):
        """table: {bits: measured_alpha}. Sets floor_bits to the cheapest
        bit-width whose measured alpha still clears alpha_target."""
        ok = [b for b, a in sorted(table.items()) if a >= self.alpha_target]
        if ok:
            self.floor_bits = min(ok)
        self._table = dict(table)
        return self.floor_bits

    def describe(self, sigmas, ages=(0, 3, 6, 12, 24)):
        rows = []
        for a in ages:
            rows.append((a, [self.bits(a, float(s)) for s in sigmas]))
        return rows


@torch.no_grad()
def apply_alpha_kv(cache, policy: AlphaKVPolicy, sigma: float,
                   frame_seqlen: int, committed_frames: int, group: int = 128,
                   mode: str = "fake") -> dict:
    """Quantize the committed region of a KV cache with alpha-guided bits.

    cache: list of per-layer dicts with 'k','v' tensors [B, kv_size, H, D]
    Returns a small report {bits_used: {...}, mean_bits: float}.
    """
    used = {}
    if mode != "fake":
        raise NotImplementedError(
            "store-mode needs the attention to read via KVStore.view(layer); "
            "see module docstring for the design.")
    for layer in cache:
        for key in ("k", "v"):
            t = layer[key] if isinstance(layer, dict) else None
            if t is None:
                continue
            n = t.shape[1]
            nf = min(committed_frames, n // max(frame_seqlen, 1))
            for f in range(nf):
                age = committed_frames - f
                b = policy.bits(age, sigma)
                used[b] = used.get(b, 0) + 1
                s0, s1 = f * frame_seqlen, (f + 1) * frame_seqlen
                t[:, s0:s1] = fake_quant_tensor(t[:, s0:s1], b, group)
    tot = sum(used.values()) or 1
    return {"bits_hist": used,
            "mean_bits": sum(b * c for b, c in used.items()) / tot}
