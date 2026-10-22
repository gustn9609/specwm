"""Spatial tiled VAE decode.

Measured: decoding 3 latent frames peaks at 7.60 GiB over resident, and
decoding 15 latent frames peaks at 7.82 GiB -- essentially FLAT in frame count.
So the cost is per-frame decoder activation, not output accumulation, and
offloading finished frames to CPU would not help. The only lever left is to
shrink the spatial extent that the decoder sees at once.

The decoder is MOSTLY convolutional, so we decode overlapping spatial tiles and
blend the seams with a linear ramp. Temporal behaviour is untouched: each tile
is decoded over its full time range in one call, so the causal feat_cache
inside the VAE stays valid.

IMPORTANT -- THIS IS APPROXIMATE, AND NOT BECAUSE OF THE OVERLAP.
wan/modules/vae2_1.py puts AttentionBlock (a GLOBAL spatial
F.scaled_dot_product_attention over the whole h*w of each frame) inside the
decoder at the middle block and at one upsample level. A tile can only attend
within itself, so no amount of overlap reproduces the untiled result. Measured
on the real VAE: 2x2 tiles cut the decode peak 7.92 -> 3.27 GiB but shift
outputs by up to 0.59 in [-1,1] units.

Use this when the memory ceiling is binding (a 24 GB card cannot fit the
untiled decode) and verify perceptually, not numerically. RMS_norm is
channel-wise so it is NOT a source of error here; the attention is.
"""
from __future__ import annotations
import torch


def _ramp(n, device, dtype):
    return torch.linspace(0, 1, n, device=device, dtype=dtype)


@torch.no_grad()
def tiled_decode(vae, z, tiles=(2, 2), overlap=8, scale=8):
    """Decode latent z [C,T,H,W] (or [B,C,T,H,W]) in spatial tiles.

    tiles   : (rows, cols) split of the latent grid
    overlap : latent-space overlap on each side of an interior seam
    scale   : VAE spatial upsampling factor (8 for Wan2.1)
    """
    squeeze = (z.dim() == 4)
    if squeeze:
        z = z.unsqueeze(0)
    B, C, T, H, W = z.shape
    R, Cc = tiles
    if R * Cc == 1:
        out = vae.decode([z[0]])[0]
        return out

    hs = [(H * i) // R for i in range(R)] + [H]
    ws = [(W * j) // Cc for j in range(Cc)] + [W]
    out = None
    wsum = None

    for i in range(R):
        for j in range(Cc):
            h0, h1 = hs[i], hs[i + 1]
            w0, w1 = ws[j], ws[j + 1]
            ph0 = max(0, h0 - overlap); ph1 = min(H, h1 + overlap)
            pw0 = max(0, w0 - overlap); pw1 = min(W, w1 + overlap)
            piece = vae.decode([z[0][:, :, ph0:ph1, pw0:pw1]])[0]   # [C,T,h*,w*]
            if out is None:
                Cout, Tout = piece.shape[0], piece.shape[1]
                out = torch.zeros(Cout, Tout, H * scale, W * scale,
                                  device=piece.device, dtype=torch.float32)
                wsum = torch.zeros(1, 1, H * scale, W * scale,
                                   device=piece.device, dtype=torch.float32)
            # feather the padded region so seams blend
            ph, pw = piece.shape[-2], piece.shape[-1]
            m = torch.ones(1, 1, ph, pw, device=piece.device, dtype=torch.float32)
            top = (h0 - ph0) * scale; bot = (ph1 - h1) * scale
            lef = (w0 - pw0) * scale; rig = (pw1 - w1) * scale
            if top > 0: m[..., :top, :] *= _ramp(top, m.device, m.dtype).view(1, 1, -1, 1)
            if bot > 0: m[..., ph - bot:, :] *= _ramp(bot, m.device, m.dtype).flip(0).view(1, 1, -1, 1)
            if lef > 0: m[..., :, :lef] *= _ramp(lef, m.device, m.dtype).view(1, 1, 1, -1)
            if rig > 0: m[..., :, pw - rig:] *= _ramp(rig, m.device, m.dtype).flip(0).view(1, 1, 1, -1)
            y0, x0 = ph0 * scale, pw0 * scale
            out[:, :, y0:y0 + ph, x0:x0 + pw] += piece.float() * m[0]
            wsum[:, :, y0:y0 + ph, x0:x0 + pw] += m
            del piece
    out = out / wsum.clamp(min=1e-6)
    return out.to(torch.float32)
