"""GPTQ: data-aware INT-b quantization that minimises OUTPUT error.

Every quantizer tried so far looked only at the weights (min-max, MSE clip
search, AWQ scaling). They all landed at ~16-17 dB PSNR for INT4, and the
layer-sensitivity scan showed the error is DIFFUSE across all 40 blocks, so
mixed precision cannot buy it back cheaply.

GPTQ is different in kind: for each layer it quantizes the weight columns one
at a time and pushes the rounding error of each column onto the columns not
yet quantized, weighted by the inverse Hessian of the layer input
(H = X^T X from calibration activations). The result is a quantized W' that
minimises ||W X - W' X||^2 -- the actual output error -- rather than
||W - W'||. Diffuse error is exactly the regime where this helps.

This is the standard blockwise OBQ formulation (Frantar et al. 2022) with
static per-group scales computed on the error-updated weights, i.e. the
"true sequential" variant within a layer. Across layers we use the simpler
non-sequential scheme (Hessians from the bf16 model), which loses a little
accuracy but needs one calibration pass instead of 40.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class HessianCollector:
    """Accumulate H = X^T X per Linear from forward hooks."""

    def __init__(self, model, min_features=1024):
        self.H, self.n, self.handles = {}, {}, []
        self.names = {}
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear) and m.in_features >= min_features:
                self.names[id(m)] = name
                self.handles.append(m.register_forward_pre_hook(self._mk(name, m.in_features)))

    def _mk(self, name, I):
        # Accumulate on CPU. All 568 Hessians together are ~51 GiB (an FFN
        # layer's alone is 13824^2 * 4 B = 730 MiB), which does not fit next
        # to the 58 GiB model. The X^T X product is formed on GPU per call and
        # the (small, one-off) result streamed to host memory.
        def pre(mod, inp):
            x = inp[0].detach().float()               # upcast FIRST: bf16 x bf16
            x = x.reshape(-1, x.shape[-1])            # would give a bf16 Hessian,
            xtx = (x.t() @ x).float().cpu()           # and Cholesky needs fp32
            if name not in self.H:
                self.H[name] = xtx
                self.n[name] = 0
            else:
                self.H[name] += xtx
            self.n[name] += x.shape[0]
            del xtx
        return pre

    def finish(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        for k in self.H:
            self.H[k] /= max(self.n[k], 1)
        return self.H


@torch.no_grad()
def gptq_quantize_weight(W, H, bits=4, group=128, blocksize=128, percdamp=0.01):
    """Return quantized-dequantized W' (same shape/dtype as W) and int codes."""
    W = W.clone().float()
    O, I = W.shape
    H = H.clone().float()                             # defensive: cusolver is fp32-only
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    # Cholesky needs H to be positive-definite. With a 13824-wide FFN input and
    # only ~2x that many correlated calibration tokens, H is rank-deficient and
    # the default 1% damping is not enough (failed at leading minor 805).
    # H is symmetric PSD, so H + lambda*I is PD for large enough lambda:
    # retry with doubling damping until the factorization goes through.
    mean_diag = torch.mean(torch.diag(H))
    Hinv = None
    d = percdamp
    for _ in range(10):
        Hd = H + torch.eye(I, device=H.device) * (d * mean_diag)
        try:
            L = torch.linalg.cholesky(Hd)
            Hinv = torch.cholesky_inverse(L)
            Hinv = torch.linalg.cholesky(Hinv, upper=True)
            break
        except torch._C._LinAlgError:
            d *= 2.0
    if Hinv is None:
        raise RuntimeError(f"Hessian not PD even at damping {d}")
    if d != percdamp:
        gptq_quantize_weight.damp_bumps = getattr(gptq_quantize_weight, "damp_bumps", 0) + 1

    qmax = 2 ** (bits - 1) - 1
    g = I if group in (-1, 0) else group
    Q = torch.zeros_like(W)
    scales = torch.zeros(O, I // g, device=W.device)

    for i1 in range(0, I, blocksize):
        i2 = min(i1 + blocksize, I)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % g == 0:
                # (re)compute this group's scale on the error-updated weights
                grp = W[:, col:col + g]
                s = (grp.abs().amax(-1, keepdim=True) / qmax).clamp(min=1e-12)
                scales[:, col // g] = s.squeeze(-1)
            s = scales[:, col // g].unsqueeze(-1)
            w = W1[:, i:i + 1]
            d = Hinv1[i, i]
            q = torch.clamp((w / s).round(), -qmax, qmax) * s
            Q1[:, i:i + 1] = q
            err = (w - q) / d
            W1[:, i:] -= err @ Hinv1[i:i + 1, i:]
            Err1[:, i:i + 1] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return Q, scales


class GPTQLinear(nn.Module):
    """Holds GPTQ-quantized weights as int8 codes + per-group scales."""

    def __init__(self, lin: nn.Linear, H, bits=4, group=128):
        super().__init__()
        H = H.to(lin.weight.device).float()               # just-in-time, one layer, fp32
        Q, scales = gptq_quantize_weight(lin.weight.data, H, bits=bits, group=group)
        del H
        O, I = Q.shape
        g = I if group in (-1, 0) else group
        qmax = 2 ** (bits - 1) - 1
        codes = torch.clamp((Q.reshape(O, I // g, g) / scales.unsqueeze(-1)).round(),
                            -qmax, qmax).to(torch.int8)
        self.register_buffer("q", codes.reshape(O, I))
        self.register_buffer("scale", scales.unsqueeze(-1).to(torch.bfloat16))
        self.bias, self.O, self.I, self.g, self.bits = lin.bias, O, I, g, bits

    def analytic_bytes(self):
        return self.O * self.I * self.bits / 8 + self.scale.numel() * 2

    def _w(self, dt):
        return (self.q.reshape(self.O, self.I // self.g, self.g).to(dt)
                * self.scale.to(dt)).reshape(self.O, self.I)

    def forward(self, x):
        return torch.nn.functional.linear(x, self._w(x.dtype), self.bias)


def apply_gptq(model, hessians, bits=4, group=128, min_features=1024, verbose=True):
    from .quantize import _swap
    by_id = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and m.in_features >= min_features:
            by_id[id(m)] = hessians.get(name)
    miss = [0]
    def make(l):
        H = by_id.get(id(l))
        if H is None:
            miss[0] += 1
            H = torch.eye(l.in_features, device=l.weight.device)
        return GPTQLinear(l, H, bits=bits, group=group)
    n = _swap(model, make, min_features=min_features)
    if verbose:
        print(f"[gptq] int{bits} g{group}: {n} Linear quantized "
              f"({miss[0]} without Hessian)", flush=True)
    return model
