"""Activation-aware INT4 (AWQ-style) weight quantization.

WHY THIS EXISTS. Plain group-wise symmetric INT4 destroyed this model:
16.75 dB PSNR, and neither proper 4-bit packing nor an MSE-optimal scale
search recovered it (a clipping search made it slightly WORSE, 15.82 dB,
which says the weight outliers carry real signal rather than noise).

Everything we tried so far looked only at the WEIGHTS. AWQ's observation is
that what matters is which weights multiply LARGE ACTIVATIONS. Using the
identity

    y = W x = (W * s) (x / s)          s = per-input-channel scale

we can amplify the columns of W that see big activations before quantizing,
so those columns land on a finer part of the 4-bit grid. The scale is folded
back by dividing the input at runtime, so the layer is mathematically
equivalent in full precision -- only the quantization error changes.

s_j = (mean_t |x_{t,j}|) ** alpha, with alpha = 0 recovering plain
quantization and alpha = 1 fully following the activation magnitude.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .quantize import _swap


class ActStats:
    """Accumulates mean |x| per input channel for chosen Linear layers."""

    def __init__(self, model, min_features=1024):
        self.sums, self.counts, self.handles = {}, {}, []
        self.min_features = min_features
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear) and m.in_features >= min_features:
                self.handles.append(m.register_forward_pre_hook(self._mk(name)))
        self.n_layers = len(self.handles)

    def _mk(self, name):
        def pre(mod, inp):
            x = inp[0]
            if x is None:
                return
            v = x.detach().abs().reshape(-1, x.shape[-1]).float()
            s = v.sum(0)
            if name in self.sums:
                self.sums[name] += s; self.counts[name] += v.shape[0]
            else:
                self.sums[name] = s; self.counts[name] = v.shape[0]
        return pre

    def finish(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        return {k: (self.sums[k] / max(self.counts[k], 1)) for k in self.sums}


class AWQInt4Linear(nn.Module):
    """Packed INT4 linear with a per-input-channel activation scale folded in."""

    def __init__(self, lin: nn.Linear, act_mean=None, alpha=0.5, group=128):
        super().__init__()
        w = lin.weight.data.float()
        O, I = w.shape
        g = I if group in (-1, 0) else min(group, I)
        assert I % g == 0 and g % 2 == 0

        if act_mean is None or alpha == 0:
            s = torch.ones(I, dtype=torch.float32, device=w.device)
        else:
            a = act_mean.to(w.device).float().clamp(min=1e-8)
            s = (a / a.mean()).pow(alpha)              # normalised so mean~1
            s = s.clamp(min=1e-3, max=1e3)
        ws = w * s.view(1, I)                          # scale COLUMNS

        qmax = 7
        wg = ws.reshape(O, I // g, g)
        scale = (wg.abs().amax(-1, keepdim=True) / qmax).clamp(min=1e-12)
        q = torch.clamp((wg / scale).round(), -qmax, qmax).to(torch.int8)
        q = (q + 8).to(torch.uint8).reshape(O, I)
        self.register_buffer("packed", (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous())
        self.register_buffer("scale", scale.to(torch.bfloat16))
        self.register_buffer("inv_s", (1.0 / s).to(torch.bfloat16))
        self.bias, self.O, self.I, self.g = lin.bias, O, I, g

    def _w(self, dt):
        p = self.packed
        lo = (p & 0x0F).to(torch.int16) - 8
        hi = ((p >> 4) & 0x0F).to(torch.int16) - 8
        q = torch.stack((lo, hi), dim=-1).reshape(self.O, self.I).to(dt)
        return (q.reshape(self.O, self.I // self.g, self.g)
                * self.scale.to(dt)).reshape(self.O, self.I)

    def forward(self, x):
        return torch.nn.functional.linear(x * self.inv_s.to(x.dtype),
                                          self._w(x.dtype), self.bias)


def quantize_awq(model, act_means, alpha=0.5, group=128, min_features=1024,
                 verbose=True):
    """Replace Linear layers with AWQInt4Linear using collected activation stats."""
    by_mod = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and m.in_features >= min_features:
            by_mod[id(m)] = act_means.get(name)
    n = _swap(model,
              lambda l: AWQInt4Linear(l, act_mean=by_mod.get(id(l)),
                                      alpha=alpha, group=group),
              min_features=min_features)
    if verbose:
        hit = sum(1 for v in by_mod.values() if v is not None)
        print(f"[awq] {n} Linear swapped (alpha={alpha}, group={group}); "
              f"{hit}/{len(by_mod)} had activation stats", flush=True)
    return model
