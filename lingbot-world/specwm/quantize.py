"""Weight-only quantization for the draft model. aarch64(GH200)-safe:
default backend is pure-torch per-channel INT8 (no bitsandbytes/torchao
dependency). Optional: torchao int4 / bnb nf4 if importable."""
import torch
import torch.nn as nn


class Int8Linear(nn.Module):
    """Per-output-channel symmetric int8 weight-only linear (dequant on the fly).
    Slower than bf16 matmul but memory is 2x smaller than bf16; for the DRAFT
    whose role is proposing (and whose quality LOSS is the thing we correct),
    fidelity-of-the-quantization-error matters, not speed."""
    def __init__(self, lin: nn.Linear):
        super().__init__()
        w = lin.weight.data.float()
        scale = w.abs().amax(dim=1, keepdim=True) / 127.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        self.register_buffer("wq", torch.clamp((w / scale).round(), -127, 127).to(torch.int8))
        self.register_buffer("scale", scale.to(torch.bfloat16))
        self.bias = lin.bias
        self.out_features, self.in_features = lin.out_features, lin.in_features

    def forward(self, x):
        w = (self.wq.to(x.dtype) * self.scale.to(x.dtype))
        return torch.nn.functional.linear(x, w, self.bias)


def quantize_module_int8(model: nn.Module, min_features: int = 1024) -> int:
    """Swap all large nn.Linear with Int8Linear in-place. Returns #swapped."""
    n = 0
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and child.in_features >= min_features:
            setattr(model, name, Int8Linear(child)); n += 1
        else:
            n += quantize_module_int8(child, min_features)
    return n


def quantize_draft(model: nn.Module, mode: str = "int8"):
    if mode == "int8":
        k = quantize_module_int8(model)
        print(f"[quantize] int8 weight-only: {k} Linear layers swapped")
        return model
    if mode == "int4-ao":
        from torchao.quantization import quantize_, int4_weight_only
        quantize_(model, int4_weight_only())
        return model
    if mode == "none":
        return model
    raise ValueError(f"unknown quant mode {mode}")


# ---------------- NF4 / group-wise quantizers (draft-quality sweep) --------
NF4_CODE = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
            0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
            0.7229568362236023, 1.0]


class GroupQuantLinear(nn.Module):
    """Weight-only quantization with per-group scales.
    kind='int': symmetric b-bit integer.  kind='nf4': 4-bit NormalFloat (QLoRA).
    group=-1 -> per-output-channel (one group per row)."""
    def __init__(self, lin: nn.Linear, bits=8, group=-1, kind="int"):
        super().__init__()
        w = lin.weight.data.float()
        O, I = w.shape
        g = I if group in (-1, 0) else min(group, I)
        assert I % g == 0, f"in_features {I} not divisible by group {g}"
        wg = w.reshape(O, I // g, g)
        if kind == "nf4":
            code = torch.tensor(NF4_CODE, dtype=torch.float32)
            scale = wg.abs().amax(-1, keepdim=True).clamp(min=1e-8)
            idx = (wg / scale).unsqueeze(-1).sub(code.view(1, 1, 1, -1)).abs().argmin(-1)
            self.register_buffer("q", idx.to(torch.uint8))
            self.register_buffer("code", code.to(torch.bfloat16))
            self.kind = "nf4"
        else:
            qmax = 2 ** (bits - 1) - 1
            scale = (wg.abs().amax(-1, keepdim=True) / qmax).clamp(min=1e-12)
            self.register_buffer("q", torch.clamp((wg / scale).round(), -qmax, qmax).to(torch.int8))
            self.kind = "int"
        self.register_buffer("scale", scale.to(torch.bfloat16))
        self.bias, self.O, self.I, self.g = lin.bias, O, I, g

    def _w(self, dt):
        if self.kind == "nf4":
            wq = self.code.to(dt)[self.q.long()]
        else:
            wq = self.q.to(dt)
        return (wq * self.scale.to(dt)).reshape(self.O, self.I)

    def forward(self, x):
        return torch.nn.functional.linear(x, self._w(x.dtype), self.bias)


def _swap(model, make, min_features=1024):
    n = 0
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and child.in_features >= min_features:
            setattr(model, name, make(child)); n += 1
        else:
            n += _swap(child, make, min_features)
    return n


def quantize_family(model: nn.Module, spec: str):
    """spec: none | int8 | int8-g128 | int4 | int4-g128 | int4-g64 | nf4 | nf4-g64"""
    if spec == "none":
        return model
    kind = "nf4" if spec.startswith("nf4") else "int"
    bits = 4 if ("4" in spec.split("-")[0]) else 8
    group = -1
    if "-g" in spec:
        group = int(spec.split("-g")[1])
    n = _swap(model, lambda l: GroupQuantLinear(l, bits=bits, group=group, kind=kind))
    print(f"[quantize] {spec}: {n} Linear layers swapped (kind={kind}, bits={bits}, group={group})")
    return model


@torch.no_grad()
def fake_quantize_family(model: nn.Module, spec: str, min_features: int = 1024) -> int:
    """In-place quantize->dequantize of Linear weights (structure unchanged).
    Numerically identical to the real quantizer; restorable via load_state_dict.
    Used for measuring quantization error / acceptance, not for memory savings."""
    if spec == "none":
        return 0
    kind = "nf4" if spec.startswith("nf4") else "int"
    bits = 4 if ("4" in spec.split("-")[0]) else 8
    group = int(spec.split("-g")[1]) if "-g" in spec else -1
    code = torch.tensor(NF4_CODE)
    n = 0
    for m in model.modules():
        if not isinstance(m, nn.Linear) or m.in_features < min_features:
            continue
        w = m.weight.data
        O, I = w.shape
        g = I if group in (-1, 0) else min(group, I)
        if I % g:
            continue
        wg = w.float().reshape(O, I // g, g)
        if kind == "nf4":
            sc = wg.abs().amax(-1, keepdim=True).clamp(min=1e-8)
            cd = code.to(w.device)
            norm = (wg / sc)
            deq = torch.empty_like(wg)
            step = max(1, 2_000_000 // max(wg.shape[1] * wg.shape[2], 1))
            for r0 in range(0, O, step):          # chunked over output rows
                blk = norm[r0:r0 + step].unsqueeze(-1)
                di = blk.sub(cd.view(1, 1, 1, -1)).abs().argmin(-1)
                deq[r0:r0 + step] = cd[di]
                del blk, di
            deq = deq * sc
        else:
            qmax = 2 ** (bits - 1) - 1
            sc = (wg.abs().amax(-1, keepdim=True) / qmax).clamp(min=1e-12)
            deq = torch.clamp((wg / sc).round(), -qmax, qmax) * sc
        m.weight.data = deq.reshape(O, I).to(w.dtype)
        n += 1
    return n
