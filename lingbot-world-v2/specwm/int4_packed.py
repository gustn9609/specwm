"""[10/19 thesis] Packed INT4 (GPTQ-4, group scales) weights WITHOUT a Triton dependency: the storage format of int4_kernels.PackedInt4LinearT
(packed [O, I/2] uint8 nibbles: even column in the low nibble, odd column in the high nibble, codes + 8; scale [O, I/G] bf16), a pure-torch
dequant, a drop-in linear that uses it, and the loader that puts a gptq_save.py checkpoint into one DiT block on the CPU (two_machine --wq gptq4
--wq-file F --lowmem 1: the block's nn.Linear modules become packed linears BEFORE the block is moved to the GPU, so the bf16 weights never sit there).
Kernel choice: make_packed(kernel="auto") takes int4_kernels.PackedInt4LinearT (fused Triton dequant, the fit24 --int4-kernel triton path) when
specwm.int4_kernels imports (triton present), else PackedInt4LinearTorch (the stack/reshape chain; same numbers, ~one extra GEMM per linear)."""
import torch, torch.nn as nn


def pack_codes(codes, scale, O, I, g):
    """int8 codes in [-7, 7] (any shape with O*I elements) + scales [O, I/g(, 1)] -> (packed uint8 [O, I/2], scale bf16 [O, I/g]) on codes' device"""
    q = (codes.to(torch.int16) + 8).to(torch.uint8).reshape(O, I)
    return (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous(), scale.reshape(O, I // g).to(torch.bfloat16).contiguous()


def dequant_torch(packed, scale, O, I, g, dt=torch.bfloat16):
    """bf16 (dt) weight [O, I] from the packed nibbles: the same numbers as int4_kernels.dequant_int4"""
    lo = (packed & 0x0F).to(torch.int16) - 8; hi = ((packed >> 4) & 0x0F).to(torch.int16) - 8
    q = torch.stack([lo, hi], -1).reshape(O, I)
    return (q.reshape(O, I // g, g).to(dt) * scale.to(dt).unsqueeze(-1)).reshape(O, I)


class PackedInt4LinearTorch(nn.Module):
    """same storage / attributes as int4_kernels.PackedInt4LinearT (packed, scale, bias, O, I, g); torch dequant then a bf16 GEMM"""
    def __init__(self, packed, scale, bias, O, I, g):
        super().__init__(); self.register_buffer("packed", packed.contiguous()); self.register_buffer("scale", scale.reshape(O, I // g).to(torch.bfloat16).contiguous())
        self.bias, self.O, self.I, self.g = bias, O, I, g
    def _w(self, dt): return dequant_torch(self.packed, self.scale, self.O, self.I, self.g, torch.bfloat16)
    def forward(self, x): return nn.functional.linear(x, self._w(x.dtype), self.bias)


def triton_class():
    """int4_kernels.PackedInt4LinearT when triton imports, else None (never raises)"""
    try:
        from specwm.int4_kernels import PackedInt4LinearT
        return PackedInt4LinearT
    except Exception:
        return None


def make_packed(packed, scale, bias, O, I, g, kernel="auto"):
    """a packed linear module from stored tensors (no re-quantization); kernel = auto | triton | torch. Returns (module, kernel used)."""
    T = triton_class() if kernel in ("auto", "triton") else None
    if kernel == "triton" and T is None: raise RuntimeError("--int4-kernel triton: specwm.int4_kernels does not import (no triton)")
    if T is None: return PackedInt4LinearTorch(packed, scale, bias, O, I, g), "torch"
    o = T.__new__(T); nn.Module.__init__(o); o.register_buffer("packed", packed.contiguous()); o.register_buffer("scale", scale.reshape(O, I // g).to(torch.bfloat16).contiguous())
    o.bias, o.O, o.I, o.g = bias, O, I, g
    return o, "triton"


def load_block(blk, prefix, layers, kernel="auto"):
    """replace every nn.Linear of DiT block `blk` whose full name prefix.<name> is in `layers` (a gptq_save.py 'layers' dict: packed, scale, bias, O, I, g)
    by a packed linear built from the stored CPU tensors; the bf16 weight is dropped (the module is replaced). Returns (n replaced, kernel used, missing names)."""
    n = 0; used = None; missing = []
    for name, mod in list(blk.named_modules()):
        for cn, ch in list(mod.named_children()):
            if not isinstance(ch, nn.Linear): continue
            full = "%s.%s%s" % (prefix, (name + ".") if name else "", cn)
            if full not in layers:
                if ch.in_features >= 1024: missing.append(full)
                continue
            L = layers[full]; assert (L["O"], L["I"]) == (ch.out_features, ch.in_features), (full, L["O"], L["I"], ch.out_features, ch.in_features)
            bias = ch.bias
            if L.get("bias") is not None and bias is None: bias = nn.Parameter(L["bias"].to(torch.bfloat16), requires_grad=False)
            m, used = make_packed(L["packed"], L["scale"], bias, L["O"], L["I"], L["g"], kernel); setattr(mod, cn, m); n += 1
    return n, used, missing


def check_file(path):
    """load a gptq_save.py checkpoint (CPU) and return (meta, n layers, names of layers with non-finite scales)"""
    Q = torch.load(path, map_location="cpu", weights_only=True); bad = [k for k, L in Q["layers"].items() if not torch.isfinite(L["scale"].float()).all()]
    return Q["meta"], len(Q["layers"]), bad


if __name__ == "__main__":   # CPU self-test: random codes -> pack -> torch dequant == direct; module forward shape
    torch.manual_seed(0); O, I, g = 256, 1024, 128
    codes = torch.randint(-7, 8, (O, I), dtype=torch.int8); scale = torch.rand(O, I // g, 1) * 0.05 + 0.01
    packed, sc = pack_codes(codes, scale, O, I, g); ref = (codes.reshape(O, I // g, g).to(torch.bfloat16) * scale.to(torch.bfloat16)).reshape(O, I)
    w = dequant_torch(packed, sc, O, I, g); assert w.shape == (O, I) and torch.equal(w, ref), "dequant mismatch"
    m = PackedInt4LinearTorch(packed, sc, None, O, I, g); x = torch.randn(3, 7, I, dtype=torch.bfloat16); y = m(x); assert y.shape == (3, 7, O), y.shape
    assert torch.allclose(y.float(), nn.functional.linear(x, ref).float()), "forward mismatch"
    print("[int4_packed] self-test ok: packed", tuple(packed.shape), packed.dtype, "scale", tuple(sc.shape), sc.dtype, "forward", tuple(y.shape), "triton class", "yes" if triton_class() else "no")
