"""INT4 weight storage with a fused Triton dequant (packed uint8 nibbles + bf16 group scales -> bf16 weight in one memory-bound pass), used by
fit24 --int4-kernel triton. At our token counts (M >= 1056 per forward) weight-only INT4 GEMMs (tinygemm, naive fused W4A16) are slower than
bf16 (int4_bench2.py), so the best INT4 path is dequantize-then-bf16-GEMM; this kernel makes the dequant ~memory-speed instead of the
stack/reshape chain of quantize.PackedInt4Linear (which costs about one extra GEMM per linear)."""
import torch, torch.nn as nn, triton, triton.language as tl
@triton.jit
def _deq_kernel(p_ptr, s_ptr, y_ptr, N, K, G: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0); kb = tl.program_id(1); kk = kb * BK + tl.arange(0, BK)   # kk indexes the packed (K/2) axis
    m = kk < K // 2
    pk = tl.load(p_ptr + row * (K // 2) + kk, mask=m, other=0)
    lo = (pk & 0x0F).to(tl.int32) - 8; hi = ((pk >> 4) & 0x0F).to(tl.int32) - 8
    sc = tl.load(s_ptr + row * (K // G) + (2 * kk) // G, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * K + 2 * kk, (lo.to(tl.float32) * sc).to(tl.bfloat16), mask=m)
    tl.store(y_ptr + row * K + 2 * kk + 1, (hi.to(tl.float32) * sc).to(tl.bfloat16), mask=m)
def dequant_int4(packed, scale, N, K, G=128, BK=1024):
    y = torch.empty(N, K, device=packed.device, dtype=torch.bfloat16)
    _deq_kernel[(N, triton.cdiv(K // 2, BK))](packed, scale, y, N, K, G=G, BK=BK); return y
class PackedInt4LinearT(nn.Module):
    """Same storage as quantize.PackedInt4Linear (packed [O, I/2] uint8, scale [O, I/G] bf16); dequant by the Triton kernel. Build from
    codes (int8 in [-7, 7], e.g. from GPTQ) + group scales, or from an existing PackedInt4Linear / AWQInt4Linear-free module."""
    def __init__(self, codes, scale, bias, O, I, g):
        super().__init__(); q = (codes.to(torch.int16) + 8).to(torch.uint8).reshape(O, I)
        self.register_buffer("packed", (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()); self.register_buffer("scale", scale.reshape(O, I // g).to(torch.bfloat16).contiguous())
        self.bias, self.O, self.I, self.g = bias, O, I, g
    @classmethod
    def from_packed(cls, m):   # quantize.PackedInt4Linear -> Triton-dequant version (no re-quantization)
        o = cls.__new__(cls); nn.Module.__init__(o); o.register_buffer("packed", m.packed); o.register_buffer("scale", m.scale.reshape(m.O, m.I // m.g).contiguous()); o.bias, o.O, o.I, o.g = m.bias, m.O, m.I, m.g; return o
    def _w(self, dt): return dequant_int4(self.packed, self.scale, self.O, self.I, self.g)
    def forward(self, x): return nn.functional.linear(x, self._w(x.dtype), self.bias)
if __name__ == "__main__":   # correctness + speed vs the stack/reshape dequant and bf16 GEMM
    import time, torch.nn.functional as F
    from specwm.quantize import PackedInt4Linear
    def t(fn, it=50):
        for _ in range(5): fn()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(it): fn()
        torch.cuda.synchronize(); return (time.time() - t0) / it * 1e3
    for (K, N) in ((5120, 5120), (5120, 13824), (13824, 5120)):
        lin = nn.Linear(K, N, bias=False, device="cuda", dtype=torch.bfloat16); x = torch.randn(3168, K, device="cuda", dtype=torch.bfloat16)
        pq = PackedInt4Linear(lin, group=128, search=False); pt = PackedInt4LinearT.from_packed(pq)
        err = (pq._w(torch.bfloat16).float() - pt._w(torch.bfloat16).float()).abs().max().item()
        print("K=%d N=%d  max|Δw| %.2e | bf16 linear %.3f ms | packed dequant+linear %.3f | triton dequant+linear %.3f | triton dequant alone %.3f" % (K, N, err, t(lambda: F.linear(x, lin.weight)), t(lambda: pq(x)), t(lambda: pt(x)), t(lambda: pt._w(torch.bfloat16))), flush=True)
