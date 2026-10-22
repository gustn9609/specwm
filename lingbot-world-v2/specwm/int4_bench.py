"""Matmul micro-benchmark for the 14B's linear shapes at our token counts: bf16 F.linear, FP8 W8A8 torch._scaled_mm (our stack), packed-INT4
dequant-to-bf16 + linear (fit24 --wq int4 as run), in-tree INT4 tinygemm torch._weight_int4pack_mm (torchao's int4 kernel, bf16 activations),
and INT4 dequant-to-FP8 + _scaled_mm. M = tokens per forward (704x384 chunk = 3168; partial commit 1056; 832x480 = 4680).
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.int4_bench"""
import torch, time, json, torch.nn.functional as F
dev = "cuda"; torch.manual_seed(0)
def t(fn, it=50):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.time() - t0) / it * 1e3
R = {}
for M in (1056, 3168, 4680):
    for (K, N) in ((5120, 5120), (5120, 13824), (13824, 5120)):
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16); w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
        r = {}
        r["bf16"] = t(lambda: F.linear(x, w))
        sw = (w.abs().amax(1, keepdim=True).float() / 448).clamp_min(1e-12); w8 = (w.float() / sw).to(torch.float8_e4m3fn); swt = sw.t().contiguous()
        def fp8():
            sa = (x.abs().amax(1, keepdim=True).float() / 448).clamp_min(1e-12); x8 = (x.float() / sa).to(torch.float8_e4m3fn)
            return torch._scaled_mm(x8, w8.t(), scale_a=sa, scale_b=swt, out_dtype=torch.bfloat16)
        r["fp8_scaled_mm"] = t(fp8)
        g = 128; wg = w.float().reshape(N, K // g, g); sc = (wg.abs().amax(-1, keepdim=True) / 7).clamp_min(1e-12); q = ((wg / sc).round().clamp(-7, 7).to(torch.int8) + 8).to(torch.uint8).reshape(N, K)
        packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous(); scb = sc.to(torch.bfloat16)
        def deq(dt):
            lo = (packed & 0x0F).to(torch.int16) - 8; hi = ((packed >> 4) & 0x0F).to(torch.int16) - 8
            return (torch.stack((lo, hi), -1).reshape(N, K // g, g).to(dt) * scb.to(dt)).reshape(N, K)
        r["int4_dequant_bf16_linear"] = t(lambda: F.linear(x, deq(torch.bfloat16)))
        def int4_fp8():
            wd = deq(torch.bfloat16); s2 = (wd.abs().amax(1, keepdim=True).float() / 448).clamp_min(1e-12); w8b = (wd.float() / s2).to(torch.float8_e4m3fn)
            sa = (x.abs().amax(1, keepdim=True).float() / 448).clamp_min(1e-12); x8 = (x.float() / sa).to(torch.float8_e4m3fn)
            return torch._scaled_mm(x8, w8b.t(), scale_a=sa, scale_b=s2.t().contiguous(), out_dtype=torch.bfloat16)
        r["int4_dequant_fp8_scaled_mm"] = t(int4_fp8)
        try:   # in-tree tinygemm (what torchao int4_weight_only uses): weight packed to the kernel's layout, bf16 activations, group scales+zeros
            inner_k = 8; wq = (q.to(torch.int32) - 8 + 8).to(torch.int32)   # unsigned 0..15 codes
            w_int4 = torch._convert_weight_to_int4pack(q.to(torch.int32).contiguous() if False else (q.to(torch.uint8)), inner_k) if hasattr(torch, "_convert_weight_to_int4pack") else None
            sz = torch.stack([sc.reshape(N, K // g).to(torch.bfloat16), torch.full((N, K // g), 8.0, device=dev, dtype=torch.bfloat16) * sc.reshape(N, K // g).to(torch.bfloat16)], -1).permute(1, 0, 2).contiguous()
            r["int4_tinygemm"] = t(lambda: torch._weight_int4pack_mm(x, w_int4, g, sz))
        except Exception as e: r["int4_tinygemm"] = "n/a: " + str(e).split("\n")[0][:120]
        R["M%d_K%d_N%d" % (M, K, N)] = r; print("M=%d K=%d N=%d  " % (M, K, N) + "  ".join("%s %.3f ms" % (k, v) if isinstance(v, float) else "%s %s" % (k, v) for k, v in r.items()), flush=True)
json.dump(R, open("/home/hk28965/specwm/paper/meeting_1012/data/int4_bench.json", "w"), indent=1)
