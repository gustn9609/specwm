"""INT4 kernels at our shapes, part 2: in-tree tinygemm (torch._weight_int4pack_mm, packed [N, K/2] uint8) and a Triton W4A16 GEMM with the
dequant fused into the K loop (group scales, symmetric). Reference: bf16 F.linear and FP8 _scaled_mm from int4_bench.py."""
import torch, time, json, torch.nn.functional as F, triton, triton.language as tl
dev = "cuda"; torch.manual_seed(0)
def t(fn, it=50):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.time() - t0) / it * 1e3
@triton.autotune(configs=[triton.Config({"BM": bm, "BN": bn, "BK": bk}, num_stages=s, num_warps=w) for bm in (64, 128) for bn in (64, 128, 256) for bk in (64, 128) for s in (3, 4) for w in (4, 8)], key=["M", "N", "K"])
@triton.jit
def w4a16_kernel(x_ptr, w_ptr, s_ptr, y_ptr, M, N, K, G: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # x [M, K] bf16 row-major; w packed [N, K//2] uint8 (low nibble = even k); s [N, K//G] bf16; y [M, N] bf16
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM); rn = pid_n * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + rk
        x = tl.load(x_ptr + rm[:, None] * K + kk[None, :], mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
        pk = tl.load(w_ptr + rn[:, None] * (K // 2) + (kk[None, :] // 2), mask=(rn[:, None] < N) & (kk[None, :] < K), other=0)   # [BN, BK] uint8
        nib = tl.where((kk[None, :] % 2) == 0, pk & 0x0F, (pk >> 4) & 0x0F).to(tl.int32) - 8
        sc = tl.load(s_ptr + rn[:, None] * (K // G) + (kk[None, :] // G), mask=(rn[:, None] < N) & (kk[None, :] < K), other=0.0)
        w = (nib.to(tl.float32) * sc.to(tl.float32)).to(tl.bfloat16)   # [BN, BK]
        acc += tl.dot(x, tl.trans(w))
    tl.store(y_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))
def w4a16(x, packed, scale, N, K, G=128):
    M = x.shape[0]; y = torch.empty(M, N, device=x.device, dtype=torch.bfloat16)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"])); w4a16_kernel[grid](x, packed, scale, y, M, N, K, G=G); return y
R = {}
for M in (1056, 3168, 4680):
    for (K, N) in ((5120, 5120), (5120, 13824), (13824, 5120)):
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16); w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02; r = {}
        r["bf16"] = t(lambda: F.linear(x, w))
        g = 128; wg = w.float().reshape(N, K // g, g); sc = (wg.abs().amax(-1, keepdim=True) / 7).clamp_min(1e-12); q = ((wg / sc).round().clamp(-7, 7).to(torch.int8) + 8).to(torch.uint8).reshape(N, K)
        packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous(); scb = sc.reshape(N, K // g).to(torch.bfloat16).contiguous()
        wdq = ((q.to(torch.float32) - 8).reshape(N, K // g, g) * sc).reshape(N, K).to(torch.bfloat16)
        y_ref = F.linear(x, wdq); y = w4a16(x, packed, scb, N, K); r["triton_w4a16_err"] = float((y.float() - y_ref.float()).abs().max() / y_ref.float().abs().max())
        r["triton_w4a16"] = t(lambda: w4a16(x, packed, scb, N, K))
        try:
            pk_hi = (q[:, 0::2] << 4 | q[:, 1::2]).to(torch.uint8).contiguous()
            for inner in (2, 4, 8):
                try:
                    w_int4 = torch._convert_weight_to_int4pack(pk_hi, inner); sz = torch.stack([scb, 8.0 * scb], -1).permute(1, 0, 2).contiguous()
                    r["int4_tinygemm_inner%d" % inner] = t(lambda: torch._weight_int4pack_mm(x, w_int4, g, sz)); break
                except Exception as e: r["int4_tinygemm_inner%d" % inner] = "n/a: " + str(e).split("\n")[0][:80]
        except Exception as e: r["int4_tinygemm"] = "n/a: " + str(e).split("\n")[0][:80]
        R["M%d_K%d_N%d" % (M, K, N)] = r; print("M=%d K=%d N=%d  " % (M, K, N) + "  ".join("%s %.3f" % (k, v) if isinstance(v, float) else "%s %s" % (k, v) for k, v in r.items()), flush=True)
json.dump(R, open("/home/hk28965/specwm/paper/meeting_1012/data/int4_bench2.json", "w"), indent=1)
