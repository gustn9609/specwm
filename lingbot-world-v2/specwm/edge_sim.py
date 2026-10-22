"""엣지 GPU 시뮬레이션 (action item #5 의 대체: 실제 엣지 GPU 가 없을 때).

(1) 메모리 상한: torch.cuda.set_per_process_memory_fraction 으로 24/16/12 GiB 캡을 걸고
    압축 구성(INT4 weight + 4bit KV store + 2x2 타일 VAE) 이 실제로 OOM 없이 도는지 판정.
    (모델 로드/양자화는 캡 밖에서 하고, 생성+디코드만 캡 안에서 측정 - 실제 엣지는 양자화된 가중치를 바로 로드함)
(2) 성능 투영: 이 GPU 에서 측정한 청크 지연을 roofline 비율로 엣지 GPU 에 투영.
    - 달성 TFLOPS 는 bf16 GEMM 마이크로벤치로 캘리브레이션
    - 청크당 FLOPs = 2 * params * tokens * forwards (DiT 선형층) + attention 항
    - 엣지 스펙(대역폭/TFLOPS) 은 EDGE 표 (공개 스펙 근사치; 필요시 수정)
    주의: 투영값이지 측정값이 아님. 보고서에 반드시 'projected' 로 표기.
"""
from __future__ import annotations
import argparse, json, os, time, gc
import numpy as np, torch

GiB = 2 ** 30
# 공개 스펙 근사 (dense bf16/fp16 TFLOPS, 메모리 대역폭 GB/s, 메모리 GB). 필요시 수정.
EDGE = {
    "this_gpu":        dict(tflops=None, bw_gbs=None, mem_gb=None),          # 런타임에 채움
    "RTX_4090":        dict(tflops=165, bw_gbs=1008, mem_gb=24),
    "RTX_5090":        dict(tflops=210, bw_gbs=1792, mem_gb=32),
    "RTX_4070_12G":    dict(tflops=58,  bw_gbs=504,  mem_gb=12),
    "Jetson_AGX_Orin": dict(tflops=42,  bw_gbs=205,  mem_gb=64),
    "Jetson_Thor":     dict(tflops=250, bw_gbs=273,  mem_gb=128),
}


def gemm_tflops(n=8192, iters=10):
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16); b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(3): a @ b
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters): a @ b
    torch.cuda.synchronize(); dt = (time.perf_counter() - t) / iters
    del a, b; torch.cuda.empty_cache()
    return 2 * n ** 3 / dt / 1e12


def bw_gbs(n_bytes=2 * GiB, iters=10):
    x = torch.empty(n_bytes // 2, device="cuda", dtype=torch.bfloat16); y = torch.empty_like(x)
    y.copy_(x); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters): y.copy_(x)
    torch.cuda.synchronize(); dt = (time.perf_counter() - t) / iters
    del x, y; torch.cuda.empty_cache()
    return 2 * n_bytes / dt / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--image", required=True); ap.add_argument("--action", required=True)
    ap.add_argument("--prompt", default="A serene lakeside scene with a lone tree.")
    ap.add_argument("--frames", type=int, default=81); ap.add_argument("--local-attn", type=int, default=18); ap.add_argument("--sink", type=int, default=6)
    ap.add_argument("--kv-bits", type=int, default=4); ap.add_argument("--caps", default="24,16,12")
    ap.add_argument("--int4-search", action="store_true", help="PackedInt4 스케일 탐색 (느리지만 PSNR 개선)"); ap.add_argument("--no-cap", action="store_true")
    ap.add_argument("--out", default="specwm_out/edge_sim")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    from .binding_v2 import V2Binding
    from .vae_tiled import tiled_decode
    from .quantize import quantize_packed_int4
    from . import kvstore
    props = torch.cuda.get_device_properties(0); total = props.total_memory
    rep = dict(gpu=props.name, total_GiB=round(total / GiB, 1), caps={}, projection={})
    rep["calib"] = dict(gemm_bf16_tflops=round(gemm_tflops(), 1), copy_bw_gbs=round(bw_gbs(), 0))
    EDGE["this_gpu"] = dict(tflops=rep["calib"]["gemm_bf16_tflops"], bw_gbs=rep["calib"]["copy_bw_gbs"], mem_gb=round(total / 1e9))
    print("[calib] %s: GEMM %.0f TFLOPS, copy %.0f GB/s" % (props.name, EDGE["this_gpu"]["tflops"], EDGE["this_gpu"]["bw_gbs"]), flush=True)

    t0 = time.time()
    b = V2Binding(a.ckpt, target_mode="fast_bf16", quant=None, sink=a.sink, local_attn=a.local_attn, swap_models=False, build_draft=False)
    cond = b.prepare(a.image, a.prompt, a.action, frame_num=a.frames)
    N = cond["num_chunks"]; sig = b.node_sigmas(); tm = b.target_model
    n_params = sum(p.numel() for p in tm.parameters())
    shp = cond["shape"]; tokens_per_chunk = int(shp[1] * (shp[2] // 2) * (shp[3] // 2))
    print("[load] %.0fs, %d chunks, %.1fB params, %d tokens/chunk" % (time.time() - t0, N, n_params / 1e9, tokens_per_chunk), flush=True)

    def gen(g_seed=1234, cap_frac=None):
        # reset_caches() 는 kvstore 가 대체하기 전에 bf16 KV 전체(~21GiB)를 잠깐 할당하므로(과도 상태), 캡은 청크 1 이후
        # (packed store 가 살아있는 정상상태) 부터 건다 -> 실제 엣지 배포가 만족해야 하는 정상상태 피크를 판정.
        b.reset_caches(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        g = torch.Generator().manual_seed(g_seed); outs, ts = [], []
        for c in range(N):
            if c == 1:
                torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                if cap_frac is not None: torch.cuda.set_per_process_memory_fraction(cap_frac)
            torch.cuda.synchronize(); t = time.perf_counter()
            z = b.chunk_noise(c, g); x0 = None
            for i in range(len(sig)):
                x0 = b.target_x0(z, i, c)
                if i < len(sig) - 1: s = float(sig[i + 1]); z = (1 - s) * x0 + s * b.randn_like(x0, g)
            b.commit(x0, c); torch.cuda.synchronize(); ts.append(time.perf_counter() - t); outs.append(x0.detach().float().cpu())
        st = sorted(ts[2:])
        return outs, dict(chunk_s_median=round(st[len(st) // 2], 4), chunk_s_max=round(max(ts[2:]), 4),
                          peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / GiB, 2), peak_reserved_GiB=round(torch.cuda.max_memory_reserved() / GiB, 2))

    def dec(lat, tiles):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.perf_counter()
        z = torch.cat(lat, dim=1).to(b.device); img = tiled_decode(b.pipe.vae, z, tiles=tiles, overlap=8, scale=8); torch.cuda.synchronize()
        info = dict(decode_s=round(time.perf_counter() - t, 3), peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / GiB, 2), peak_reserved_GiB=round(torch.cuda.max_memory_reserved() / GiB, 2))
        del z; return img.detach().cpu(), info

    # 기준 (캡 없음, bf16)
    outs, i0 = gen(); img_ref, d0 = dec(outs, (1, 1)); rep["bf16_uncapped"] = dict(gen=i0, decode=d0); del outs
    print("[bf16] %s %s" % (i0, d0), flush=True)
    # 압축 구성
    kvstore.install(tm, bits=a.kv_bits); quantize_packed_int4(tm, group=128, search=a.int4_search); gc.collect(); torch.cuda.empty_cache()
    wbytes = sum(p.numel() * p.element_size() for p in tm.parameters()) + sum(q.numel() * q.element_size() for q in tm.buffers())
    rep["compressed"] = dict(weight_GiB=round(wbytes / GiB, 2), kv_bits=a.kv_bits, int4_search=a.int4_search)
    outs, i1 = gen(); img_c, d1 = dec(outs, (2, 2)); rep["compressed"].update(gen_uncapped=i1, decode_uncapped=d1, kv_store_GiB=round(kvstore.total_bytes(tm) / GiB, 2))
    mse = float((img_c.float() - img_ref.float()).pow(2).mean()); rep["compressed"]["psnr_vs_bf16"] = round(float(10 * np.log10(4.0 / max(mse, 1e-12))), 2)
    print("[compressed] %s %s psnr %.2f" % (i1, d1, rep["compressed"]["psnr_vs_bf16"]), flush=True); del outs, img_c

    # 메모리 캡 판정
    resident = torch.cuda.memory_allocated() / GiB
    for cap in ([] if a.no_cap else [float(c) for c in a.caps.split(",")]):
        gc.collect(); torch.cuda.empty_cache()
        frac = min(0.999, cap * GiB / total)
        row = dict(cap_GiB=cap, resident_before_GiB=round(resident, 2), note="cap applied from chunk 1 (steady state, packed KV live)")
        try:
            outs, gi = gen(cap_frac=frac); row["gen"] = gi
            _, di = dec(outs, (2, 2)); row["decode"] = di; row["verdict"] = "FITS"; del outs
        except torch.OutOfMemoryError as e:
            row["verdict"] = "OOM"; row["error"] = str(e)[:200]; gc.collect(); torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(1.0)
        rep["caps"]["%gG" % cap] = row; print("[cap %gG] %s" % (cap, row.get("verdict")), row.get("gen", ""), flush=True)
        json.dump(rep, open(os.path.join(a.out, "edge_sim.json"), "w"), indent=1)

    # roofline 투영
    forwards = len(sig) + 1                                   # 4 노드 + commit
    flops_chunk = 2 * n_params * tokens_per_chunk * forwards
    kv_tokens = a.local_attn * (shp[2] // 2) * (shp[3] // 2)
    attn_flops = 4 * tokens_per_chunk * kv_tokens * 5120 * 40 * forwards        # q·k + p·v, dim 5120, 40 blocks (근사)
    flops_chunk += attn_flops
    bytes_chunk = wbytes * forwards + kvstore.total_bytes(tm) * forwards
    meas = rep["compressed"]["gen_uncapped"]["chunk_s_median"]
    here = EDGE["this_gpu"]; eff = flops_chunk / meas / 1e12 / here["tflops"]      # 측정 대비 roofline 효율
    rep["projection_model"] = dict(flops_per_chunk_T=round(flops_chunk / 1e12, 1), bytes_per_chunk_GiB=round(bytes_chunk / GiB, 2),
                                   measured_chunk_s=meas, achieved_eff_vs_gemm=round(eff, 3),
                                   note="t_proj = max(flops/(tflops*eff), bytes/bw); 동일 커널 효율 가정. 측정값 아님(projected).")
    for name, s in EDGE.items():
        t_c = flops_chunk / (s["tflops"] * 1e12 * eff); t_m = bytes_chunk / (s["bw_gbs"] * 1e9)
        t = max(t_c, t_m)
        rep["projection"][name] = dict(**s, chunk_s_projected=round(t, 3), bound="compute" if t_c >= t_m else "memory",
                                       fps_projected=round(12.0 / t, 2), fits_memory=(s["mem_gb"] >= rep["compressed"]["gen_uncapped"]["peak_reserved_GiB"] * 1.0737))
    json.dump(rep, open(os.path.join(a.out, "edge_sim.json"), "w"), indent=1)
    print("\n%-18s %7s %8s %6s %10s %8s %7s" % ("device", "TFLOPS", "GB/s", "mem", "chunk_s*", "fps*", "bound"))
    for k, v in rep["projection"].items():
        print("%-18s %7.0f %8.0f %6.0f %10.3f %8.2f %7s" % (k, v["tflops"], v["bw_gbs"], v["mem_gb"], v["chunk_s_projected"], v["fps_projected"], v["bound"]))
    print("* projected (roofline), not measured.\n[EDGE_SIM_DONE]", flush=True)


if __name__ == "__main__":
    main()
