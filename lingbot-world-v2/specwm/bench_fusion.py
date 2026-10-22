"""Does kernel fusion (torch.compile) recover the elementwise 33.9%?

Kernel-level profiling of one chunk (2,279.9 ms of GPU work) gave:
    GEMM bf16    882.9 ms  38.7%
    elementwise  772.9 ms  33.9%   <- 11,637 launches, memory bound
    attention    434.0 ms  19.0%   (already cuDNN flash)
    GEMM fp32    165.8 ms   7.3%   (autocast(float32) region)
Elementwise is the largest fusable target, so we test torch.compile.

PREDICTION TO CHECK: the local-attention path calls .item() on the cache
index tensors ~630x per chunk. Each is a host sync and therefore a DYNAMO
GRAPH BREAK, which would fragment exactly the region we want fused. We log
graph-break counts so a null result is explainable rather than mysterious.

Also measures the last empty cell of the 24 GB budget map: VAE decode peak
when decoding PER CHUNK (streaming) instead of the whole clip at once.
"""
from __future__ import annotations
import argparse, json, os, time
import torch

GiB = 2 ** 30


def dynamo_stats():
    try:
        from torch._dynamo.utils import counters
        gb = dict(counters.get("graph_break", {}))
        return dict(graph_breaks=int(sum(gb.values())),
                    top=sorted(gb.items(), key=lambda x: -x[1])[:6])
    except Exception as e:
        return dict(error=str(e))


def reset_dynamo():
    try:
        from torch._dynamo.utils import counters
        counters.clear()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--action", required=True)
    ap.add_argument("--prompt", default="A serene lakeside scene with a lone tree.")
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--local-attn", type=int, default=18)
    ap.add_argument("--sink", type=int, default=6)
    ap.add_argument("--modes", nargs="*", default=["baseline", "compile"])
    ap.add_argument("--compile-mode", default="default",
                    choices=["default", "reduce-overhead", "max-autotune"])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default="specwm_out/fusion")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from .binding_v2 import V2Binding
    rep = {"compile_mode": a.compile_mode, "modes": {}}

    t0 = time.time()
    b = V2Binding(a.ckpt, target_mode="fast_bf16", quant=None,
                  sink=a.sink, local_attn=a.local_attn,
                  swap_models=False, build_draft=False)
    cond = b.prepare(a.image, a.prompt, a.action, frame_num=a.frames)
    N = cond["num_chunks"]
    sig = b.node_sigmas()
    print(f"[load] {time.time()-t0:.0f}s, {N} chunks", flush=True)
    tm = b.target_model
    orig_blocks = list(tm.blocks)

    def run_chunk(c, gen):
        z = b.chunk_noise(c, gen); x0 = None
        for i in range(len(sig)):
            x0 = b.target_x0(z, i, c)
            if i < len(sig) - 1:
                s = float(sig[i + 1]); z = (1 - s) * x0 + s * b.randn_like(x0, gen)
        b.commit(x0, c)
        return x0

    outs_for_vae = None
    for mode in a.modes:
        print(f"\n{'='*64}\n=== {mode}\n{'='*64}", flush=True)
        # restore pristine blocks, then optionally compile
        for i, blk in enumerate(orig_blocks):
            tm.blocks[i] = blk
        reset_dynamo()
        compile_s = 0.0
        if mode == "compile":
            tc = time.time()
            for i in range(len(tm.blocks)):
                tm.blocks[i] = torch.compile(tm.blocks[i], mode=a.compile_mode,
                                             dynamic=False)
            compile_s = time.time() - tc
        b.reset_caches()
        gen = torch.Generator().manual_seed(1234)

        # warmup (compilation happens on the first chunk)
        tw = time.time()
        for c in range(min(a.warmup, N)):
            run_chunk(c, gen)
        torch.cuda.synchronize()
        warm_s = time.time() - tw

        # timed steady-state chunks
        times, outs = [], []
        torch.cuda.reset_peak_memory_stats()
        for c in range(a.warmup, N):
            torch.cuda.synchronize(); t = time.perf_counter()
            outs.append(run_chunk(c, gen))
            torch.cuda.synchronize(); times.append(time.perf_counter() - t)
        m = sorted(times)[len(times) // 2]
        rep["modes"][mode] = dict(
            compile_s=round(compile_s, 1), warmup_s=round(warm_s, 1),
            chunk_s_median=round(m, 4),
            chunk_s_all=[round(x, 4) for x in times],
            peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / GiB, 2),
            peak_reserved_GiB=round(torch.cuda.max_memory_reserved() / GiB, 2),
            dynamo=dynamo_stats())
        print(f"[{mode}] chunk median {m:.3f}s  "
              f"(compile {compile_s:.0f}s, warmup {warm_s:.0f}s)", flush=True)
        print(f"[{mode}] dynamo: {rep['modes'][mode]['dynamo']}", flush=True)
        if outs_for_vae is None:
            outs_for_vae = outs

    # ── VAE decode: streaming (per chunk) vs whole clip ──
    print(f"\n{'='*64}\n=== VAE decode\n{'='*64}", flush=True)
    vae = {}
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t = time.time()
        for o in outs_for_vae:
            _ = b.pipe.vae.decode([o.to(b.device)])[0]
            del _
        torch.cuda.synchronize()
        vae["per_chunk"] = dict(
            wall_s=round(time.time() - t, 2),
            peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / GiB, 2),
            peak_reserved_GiB=round(torch.cuda.max_memory_reserved() / GiB, 2),
            over_resident_GiB=round((torch.cuda.max_memory_allocated() - base) / GiB, 2),
            chunks=len(outs_for_vae))
    except Exception as e:
        vae["per_chunk"] = {"error": f"{type(e).__name__}: {e}"}
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        lat = torch.cat(outs_for_vae, dim=1).to(b.device)
        t = time.time()
        _ = b.pipe.vae.decode([lat])[0]
        torch.cuda.synchronize()
        vae["whole_clip"] = dict(
            wall_s=round(time.time() - t, 2),
            peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / GiB, 2),
            peak_reserved_GiB=round(torch.cuda.max_memory_reserved() / GiB, 2),
            over_resident_GiB=round((torch.cuda.max_memory_allocated() - base) / GiB, 2))
        del _, lat
    except Exception as e:
        vae["whole_clip"] = {"error": f"{type(e).__name__}: {e}"}
    rep["vae"] = vae

    with open(os.path.join(a.out, "fusion.json"), "w") as f:
        json.dump(rep, f, indent=1)

    print("\n" + "=" * 76)
    print(f"{'mode':12}{'chunk s':>10}{'vs base':>10}{'peak alloc':>13}{'graph breaks':>15}")
    print("-" * 76)
    base_t = rep["modes"].get("baseline", {}).get("chunk_s_median")
    for k, v in rep["modes"].items():
        sp = f"{base_t/v['chunk_s_median']:.2f}x" if base_t else "-"
        print(f"{k:12}{v['chunk_s_median']:>10.3f}{sp:>10}"
              f"{v['peak_alloc_GiB']:>12.2f}G{str(v['dynamo'].get('graph_breaks','?')):>15}")
    print("\nVAE decode:")
    for k, v in vae.items():
        if "error" in v: print(f"  {k:12} ERROR {v['error'][:50]}")
        else: print(f"  {k:12} {v['wall_s']:>6.2f}s   peak {v['peak_alloc_GiB']:>6.2f}G alloc / "
                    f"{v['peak_reserved_GiB']:>6.2f}G resv   상주초과 {v['over_resident_GiB']:>5.2f}G")
    print(f"\nsaved {os.path.join(a.out,'fusion.json')}")


if __name__ == "__main__":
    main()
