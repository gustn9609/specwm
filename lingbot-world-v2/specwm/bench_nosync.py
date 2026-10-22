"""Two experiments in one job.

A) Does removing the .item() host syncs unlock kernel fusion?
   baseline -> nosync -> nosync+compile. Correctness is checked by comparing
   the produced latents against baseline: the patch rewrites the KV-cache
   bookkeeping, so a speedup that changes the output is worthless.

B) Does spatial tiling shrink the VAE decode peak (7.6 GiB, the cell that
   currently pushes the 24 GB budget to 25.6 GB)?
"""
from __future__ import annotations
import argparse, json, os, time
import torch

GiB = 2 ** 30


def gb_stats():
    try:
        from torch._dynamo.utils import counters
        d = dict(counters.get("graph_break", {}))
        return int(sum(d.values())), sorted(d.items(), key=lambda x: -x[1])[:4]
    except Exception:
        return -1, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--action", required=True)
    ap.add_argument("--prompt", default="A serene lakeside scene with a lone tree.")
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--local-attn", type=int, default=18)
    ap.add_argument("--sink", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--tiles", nargs="*", default=["1x1", "2x2", "3x3"])
    ap.add_argument("--out", default="specwm_out/nosync")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from .binding_v2 import V2Binding
    from . import nosync_patch
    from .vae_tiled import tiled_decode

    rep = {"modes": {}, "vae": {}}
    t0 = time.time()
    b = V2Binding(a.ckpt, target_mode="fast_bf16", quant=None, sink=a.sink,
                  local_attn=a.local_attn, swap_models=False, build_draft=False)
    cond = b.prepare(a.image, a.prompt, a.action, frame_num=a.frames)
    N = cond["num_chunks"]; sig = b.node_sigmas()
    print(f"[load] {time.time()-t0:.0f}s, {N} chunks", flush=True)
    tm = b.target_model
    pristine = list(tm.blocks)

    def run_all(seed=1234):
        b.reset_caches()
        g = torch.Generator().manual_seed(seed)
        outs, times = [], []
        for c in range(N):
            torch.cuda.synchronize(); t = time.perf_counter()
            z = b.chunk_noise(c, g); x0 = None
            for i in range(len(sig)):
                x0 = b.target_x0(z, i, c)
                if i < len(sig) - 1:
                    s = float(sig[i + 1]); z = (1 - s) * x0 + s * b.randn_like(x0, g)
            b.commit(x0, c)
            torch.cuda.synchronize(); times.append(time.perf_counter() - t)
            outs.append(x0.detach().float().cpu())
        return outs, times

    ref = None
    for mode in ["baseline", "nosync", "nosync+compile"]:
        print(f"\n{'='*62}\n=== {mode}\n{'='*62}", flush=True)
        for i, blk in enumerate(pristine):
            tm.blocks[i] = blk
        try:
            from torch._dynamo.utils import counters; counters.clear()
            torch._dynamo.reset()
        except Exception:
            pass
        # rebuild pristine forwards for the baseline
        if mode == "baseline":
            for m in tm.modules():
                if getattr(m, "_nosync", False):
                    del m.forward; m._nosync = False
        else:
            nosync_patch.apply(tm)
        comp_s = 0.0
        if mode.endswith("compile"):
            tc = time.time()
            for i in range(len(tm.blocks)):
                tm.blocks[i] = torch.compile(tm.blocks[i], dynamic=False)
            comp_s = time.time() - tc

        run_all()                                   # warmup / compile
        outs, times = run_all()                     # measured
        steady = sorted(times[a.warmup:])
        med = steady[len(steady) // 2]
        n_gb, top_gb = gb_stats()
        drift = None
        if ref is None:
            ref = outs
        else:
            drift = max(float((x - y).abs().max()) for x, y in zip(outs, ref))
        rep["modes"][mode] = dict(
            chunk_s_median=round(med, 4), compile_s=round(comp_s, 1),
            all_s=[round(x, 4) for x in times],
            peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / GiB, 2),
            graph_breaks=n_gb, graph_break_top=[t[0][:60] for t in top_gb],
            max_abs_diff_vs_baseline=drift)
        print(f"[{mode}] chunk {med:.3f}s  breaks={n_gb}  "
              f"diff_vs_base={drift}", flush=True)

    # ── B: VAE tiling ──
    print(f"\n{'='*62}\n=== VAE tiling\n{'='*62}", flush=True)
    lat = torch.cat([o for o in ref], dim=1).to(b.device)
    base_img = None
    for spec in a.tiles:
        r, c = (int(v) for v in spec.lower().split("x"))
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            resident = torch.cuda.memory_allocated()
            t = time.time()
            img = tiled_decode(b.pipe.vae, lat, tiles=(r, c), overlap=8, scale=8)
            torch.cuda.synchronize()
            d = None
            if base_img is None:
                base_img = img.detach().cpu()
            else:
                d = float((img.detach().cpu() - base_img).abs().max())
            rep["vae"][spec] = dict(
                wall_s=round(time.time() - t, 2),
                peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / GiB, 2),
                over_resident_GiB=round(
                    (torch.cuda.max_memory_allocated() - resident) / GiB, 2),
                max_abs_diff_vs_1x1=d)
            print(f"[vae {spec}] {rep['vae'][spec]}", flush=True)
            del img
        except Exception as e:
            rep["vae"][spec] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[vae {spec}] ERROR {e}", flush=True)

    with open(os.path.join(a.out, "nosync.json"), "w") as f:
        json.dump(rep, f, indent=1)

    print("\n" + "=" * 84)
    print(f"{'mode':18}{'chunk s':>10}{'vs base':>9}{'breaks':>9}{'출력차이':>14}")
    print("-" * 84)
    bt = rep["modes"]["baseline"]["chunk_s_median"]
    for k, v in rep["modes"].items():
        d = v["max_abs_diff_vs_baseline"]
        print(f"{k:18}{v['chunk_s_median']:>10.3f}{bt/v['chunk_s_median']:>8.2f}x"
              f"{v['graph_breaks']:>9}{('-' if d is None else f'{d:.2e}'):>14}")
    print(f"\n{'VAE tiles':12}{'wall s':>9}{'peak':>10}{'상주초과':>12}{'1x1 대비 차이':>16}")
    print("-" * 84)
    for k, v in rep["vae"].items():
        if "error" in v: print(f"{k:12} ERROR {v['error'][:48]}")
        else:
            d = v["max_abs_diff_vs_1x1"]
            print(f"{k:12}{v['wall_s']:>9.2f}{v['peak_alloc_GiB']:>9.2f}G"
                  f"{v['over_resident_GiB']:>11.2f}G"
                  f"{('-' if d is None else f'{d:.2e}'):>16}")


if __name__ == "__main__":
    main()
