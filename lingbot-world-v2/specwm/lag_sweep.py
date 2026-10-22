"""Accuracy-vs-speed curve for the delta-lag knob (paper figure).

Loads the 14B model ONCE and reuses it for every point, so the sweep costs one
model load instead of one per config.

Points:
  target        pure target chain, no speculation      -> the reference output
  target(seed2) same, different seed                   -> NOISE FLOOR
  lag=0         exact tile-factorized speculation      -> should sit at the floor
  lag=1..4      approximate, staleness carried k nodes -> deviation should grow

Accuracy metric: latent PSNR against the `target` reference. Because a
different RNG path is consumed, even an EXACT sampler does not reproduce the
reference bit-for-bit, so the honest baseline is the seed-to-seed noise floor:
any lag whose deviation stays near that floor is empirically indistinguishable
from exact sampling.
"""
from __future__ import annotations
import argparse, json, os, time, torch


def psnr(a, b):
    mse = float((a.float() - b.float()).pow(2).mean())
    rng = float(a.float().max() - a.float().min()) or 1.0
    return 10.0 * torch.log10(torch.tensor(rng * rng / max(mse, 1e-12))).item(), mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--action", required=True)
    ap.add_argument("--prompt", default="A serene lakeside scene with a lone tree.")
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--local-attn", type=int, default=18)
    ap.add_argument("--sink", type=int, default=6)
    ap.add_argument("--quant", default="int8-g128")
    ap.add_argument("--tile-mode", default="quad")
    ap.add_argument("--lags", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--trees", type=int, nargs="*", default=[2, 4],
                    help="tree-speculation branch counts K (EXACT)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--seed2", type=int, default=777)
    ap.add_argument("--out", default="specwm_out/lagsweep")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from .binding_v2 import V2Binding
    from .core import SpecStats
    from .tiled import TiledChunkSpec
    from .tree import TreeChunkSpec
    from .l2_compare import plain_chunks

    t0 = time.time()
    b = V2Binding(a.ckpt, target_mode="fast_bf16", quant=a.quant,
                  sink=a.sink, local_attn=a.local_attn, swap_models=True)
    cond = b.prepare(a.image, a.prompt, a.action, frame_num=a.frames)
    N = cond["num_chunks"]
    print(f"[load] {time.time()-t0:.0f}s, {N} chunks", flush=True)

    rows, lats = [], {}

    def save(tag, chunks):
        lat = torch.cat(chunks, dim=1).cpu()
        lats[tag] = lat
        torch.save(lat, os.path.join(a.out, f"latent_{tag}.pt"))
        try:
            import imageio
            v = b.pipe.vae.decode([lat.to(b.device)])[0]
            v = ((v.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 3, 0).cpu().numpy()
            imageio.mimwrite(os.path.join(a.out, f"{tag}.mp4"), list(v), fps=16, quality=8)
        except Exception as e:
            print(f"[{tag}] video failed: {e}", flush=True)

    # ---------- reference + noise floor ----------
    for tag, sd in [("target", a.seed), ("target_seed2", a.seed2)]:
        b.reset_caches(); torch.cuda.reset_peak_memory_stats()
        t = time.time(); ch = plain_chunks(b, N, "target", sd); w = time.time() - t
        save(tag, ch)
        rows.append(dict(point=tag, lag=None, wallclock_s=round(w, 1),
                         target_passes=N * len(b.node_sigmas()),
                         peak_mem_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 1)))
        print(f"[{tag}] {w:.1f}s", flush=True)

    # ---------- lag sweep ----------
    for lag in a.lags:
        b.reset_caches(); torch.cuda.reset_peak_memory_stats()
        st = SpecStats()
        spec = TiledChunkSpec(b, b.node_sigmas(), st, tile_mode=a.tile_mode, lag=lag)
        g = torch.Generator().manual_seed(a.seed)
        t = time.time()
        ch = [spec.run_chunk(c, cond, g) for c in range(N)]
        w = time.time() - t
        save(f"lag{lag}", ch)
        sm = st.summary()
        rows.append(dict(point=f"lag{lag}", lag=lag, wallclock_s=round(w, 1),
                         target_passes=sm["target_passes"], draft_calls=sm["draft_calls"],
                         accept=round(sm["overall_accept"], 4),
                         mean_rounds=round(sm["mean_rounds"], 3),
                         residual_fallbacks=sm["residual_fallbacks"],
                         peak_mem_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 1)))
        print(f"[lag{lag}] {w:.1f}s passes={sm['target_passes']} "
              f"a={sm['overall_accept']:.3f}", flush=True)

    # ---------- tree speculation (exact, pre-rolled branches) ----------
    for K in a.trees:
        b.reset_caches(); torch.cuda.reset_peak_memory_stats()
        st = SpecStats()
        spec = TreeChunkSpec(b, b.node_sigmas(), st, K=K)
        g = torch.Generator().manual_seed(a.seed)
        t = time.time()
        ch = [spec.run_chunk(c, cond, g) for c in range(N)]
        w = time.time() - t
        save(f"tree{K}", ch)
        sm = st.summary()
        rows.append(dict(point=f"tree{K}", lag=None, tree_K=K, wallclock_s=round(w, 1),
                         target_passes=sm["target_passes"],
                         target_evals=sm.get("target_evals"),
                         draft_calls=sm["draft_calls"],
                         accept=round(sm["overall_accept"], 4),
                         mean_rounds=round(sm["mean_rounds"], 3),
                         residual_fallbacks=sm["residual_fallbacks"],
                         peak_mem_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 1)))
        print(f"[tree{K}] {w:.1f}s passes={sm['target_passes']} "
              f"a={sm['overall_accept']:.3f}", flush=True)

    # ---------- accuracy vs the reference ----------
    ref = lats["target"]
    for r in rows:
        p, m = psnr(ref, lats[r["point"]])
        r["psnr_vs_target"] = round(p, 2)
        r["mse_vs_target"] = m
    floor = [r for r in rows if r["point"] == "target_seed2"][0]["psnr_vs_target"]
    for r in rows:
        r["above_noise_floor"] = bool(r["psnr_vs_target"] >= floor)
    with open(os.path.join(a.out, "lagsweep.json"), "w") as f:
        json.dump(dict(noise_floor_psnr=floor, rows=rows), f, indent=1)

    print("\n" + "=" * 88)
    print(f"noise floor (target vs target seed2) = {floor:.2f} dB PSNR")
    print(f"{'point':14s}{'wall s':>9s}{'passes':>8s}{'alpha':>8s}"
          f"{'rounds':>8s}{'PSNR dB':>10s}{'>=floor':>9s}")
    print("-" * 88)
    for r in rows:
        print(f"{r['point']:14s}{r['wallclock_s']:9.1f}{r['target_passes']:8d}"
              f"{r.get('accept', float('nan')):8.3f}{r.get('mean_rounds', float('nan')):8.2f}"
              f"{r['psnr_vs_target']:10.2f}{str(r['above_noise_floor']):>9s}")
    print("=" * 88)


if __name__ == "__main__":
    main()
