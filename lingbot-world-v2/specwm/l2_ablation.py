"""ONE job that measures all four features in sequence.

Configs (each re-inits the binding, ~90 s model load, then generates 7 chunks):
  A baseline    int8-g128, separate caches            <- today's reference
  B share-kv    int8-g128, draft reads target cache   <- feature (1)
  C fp8         fp8 draft + share-kv                  <- feature (3)
  D full        C + quad tiles + delta-lag + alpha-KV <- features (2)+(4)

Reports per config: peak GPU memory, wallclock, acceptance, rounds, target
passes, and (for D) the KV bit histogram.  Writes ablation.json + prints a
table.  Designed to be submitted ONCE.
"""
import argparse, json, os, time, torch


# share_kv is OFF in every config. Measured on the 14B model, sharing the KV
# cache saved 21.1 GiB but drove acceptance to 0 from the very first node:
# the cache is a STATEFUL object (advancing local/global_end_index plus a
# sliding-window eviction that physically rolls its contents), so the draft
# does not merely read it -- it mutates it and corrupts the target.
# Memory is reclaimed instead by quantizing the draft's OWN cache (E2).
#
# Each config changes exactly ONE thing vs A_baseline, so effects are isolated.
CONFIGS = [
    ("A_baseline",    dict(quant="int8-g128", share_kv=False, tile=None,   lag=0, kvq=None)),
    ("C2_fp8",        dict(quant="fp8",       share_kv=False, tile=None,   lag=0, kvq=None)),
    ("D2_tile_exact", dict(quant="int8-g128", share_kv=False, tile="quad", lag=0, kvq=None)),
    ("D2_tile_lag2",  dict(quant="int8-g128", share_kv=False, tile="quad", lag=2, kvq=None)),
    ("E2_kvquant",    dict(quant="int8-g128", share_kv=False, tile=None,   lag=0, kvq=0.9)),
]


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--action", required=True)
    ap.add_argument("--prompt", default="A serene lakeside scene with a lone tree.")
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--local-attn", type=int, default=18)
    ap.add_argument("--sink", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--only", nargs="*", default=None, help="subset of config names")
    ap.add_argument("--out", default="specwm_out/ablation")
    return ap.parse_args()


def run_one(a, name, cfg, report):
    from .binding_v2 import V2Binding
    from .core import SpecStats
    from .l2_spec_chunk import ChunkSpec
    from .tiled import TiledChunkSpec
    from .kvquant import AlphaKVPolicy
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t_load = time.time()
    pol = AlphaKVPolicy(alpha_target=cfg["kvq"]) if cfg["kvq"] else None
    b = V2Binding(a.ckpt, target_mode="fast_bf16", quant=cfg["quant"],
                  sink=a.sink, local_attn=a.local_attn,
                  share_kv=cfg["share_kv"], kv_policy=pol,
                  swap_models=True)
    cond = b.prepare(a.image, a.prompt, a.action, frame_num=a.frames)
    mem_after_setup = torch.cuda.memory_allocated() / 2**30
    st = SpecStats()
    spec = (TiledChunkSpec(b, b.node_sigmas(), st, tile_mode=cfg["tile"], lag=cfg["lag"])
            if cfg["tile"] else ChunkSpec(b, b.node_sigmas(), st, target_substeps=1))
    gen = torch.Generator().manual_seed(a.seed)
    t0 = time.time()
    chunks = [spec.run_chunk(c, cond, gen) for c in range(cond["num_chunks"])]
    wall = time.time() - t0
    sm = st.summary()
    rec = dict(config=name, **{k: (v if not isinstance(v, bool) else int(v))
                               for k, v in cfg.items()},
               load_s=round(t_load and time.time() - t_load - wall, 1),
               setup_mem_GiB=round(mem_after_setup, 1),
               peak_mem_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 1),
               wallclock_s=round(wall, 1),
               accept=round(sm["overall_accept"], 4),
               mean_rounds=round(sm["mean_rounds"], 3),
               target_passes=sm["target_passes"],
               draft_calls=sm["draft_calls"],
               residual_fallbacks=sm["residual_fallbacks"],
               per_node_accept=sm["per_node_accept_emp"])
    if getattr(b, "_last_kv_report", None):
        rec["kv_bits"] = b._last_kv_report
    report.append(rec)
    print("\n" + json.dumps(rec, indent=1), flush=True)
    # save video for visual check
    try:
        import imageio
        lat = torch.cat(chunks, dim=1)
        vid = b.pipe.vae.decode([lat.to(b.device)])[0]
        v = ((vid.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 3, 0).cpu().numpy()
        imageio.mimwrite(os.path.join(a.out, f"{name}.mp4"), list(v), fps=16, quality=8)
        torch.save(lat.cpu(), os.path.join(a.out, f"latent_{name}.pt"))
    except Exception as e:
        print(f"[{name}] video save failed: {e}", flush=True)
    del b, spec
    import gc; gc.collect(); torch.cuda.empty_cache()


def main():
    a = get_args(); os.makedirs(a.out, exist_ok=True)
    report = []
    names = a.only or [n for n, _ in CONFIGS]
    for name, cfg in CONFIGS:
        if name not in names:
            continue
        print(f"\n{'='*70}\n=== {name}  {cfg}\n{'='*70}", flush=True)
        try:
            run_one(a, name, cfg, report)
        except Exception as e:
            print(f"[{name}] FAILED: {type(e).__name__}: {e}", flush=True)
            report.append(dict(config=name, error=f"{type(e).__name__}: {e}"))
        with open(os.path.join(a.out, "ablation.json"), "w") as f:
            json.dump(report, f, indent=1)
    # summary table
    print("\n\n" + "=" * 96)
    print(f"{'config':12s} {'peak GiB':>9s} {'wall s':>8s} {'accept':>8s} "
          f"{'rounds':>7s} {'passes':>7s} {'fallbk':>7s}")
    print("-" * 96)
    for r in report:
        if "error" in r:
            print(f"{r['config']:12s}  ERROR: {r['error'][:60]}")
        else:
            print(f"{r['config']:12s} {r['peak_mem_GiB']:9.1f} {r['wallclock_s']:8.1f} "
                  f"{r['accept']:8.3f} {r['mean_rounds']:7.2f} {r['target_passes']:7d} "
                  f"{r['residual_fallbacks']:7d}")
    print("=" * 96)
    print(f"saved {os.path.join(a.out, 'ablation.json')}")


if __name__ == "__main__":
    main()
