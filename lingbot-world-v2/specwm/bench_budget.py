"""Reduce each budget item ONE AT A TIME and compare against the untouched model.

The 24 GB map has four rows. Each is a different kind of claim, so each needs a
different measurement:

  weight   35.1 GiB  -> INT4-g128     real storage change; measure bytes, time, quality
  KV cache 21.3 GiB  -> alpha-KV 5.4b fake-quant only (store mode unimplemented),
                                      so quality is measured and memory is analytic
  VAE peak  7.9 GiB  -> 2x2 tiling    real; known-approximate (global attention
                                      inside the decoder), so quality matters most
  activation 2.0 GiB -> (no knob)     a consequence of chunk size, reported only

ORDER MATTERS. Weight quantisation is destructive and cannot be undone in
process, so it runs LAST; everything before it shares one model load.

Quality is reported both in latent space and, after decoding, as PSNR against
the baseline pixels -- a latent delta is hard to interpret on its own.
"""
from __future__ import annotations
import argparse, json, os, time
import torch

GiB = 2 ** 30


def wbytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) + \
           sum(b.numel() * b.element_size() for b in model.buffers())


def psnr(a, b):
    a = a.float(); b = b.float()
    mse = float((a - b).pow(2).mean())
    rng = float(a.max() - a.min()) or 1.0
    return 99.0 if mse < 1e-12 else float(10 * torch.log10(torch.tensor(rng * rng / mse)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--action", required=True)
    ap.add_argument("--prompt", default="A serene lakeside scene with a lone tree.")
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--local-attn", type=int, default=18)
    ap.add_argument("--sink", type=int, default=6)
    ap.add_argument("--weight-spec", default="int4-g128")
    ap.add_argument("--weight-backend", default="ours",
                    choices=["ours", "torchao", "packed", "packed-mm", "awq"])
    ap.add_argument("--awq-alpha", type=float, default=0.25)
    ap.add_argument("--save-videos", action="store_true",
                    help="write mp4 per variant -- reference PSNR cannot judge "
                         "quality, only a human can")
    ap.add_argument("--kv-alpha", type=float, default=0.9)
    ap.add_argument("--tiles", default="2x2")
    ap.add_argument("--out", default="specwm_out/budget")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from .binding_v2 import V2Binding
    from .vae_tiled import tiled_decode
    from .kvquant import AlphaKVPolicy
    from .quantize import quantize_family

    rep = {"rows": {}}
    t0 = time.time()
    b = V2Binding(a.ckpt, target_mode="fast_bf16", quant=None, sink=a.sink,
                  local_attn=a.local_attn, swap_models=False, build_draft=False)
    cond = b.prepare(a.image, a.prompt, a.action, frame_num=a.frames)
    N = cond["num_chunks"]; sig = b.node_sigmas()
    tm = b.target_model
    print(f"[load] {time.time()-t0:.0f}s, {N} chunks, weights "
          f"{wbytes(tm)/GiB:.2f} GiB", flush=True)

    def gen_all(tag, seed=1234):
        b.reset_caches()
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        g = torch.Generator().manual_seed(seed)
        outs, ts = [], []
        for c in range(N):
            torch.cuda.synchronize(); t = time.perf_counter()
            z = b.chunk_noise(c, g); x0 = None
            for i in range(len(sig)):
                x0 = b.target_x0(z, i, c)
                if i < len(sig) - 1:
                    s = float(sig[i + 1]); z = (1 - s) * x0 + s * b.randn_like(x0, g)
            b.commit(x0, c)
            torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
            outs.append(x0.detach().float().cpu())
        st = sorted(ts[2:])
        info = dict(chunk_s_median=round(st[len(st) // 2], 4),
                    peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / GiB, 2),
                    weight_GiB=round(wbytes(tm) / GiB, 2))
        print(f"[{tag}] {info}", flush=True)
        return outs, info

    def save_mp4(img, name):
        if not a.save_videos:
            return
        # PNG frames first: every video codec available on this cluster failed
        # to decode its own output, so the frames are written directly with
        # matplotlib (pure python, no codec) for the visual comparison.
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as _plt
            v = ((img.clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).cpu().numpy()
            for fi in (10, 40, 70):
                if fi < v.shape[0]:
                    _plt.imsave(os.path.join(a.out, f"{name}__f{fi}.png"),
                                v[fi].clip(0, 1))
            print(f"[png] {name} frames saved", flush=True)
        except Exception as e:
            print(f"[png] {name} failed: {e}", flush=True)
        try:
            import imageio
            v = ((img.clamp(-1, 1) + 1) / 2 * 255).byte()
            v = v.permute(1, 2, 3, 0).cpu().numpy()
            imageio.mimwrite(os.path.join(a.out, f"{name}.mp4"), list(v),
                             fps=16, quality=8)
            print(f"[video] {name}.mp4", flush=True)
        except Exception as e:
            print(f"[video] {name} failed: {e}", flush=True)

    def decode(lat_cpu, tiles=(1, 1)):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        res = torch.cuda.memory_allocated()
        lat = torch.cat(lat_cpu, dim=1).to(b.device)
        t = time.time()
        img = tiled_decode(b.pipe.vae, lat, tiles=tiles, overlap=8, scale=8)
        torch.cuda.synchronize()
        info = dict(wall_s=round(time.time() - t, 2),
                    peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / GiB, 2),
                    over_resident_GiB=round(
                        (torch.cuda.max_memory_allocated() - res) / GiB, 2))
        del lat
        return img.detach().cpu(), info

    # ── 0. baseline ──────────────────────────────────────────
    ref, info = gen_all("baseline")
    img_ref, dinfo = decode(ref, (1, 1))
    save_mp4(img_ref, "00_baseline_bf16")
    rep["rows"]["baseline"] = dict(**info, decode=dinfo,
                                   latent_max_diff=0.0, pixel_psnr=99.0)

    # NOISE FLOOR. This model is a stochastic sampler, so a quantized variant
    # can produce an equally good video that is simply a DIFFERENT sample.
    # PSNR against the reference cannot tell that apart from real degradation,
    # so we measure how far two bf16 runs with different seeds already are.
    # Any variant at or above this floor is indistinguishable from resampling.
    outs2, info2 = gen_all("baseline-seed2", seed=4321)
    img2, _ = decode(outs2, (1, 1))
    save_mp4(img2, "01_bf16_seed2_NOISEFLOOR")
    rep["rows"]["noise_floor(seed2)"] = dict(
        **info2,
        latent_max_diff=round(max(float((x - y).abs().max())
                                  for x, y in zip(outs2, ref)), 5),
        pixel_psnr=round(psnr(img2, img_ref), 2),
        note="two bf16 runs, different seed -- this is the floor")
    del outs2, img2

    # ── 1. KV cache quantisation (non-destructive) ───────────
    b.kv_policy = AlphaKVPolicy(alpha_target=a.kv_alpha)
    outs, info = gen_all("kv-quant")
    kvrep = getattr(b, "_last_kv_report", None)
    img, _ = decode(outs, (1, 1))
    save_mp4(img, "02_kv_quant")
    rep["rows"]["kv_quant"] = dict(
        **info, kv_bits=kvrep,
        latent_max_diff=round(max(float((x - y).abs().max())
                                  for x, y in zip(outs, ref)), 5),
        pixel_psnr=round(psnr(img, img_ref), 2),
        note="fake-quant: quality is real, memory saving is analytic")
    b.kv_policy = None
    del outs, img

    # ── 2. VAE tiling (decode-side only) ─────────────────────
    r, c = (int(v) for v in a.tiles.lower().split("x"))
    img_t, dinfo_t = decode(ref, (r, c))
    save_mp4(img_t, "03_vae_tiled")
    rep["rows"]["vae_tiling"] = dict(
        chunk_s_median=rep["rows"]["baseline"]["chunk_s_median"],
        peak_alloc_GiB=rep["rows"]["baseline"]["peak_alloc_GiB"],
        weight_GiB=rep["rows"]["baseline"]["weight_GiB"],
        decode=dinfo_t, latent_max_diff=0.0,
        pixel_psnr=round(psnr(img_t, img_ref), 2), tiles=a.tiles)
    del img_t

    # ── 3. weight quantisation (destructive -> last) ─────────
    before = wbytes(tm)
    g = int(a.weight_spec.split("-g")[1]) if "-g" in a.weight_spec else 128
    if a.weight_backend == "torchao":
        from .quantize import quantize_torchao
        quantize_torchao(tm, spec=a.weight_spec.split("-")[0], group=g)
    elif a.weight_backend == "awq":
        # activation stats first: one calibration chunk through the bf16 model
        from .awq import ActStats, quantize_awq
        st = ActStats(tm, min_features=1024)
        b.reset_caches()
        gcal = torch.Generator().manual_seed(7)
        zc = b.chunk_noise(0, gcal)
        for i in range(len(sig)):
            _x0 = b.target_x0(zc, i, 0)
            if i < len(sig) - 1:
                ss = float(sig[i + 1]); zc = (1 - ss) * _x0 + ss * b.randn_like(_x0, gcal)
        means = st.finish()
        print(f"[awq] calibration collected for {len(means)} layers", flush=True)
        quantize_awq(tm, means, alpha=a.awq_alpha, group=g)
    elif a.weight_backend in ("packed", "packed-mm"):
        from .quantize import quantize_packed_int4
        quantize_packed_int4(tm, group=g, search=(a.weight_backend == "packed"))
    else:
        quantize_family(tm, a.weight_spec)
    torch.cuda.empty_cache()
    after = wbytes(tm)
    outs, info = gen_all(f"weight-{a.weight_spec}")
    img, _ = decode(outs, (1, 1))
    save_mp4(img, f"04_weight_{a.weight_backend}_{a.weight_spec}")
    rep["rows"]["weight_quant"] = dict(
        **info, spec=a.weight_spec, backend=a.weight_backend,
        weight_before_GiB=round(before / GiB, 2), weight_after_GiB=round(after / GiB, 2),
        latent_max_diff=round(max(float((x - y).abs().max())
                                  for x, y in zip(outs, ref)), 5),
        pixel_psnr=round(psnr(img, img_ref), 2))

    with open(os.path.join(a.out, "budget.json"), "w") as f:
        json.dump(rep, f, indent=1)

    print("\n" + "=" * 96)
    print(f"{'항목':16}{'weight':>9}{'gen peak':>10}{'chunk s':>9}"
          f"{'decode peak':>13}{'latent차':>10}{'pixel PSNR':>12}")
    print("-" * 96)
    for k, v in rep["rows"].items():
        d = v.get("decode", {})
        print(f"{k:16}{v['weight_GiB']:>8.1f}G{v['peak_alloc_GiB']:>9.1f}G"
              f"{v['chunk_s_median']:>9.3f}{d.get('over_resident_GiB', 0):>12.2f}G"
              f"{v['latent_max_diff']:>10.4f}{v['pixel_psnr']:>12.2f}")
    print("=" * 96)
    print(f"saved {os.path.join(a.out,'budget.json')}")


if __name__ == "__main__":
    main()
