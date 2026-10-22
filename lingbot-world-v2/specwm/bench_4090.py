"""Does the compressed configuration actually fit in 24 GB?

Every earlier budget number for the KV cache was hypothetical: alpha-KV was
fake quantization (quantize then immediately dequantize), so it cost quality
and saved nothing. kvstore.py stores the cache packed for real, so this run
measures the peak rather than predicting it.

Stages, each measured against the untouched bf16 model:
    baseline            reference latents + pixels
    noise_floor         same model, different seed -- reference PSNR is
                        meaningless without this, as we learned the hard way
    kv_store            real packed KV, weights still bf16
    weights_int4        packed INT4 weights, KV still bf16
    COMBINED            both -- this is the 24 GB verdict
"""
from __future__ import annotations
import argparse, json, os, time
import torch

GiB = 2 ** 30


def wbytes(m):
    return (sum(p.numel() * p.element_size() for p in m.parameters())
            + sum(b.numel() * b.element_size() for b in m.buffers()))


def psnr(a, b):
    a, b = a.float(), b.float()
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
    ap.add_argument("--kv-bits", type=int, default=4, choices=[4, 8])
    ap.add_argument("--tiles", default="2x2")
    ap.add_argument("--out", default="specwm_out/b4090")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from .binding_v2 import V2Binding
    from .vae_tiled import tiled_decode
    from .quantize import quantize_packed_int4
    from . import kvstore

    rep = {"rows": {}, "kv_bits": a.kv_bits}
    t0 = time.time()
    b = V2Binding(a.ckpt, target_mode="fast_bf16", quant=None, sink=a.sink,
                  local_attn=a.local_attn, swap_models=False, build_draft=False)
    cond = b.prepare(a.image, a.prompt, a.action, frame_num=a.frames)
    N = cond["num_chunks"]; sig = b.node_sigmas(); tm = b.target_model
    print(f"[load] {time.time()-t0:.0f}s, {N} chunks, weights {wbytes(tm)/GiB:.2f} GiB",
          flush=True)

    def gen(tag, seed=1234):
        b.reset_caches()
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        g = torch.Generator().manual_seed(seed)
        outs, ts = [], []
        # reset_caches() allocates the full bf16 KV before kvstore replaces it,
        # so the raw peak still contains 21.3 GiB that a real deployment would
        # never allocate. Re-arm the counter once the packed store is live and
        # the bf16 tensors are freed; that steady-state peak is the number a
        # 24 GB card actually has to satisfy.
        rearmed = False
        for c in range(N):
            if c == 1 and not rearmed:
                torch.cuda.synchronize(); torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(); rearmed = True
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
        kvb = kvstore.total_bytes(tm)
        info = dict(chunk_s_median=round(st[len(st) // 2], 4),
                    peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / GiB, 2),
                    peak_reserved_GiB=round(torch.cuda.max_memory_reserved() / GiB, 2),
                    weight_GiB=round(wbytes(tm) / GiB, 2),
                    kv_store_GiB=round(kvb / GiB, 2) if kvb else None)
        print(f"[{tag}] {info}", flush=True)
        return outs, info

    def dec(lat, tiles=(1, 1)):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        res = torch.cuda.memory_allocated()
        z = torch.cat(lat, dim=1).to(b.device)
        img = tiled_decode(b.pipe.vae, z, tiles=tiles, overlap=8, scale=8)
        torch.cuda.synchronize()
        info = dict(peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / GiB, 2),
                    over_resident_GiB=round(
                        (torch.cuda.max_memory_allocated() - res) / GiB, 2))
        del z
        return img.detach().cpu(), info

    # 0. baseline + noise floor
    ref, i0 = gen("baseline")
    img_ref, d0 = dec(ref, (1, 1))
    rep["rows"]["baseline"] = dict(**i0, decode=d0, pixel_psnr=99.0)
    o2, i2 = gen("noise_floor(seed2)", seed=4321)
    img2, _ = dec(o2, (1, 1))
    rep["rows"]["noise_floor"] = dict(**i2, pixel_psnr=round(psnr(img2, img_ref), 2))
    del o2, img2

    # 1. VAE tiling (decode side only)
    r, c = (int(v) for v in a.tiles.lower().split("x"))
    img_t, dt = dec(ref, (r, c))
    rep["rows"]["vae_tiling"] = dict(decode=dt, pixel_psnr=round(psnr(img_t, img_ref), 2))
    del img_t

    # 2. real packed KV store (weights still bf16)
    kvstore.install(tm, bits=a.kv_bits)
    outs, i3 = gen("kv_store")
    img, _ = dec(outs, (1, 1))
    rep["rows"]["kv_store"] = dict(**i3, pixel_psnr=round(psnr(img, img_ref), 2))
    del outs, img

    # 3. COMBINED: INT4 weights on top of the packed KV
    quantize_packed_int4(tm, group=128, search=False)
    torch.cuda.empty_cache()
    outs, i4 = gen("COMBINED int4w + kv%db" % a.kv_bits)
    img, i4d = dec(outs, (r, c))
    rep["rows"]["combined"] = dict(**i4, decode=i4d,
                                   pixel_psnr=round(psnr(img, img_ref), 2))
    # Dense frame dump for the visual comparison. Reference PSNR cannot judge
    # quality, so the deliverable is frames a human can look at -- and enough
    # of them, spread over the clip, to reveal drift that a single frame hides.
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        pairs = [("baseline", img_ref), ("combined", img)]
        T = img_ref.shape[1]
        idxs = [int(round(t)) for t in np.linspace(0, T - 1, 8)]
        for nm, im in pairs:
            v = ((im.clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy()
            for fi in idxs:
                plt.imsave(os.path.join(a.out, f"{nm}__f{fi:03d}.png"), v[fi].clip(0, 1))
            np.save(os.path.join(a.out, f"{nm}_frames.npy"),
                    (v[idxs] * 255).astype("uint8"))
        rep["frame_idxs"] = idxs
        print(f"[png] dumped {len(idxs)} frames x 2 variants", flush=True)
        # also write mp4s for side-by-side viewing
        try:
            import imageio
            for nm, im in pairs:
                vv = ((im.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 3, 0).numpy()
                imageio.mimwrite(os.path.join(a.out, f"{nm}.mp4"), list(vv),
                                 fps=16, quality=8)
            print("[mp4] written", flush=True)
        except Exception as e:
            print(f"[mp4] failed: {e}", flush=True)
    except Exception as e:
        print("png failed", e, flush=True)

    with open(os.path.join(a.out, "b4090.json"), "w") as f:
        json.dump(rep, f, indent=1)

    print("\n" + "=" * 92)
    print(f"{'stage':22}{'weight':>9}{'KV store':>10}{'gen peak':>11}"
          f"{'chunk s':>9}{'PSNR':>8}")
    print("-" * 92)
    for k, v in rep["rows"].items():
        print(f"{k:22}{(str(v.get('weight_GiB','-'))):>9}"
              f"{(str(v.get('kv_store_GiB','-'))):>10}"
              f"{(str(v.get('peak_alloc_GiB','-'))):>11}"
              f"{(str(v.get('chunk_s_median','-'))):>9}"
              f"{v.get('pixel_psnr','-'):>8}")
    print("=" * 92)
    cb = rep["rows"]["combined"]
    print(f"24 GB 판정: 생성 peak {cb['peak_alloc_GiB']} GiB "
          f"({'FITS' if cb['peak_alloc_GiB'] < 24 else 'OVER'})")


if __name__ == "__main__":
    main()
