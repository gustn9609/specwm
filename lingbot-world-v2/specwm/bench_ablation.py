"""One-factor-at-a-time ablation from the bf16 original to the 24 GB build.

Each arm changes exactly ONE thing versus the untouched model, and every arm
writes a video, because reference PSNR has repeatedly proved unable to tell
"degraded" from "a different but equally valid sample" -- only looking at the
output can.

    A_original      bf16 weights, bf16 KV, untiled decode      (reference)
    B_seed2         identical model, different seed            (NOISE FLOOR)
    C_vae_tiled     original latents, 2x2 tiled decode
    D_kv4           4-bit packed KV, weights untouched
    E_int4w         INT4 packed weights, KV untouched
    F_combined      D + E + C  -- the configuration that fits 24 GB

ORDERING CONSTRAINT. Weight quantization cannot be undone in process, so the
KV-only arm runs first and is reverted (kvstore.uninstall) before the weight
arm; the combined arm re-installs it at the end. One model load serves all six.
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
    ap.add_argument("--out", default="specwm_out/ablation_video")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from .binding_v2 import V2Binding
    from .vae_tiled import tiled_decode
    from .quantize import quantize_packed_int4
    from . import kvstore

    rep = {"kv_bits": a.kv_bits, "tiles": a.tiles, "arms": {}}
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
        outs, ts, rearmed = [], [], False
        for c in range(N):
            if c == 1 and not rearmed:
                # the bf16 cache reset_caches() allocates is discarded once the
                # packed store is live; re-arm so the peak reflects steady state
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
        st = sorted(ts[2:]); kvb = kvstore.total_bytes(tm)
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
        info = dict(decode_over_resident_GiB=round(
            (torch.cuda.max_memory_allocated() - res) / GiB, 2))
        del z
        return img.detach().cpu(), info

    import numpy as np
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FR = None

    def emit(name, img, ref=None):
        nonlocal FR
        T = img.shape[1]
        if FR is None:
            FR = [int(round(t)) for t in np.linspace(0, T - 1, 8)]
            rep["frame_idxs"] = FR
        v = ((img.clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy()
        for fi in FR:
            plt.imsave(os.path.join(a.out, f"{name}__f{fi:03d}.jpg"), v[fi].clip(0, 1))
        try:
            import imageio
            imageio.mimwrite(os.path.join(a.out, f"{name}.mp4"),
                             list((v * 255).astype("uint8")), fps=16, quality=8)
            print(f"[video] {name}.mp4", flush=True)
        except Exception as e:
            print(f"[video] {name} failed: {e}", flush=True)
        return None if ref is None else round(psnr(img, ref), 2)

    r, c = (int(x) for x in a.tiles.lower().split("x"))

    # ---- A: original -------------------------------------------------
    ref_lat, iA = gen("A_original")
    img_ref, dA = dec(ref_lat, (1, 1))
    emit("A_original", img_ref)
    rep["arms"]["A_original"] = dict(**iA, **dA, pixel_psnr=99.0,
                                     changed="아무것도 (기준)")

    # ---- B: noise floor ---------------------------------------------
    o, iB = gen("B_seed2", seed=4321)
    img, dB = dec(o, (1, 1))
    rep["arms"]["B_seed2_noisefloor"] = dict(
        **iB, **dB, pixel_psnr=emit("B_seed2_noisefloor", img, img_ref),
        changed="시드만 (노이즈 플로어)")
    del o, img

    # ---- C: VAE tiling only -----------------------------------------
    img, dC = dec(ref_lat, (r, c))
    rep["arms"]["C_vae_tiled"] = dict(
        **{k: iA[k] for k in ("chunk_s_median", "peak_alloc_GiB", "weight_GiB")},
        **dC, pixel_psnr=emit("C_vae_tiled", img, img_ref),
        changed=f"VAE 디코드만 {a.tiles} 타일")
    del img

    # ---- D: KV only --------------------------------------------------
    kvstore.install(tm, bits=a.kv_bits)
    o, iD = gen("D_kv%db" % a.kv_bits)
    img, dD = dec(o, (1, 1))
    rep["arms"][f"D_kv{a.kv_bits}bit"] = dict(
        **iD, **dD, pixel_psnr=emit(f"D_kv{a.kv_bits}bit", img, img_ref),
        changed=f"KV 캐시만 {a.kv_bits}비트")
    del o, img
    kvstore.uninstall(tm)
    torch.cuda.empty_cache()

    # ---- E: weights only ---------------------------------------------
    quantize_packed_int4(tm, group=128, search=False)
    torch.cuda.empty_cache()
    o, iE = gen("E_int4w")
    img, dE = dec(o, (1, 1))
    rep["arms"]["E_int4_weights"] = dict(
        **iE, **dE, pixel_psnr=emit("E_int4_weights", img, img_ref),
        changed="가중치만 INT4")
    del o, img

    # ---- F: combined --------------------------------------------------
    kvstore.install(tm, bits=a.kv_bits)
    o, iF = gen("F_combined")
    img, dF = dec(o, (r, c))
    rep["arms"]["F_combined"] = dict(
        **iF, **dF, pixel_psnr=emit("F_combined", img, img_ref),
        changed=f"가중치 INT4 + KV {a.kv_bits}비트 + {a.tiles} 타일")

    with open(os.path.join(a.out, "ablation.json"), "w") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)

    print("\n" + "=" * 104)
    print(f"{'arm':22}{'바꾼 것':30}{'weight':>9}{'KV':>8}{'gen peak':>10}"
          f"{'chunk s':>9}{'PSNR':>8}")
    print("-" * 104)
    for k, v in rep["arms"].items():
        print(f"{k:22}{v['changed'][:28]:30}"
              f"{str(v.get('weight_GiB','-')):>9}{str(v.get('kv_store_GiB') or '-'):>8}"
              f"{str(v.get('peak_alloc_GiB','-')):>10}"
              f"{str(v.get('chunk_s_median','-')):>9}{v.get('pixel_psnr','-'):>8}")
    print("=" * 104)


if __name__ == "__main__":
    main()
