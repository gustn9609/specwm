"""The pairwise combinations the one-factor ablation left out.

Single-factor arms (C, D, E) and the full stack (F) were measured already; the
pairs were not, and they answer a different question -- what does DROPPING one
compression buy back?

    CD   bf16 weights + 4-bit KV + tiled decode    weights untouched
    CE   INT4 weights + bf16 KV  + tiled decode    KV untouched
    DE   INT4 weights + 4-bit KV + untiled decode  = F without tiling

CD matters most: it keeps full-precision weights, so there is no weight
unpacking on the critical path and it should be the fastest compressed arm.
Its generation peak is still ~44 GiB though, so it is not a 24 GB candidate --
it is the configuration for a bigger card.

Weight quantization is irreversible in process, so CD runs first, its KV store
is removed, and only then are the weights quantized for CE and DE.
"""
from __future__ import annotations
import argparse, json, os, time
import torch
import numpy as np

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
    ap.add_argument("--kv-bits", type=int, default=4)
    ap.add_argument("--tiles", default="2x2")
    ap.add_argument("--out", default="specwm_out/pairs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from .binding_v2 import V2Binding
    from .vae_tiled import tiled_decode
    from .quantize import quantize_packed_int4
    from . import kvstore
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rep = {"arms": {}}
    t0 = time.time()
    b = V2Binding(a.ckpt, target_mode="fast_bf16", quant=None, sink=a.sink,
                  local_attn=a.local_attn, swap_models=False, build_draft=False)
    cond = b.prepare(a.image, a.prompt, a.action, frame_num=a.frames)
    N = cond["num_chunks"]; sig = b.node_sigmas(); tm = b.target_model
    print(f"[load] {time.time()-t0:.0f}s, {N} chunks", flush=True)
    r, c = (int(x) for x in a.tiles.lower().split("x"))

    def gen(tag, seed=1234):
        b.reset_caches()
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        g = torch.Generator().manual_seed(seed)
        outs, ts, rearm = [], [], False
        for ci in range(N):
            if ci == 1 and not rearm:
                torch.cuda.synchronize(); torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(); rearm = True
            torch.cuda.synchronize(); t = time.perf_counter()
            z = b.chunk_noise(ci, g); x0 = None
            for i in range(len(sig)):
                x0 = b.target_x0(z, i, ci)
                if i < len(sig) - 1:
                    s = float(sig[i + 1]); z = (1 - s) * x0 + s * b.randn_like(x0, g)
            b.commit(x0, ci)
            torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
            outs.append(x0.detach().float().cpu())
        st = sorted(ts[2:]); kvb = kvstore.total_bytes(tm)
        info = dict(chunk_s_median=round(st[len(st) // 2], 4),
                    peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / GiB, 2),
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

    FR = None
    def emit(name, img, ref):
        nonlocal FR
        if FR is None:
            FR = [int(round(t)) for t in np.linspace(0, img.shape[1] - 1, 8)]
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
            print(f"[video] {name}: {e}", flush=True)
        return None if ref is None else round(psnr(img, ref), 2)

    # reference (bf16, untiled) -- needed for PSNR, not re-emitted
    ref_lat, _ = gen("ref")
    img_ref, _ = dec(ref_lat, (1, 1))

    # ---- CD : weights untouched --------------------------------------
    kvstore.install(tm, bits=a.kv_bits)
    o, i1 = gen("CD_kv4_tiled_bf16w")
    img, d1 = dec(o, (r, c))
    rep["arms"]["CD_bf16w_kv4_tiled"] = dict(
        **i1, **d1, pixel_psnr=emit("CD_bf16w_kv4_tiled", img, img_ref),
        changed=f"KV {a.kv_bits}비트 + {a.tiles} 타일 (가중치 bf16 유지)")
    del o, img
    kvstore.uninstall(tm); torch.cuda.empty_cache()

    # ---- CE : KV untouched -------------------------------------------
    quantize_packed_int4(tm, group=128, search=False)
    torch.cuda.empty_cache()
    o, i2 = gen("CE_int4w_tiled_bf16kv")
    img, d2 = dec(o, (r, c))
    rep["arms"]["CE_int4w_bf16kv_tiled"] = dict(
        **i2, **d2, pixel_psnr=emit("CE_int4w_bf16kv_tiled", img, img_ref),
        changed=f"가중치 INT4 + {a.tiles} 타일 (KV bf16 유지)")
    del o, img

    # ---- DE : no tiling ----------------------------------------------
    kvstore.install(tm, bits=a.kv_bits)
    o, i3 = gen("DE_int4w_kv4_untiled")
    img, d3 = dec(o, (1, 1))
    rep["arms"]["DE_int4w_kv4_untiled"] = dict(
        **i3, **d3, pixel_psnr=emit("DE_int4w_kv4_untiled", img, img_ref),
        changed=f"가중치 INT4 + KV {a.kv_bits}비트 (타일링 없음)")

    with open(os.path.join(a.out, "pairs.json"), "w") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)

    print("\n" + "=" * 100)
    print(f"{'arm':26}{'weight':>9}{'KV':>8}{'gen peak':>10}{'decode':>9}"
          f"{'chunk s':>9}{'PSNR':>8}")
    print("-" * 100)
    for k, v in rep["arms"].items():
        print(f"{k:26}{v['weight_GiB']:>8}G{str(v.get('kv_store_GiB') or '-'):>8}"
              f"{v['peak_alloc_GiB']:>9}G{v['decode_over_resident_GiB']:>8}G"
              f"{v['chunk_s_median']:>9}{v['pixel_psnr']:>8}")
    print("=" * 100)


if __name__ == "__main__":
    main()
