"""Final 24 GB configuration with the improved KV, and the tail-size tradeoff.

Keeping the most recent chunk in bf16 lifted 4-bit KV from 15.72 to 20.85 dB --
far more than any change to HOW the bulk is quantized (asymmetric +0.9,
head_dim groups -0.5, frozen per-channel -3.3). What matters is WHICH tokens
stay precise, not the quantizer.

But the tail costs memory (KV 5.34 -> 10.04 GiB), which threatens the 24 GB
budget once INT4 weights are added. So sweep the tail size and report where
quality saturates versus where the budget breaks.
"""
from __future__ import annotations
import argparse, json, os, time
import torch, numpy as np

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
    ap.add_argument("--tails", type=int, nargs="*", default=[0, 1560, 3120, 4680])
    ap.add_argument("--tiles", default="2x2")
    ap.add_argument("--out", default="specwm_out/final")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from .binding_v2 import V2Binding
    from .vae_tiled import tiled_decode
    from .quantize import quantize_packed_int4
    from . import kvstore2
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rep = {"arms": {}}
    t0 = time.time()
    b = V2Binding(a.ckpt, target_mode="fast_bf16", quant=None, sink=a.sink,
                  local_attn=a.local_attn, swap_models=False, build_draft=False)
    cond = b.prepare(a.image, a.prompt, a.action, frame_num=a.frames)
    N = cond["num_chunks"]; sig = b.node_sigmas(); tm = b.target_model
    r, c = (int(x) for x in a.tiles.lower().split("x"))
    print(f"[load] {time.time()-t0:.0f}s, {N} chunks", flush=True)

    def gen(tag):
        b.reset_caches()
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        g = torch.Generator().manual_seed(1234)
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
        st = sorted(ts[2:]); kb = kvstore2.total_bytes(tm)
        info = dict(chunk_s_median=round(st[len(st) // 2], 4),
                    peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / GiB, 2),
                    peak_reserved_GiB=round(torch.cuda.max_memory_reserved() / GiB, 2),
                    weight_GiB=round(wbytes(tm) / GiB, 2),
                    kv_store_GiB=round(kb / GiB, 2) if kb else None)
        print(f"[{tag}] {info}", flush=True)
        return outs, info

    def dec(lat, tiles):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        res = torch.cuda.memory_allocated()
        z = torch.cat(lat, dim=1).to(b.device)
        img = tiled_decode(b.pipe.vae, z, tiles=tiles, overlap=8, scale=8)
        torch.cuda.synchronize(); del z
        return img.detach().cpu(), round(
            (torch.cuda.max_memory_allocated() - res) / GiB, 2)

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

    # bf16 reference
    ref_lat, _ = gen("reference")
    img_ref, _ = dec(ref_lat, (1, 1))
    emit("R_reference", img_ref, None)

    # INT4 weights, then sweep the tail (weights are quantized once, non-revertible)
    quantize_packed_int4(tm, group=128, search=False)
    torch.cuda.empty_cache()
    for tail in a.tails:
        key = f"FINAL_tail{tail}"
        kvstore2.install(tm, bits=4, k_axis="token", v_axis="token",
                         asym=True, group=32, hp_tail=tail)
        o, info = gen(key)
        img, dpk = dec(o, (r, c))
        rep["arms"][key] = dict(
            **info, decode_over_resident_GiB=dpk, hp_tail=tail,
            tail_frames=round(tail / 1560, 2),
            pixel_psnr=emit(key, img, img_ref),
            fits_24GB=bool(max(info["peak_reserved_GiB"],
                               info["peak_alloc_GiB"] + dpk) < 24))
        del o, img
        kvstore2.uninstall(tm); torch.cuda.empty_cache()

    with open(os.path.join(a.out, "final.json"), "w") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    print("\n" + "=" * 104)
    print(f"{'구성':20}{'tail(프레임)':>13}{'weight':>9}{'KV':>8}"
          f"{'gen peak':>10}{'디코드':>8}{'chunk s':>9}{'PSNR':>8}{'24GB':>7}")
    print("-" * 104)
    for k, v in rep["arms"].items():
        print(f"{k:20}{v['tail_frames']:>13}{v['weight_GiB']:>8}G"
              f"{str(v.get('kv_store_GiB') or '-'):>8}{v['peak_alloc_GiB']:>9}G"
              f"{v['decode_over_resident_GiB']:>7}G{v['chunk_s_median']:>9}"
              f"{v['pixel_psnr']:>8}{('OK' if v['fits_24GB'] else 'NO'):>7}")
    print("=" * 104)
    print("기준: 노이즈 플로어 13.73 / 기존 4비트 KV 15.72 / 최근청크 bf16 20.85")


if __name__ == "__main__":
    main()
