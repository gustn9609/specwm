"""Original vs final build: real peak memory, quality, and videos for the web page.

Arms (one model load):
    ORIGINAL     bf16 weights, bf16 KV, untiled decode
    SEED2        same model, different seed           <- noise floor
    FINAL6       GPTQ int6 + 4-bit asym KV + 2x2 tiled decode
    FINAL5       GPTQ int5 + 4-bit asym KV + 2x2 tiled decode   (safety margin)

MEMORY CAVEAT, STATED UP FRONT. GPTQ codes here live in int8 storage because
5/6-bit values do not align to bytes, so the MEASURED peak carries weights at
17.6 GiB. A packed kernel would hold them at 13.2 (int6) / 11.1 (int5) GiB.
Nothing else differs, so we report both the measured peak and the
packed-equivalent (measured minus the fixed weight-size delta). The second
number is the 24 GB verdict; the first is what this run physically used.
"""
from __future__ import annotations
import argparse, json, os, time
import torch, torch.nn as nn, numpy as np

GiB = 2 ** 30


def wbytes(m):
    return (sum(p.numel() * p.element_size() for p in m.parameters())
            + sum(b.numel() * b.element_size() for b in m.buffers()))


def psnr(a, b):
    a, b = a.float(), b.float()
    mse = float((a - b).pow(2).mean())
    rng = float(a.max() - a.min()) or 1.0
    return 99.0 if mse < 1e-12 else float(10 * torch.log10(torch.tensor(rng * rng / mse)))


def parent_and_leaf(model, name):
    parts = name.split("."); p = model
    for q in parts[:-1]:
        p = getattr(p, q)
    return p, parts[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--action", required=True)
    ap.add_argument("--prompt", default="A serene lakeside scene with a lone tree.")
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--local-attn", type=int, default=18)
    ap.add_argument("--sink", type=int, default=6)
    ap.add_argument("--bits", type=int, nargs="*", default=[6, 5])
    ap.add_argument("--tiles", default="2x2")
    ap.add_argument("--out", default="specwm_out/showdown")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from .binding_v2 import V2Binding
    from .vae_tiled import tiled_decode
    from .quantize import analytic_weight_bytes
    from .gptq import HessianCollector, GPTQLinear
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
    print(f"[load] {time.time()-t0:.0f}s", flush=True)

    orig = {n: m for n, m in tm.named_modules()
            if isinstance(m, nn.Linear) and m.in_features >= 1024}

    def restore():
        for name, m in orig.items():
            p, leaf = parent_and_leaf(tm, name); setattr(p, leaf, m)
        torch.cuda.empty_cache()

    def gen(tag, seed=1234, rearm=False):
        b.reset_caches()
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        g = torch.Generator().manual_seed(seed)
        outs, ts, done = [], [], False
        for ci in range(N):
            if rearm and ci == 1 and not done:
                torch.cuda.synchronize(); torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(); done = True
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
                    weight_stored_GiB=round(wbytes(tm) / GiB, 2),
                    weight_packed_GiB=round(analytic_weight_bytes(tm) / GiB, 2),
                    kv_store_GiB=round(kb / GiB, 2) if kb else None)
        print(f"[{tag}] {info}", flush=True)
        return outs, info

    def dec(lat, tiles):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        res = torch.cuda.memory_allocated()
        z = torch.cat(lat, dim=1).to(b.device)
        img = tiled_decode(b.pipe.vae, z, tiles=tiles, overlap=8, scale=8)
        torch.cuda.synchronize(); del z
        return img.detach().cpu(), round((torch.cuda.max_memory_allocated() - res) / GiB, 2)

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

    # ---- ORIGINAL + noise floor ----
    ref_lat, i0 = gen("ORIGINAL")
    img_ref, d0 = dec(ref_lat, (1, 1))
    rep["arms"]["ORIGINAL"] = dict(**i0, decode_GiB=d0, pixel_psnr=99.0, emit="ORIGINAL")
    emit("ORIGINAL", img_ref, None)
    o, i1 = gen("SEED2", seed=4321)
    img, d1 = dec(o, (1, 1))
    rep["arms"]["SEED2_noisefloor"] = dict(**i1, decode_GiB=d1,
                                           pixel_psnr=emit("SEED2_noisefloor", img, img_ref))
    del o, img

    # ---- Hessians (3 calibration chunks on the bf16 model) ----
    t = time.time()
    hc = HessianCollector(tm, min_features=1024)
    b.reset_caches(); g = torch.Generator().manual_seed(7)
    for ci in range(min(3, N)):
        z = b.chunk_noise(ci, g); x0 = None
        for i in range(len(sig)):
            x0 = b.target_x0(z, i, ci)
            if i < len(sig) - 1:
                s = float(sig[i + 1]); z = (1 - s) * x0 + s * b.randn_like(x0, g)
        b.commit(x0, ci)
    Hs = hc.finish(); torch.cuda.empty_cache()
    print(f"[hessian] {len(Hs)} layers, {time.time()-t:.0f}s", flush=True)

    # ---- FINAL builds ----
    # Free everything the earlier arms left resident (bf16 KV cache ~21 GiB,
    # decode buffers) BEFORE quantizing: GPTQ needs ~1.5 GiB of fp32 working
    # set per FFN layer (Hessian + inverse + Cholesky factors) and the first
    # attempt OOM'd at 68.8 GiB in use with only 2.7 free.
    import gc
    b._caches = {}
    gc.collect(); torch.cuda.empty_cache()
    print(f"[mem] before GPTQ: {torch.cuda.memory_allocated()/GiB:.1f}G alloc, "
          f"{torch.cuda.memory_reserved()/GiB:.1f}G reserved", flush=True)
    for bits in a.bits:
        restore(); gc.collect(); torch.cuda.empty_cache()
        t = time.time()
        for name, lin in orig.items():
            p, leaf = parent_and_leaf(tm, name)
            setattr(p, leaf, GPTQLinear(lin, Hs[name], bits=bits, group=128))
        torch.cuda.empty_cache(); qt = time.time() - t
        kvstore2.install(tm, bits=4, k_axis="token", v_axis="token", asym=True)
        key = f"FINAL_gptq{bits}_kv4_tiled"
        o, info = gen(key, rearm=True)
        img, dk = dec(o, (r, c))
        delta = info["weight_stored_GiB"] - info["weight_packed_GiB"]
        rep["arms"][key] = dict(
            **info, decode_GiB=dk, quant_s=round(qt, 0),
            peak_packed_equiv_GiB=round(info["peak_alloc_GiB"] - delta, 2),
            peak_with_decode_packed_GiB=round(info["peak_alloc_GiB"] - delta + dk, 2),
            fits_24GB=bool(info["peak_alloc_GiB"] - delta + dk < 24),
            pixel_psnr=emit(key, img, img_ref))
        print(f"[{key}] packed-equiv peak {info['peak_alloc_GiB']-delta:.2f}G "
              f"(+decode {dk}G) PSNR {rep['arms'][key]['pixel_psnr']}", flush=True)
        del o, img
        kvstore2.uninstall(tm)
    restore()

    with open(os.path.join(a.out, "showdown.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print("\n" + "=" * 100)
    print(f"{'arm':24}{'weight':>9}{'KV':>7}{'peak(측정)':>11}{'peak(패킹환산)':>14}"
          f"{'디코드':>7}{'chunk':>7}{'PSNR':>7}{'24GB':>6}")
    print("-" * 100)
    for k, v in rep["arms"].items():
        pe = v.get("peak_packed_equiv_GiB", v["peak_alloc_GiB"])
        fit = v.get("fits_24GB", False)
        print(f"{k:24}{v['weight_packed_GiB']:>8}G{str(v.get('kv_store_GiB') or '-'):>7}"
              f"{v['peak_alloc_GiB']:>10}G{pe:>13}G{v['decode_GiB']:>6}G"
              f"{v['chunk_s_median']:>7}{v['pixel_psnr']:>7}{('OK' if fit else '-'):>6}")
    print("=" * 100)


if __name__ == "__main__":
    main()
