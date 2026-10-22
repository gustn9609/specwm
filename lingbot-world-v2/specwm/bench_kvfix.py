"""Does the K/V-aware quantization recover the quality the naive store lost?

The naive 4-bit arm degrades visibly. Synthetic tests say the cause is using
one strategy for both tensors; here we check that on the real model, one
change at a time, with a video for each so it can be judged by eye.

    D0  naive           per-token symmetric for both K and V   (current)
    D1  +asymmetric     per-token asymmetric for both
    D2  +K per-channel  K per-channel asym, V per-token asym   (the fix)
    D3  D2 + hp tail    D2 plus the most recent chunk in bf16
"""
from __future__ import annotations
import argparse, json, os, time
import torch, numpy as np

GiB = 2 ** 30


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
    ap.add_argument("--out", default="specwm_out/kvfix")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from .binding_v2 import V2Binding
    from .vae_tiled import tiled_decode
    from . import kvstore, kvstore2
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rep = {"arms": {}}
    t0 = time.time()
    b = V2Binding(a.ckpt, target_mode="fast_bf16", quant=None, sink=a.sink,
                  local_attn=a.local_attn, swap_models=False, build_draft=False)
    cond = b.prepare(a.image, a.prompt, a.action, frame_num=a.frames)
    N = cond["num_chunks"]; sig = b.node_sigmas(); tm = b.target_model
    print(f"[load] {time.time()-t0:.0f}s, {N} chunks", flush=True)

    def gen(tag, bytes_fn):
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
        st = sorted(ts[2:]); kb = bytes_fn(tm)
        info = dict(chunk_s_median=round(st[len(st) // 2], 4),
                    peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / GiB, 2),
                    kv_store_GiB=round(kb / GiB, 2) if kb else None)
        print(f"[{tag}] {info}", flush=True)
        return outs, info

    def dec(lat):
        z = torch.cat(lat, dim=1).to(b.device)
        img = tiled_decode(b.pipe.vae, z, tiles=(1, 1), overlap=8, scale=8)
        torch.cuda.synchronize(); del z
        return img.detach().cpu()

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

    ref_lat, _ = gen("reference_bf16", lambda m: 0)
    img_ref = dec(ref_lat)
    emit("R_reference", img_ref, None)

    def arm(key, label, install, bytes_fn, uninstall):
        install()
        o, info = gen(key, bytes_fn)
        img = dec(o)
        rep["arms"][key] = dict(**info, label=label,
                                pixel_psnr=emit(key, img, img_ref))
        del o, img
        uninstall(); torch.cuda.empty_cache()

    arm("D0_naive", "per-token 대칭 (K,V 동일) — 기존",
        lambda: kvstore.install(tm, bits=4),
        kvstore.total_bytes, lambda: kvstore.uninstall(tm))
    arm("D1_asym", "per-token 비대칭",
        lambda: kvstore2.install(tm, bits=4, k_axis="token", v_axis="token", asym=True),
        kvstore2.total_bytes, lambda: kvstore2.uninstall(tm))
    arm("D4_group32", "per-token 비대칭 + head_dim 그룹 32",
        lambda: kvstore2.install(tm, bits=4, k_axis="token", v_axis="token",
                                 asym=True, group=32),
        kvstore2.total_bytes, lambda: kvstore2.uninstall(tm))
    arm("D5_kfrozen", "K는 채널고정 · V는 토큰별 (비대칭)",
        lambda: kvstore2.install(tm, bits=4, k_axis="channel_frozen",
                                 v_axis="token", asym=True),
        kvstore2.total_bytes, lambda: kvstore2.uninstall(tm))
    arm("D6_group32_tail", "D4 + 최근 1청크 bf16",
        lambda: kvstore2.install(tm, bits=4, k_axis="token", v_axis="token",
                                 asym=True, group=32, hp_tail=4680),
        kvstore2.total_bytes, lambda: kvstore2.uninstall(tm))

    with open(os.path.join(a.out, "kvfix.json"), "w") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    print("\n" + "=" * 96)
    print(f"{'arm':16}{'방식':42}{'KV':>8}{'chunk s':>9}{'PSNR':>8}")
    print("-" * 96)
    for k, v in rep["arms"].items():
        print(f"{k:16}{v['label'][:40]:42}{str(v.get('kv_store_GiB') or '-'):>8}"
              f"{v['chunk_s_median']:>9}{v['pixel_psnr']:>8}")
    print("=" * 96)
    print("기준: 노이즈 플로어 13.73 dB / 기존 naive 15.72 dB")


if __name__ == "__main__":
    main()
