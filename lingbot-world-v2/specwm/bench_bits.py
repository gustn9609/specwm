"""Bit-width curve (naive vs GPTQ) in ONE job -- the allocation is shared.

Known points: naive INT4 16.75 dB (fits 24 GB), naive INT8 21.20 dB (does not).
Unknown: the shape of the curve between them, and whether data-aware GPTQ
shifts it. Five arms after one model load:

    naive int5, naive int6            <- where does naive land in between?
    gptq  int4, gptq int5, gptq int6  <- how much does output-aware help?

Memory is reported ANALYTICALLY for 5/6-bit (int8 storage here, no cross-byte
packing); that is the size a production kernel would give.

All arms restore the original bf16 Linear modules before the next.
"""
from __future__ import annotations
import argparse, json, os, time
import torch, torch.nn as nn, numpy as np

GiB = 2 ** 30


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
    ap.add_argument("--out", default="specwm_out/bits")
    ap.add_argument("--skip-naive", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from .binding_v2 import V2Binding
    from .quantize import PackedInt4Linear, IntNLinear, analytic_weight_bytes
    from .gptq import HessianCollector, GPTQLinear
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rep = {"arms": {}}
    t0 = time.time()
    b = V2Binding(a.ckpt, target_mode="fast_bf16", quant=None, sink=a.sink,
                  local_attn=a.local_attn, swap_models=False, build_draft=False)
    cond = b.prepare(a.image, a.prompt, a.action, frame_num=a.frames)
    N = cond["num_chunks"]; sig = b.node_sigmas(); tm = b.target_model
    print(f"[load] {time.time()-t0:.0f}s", flush=True)

    orig = {n: m for n, m in tm.named_modules()
            if isinstance(m, nn.Linear) and m.in_features >= 1024}

    def restore():
        for name, m in orig.items():
            p, leaf = parent_and_leaf(tm, name); setattr(p, leaf, m)
        torch.cuda.empty_cache()

    def gen(seed=1234, hooks_ok=True):
        b.reset_caches()
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        g = torch.Generator().manual_seed(seed)
        outs, ts = [], []
        for ci in range(N):
            torch.cuda.synchronize(); t = time.perf_counter()
            z = b.chunk_noise(ci, g); x0 = None
            for i in range(len(sig)):
                x0 = b.target_x0(z, i, ci)
                if i < len(sig) - 1:
                    s = float(sig[i + 1]); z = (1 - s) * x0 + s * b.randn_like(x0, g)
            b.commit(x0, ci)
            torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
            outs.append(x0.detach().float().cpu())
        st = sorted(ts[2:])
        return outs, round(st[len(st) // 2], 4)

    def dec(lat):
        z = torch.cat(lat, dim=1).to(b.device)
        img = b.pipe.vae.decode([z])[0].detach().cpu(); del z; return img

    FR = None
    def emit(name, img, ref):
        nonlocal FR
        if FR is None:
            FR = [int(round(t)) for t in np.linspace(0, img.shape[1] - 1, 8)]
        v = ((img.clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy()
        for fi in FR:
            plt.imsave(os.path.join(a.out, f"{name}__f{fi:03d}.jpg"), v[fi].clip(0, 1))
        try:
            import imageio
            imageio.mimwrite(os.path.join(a.out, f"{name}.mp4"),
                             list((v * 255).astype("uint8")), fps=16, quality=8)
        except Exception as e:
            print(f"[video] {name}: {e}", flush=True)
        return None if ref is None else round(psnr(img, ref), 2)

    ref_lat, _ = gen()
    img_ref = dec(ref_lat); emit("R_reference", img_ref, None)

    # ---- Hessians: one calibration chunk through the bf16 model ----
    t = time.time()
    hc = HessianCollector(tm, min_features=1024)
    b.reset_caches()
    g = torch.Generator().manual_seed(7)
    # 3 chunks of calibration (~56k tokens/layer) instead of 1: the FFN input
    # is 13824-wide and one chunk left the Hessian rank-deficient.
    for ci in range(min(3, N)):
        z = b.chunk_noise(ci, g); x0 = None
        for i in range(len(sig)):
            x0 = b.target_x0(z, i, ci)
            if i < len(sig) - 1:
                s = float(sig[i + 1]); z = (1 - s) * x0 + s * b.randn_like(x0, g)
        b.commit(x0, ci)
    Hs = hc.finish()
    hbytes = sum(v.numel() * 4 for v in Hs.values())
    print(f"[hessian] {len(Hs)} layers, {hbytes/GiB:.2f} GiB on CPU, {time.time()-t:.0f}s, "
          f"GPU alloc {torch.cuda.memory_allocated()/GiB:.1f}G", flush=True)
    torch.cuda.empty_cache()

    def arm(key, label, maker):
        restore()
        t = time.time()
        for name, lin in orig.items():
            p, leaf = parent_and_leaf(tm, name)
            setattr(p, leaf, maker(name, lin))
        torch.cuda.empty_cache(); qt = time.time() - t
        outs, cs = gen()
        img = dec(outs)
        p = emit(key, img, img_ref)
        w = analytic_weight_bytes(tm) / GiB
        rep["arms"][key] = dict(label=label, pixel_psnr=p, chunk_s_median=cs,
                                analytic_weight_GiB=round(w, 2), quant_s=round(qt, 1))
        from .gptq import gptq_quantize_weight as _gq
        bumps = getattr(_gq, "damp_bumps", 0)
        print(f"[{key}] PSNR {p}  weight(analytic) {w:.2f}G  chunk {cs}s  quant {qt:.0f}s"
              f"  damp_bumps={bumps}", flush=True)
        del outs, img

    if not a.skip_naive:
        arm("N5", "naive int5 g128", lambda n, l: IntNLinear(l, bits=5, group=128))
        arm("N6", "naive int6 g128", lambda n, l: IntNLinear(l, bits=6, group=128))
    arm("G4", "GPTQ int4 g128", lambda n, l: GPTQLinear(l, Hs[n], bits=4, group=128))
    arm("G5", "GPTQ int5 g128", lambda n, l: GPTQLinear(l, Hs[n], bits=5, group=128))
    arm("G6", "GPTQ int6 g128", lambda n, l: GPTQLinear(l, Hs[n], bits=6, group=128))
    restore()

    with open(os.path.join(a.out, "bits.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print("\n" + "=" * 78)
    print(f"{'arm':6}{'방식':18}{'weight(분석)':>13}{'chunk s':>9}{'PSNR':>8}")
    print("-" * 78)
    print(f"{'N4':6}{'naive int4 (기존)':18}{8.92:>12.2f}G{'3.87':>9}{16.75:>8}")
    for k, v in rep["arms"].items():
        print(f"{k:6}{v['label']:18}{v['analytic_weight_GiB']:>12.2f}G"
              f"{v['chunk_s_median']:>9}{v['pixel_psnr']:>8}")
    print(f"{'N8':6}{'naive int8 (기존)':18}{17.55:>12.2f}G{'3.17':>9}{21.20:>8}")
    print("=" * 78)
    print("노이즈 플로어 13.73 / 24GB 한계: 가중치 ≲ 13 G (KV 6.5 + act 4 포함)")


if __name__ == "__main__":
    main()
