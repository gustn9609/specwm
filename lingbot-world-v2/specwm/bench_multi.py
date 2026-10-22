"""Original vs final build on THREE scenes at 10 s (161 frames, 14 chunks).

Why this matters beyond "more samples":
  * 14 chunks means the local_attn=18 window (6 chunks) fills and EVICTS more
    than twice, so the packed KV store's rolling behaviour is actually
    exercised -- the 5 s clips never evicted.
  * GPTQ is calibrated ONCE on a mix of all three scenes, as a deployment
    would, then every scene is evaluated with that single quantized model.
    Scene 00 is in-distribution for the calibration; 01/02 test generalisation.

Order (weight quantization is destructive): all ORIGINAL runs first, then
Hessians, then quantize once, then all FINAL runs.
"""
from __future__ import annotations
import argparse, json, os, time
import torch, torch.nn as nn, numpy as np

GiB = 2 ** 30
SCENES = {
    "00": "A sweeping cinematic journey along the Great Wall of China, winding through "
          "golden autumn hills under a brilliant blue sky as the camera glides forward.",
    "01": "A stone footbridge over a calm river at golden hour, sunlight glinting on "
          "the water, the camera moving across the bridge and turning to survey the banks.",
    "02": "A grand temple courtyard on a bright day, visitors walking slowly, clouds "
          "drifting overhead, the camera holding a steady view of the main hall.",
}


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
    ap.add_argument("--scenes", nargs="*", default=["00", "01", "02"])
    ap.add_argument("--frames", type=int, default=161)
    ap.add_argument("--local-attn", type=int, default=18)
    ap.add_argument("--sink", type=int, default=6)
    ap.add_argument("--bits", type=int, default=6)
    ap.add_argument("--tiles", default="2x2")
    ap.add_argument("--out", default="specwm_out/multi")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from .binding_v2 import V2Binding
    from .vae_tiled import tiled_decode
    from .quantize import analytic_weight_bytes
    from .gptq import HessianCollector, GPTQLinear
    from . import kvstore2
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import gc

    rep = {"bits": a.bits, "frames": a.frames, "scenes": {}}
    r, c = (int(x) for x in a.tiles.lower().split("x"))

    t0 = time.time()
    b = V2Binding(a.ckpt, target_mode="fast_bf16", quant=None, sink=a.sink,
                  local_attn=a.local_attn, swap_models=False, build_draft=False)
    tm = b.target_model
    print(f"[load] {time.time()-t0:.0f}s", flush=True)
    orig = {n: m for n, m in tm.named_modules()
            if isinstance(m, nn.Linear) and m.in_features >= 1024}
    sig = b.node_sigmas()

    def prepare(sc):
        return b.prepare(f"examples/{sc}/image.jpg", SCENES[sc], f"examples/{sc}",
                         frame_num=a.frames)

    def gen(N, tag, seed=1234, rearm=False):
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
                    total_s=round(sum(ts), 1),
                    peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / GiB, 2),
                    weight_packed_GiB=round(analytic_weight_bytes(tm) / GiB, 2),
                    kv_store_GiB=round(kb / GiB, 2) if kb else None)
        print(f"[{tag}] {info}", flush=True)
        return outs, info

    def dec(lat, tiles):
        z = torch.cat(lat, dim=1).to(b.device)
        img = tiled_decode(b.pipe.vae, z, tiles=tiles, overlap=8, scale=8)
        torch.cuda.synchronize(); del z
        return img.detach().cpu()

    def emit(name, img, ref):
        T = img.shape[1]
        FR = [int(round(t)) for t in np.linspace(0, T - 1, 10)]
        v = ((img.clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy()
        for fi in FR:
            plt.imsave(os.path.join(a.out, f"{name}__f{fi:03d}.jpg"), v[fi].clip(0, 1))
        try:
            import imageio
            imageio.mimwrite(os.path.join(a.out, f"{name}.mp4"),
                             list((v * 255).astype("uint8")), fps=16, quality=8)
            print(f"[video] {name}.mp4 ({T} frames)", flush=True)
        except Exception as e:
            print(f"[video] {name}: {e}", flush=True)
        # per-chunk drift curve vs reference
        drift = None
        if ref is not None:
            d = (img.float() - ref.float()).abs().mean(dim=(0, 2, 3))   # per frame
            drift = [round(float(x), 4) for x in d[::16]]               # every 1 s
        return (None if ref is None else round(psnr(img, ref), 2)), FR, drift

    # ---- pass 1: ORIGINAL on every scene ----
    refs = {}
    for sc in a.scenes:
        cond = prepare(sc); N = cond["num_chunks"]
        lat, info = gen(N, f"ORIG_{sc}")
        img = dec(lat, (1, 1))
        p, FR, _ = emit(f"ORIG_{sc}", img, None)
        refs[sc] = img
        rep["scenes"][sc] = {"prompt": SCENES[sc], "num_chunks": N, "frame_idxs": FR,
                             "original": dict(**info, pixel_psnr=99.0)}
        del lat
    # noise floor on one scene (cheap sanity anchor)
    cond = prepare(a.scenes[0]); N = cond["num_chunks"]
    lat, info = gen(N, f"SEED2_{a.scenes[0]}", seed=4321)
    img = dec(lat, (1, 1))
    p, _, drift = emit(f"SEED2_{a.scenes[0]}", img, refs[a.scenes[0]])
    rep["scenes"][a.scenes[0]]["noise_floor"] = dict(**info, pixel_psnr=p, drift_per_s=drift)
    del lat, img

    # ---- Hessians: mixed calibration, 1 chunk per scene ----
    t = time.time()
    b._caches = {}; gc.collect(); torch.cuda.empty_cache()
    hc = HessianCollector(tm, min_features=1024)
    for sc in a.scenes:
        prepare(sc); b.reset_caches()
        g = torch.Generator().manual_seed(7)
        z = b.chunk_noise(0, g); x0 = None
        for i in range(len(sig)):
            x0 = b.target_x0(z, i, 0)
            if i < len(sig) - 1:
                s = float(sig[i + 1]); z = (1 - s) * x0 + s * b.randn_like(x0, g)
        b.commit(x0, 0)
    Hs = hc.finish()
    b._caches = {}; gc.collect(); torch.cuda.empty_cache()
    print(f"[hessian] {len(Hs)} layers from {len(a.scenes)} scenes, {time.time()-t:.0f}s, "
          f"GPU {torch.cuda.memory_allocated()/GiB:.1f}G", flush=True)

    # ---- quantize once (weights destructive, originals -> CPU) ----
    t = time.time()
    for name, lin in orig.items():
        p_, leaf = parent_and_leaf(tm, name)
        q = GPTQLinear(lin, Hs[name], bits=a.bits, group=128)
        # GPTQLinear aliases lin.bias; detach a GPU copy BEFORE we drop the
        # original, or lin.to("cpu") drags the bias along and the forward
        # dies with a cuda/cpu mismatch (this happened -- 22 min lost).
        if q.bias is not None:
            q.bias = nn.Parameter(q.bias.detach().clone(), requires_grad=False)
        setattr(p_, leaf, q)
        lin.weight.data = torch.empty(0)          # free the bf16 weight only
    del Hs; gc.collect(); torch.cuda.empty_cache()
    print(f"[gptq] int{a.bits} done in {time.time()-t:.0f}s, "
          f"GPU {torch.cuda.memory_allocated()/GiB:.1f}G", flush=True)

    # ---- pass 2: FINAL on every scene ----
    for sc in a.scenes:
        cond = prepare(sc); N = cond["num_chunks"]
        kvstore2.install(tm, bits=4, k_axis="token", v_axis="token", asym=True, verbose=False)
        lat, info = gen(N, f"FINAL_{sc}", rearm=True)
        img = dec(lat, (r, c))
        p, _, drift = emit(f"FINAL_{sc}", img, refs[sc])
        rep["scenes"][sc]["final"] = dict(**info, pixel_psnr=p, drift_per_s=drift)
        kvstore2.uninstall(tm, verbose=False)
        del lat, img

    with open(os.path.join(a.out, "multi.json"), "w") as f:
        json.dump(rep, f, indent=1)

    print("\n" + "=" * 92)
    print(f"{'scene':7}{'arm':10}{'peak':>8}{'chunk s':>9}{'total s':>9}{'PSNR':>8}   drift(1s→10s)")
    print("-" * 92)
    for sc, v in rep["scenes"].items():
        for arm in ("original", "noise_floor", "final"):
            if arm not in v: continue
            x = v[arm]; d = x.get("drift_per_s")
            ds = "" if not d else f"{d[0]:.3f} → {d[-1]:.3f}"
            print(f"{sc:7}{arm:10}{x['peak_alloc_GiB']:>7}G{x['chunk_s_median']:>9}"
                  f"{x['total_s']:>9}{x['pixel_psnr']:>8}   {ds}")
    print("=" * 92)


if __name__ == "__main__":
    main()
