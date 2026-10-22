"""Base/Fast와 동일한 입력으로 Ours(4090 압축본)를 생성해 3자 비교를 완성한다.

케이스 A: 실제 pose 재방문 궤적 161프레임 (기억)
케이스 B: 손 조작 81프레임 (물리)

GPTQ 보정은 케이스마다 따로 한다 (Hessian이 입력 분포에 의존하므로).
"""
import argparse, json, os, time, gc
import numpy as np, torch

GiB = 2 ** 30


def psnr(a, b):
    a, b = a.float(), b.float()
    mse = float((a - b).pow(2).mean())
    rng = float(a.max() - a.min()) or 1.0
    return 99.0 if mse < 1e-12 else float(10 * torch.log10(torch.tensor(rng * rng / mse)))


def hf_energy(img):
    g = img.float().mean(0)
    f = torch.fft.fftshift(torch.fft.fft2(g), dim=(-2, -1)).abs()
    T, H, W = g.shape
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    r = ((yy - H / 2) ** 2 + (xx - W / 2) ** 2).sqrt()
    hf = (r > min(H, W) / 4).to(f.dtype)
    return float((f * hf).sum(dim=(-2, -1)).mean() / (f.sum(dim=(-2, -1)).mean() + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--scene-root", default="examples")
    ap.add_argument("--action-dir", required=True)
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--bits", type=int, default=6)
    ap.add_argument("--kv-bits", type=int, default=4)
    ap.add_argument("--tiles", default="2x2")
    ap.add_argument("--local-attn", type=int, default=18)
    ap.add_argument("--sink", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    r, c = (int(x) for x in a.tiles.split("x"))

    from .binding_v2 import V2Binding
    from .vae_tiled import tiled_decode
    from .gptq import HessianCollector, GPTQLinear
    from . import kvstore2

    if a.prompt_file and os.path.exists(a.prompt_file):
        a.prompt = open(a.prompt_file).read().strip()
    src = os.path.join(a.scene_root, a.scene)

    t0 = time.time()
    b = V2Binding(a.ckpt, target_mode="fast_bf16", quant=None, sink=a.sink,
                  local_attn=a.local_attn, swap_models=False, build_draft=False,
                  timesteps_index=(0, 250, 500, 750))
    cond = b.prepare(os.path.join(src, "image.jpg"), a.prompt, a.action_dir,
                     frame_num=a.frames)
    N = cond["num_chunks"]; sig = b.node_sigmas()
    tm = b.target_model
    print(f"[load] {time.time()-t0:.0f}s, {N} chunks, {len(sig)} steps", flush=True)

    def generate(tag, seed=1234):
        b.reset_caches()
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        g = torch.Generator().manual_seed(seed); outs = []; ts = []
        for ci in range(N):
            tc = time.time()
            z = b.chunk_noise(ci, g); x0 = None
            for i in range(len(sig)):
                x0 = b.target_x0(z, i, ci)
                if i < len(sig) - 1:
                    s = float(sig[i + 1]); z = (1 - s) * x0 + s * b.randn_like(x0, g)
            b.commit(x0, ci); outs.append(x0.detach().float().cpu())
            ts.append(time.time() - tc)
            if ci == 1:                      # 워밍업 이후 재무장
                torch.cuda.reset_peak_memory_stats()
        lat = torch.cat(outs, dim=1)
        st = sorted(ts[2:]) or ts
        return lat, dict(chunk_s_median=round(st[len(st)//2], 4),
                         gen_s=round(sum(ts), 1),
                         peak_alloc_GiB=round(torch.cuda.max_memory_allocated()/GiB, 2))

    def decode(lat, tiles):
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        z = lat.to(b.device)
        img = (tiled_decode(b.pipe.vae, z, tiles=tiles) if tiles != (1, 1)
               else b.pipe.vae.decode([z])[0]).detach().cpu()
        over = (torch.cuda.max_memory_allocated() - before) / GiB
        del z; torch.cuda.empty_cache()
        return img, round(over, 2)

    def save(tag, img, ref=None):
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        v = ((img.clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy()
        T = v.shape[0]
        d = os.path.join(a.out, tag); os.makedirs(d, exist_ok=True)
        for name, fi in {"first": 0, "q1": T//3, "mid": T//2,
                         "q3": 2*T//3, "last": T-1}.items():
            plt.imsave(os.path.join(d, f"{name}.jpg"), v[fi].clip(0, 1))
        np.save(os.path.join(d, "frames_all.npy"), (v*255).astype("uint8")[::4])
        try:
            import imageio
            imageio.mimwrite(os.path.join(d, "video.mp4"),
                             list((v*255).astype("uint8")), fps=16, quality=8)
        except Exception as e:
            print("[video]", e, flush=True)
        return dict(frames=T, hf_energy=round(hf_energy(img), 5),
                    flicker=round(float((img[:, 1:]-img[:, :-1]).abs().mean()), 5),
                    pixel_psnr=(round(psnr(img, ref), 2) if ref is not None else 99.0))

    rep = {"arms": {}, "config": vars(a)}

    # ---------- 1) 참조: Fast bf16 ----------
    lat, info = generate("FAST_bf16")
    img_ref, dk = decode(lat, (1, 1))
    rep["arms"]["FAST_bf16"] = dict(**info, decode_GiB=dk, **save("FAST_bf16", img_ref))
    print("[FAST_bf16]", rep["arms"]["FAST_bf16"], flush=True)
    del lat

    # ---------- 2) 노이즈 바닥 ----------
    lat, info = generate("SEED2", seed=4321)
    img, dk = decode(lat, (1, 1))
    rep["arms"]["SEED2_noisefloor"] = dict(**info, decode_GiB=dk,
                                           **save("SEED2_noisefloor", img, img_ref))
    print("[SEED2]", rep["arms"]["SEED2_noisefloor"], flush=True)
    del lat, img

    # ---------- 3) GPTQ 보정 ----------
    t = time.time()
    hc = HessianCollector(tm, min_features=1024)
    b.reset_caches(); g = torch.Generator().manual_seed(7)
    for ci in range(min(3, N)):
        z = b.chunk_noise(ci, g); x0 = None
        for i in range(len(sig)):
            x0 = b.target_x0(z, i, ci)
            if i < len(sig)-1:
                s = float(sig[i+1]); z = (1-s)*x0 + s*b.randn_like(x0, g)
        b.commit(x0, ci)
    Hs = hc.finish(); torch.cuda.empty_cache()
    print(f"[hessian] {len(Hs)} layers, {time.time()-t:.0f}s", flush=True)

    import torch.nn as nn
    def parent_and_leaf(model, name):
        parts = name.split("."); p = model
        for q in parts[:-1]:
            p = p[int(q)] if q.isdigit() else getattr(p, q)
        return p, parts[-1]
    orig = {n: m for n, m in tm.named_modules()
            if isinstance(m, nn.Linear) and min(m.in_features, m.out_features) >= 1024}

    b._caches = {}; gc.collect(); torch.cuda.empty_cache()
    print(f"[mem] GPTQ 직전 {torch.cuda.memory_allocated()/GiB:.1f}G", flush=True)

    # ---------- 4) Ours ----------
    t = time.time()
    for name, lin in orig.items():
        p, leaf = parent_and_leaf(tm, name)
        setattr(p, leaf, GPTQLinear(lin, Hs[name], bits=a.bits, group=128))
    torch.cuda.empty_cache(); qt = time.time() - t
    kvstore2.install(tm, bits=a.kv_bits, k_axis="token", v_axis="token", asym=True)
    key = f"OURS_gptq{a.bits}_kv{a.kv_bits}_tiled"
    lat, info = generate(key)
    img, dk = decode(lat, (r, c))
    rep["arms"][key] = dict(**info, decode_GiB=dk, quant_s=round(qt, 0),
                            **save(key, img, img_ref))
    print("[" + key + "]", rep["arms"][key], flush=True)
    kvstore2.uninstall(tm)

    json.dump(rep, open(os.path.join(a.out, "ours_same.json"), "w"), indent=1)
    print("\n=== 요약 ===", flush=True)
    for k, v in rep["arms"].items():
        print("  %-26s gen=%7.1fs chunk=%6.3fs peak=%6.2fG psnr=%6.2f flicker=%.4f"
              % (k, v["gen_s"], v["chunk_s_median"], v["peak_alloc_GiB"],
                 v["pixel_psnr"], v["flicker"]), flush=True)
    print("[OURS_SAME_DONE]", flush=True)


if __name__ == "__main__":
    main()
