"""55초 클립 창 스윕: '4.5초 창 vs 전체 문맥' 격차가 긴 지평선에서도 창 확대로 메워지는가.

10초 클립에서는 창 36 이상이 이미 클립 전체(39잠재)를 덮어 포화됐다.
55초(222잠재)에서는 창 18 이 클립의 8% 만 본다. 여기서도 창 확대가
Base 를 따라잡으면 '망각 = 예산 문제' 가 긴 지평선에서 확정된다.
"""
import argparse, json, os, time, gc
import numpy as np, torch

GiB = 2 ** 30
ARMS = [("W18", 18, None), ("W36", 36, 4), ("W72", 72, 4),
        ("W144", 144, 4), ("W222_gptq", 222, 4)]


def hf_energy(img):
    g = img.float().mean(0)
    f = torch.fft.fftshift(torch.fft.fft2(g), dim=(-2, -1)).abs()
    T, H, W = g.shape
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    r = ((yy - H / 2) ** 2 + (xx - W / 2) ** 2).sqrt()
    hf = (r > min(H, W) / 4).to(f.dtype)
    return float((f * hf).sum(dim=(-2, -1)).mean() / (f.sum(dim=(-2, -1)).mean() + 1e-9))


def parent_and_leaf(model, name):
    parts = name.split("."); p = model
    for q in parts[:-1]:
        p = p[int(q)] if q.isdigit() else getattr(p, q)
    return p, parts[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--action-root", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--frames", type=int, default=893)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from .binding_v2 import V2Binding
    from .vae_tiled import tiled_decode
    from . import kvstore2, lowbit_alloc

    rp = os.path.join(a.out, "win60.json")
    rep = json.load(open(rp)) if os.path.exists(rp) else {"arms": {}}
    prompt = open(a.prompt_file).read().strip()

    for tag, la, kvb in ARMS:
        if tag in rep["arms"]:
            print("[%s] 이미 있음" % tag, flush=True); continue
        print("\n" + "=" * 68, flush=True); print("[%s] 창 %d (%.1f초)" % (tag, la, la * 4 / 16), flush=True)
        t0 = time.time(); b = None
        try:
            b = V2Binding(a.ckpt, target_mode="fast_bf16", quant=None, sink=6,
                          local_attn=la, swap_models=False, build_draft=False,
                          timesteps_index=(0, 250, 500, 750))
            if kvb: lowbit_alloc.patch(b.pipe, verbose=False)
            cond = b.prepare(os.path.join("examples", "00", "image.jpg"), prompt,
                             a.action_root, frame_num=a.frames)
            N = cond["num_chunks"]; sig = b.node_sigmas(); tm = b.target_model
            if kvb:
                kvstore2.install(tm, bits=kvb, k_axis="token", v_axis="token",
                                 asym=True, verbose=False)
            print("[%s] 로드 %.0fs, %d청크" % (tag, time.time() - t0, N), flush=True)

            # 창 222 는 GPTQ int6 가중치라야 95 GiB 안에 들어간다
            qs = 0.0
            if tag.endswith("gptq"):
                from .gptq import HessianCollector, GPTQLinear
                th = time.time(); hc = HessianCollector(tm, min_features=1024)
                b.reset_caches(); g0 = torch.Generator().manual_seed(7)
                for ci in range(3):
                    z = b.chunk_noise(ci, g0); x0 = None
                    for i in range(len(sig)):
                        x0 = b.target_x0(z, i, ci)
                        if i < len(sig) - 1:
                            s = float(sig[i + 1]); z = (1 - s) * x0 + s * b.randn_like(x0, g0)
                    b.commit(x0, ci)
                Hs = hc.finish(); b.reset_caches()
                names = [n for n, m in tm.named_modules()
                         if isinstance(m, torch.nn.Linear) and min(m.in_features, m.out_features) >= 1024]
                for n_ in names:
                    par, leaf = parent_and_leaf(tm, n_)
                    lin = getattr(par, leaf)
                    setattr(par, leaf, GPTQLinear(lin, Hs[n_], bits=6, group=128))
                    del lin
                del Hs; gc.collect(); torch.cuda.empty_cache()
                qs = time.time() - th
                print("[%s] GPTQ %.0fs" % (tag, qs), flush=True)

            b.reset_caches(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            g = torch.Generator().manual_seed(1234); outs = []; ts = []
            for ci in range(N):
                tc = time.time()
                z = b.chunk_noise(ci, g); x0 = None
                for i in range(len(sig)):
                    x0 = b.target_x0(z, i, ci)
                    if i < len(sig) - 1:
                        s = float(sig[i + 1]); z = (1 - s) * x0 + s * b.randn_like(x0, g)
                b.commit(x0, ci); outs.append(x0.detach().float().cpu()); ts.append(time.time() - tc)
                if ci == 1: torch.cuda.reset_peak_memory_stats()
                if ci % 20 == 0: print("   청크 %d/%d  %.1fs" % (ci, N, time.time() - tc), flush=True)
            lat = torch.cat(outs, dim=1); peak = torch.cuda.max_memory_allocated() / GiB
            img = tiled_decode(b.pipe.vae, lat.to(b.device), tiles=(2, 2)).detach().cpu()
            v = ((img.clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy()
            d = os.path.join(a.out, tag); os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, "frames_all.npy"), (v * 255).astype("uint8")[::4])
            st = sorted(ts[2:]) or ts
            rep["arms"][tag] = dict(
                window=la, window_s=round(la * 4 / 16, 1), kv_bits=kvb,
                gptq=tag.endswith("gptq"), frames=int(v.shape[0]), chunks=N,
                gen_s=round(sum(ts), 1), chunk_s_median=round(st[len(st) // 2], 4),
                peak_alloc_GiB=round(peak, 2),
                kv_GiB=round(kvstore2.total_bytes(tm) / GiB, 2) if kvb else None,
                quant_s=round(qs, 0), hf_energy=round(hf_energy(img), 5),
                flicker=round(float((img[:, 1:] - img[:, :-1]).abs().mean()), 5))
            print("[%s] %s" % (tag, rep["arms"][tag]), flush=True)
            del lat, img, outs
        except Exception as e:
            rep["arms"][tag] = {"error": str(e)[:300]}
            print("[%s] 실패: %s" % (tag, str(e)[:300]), flush=True)
        finally:
            if b is not None:
                try: kvstore2.uninstall(b.target_model, verbose=False)
                except Exception: pass
                b._caches = {}; del b
            gc.collect(); torch.cuda.empty_cache()
            json.dump(rep, open(rp, "w"), indent=1)
    print("[WIN60_DONE]", flush=True)


if __name__ == "__main__":
    main()
