"""빠진 OURS 칸 채우기: GPTQ int6 + KV 4bit + 2x2 타일 디코딩, 장면 x 창.

주의: GPTQ 양자화 후 원본 bf16 Linear 참조를 즉시 버려야 한다.
(bench_ours_same 은 restore 용으로 붙들고 있어 피크가 34.5 GiB 부풀었다)
"""
import argparse, json, os, time, gc
import numpy as np, torch, torch.nn as nn

GiB = 2 ** 30
CELLS = [("00", 36), ("00", 72), ("01", 18), ("01", 36), ("01", 72),
         ("05", 18), ("05", 36), ("05", 72)]


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
    ap.add_argument("--prompt-root", required=True)
    ap.add_argument("--frames", type=int, default=161)
    ap.add_argument("--bits", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from .binding_v2 import V2Binding
    from .vae_tiled import tiled_decode
    from .gptq import HessianCollector, GPTQLinear
    from . import kvstore2, lowbit_alloc

    rep = {"arms": {}}
    rp = os.path.join(a.out, "fill_ours.json")
    if os.path.exists(rp):
        rep = json.load(open(rp))

    for scene, la in CELLS:
        tag = "OURS_S%s_W%d" % (scene, la)
        if tag in rep["arms"]:
            print("[%s] 이미 있음, 건너뜀" % tag, flush=True); continue
        print("\n" + "=" * 70, flush=True); print("[%s] 시작" % tag, flush=True)
        t0 = time.time(); b = None
        try:
            prompt = open(os.path.join(a.prompt_root, "p_%s.txt" % scene)).read().strip()
            action = os.path.join(a.action_root, "rt2_action_%s" % scene)
            b = V2Binding(a.ckpt, target_mode="fast_bf16", quant=None, sink=6,
                          local_attn=la, swap_models=False, build_draft=False,
                          timesteps_index=(0, 250, 500, 750))
            lowbit_alloc.patch(b.pipe, verbose=False)
            cond = b.prepare(os.path.join("examples", scene, "image.jpg"), prompt,
                             action, frame_num=a.frames)
            N = cond["num_chunks"]; sig = b.node_sigmas(); tm = b.target_model
            print("[%s] 로드 %.0fs, %d청크" % (tag, time.time() - t0, N), flush=True)

            # lowbit 패치는 KV 를 빈 텐서로 만든다. kvstore2 가 그 자리를 대신하므로
            # 반드시 첫 forward 이전에 설치해야 한다 (bf16 KV 경로는 이미 죽어 있다).
            kvstore2.install(tm, bits=4, k_axis="token", v_axis="token",
                             asym=True, verbose=False)

            # --- Hessian (4비트 KV 상태에서 3청크 = 배포 조건과 동일) ---
            th = time.time()
            hc = HessianCollector(tm, min_features=1024)
            b.reset_caches(); g = torch.Generator().manual_seed(7)
            for ci in range(min(3, N)):
                z = b.chunk_noise(ci, g); x0 = None
                for i in range(len(sig)):
                    x0 = b.target_x0(z, i, ci)
                    if i < len(sig) - 1:
                        s = float(sig[i + 1]); z = (1 - s) * x0 + s * b.randn_like(x0, g)
                b.commit(x0, ci)
            Hs = hc.finish(); b.reset_caches()
            gc.collect(); torch.cuda.empty_cache()
            print("[%s] Hessian %d층 %.0fs" % (tag, len(Hs), time.time() - th), flush=True)

            # --- GPTQ 교체 후 원본 즉시 해제 ---
            tq = time.time()
            names = [n for n, m in tm.named_modules()
                     if isinstance(m, nn.Linear) and min(m.in_features, m.out_features) >= 1024]
            for n_ in names:
                p, leaf = parent_and_leaf(tm, n_)
                lin = getattr(p, leaf)
                setattr(p, leaf, GPTQLinear(lin, Hs[n_], bits=a.bits, group=128))
                del lin
            del Hs; gc.collect(); torch.cuda.empty_cache()
            qs = time.time() - tq

            # --- 생성 ---
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
            lat = torch.cat(outs, dim=1); peak = torch.cuda.max_memory_allocated() / GiB
            img = tiled_decode(b.pipe.vae, lat.to(b.device), tiles=(2, 2)).detach().cpu()
            v = ((img.clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy()
            d = os.path.join(a.out, tag); os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, "frames_all.npy"), (v * 255).astype("uint8")[::4])
            try:
                import imageio
                imageio.mimwrite(os.path.join(d, "video.mp4"),
                                 list((v * 255).astype("uint8")), fps=16, quality=8)
            except Exception as e:
                print("[video]", e, flush=True)
            st = sorted(ts[2:]) or ts
            rep["arms"][tag] = dict(
                scene=scene, window=la, window_s=round(la * 4 / 16.0, 1), bits=a.bits,
                kv_bits=4, frames=int(v.shape[0]), gen_s=round(sum(ts), 1),
                chunk_s_median=round(st[len(st) // 2], 4),
                peak_alloc_GiB=round(peak, 2),
                kv_GiB=round(kvstore2.total_bytes(tm) / GiB, 2), quant_s=round(qs, 0),
                hf_energy=round(hf_energy(img), 5),
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
    print("[FILL_OURS_DONE]", flush=True)


if __name__ == "__main__":
    main()
