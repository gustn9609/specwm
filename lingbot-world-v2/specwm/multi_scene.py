"""실험 1: 창 크기 x 장면 x 시드. '메모리 예산 -> 기억 지평선' 곡선을 주장으로 만든다.

장면 00 / 01 / 05, 창 18 / 36 / 72, 시드 2개, 4비트 KV + lowbit_alloc.
창 18 bf16 기준선은 시드마다 1개. 모델은 (장면, 창) 마다 1회만 로드.
"""
import argparse, json, os, time, gc
import numpy as np, torch

GiB = 2 ** 30
SCENES = ["00", "01", "05"]
WINDOWS = [18, 36, 72]
SEEDS = [1234, 4321]


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
    ap.add_argument("--action-root", required=True)      # <root>/rt2_action_<scene>
    ap.add_argument("--prompt-root", required=True)      # <root>/p_<scene>.txt
    ap.add_argument("--frames", type=int, default=161)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from .binding_v2 import V2Binding
    from . import kvstore2, lowbit_alloc

    rep = {"arms": {}, "config": vars(a)}
    configs = [(sc, 18, None) for sc in SCENES] + \
              [(sc, w, 4) for sc in SCENES for w in WINDOWS]

    for scene, la, kvb in configs:
        tag_base = "S%s_W%d_%s" % (scene, la, "kv4" if kvb else "bf16")
        prompt = open(os.path.join(a.prompt_root, "p_%s.txt" % scene)).read().strip()
        action = os.path.join(a.action_root, "rt2_action_%s" % scene)
        image = os.path.join("examples", scene, "image.jpg")
        print("\n" + "=" * 70, flush=True)
        print("[%s] 로드" % tag_base, flush=True)
        t0 = time.time(); b = None
        try:
            b = V2Binding(a.ckpt, target_mode="fast_bf16", quant=None, sink=6,
                          local_attn=la, swap_models=False, build_draft=False,
                          timesteps_index=(0, 250, 500, 750))
            if kvb:
                lowbit_alloc.patch(b.pipe, verbose=False)
            cond = b.prepare(image, prompt, action, frame_num=a.frames)
            N = cond["num_chunks"]; sig = b.node_sigmas()
            if kvb:
                kvstore2.install(b.target_model, bits=kvb, k_axis="token",
                                 v_axis="token", asym=True, verbose=False)
            print("[%s] 로드 %.0f초, %d청크" % (tag_base, time.time() - t0, N), flush=True)

            for seed in SEEDS:
                tag = "%s_s%d" % (tag_base, seed)
                b.reset_caches(); torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
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
                    if ci == 1:
                        torch.cuda.reset_peak_memory_stats()
                lat = torch.cat(outs, dim=1); peak = torch.cuda.max_memory_allocated() / GiB
                img = b.pipe.vae.decode([lat.to(b.device)])[0].detach().cpu()
                v = ((img.clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy()
                d = os.path.join(a.out, tag); os.makedirs(d, exist_ok=True)
                np.save(os.path.join(d, "frames_all.npy"), (v * 255).astype("uint8")[::4])
                st = sorted(ts[2:]) or ts
                rep["arms"][tag] = dict(
                    scene=scene, window=la, window_s=round(la * 4 / 16.0, 1),
                    kv_bits=kvb, seed=seed, frames=int(v.shape[0]),
                    gen_s=round(sum(ts), 1), chunk_s_median=round(st[len(st) // 2], 4),
                    peak_alloc_GiB=round(peak, 2),
                    hf_energy=round(hf_energy(img), 5),
                    flicker=round(float((img[:, 1:] - img[:, :-1]).abs().mean()), 5))
                print("[%s] gen %.1fs peak %.2fG flicker %.4f"
                      % (tag, sum(ts), peak, rep["arms"][tag]["flicker"]), flush=True)
                del lat, img, outs
                json.dump(rep, open(os.path.join(a.out, "multi_scene.json"), "w"), indent=1)
        except Exception as e:
            rep["arms"][tag_base + "_ERR"] = {"error": str(e)[:300]}
            print("[%s] 실패: %s" % (tag_base, str(e)[:300]), flush=True)
        finally:
            if b is not None:
                try:
                    kvstore2.uninstall(b.target_model, verbose=False)
                except Exception:
                    pass
                b._caches = {}; del b
            gc.collect(); torch.cuda.empty_cache()

    json.dump(rep, open(os.path.join(a.out, "multi_scene.json"), "w"), indent=1)
    n_ok = sum(1 for v in rep["arms"].values() if "error" not in v)
    print("\n완료: %d arm 성공 / %d 실패" % (n_ok, len(rep["arms"]) - n_ok), flush=True)
    print("[MULTI_SCENE_DONE]", flush=True)


if __name__ == "__main__":
    main()
