"""창 크기 x KV 정밀도 스윕: KV 압축이 같은 메모리로 더 긴 기억을 사는가.

핵심 비교
  W18_bf16   창 18 (4.5초), bf16 KV  -> 21.3 GiB   [현재 배포 설정, 유지율 0.642 측정됨]
  W72_kv4    창 72 (18초),  4비트 KV -> 21.5 GiB   [같은 메모리, 4배 긴 창]
부수 확인
  W18_kv4    압축만의 영향 (창 고정)
  W36_kv4    중간 지점
  W18_sink0  sink 토큰 기여도 (SolarWM은 sink=0)
모델을 한 번만 올리고 arm마다 창을 바꿔 재생성한다.
"""
import argparse, json, os, time, gc
import numpy as np, torch

GiB = 2 ** 30


def hf_energy(img):
    g = img.float().mean(0)
    f = torch.fft.fftshift(torch.fft.fft2(g), dim=(-2, -1)).abs()
    T, H, W = g.shape
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    r = ((yy - H / 2) ** 2 + (xx - W / 2) ** 2).sqrt()
    hf = (r > min(H, W) / 4).to(f.dtype)
    return float((f * hf).sum(dim=(-2, -1)).mean() / (f.sum(dim=(-2, -1)).mean() + 1e-9))


ARMS = [
    ("W18_bf16",  18, None, 6),
    ("W18_kv4",   18, 4,    6),
    ("W36_kv4",   36, 4,    6),
    ("W72_kv4",   72, 4,    6),
    ("W18_sink0", 18, None, 0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scene", default="00")
    ap.add_argument("--scene-root", default="examples")
    ap.add_argument("--action-dir", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--frames", type=int, default=161)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from .binding_v2 import V2Binding
    from . import kvstore2

    prompt = open(a.prompt_file).read().strip()
    src = os.path.join(a.scene_root, a.scene)
    image = os.path.join(src, "image.jpg")
    rep = {"arms": {}, "config": vars(a)}

    for tag, la, kvb, sink in ARMS:
        print("\n" + "=" * 70, flush=True)
        print("[%s] 창=%d (%.1f초) KV=%s sink=%d"
              % (tag, la, la * 4 / 16.0, "4bit" if kvb else "bf16", sink), flush=True)
        t0 = time.time()
        b = None
        try:
            b = V2Binding(a.ckpt, target_mode="fast_bf16", quant=None, sink=sink,
                          local_attn=la, swap_models=False, build_draft=False,
                          timesteps_index=(0, 250, 500, 750))
            cond = b.prepare(image, prompt, a.action_dir, frame_num=a.frames)
            N = cond["num_chunks"]; sig = b.node_sigmas()
            if kvb:
                kvstore2.install(b.target_model, bits=kvb, k_axis="token",
                                 v_axis="token", asym=True, verbose=False)
            b.reset_caches()
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            g = torch.Generator().manual_seed(1234); outs = []; ts = []
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
            lat = torch.cat(outs, dim=1)
            peak = torch.cuda.max_memory_allocated() / GiB
            img = b.pipe.vae.decode([lat.to(b.device)])[0].detach().cpu()

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
            info = dict(window=la, window_s=round(la * 4 / 16.0, 1),
                        kv_bits=kvb, sink=sink, frames=int(v.shape[0]),
                        gen_s=round(sum(ts), 1),
                        chunk_s_median=round(st[len(st) // 2], 4),
                        peak_alloc_GiB=round(peak, 2),
                        kv_GiB=round(kvstore2.total_bytes(b.target_model) / GiB, 2) if kvb else None,
                        hf_energy=round(hf_energy(img), 5),
                        flicker=round(float((img[:, 1:] - img[:, :-1]).abs().mean()), 5))
            rep["arms"][tag] = info
            print("[%s] %s" % (tag, info), flush=True)
            del lat, img, outs
        except Exception as e:
            rep["arms"][tag] = {"error": str(e)[:300]}
            print("[%s] 실패: %s" % (tag, str(e)[:300]), flush=True)
        finally:
            if b is not None:
                try:
                    kvstore2.uninstall(b.target_model, verbose=False)
                except Exception:
                    pass
                b._caches = {}
                del b
            gc.collect(); torch.cuda.empty_cache()
            print("[정리] %.1f GiB 남음, %.0f초 경과"
                  % (torch.cuda.memory_allocated() / GiB, time.time() - t0), flush=True)

    json.dump(rep, open(os.path.join(a.out, "win_sweep.json"), "w"), indent=1)
    print("\n=== 요약 ===", flush=True)
    print("%-11s %6s %8s %6s %9s %9s %9s"
          % ("arm", "창(초)", "KV", "sink", "gen", "peak", "flicker"), flush=True)
    for k, v in rep["arms"].items():
        if "error" in v:
            print("%-11s  실패: %s" % (k, v["error"][:60]), flush=True); continue
        print("%-11s %6.1f %8s %6d %8.1fs %8.2fG %9.4f"
              % (k, v["window_s"], "4bit" if v["kv_bits"] else "bf16",
                 v["sink"], v["gen_s"], v["peak_alloc_GiB"], v["flicker"]), flush=True)
    print("[WIN_SWEEP_DONE]", flush=True)


if __name__ == "__main__":
    main()
