"""Base 를 '공식 파이프라인(UniPC ODE)' 으로 생성한다.

내 이전 Base 루프는 매 스텝 x0 에 새 노이즈를 재주입하는 방식이라 공식과 달랐고,
카메라 회전이 큰 궤적에서 t~2s 이후 상하반전/좌우대칭으로 붕괴했다.
여기서는 wan.WanI2V.generate() 를 그대로 호출해 그 차이를 제거한다.
"""
import argparse, json, os, time
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--action-path", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--frames", type=int, default=161)
    ap.add_argument("--steps", type=int, default=70)          # cfg.sample_steps
    ap.add_argument("--shift", type=float, default=10.0)      # cfg.sample_shift
    ap.add_argument("--guide", type=float, nargs=2, default=(5.0, 5.0))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-area", type=int, default=480 * 832, help="잠재 면적 예산 (55s 클립은 480x832 에서 OOM -> 184320=360x512)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--worldcache", type=int, default=0, help="WorldCache-style feature caching: probe depth k (0=off)")
    ap.add_argument("--wc-tau0", type=float, default=0.08); ap.add_argument("--wc-alpha", type=float, default=2.0); ap.add_argument("--wc-beta-s", type=float, default=0.12); ap.add_argument("--wc-beta-d", type=float, default=4.0); ap.add_argument("--wc-thresh", type=float, default=0.06, help="official: rel_l1_thresh"); ap.add_argument("--wc-stepidx", action="store_true", help="official: 진행률을 스텝 인덱스(cnt/num_steps)로 — high expert 구간도 warm-up 후 캐시 가능"); ap.add_argument("--wc-ret", type=float, default=0.2, help="official: ret_ratio (초기 캐시 금지 구간)"); ap.add_argument("--wc-oalpha", type=float, default=0.2, help="official: motion_sensitivity"); ap.add_argument("--wc-gamma-max", type=float, default=2.0, help="OSI 외삽 gain 상한 (1.0 = 보간만, 외삽 없음)"); ap.add_argument("--wc-official", action="store_true", help="공식 WorldCache(WAN2.1) 판정 논리 이식판 사용: probe 3, thresh 0.06, ret 0.2, α5, β0.12, OSI, decay, AdUC")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    import wan
    from wan.configs import WAN_CONFIGS
    from PIL import Image
    if os.environ.get("SPECWM_NO_FLASH", "0") == "1":
        from specwm import sdpa_patch; sdpa_patch.apply()
    cfg = WAN_CONFIGS["i2v-A14B"]
    t0 = time.time()
    pipe = wan.WanI2V(config=cfg, checkpoint_dir=a.ckpt, device_id=0, rank=0,
                      t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
                      convert_model_dtype=True)
    print("[load] %.0fs" % (time.time() - t0), flush=True)
    wc = None
    if a.worldcache > 0:
        from specwm import worldcache
        sc = worldcache.StepCounter(a.steps, 2) if (a.wc_official and a.wc_stepidx) else None
        wc = [(worldcache.install_official(m, k=a.worldcache, thresh=a.wc_thresh, ret_ratio=a.wc_ret, alpha=a.wc_oalpha, step_counter=sc) if a.wc_official else worldcache.install(m, k=a.worldcache, tau0=a.wc_tau0, alpha=a.wc_alpha, beta_s=a.wc_beta_s, beta_d=a.wc_beta_d, gmax=a.wc_gamma_max)) for m in (pipe.low_noise_model, pipe.high_noise_model)]
        print("[worldcache] installed k=%d" % a.worldcache, flush=True)
    prompt = open(a.prompt_file).read().strip()
    img = Image.open(a.image).convert("RGB")

    torch.cuda.reset_peak_memory_stats(); tg = time.time()
    video = pipe.generate(prompt, img, action_path=a.action_path, allow_act2cam=False,
                          vis_ui=False, max_area=a.max_area, frame_num=a.frames,
                          shift=a.shift, sample_solver="unipc", sampling_steps=a.steps,
                          guide_scale=tuple(a.guide), seed=a.seed, offload_model=True)
    gen_s = time.time() - tg; peak = torch.cuda.max_memory_allocated() / GiB
    img_t = video.detach().float().cpu()          # [C, T, H, W] in [-1, 1]
    T = img_t.shape[1]
    if wc is not None:
        from specwm import worldcache
        print("[worldcache] low=%s high=%s" % (worldcache.stats(pipe.low_noise_model), worldcache.stats(pipe.high_noise_model)), flush=True)
    rep = dict(model="base_official", worldcache=(dict(k=a.worldcache, official=a.wc_official, stepidx=a.wc_stepidx, thresh=a.wc_thresh, ret=a.wc_ret, low=worldcache.stats(pipe.low_noise_model), high=worldcache.stats(pipe.high_noise_model)) if wc is not None else None), solver="unipc", steps=a.steps, shift=a.shift, max_area=a.max_area,
               guide=list(a.guide), frames=T, gen_s=round(gen_s, 1), peak_alloc_GiB=round(peak, 2),
               hf_energy=round(hf_energy(img_t), 5),
               flicker=round(float((img_t[:, 1:] - img_t[:, :-1]).abs().mean()), 5))
    v = ((img_t.clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy()
    np.save(os.path.join(a.out, "frames_all.npy"), (v * 255).astype("uint8")[::4])
    try:
        import imageio
        imageio.mimwrite(os.path.join(a.out, "video.mp4"), list((v * 255).astype("uint8")), fps=16, quality=8)
    except Exception as e:
        print("[video]", e, flush=True)
    json.dump(rep, open(os.path.join(a.out, "revisit.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1), flush=True); print("[BASE_OFFICIAL_DONE]", flush=True)


if __name__ == "__main__":
    main()
