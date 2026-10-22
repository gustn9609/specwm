"""실험 2: Fast 가 생성한 잠재값을 Base 의 저노이즈 전문가로 마지막 몇 스텝만 다듬는다.

근거: per-node alpha = 0.714 / 0.286 / 0.000 -- Fast 는 저노이즈 마무리를 배운 적이 없다
(고노이즈 전문가만 상속). 물리/디테일 손실이 이 구간에서 나므로, 그 구간만 교사에게 맡긴다.

절차 (SDEdit 방식):
  x0 <- Fast 잠재
  for sigma in schedule:            # 큰 sigma 부터 작은 sigma 순
      z = (1-s)*x0 + s*noise         # 재노이즈
      x0 = Base_low(z, t(s))         # 저노이즈 전문가로 x0 재예측
비용: Base 1 스텝 (양방향, 전체 프레임) 당 ~30초. 2-3 스텝이면 1-2 분.
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


def psnr(a, b):
    a, b = a.float(), b.float()
    mse = float((a - b).pow(2).mean()); rng = float(a.max() - a.min()) or 1.0
    return 99.0 if mse < 1e-12 else float(10 * torch.log10(torch.tensor(rng * rng / mse)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--action-dir", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--fast-latent", default=None)      # Fast 가 만든 latent.pt [16,T,H,W]
    ap.add_argument("--frames", type=int, required=True)  # Base prepare 에 줄 프레임 수 (잠재 T 와 맞게)
    ap.add_argument("--guide", type=float, default=5.0)
    ap.add_argument("--schedules", default="0.25;0.15;0.25,0.10;0.40,0.20,0.08")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-high", action="store_true", help="D1b: sigma 가 boundary 위이면 high-noise expert 로 (전체 target 모델 SDEdit)")
    ap.add_argument("--image", default=None, help="입력 이미지 경로 override (기본 examples/<scene>/image.jpg)")
    ap.add_argument("--max-area", type=int, default=480 * 832, help="R1: 저해상도 target (184320=360p 급)")
    ap.add_argument("--frame-offset", type=int, default=0, help="--fast-video 에서 시작 프레임 (C2 anchor: 1 프레임만)")
    ap.add_argument("--fast-video", default=None, help="R1: Fast latent 대신 Fast 영상을 target 해상도로 리사이즈 후 VAE 인코딩")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    prompt = open(a.prompt_file).read().strip()

    from .binding_v1 import V1Binding
    t0 = time.time()
    b = V1Binding(a.ckpt, t5_cpu=True)
    models = {"high": b.pipe.high_noise_model, "low": b.pipe.low_noise_model}
    models["low"].to(b.device); models["high"].cpu(); torch.cuda.empty_cache()   # allow-high 시 high expert 는 필요할 때만 GPU 로 스왑
    cond = b.prepare(a.image or os.path.join("examples", a.scene, "image.jpg"), prompt,
                     a.action_dir, frame_num=a.frames, max_area=a.max_area)
    sig_all, ts_all = b.node_grid(70)
    print("[load] %.0f초, boundary t=%.0f, cond shape=%s" % (
        time.time() - t0, b.boundary_t, cond["shape"]), flush=True)

    if a.fast_video:
        import imageio
        V = np.stack([np.asarray(f) for f in imageio.get_reader(a.fast_video)])[a.frame_offset:a.frame_offset + a.frames]
        h, w = cond["hw"] if "hw" in cond else (cond["shape"][2] * 8, cond["shape"][3] * 8)
        x = torch.from_numpy(V).permute(3, 0, 1, 2).float() / 127.5 - 1
        x = torch.nn.functional.interpolate(x.permute(1, 0, 2, 3), size=(h, w), mode="bicubic", align_corners=False).clamp(-1, 1).permute(1, 0, 2, 3)
        with torch.no_grad(): x0_fast = b.pipe.vae.encode([x.to(b.device)])[0].float().cpu()
        print("[fast-video] %d frames -> %dx%d -> latent %s" % (len(V), h, w, tuple(x0_fast.shape)), flush=True)
    else:
        x0_fast = torch.load(a.fast_latent, map_location="cpu").float()
    T_base = cond["shape"][1]
    if x0_fast.shape[1] > T_base: print("[trunc] Fast latent %d -> %d" % (x0_fast.shape[1], T_base), flush=True); x0_fast = x0_fast[:, :T_base]
    if x0_fast.shape[1] != T_base:
        raise SystemExit("잠재 프레임 불일치: Fast %d vs Base %d. --frames 를 조정할 것"
                         % (x0_fast.shape[1], T_base))
    x0_fast = x0_fast.to(b.device)

    def nearest(s_target):
        i = int(torch.argmin((sig_all - s_target).abs()))
        return float(sig_all[i]), float(ts_all[i])

    @torch.no_grad()
    def refine(x0, sched, gen):
        x = x0.clone(); steps = []
        for s_req in sched:
            s, t_val = nearest(s_req)
            if t_val >= b.boundary_t and not a.allow_high:
                raise SystemExit("sigma %.3f -> t=%.0f 는 고노이즈 구간(boundary %.0f). 저노이즈만 허용"
                                 % (s, t_val, b.boundary_t))
            use_high = a.allow_high and t_val >= b.boundary_t
            if use_high: models["low"].cpu(); models["high"].to(b.device); torch.cuda.empty_cache()
            elif next(models["high"].parameters()).device.type == "cuda": models["high"].cpu(); models["low"].to(b.device); torch.cuda.empty_cache()
            mdl = models["high"] if use_high else models["low"]
            noise = torch.randn(x.shape, generator=gen, device="cpu").to(b.device)
            z = (1 - s) * x + s * noise
            tt = torch.full((1,), t_val, device=b.device)
            zin = [z.to(b.param_dtype)]
            with torch.amp.autocast("cuda", dtype=b.param_dtype):
                fc = mdl(zin, t=tt, context=[cond["ctx"][0]], seq_len=cond["seq_len"],
                         y=[cond["y"]], dit_cond_dict=cond["dit_cond"])[0]
                fu = mdl(zin, t=tt, context=cond["ctxn"], seq_len=cond["seq_len"],
                         y=[cond["y"]], dit_cond_dict=cond["dit_cond"])[0]
            flow = fu + a.guide * (fc - fu)
            x = z - s * flow.float()
            steps.append(dict(sigma_req=s_req, sigma=round(s, 4), t=round(t_val, 1)))
        return x, steps

    def decode_save(x, tag, ref=None):
        img = b.pipe.vae.decode([x])[0].detach().cpu()
        v = ((img.clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy()
        d = os.path.join(a.out, tag); os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "frames_all.npy"), (v * 255).astype("uint8")[::4])
        try:
            import imageio
            imageio.mimwrite(os.path.join(d, "video.mp4"), list((v * 255).astype("uint8")),
                             fps=16, quality=8)
        except Exception as e:
            print("[video]", e, flush=True)
        return img, dict(frames=int(v.shape[0]), hf_energy=round(hf_energy(img), 5),
                         flicker=round(float((img[:, 1:] - img[:, :-1]).abs().mean()), 5),
                         psnr_vs_fast=(round(psnr(img, ref), 2) if ref is not None else 99.0))

    rep = {"arms": {}, "config": vars(a)}
    img_ref, m = decode_save(x0_fast, "FAST_asis")
    rep["arms"]["FAST_asis"] = dict(**m, steps=[], refine_s=0.0)
    print("[FAST_asis] %s" % m, flush=True)

    for sched_str in a.schedules.split(";"):
        sched = [float(s) for s in sched_str.split(",")]
        tag = "REFINE_" + "_".join("%.2f" % s for s in sched)
        gen = torch.Generator().manual_seed(1234)
        torch.cuda.reset_peak_memory_stats(); t = time.time()
        x, steps = refine(x0_fast, sched, gen); torch.cuda.synchronize()
        rs = time.time() - t
        img, m = decode_save(x, tag, ref=img_ref)
        rep["arms"][tag] = dict(**m, steps=steps, refine_s=round(rs, 1),
                                peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / GiB, 2))
        print("[%s] refine %.1fs peak %.2fG | %s" % (tag, rs, rep["arms"][tag]["peak_alloc_GiB"], m),
              flush=True)
        del x, img
        json.dump(rep, open(os.path.join(a.out, "hybrid.json"), "w"), indent=1)

    print("\n=== 요약 (Fast 대비) ===", flush=True)
    base = rep["arms"]["FAST_asis"]
    print("%-24s %8s %9s %9s %8s" % ("arm", "refine", "flicker", "hf", "psnr"), flush=True)
    for k, v in rep["arms"].items():
        print("%-24s %7.1fs %9.4f %9.4f %8.2f" % (
            k, v["refine_s"], v["flicker"], v["hf_energy"], v["psnr_vs_fast"]), flush=True)
    print("[HYBRID_DONE]", flush=True)


if __name__ == "__main__":
    main()
