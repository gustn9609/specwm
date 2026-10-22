"""Pyramid-Flow 식 업링크 그리드 (edge -> cloud 핸드오프, action item #4).

질문: Fast(edge) 가 만든 상태를 cloud(Base) 로 보낼 때, 더 노이즈가 많고(sigma) 더 낮은 해상도(res) 로
보내도 cloud 가 정제한 결과의 품질이 유지되는가?  업링크 바이트 수 vs 품질 곡선을 얻는다.

절차 (arm 당):
  x0_fast   : Fast 영상 -> Base 해상도(480x832) 리사이즈 -> Base VAE 인코딩  [16, T, H, W]
  payload   : down(x0_fast, res) -> quant(bits)                      <- 업링크로 보내는 것 (bytes 기록)
  cloud     : up(payload) -> z = (1-s)*x + s*eps  (Pyramid-Flow 의 'corrupted upsample + renoise')
              -> Base 전문가로 K 스텝 정제 (sigma_start 이하 노드를 균등 선택, 재노이즈 ancestral 체인)
  기록      : frames_all.npy(4프레임 간격), video.mp4, psnr_vs_fast, flicker, hf_energy, 스텝 시간, 피크 메모리
DINO 거리(Base 참조 대비, 노이즈 플로어 포함) 는 scripts/floor_analysis.py 에서 계산한다.

arm 이름: r{res}_b{bits}_s{sigma}   (+ FAST_asis, UPONLY_r{res}_b{bits} = 정제 없이 down/up 만)
"""
import argparse, json, os, time, gc
import numpy as np, torch
import torch.nn.functional as F

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
    mse = float((a - b).pow(2).mean()); rng = 2.0
    return 99.0 if mse < 1e-12 else float(10 * np.log10(rng * rng / mse))


def read_video(path):
    import cv2; cv2.setNumThreads(2)
    c = cv2.VideoCapture(path); fr = []
    while True:
        ok, f = c.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return np.stack(fr)


def resample(x, hw):
    """x [C, T, H, W] -> spatial resize to hw (bilinear, antialias when shrinking)."""
    C, T, H, W = x.shape
    if (H, W) == tuple(hw): return x
    y = F.interpolate(x.permute(1, 0, 2, 3), size=tuple(hw), mode="bilinear",
                      align_corners=False, antialias=(hw[0] < H))
    return y.permute(1, 0, 2, 3).contiguous()


def quantize(x, bits):
    """대칭 absmax 양자화 (채널 x 프레임 별 스케일). 16 = bf16 캐스트. 반환: (복원값, payload bytes)."""
    if bits >= 16:
        return x.to(torch.bfloat16).float(), x.numel() * 2
    qmax = 2 ** (bits - 1) - 1
    s = x.abs().amax(dim=(2, 3), keepdim=True).clamp(min=1e-8) / qmax
    q = (x / s).round().clamp(-qmax, qmax)
    nbytes = x.numel() * bits / 8 + s.numel() * 2
    return q * s, int(nbytes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--scene", required=True)
    ap.add_argument("--image", required=True); ap.add_argument("--action-dir", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--fast-video", required=True, help="Fast(edge) 결과 video.mp4 (fast_alone)")
    ap.add_argument("--sigmas", default="0.2,0.4,0.625,0.833,0.937")
    ap.add_argument("--res", default="1.0,0.75,0.5")
    ap.add_argument("--bits", default="16,8")
    ap.add_argument("--steps", type=int, default=8, help="cloud 정제 스텝 수 K")
    ap.add_argument("--guide", type=float, default=5.0); ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--arms", default="", help="실행할 arm 필터(콤마). 비면 전체")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rep_path = os.path.join(a.out, "handoff_grid.json")
    rep = json.load(open(rep_path)) if os.path.exists(rep_path) else {"config": vars(a), "arms": {}}

    from .binding_v1 import V1Binding
    t0 = time.time()
    b = V1Binding(a.ckpt, t5_cpu=True); dev = b.device
    models = {"high": b.pipe.high_noise_model, "low": b.pipe.low_noise_model}
    resident = {"name": None}

    def use_expert(name):
        if resident["name"] == name: return models[name]
        for k, m in models.items():
            if k != name: m.cpu()
        torch.cuda.empty_cache(); models[name].to(dev); resident["name"] = name
        return models[name]

    prompt = open(a.prompt_file).read().strip()
    V = read_video(a.fast_video)
    T_use = ((len(V) - 1) // 4) * 4 + 1
    cond = b.prepare(a.image, prompt, a.action_dir, frame_num=T_use)
    lat_f, H, W = cond["lat"]; hw = cond["hw"]
    assert T_use == 4 * (lat_f - 1) + 1, (T_use, lat_f)
    sig_all, ts_all = b.node_grid(70)
    sig_all = sig_all.float(); ts_all = ts_all.float()
    print("[load] %.0fs  fast video %s -> use %d frames, base latent %s, hw %s, boundary t=%.0f" % (
        time.time() - t0, V.shape, T_use, (16, lat_f, H, W), hw, b.boundary_t), flush=True)

    # Fast 영상 -> Base 해상도 -> Base VAE 인코딩
    x = torch.from_numpy(V[:T_use]).permute(3, 0, 1, 2).float() / 127.5 - 1.0
    x = resample(x, hw)
    with torch.no_grad():
        x0_fast = b.pipe.vae.encode([x.to(dev)])[0].float()
    del x; torch.cuda.empty_cache()
    assert tuple(x0_fast.shape) == (16, lat_f, H, W), x0_fast.shape
    full_bytes_bf16 = x0_fast.numel() * 2
    px_bytes = int(T_use * hw[0] * hw[1] * 3)

    def nodes_below(s_start, K):
        idx = torch.nonzero(sig_all <= s_start + 1e-6).flatten()
        if len(idx) == 0: idx = torch.tensor([len(sig_all) - 1])
        pick = torch.linspace(0, len(idx) - 1, K).round().long().unique()
        return [int(i) for i in idx[pick]]

    @torch.no_grad()
    def x0_pred(z, i):
        s = float(sig_all[i]); t_val = float(ts_all[i])
        m = use_expert("high" if t_val >= b.boundary_t else "low")
        tt = torch.full((1,), t_val, device=dev); zin = [z.to(b.param_dtype)]
        with torch.amp.autocast("cuda", dtype=b.param_dtype):
            fc = m(zin, t=tt, context=[cond["ctx"][0]], seq_len=cond["seq_len"], y=[cond["y"]], dit_cond_dict=cond["dit_cond"])[0]
            fu = m(zin, t=tt, context=cond["ctxn"], seq_len=cond["seq_len"], y=[cond["y"]], dit_cond_dict=cond["dit_cond"])[0]
        flow = fu + a.guide * (fc - fu)
        return z - s * flow.float(), s, t_val

    @torch.no_grad()
    def refine(x_init, s_start, K, gen):
        ids = nodes_below(s_start, K); steps = []
        s0 = float(sig_all[ids[0]])
        eps = torch.randn(x_init.shape, generator=gen).to(dev)
        z = (1 - s0) * x_init + s0 * eps; x0 = x_init
        for j, i in enumerate(ids):
            torch.cuda.synchronize(); t1 = time.time()
            x0, s, t_val = x0_pred(z, i)
            torch.cuda.synchronize(); steps.append(dict(sigma=round(s, 4), t=round(t_val, 1), s=round(time.time() - t1, 1)))
            if j + 1 < len(ids):
                sn = float(sig_all[ids[j + 1]]); z = (1 - sn) * x0 + sn * torch.randn(x0.shape, generator=gen).to(dev)
        return x0, steps

    def decode_save(x, tag, ref=None):
        with torch.no_grad(): img = b.pipe.vae.decode([x.to(dev)])[0].detach().cpu()
        v = ((img.clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy()
        d = os.path.join(a.out, tag); os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "frames_all.npy"), (v * 255).astype("uint8")[::4])
        try:
            import imageio; imageio.mimwrite(os.path.join(d, "video.mp4"), list((v * 255).astype("uint8")), fps=16, quality=8)
        except Exception as e: print("[video]", e, flush=True)
        return img, dict(frames=int(v.shape[0]), hf_energy=round(hf_energy(img), 5),
                         flicker=round(float((img[:, 1:] - img[:, :-1]).abs().mean()), 5),
                         psnr_vs_fast=(round(psnr(img, ref), 2) if ref is not None else 99.0))

    def done(tag): return tag in rep["arms"] and "error" not in rep["arms"][tag]
    def save_rep(): json.dump(rep, open(rep_path, "w"), indent=1)

    # 기준: Fast 그대로 (Base VAE 왕복만)
    if not done("FAST_asis"):
        img_ref, m = decode_save(x0_fast, "FAST_asis")
        rep["arms"]["FAST_asis"] = dict(**m, res=1.0, bits=16, sigma=0.0, steps=[], refine_s=0.0,
                                        payload_bytes=full_bytes_bf16, payload_ratio_vs_bf16=1.0, payload_ratio_vs_px=full_bytes_bf16 / px_bytes)
        save_rep(); print("[FAST_asis] %s" % m, flush=True)
    else:
        with torch.no_grad(): img_ref = b.pipe.vae.decode([x0_fast.to(dev)])[0].detach().cpu()

    RES = [float(r) for r in a.res.split(",")]; BITS = [int(v) for v in a.bits.split(",")]; SIG = [float(s) for s in a.sigmas.split(",")]
    keep = set(a.arms.split(",")) if a.arms else None
    grid = []
    for r in RES:
        for bits in BITS:
            grid.append(("UPONLY_r%g_b%d" % (r, bits), r, bits, None))
            for s in SIG: grid.append(("r%g_b%d_s%g" % (r, bits, s), r, bits, s))
    print("[grid] %d arms (K=%d, 정제 스텝 시간 ~ Base 1 forward x2 CFG)" % (len(grid), a.steps), flush=True)

    for tag, r, bits, s in grid:
        if keep and tag not in keep: continue
        if done(tag): print("[%s] 이미 있음" % tag, flush=True); continue
        try:
            gen = torch.Generator().manual_seed(a.seed)
            lo_hw = (max(8, int(round(H * r))), max(8, int(round(W * r))))
            payload, nbytes = quantize(resample(x0_fast, lo_hw), bits)
            x_up = resample(payload, (H, W)).to(dev)
            torch.cuda.reset_peak_memory_stats(); t1 = time.time()
            if s is None: x, steps = x_up, []
            else: x, steps = refine(x_up, s, a.steps, gen)
            torch.cuda.synchronize(); rs = time.time() - t1
            img, m = decode_save(x, tag, ref=img_ref)
            rep["arms"][tag] = dict(**m, res=r, bits=bits, sigma=(s or 0.0), lo_hw=list(lo_hw), steps=steps, refine_s=round(rs, 1),
                                    peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / GiB, 2),
                                    payload_bytes=int(nbytes), payload_ratio_vs_bf16=round(nbytes / full_bytes_bf16, 4),
                                    payload_ratio_vs_px=round(nbytes / px_bytes, 5))
            torch.save(x.detach().cpu().to(torch.float16), os.path.join(a.out, tag, "latent.pt"))
            print("[%s] refine %.0fs (%d steps) payload %.1f MB (%.2fx bf16) | %s" % (tag, rs, len(steps), nbytes / 2 ** 20, nbytes / full_bytes_bf16, m), flush=True)
            del x, img
        except Exception as e:
            rep["arms"][tag] = {"error": str(e)[:300]}; print("[%s] 실패: %s" % (tag, str(e)[:300]), flush=True)
        save_rep(); gc.collect(); torch.cuda.empty_cache()

    print("\n=== 요약 (Fast 대비) ===", flush=True)
    print("%-22s %5s %4s %6s %8s %9s %8s %8s" % ("arm", "res", "bit", "sigma", "MB", "flicker", "hf", "psnr"), flush=True)
    for k, v in rep["arms"].items():
        if "error" in v: print("%-22s ERROR %s" % (k, v["error"][:60])); continue
        print("%-22s %5g %4d %6g %8.1f %9.4f %8.4f %8.2f" % (k, v["res"], v["bits"], v["sigma"], v["payload_bytes"] / 2 ** 20, v["flicker"], v["hf_energy"], v["psnr_vs_fast"]), flush=True)
    print("[HANDOFF_GRID_DONE]", flush=True)


if __name__ == "__main__":
    main()
