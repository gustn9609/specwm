"""E3 원인분리 + E3' training-free 경로. Base(v1) 가 prefix 를 받아 나머지를 재생성한다.

--mode mask    : I2V 마스크를 prefix 픽셀프레임까지 1 로 (E3 원본 방식)
--mode repaint : 마스크는 공식(첫 프레임만). 매 스텝 prefix 잠재 영역을 q(x_t|x0=prefix) 로 덮어쓴다
                 (flow: x_t = (1-s) x0 + s eps). 마지막에 x0 로 고정. 모든 프레임이 같은 노이즈 수준 → in-distribution.
환경변수 SPECWM_CAUSAL_FRAMES=N + SPECWM_LAT_F 로 Base-causal target (E3'-b) 이 된다 (sdpa_patch).
"""
import argparse, copy, json, os, time
import numpy as np, torch


def read_video(path):
    import cv2; cv2.setNumThreads(1)
    c = cv2.VideoCapture(path); fr = []
    while True:
        ok, f = c.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return np.stack(fr)


_SCH_KEYS = ["model_outputs", "timestep_list", "lower_order_nums", "_step_index", "this_order", "last_sample"]


def _sch_state(sch):
    return {k: copy.deepcopy(getattr(sch, k)) for k in _SCH_KEYS if hasattr(sch, k)}


def _sch_restore(sch, st):
    for k, v in st.items():
        setattr(sch, k, copy.deepcopy(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--image", required=True)
    ap.add_argument("--action-path", required=True); ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--prefix-video", required=True); ap.add_argument("--prefix-latents", type=int, default=9)
    ap.add_argument("--mode", choices=["mask", "repaint"], default="mask")
    ap.add_argument("--frames", type=int, default=161); ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--shift", type=float, default=10.0); ap.add_argument("--guide", type=float, nargs=2, default=(5.0, 5.0))
    ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--out", required=True)
    ap.add_argument("--resident", action="store_true", help="두 expert 를 GPU 에 상주 (클라우드 서빙 가정; 스왑 오버헤드 제거)")
    ap.add_argument("--resample", type=int, default=1, help="RePaint resampling u: 스텝마다 되돌아갔다 오기를 u회")
    ap.add_argument("--full-frames", action="store_true", help="frames_all.npy 를 16fps 전체로 저장 (기본은 4배 서브샘플)")
    ap.add_argument("--blend-latents", type=int, default=0, help="seam removal: after the hard prefix (L latents), blend the next B latents with the (laundered) source video at linearly decaying weight")
    ap.add_argument("--sdedit-sigma", type=float, default=0.0, help="specificity: start from the laundered source video noised to this sigma instead of pure noise (skips steps with sigma > s0)")
    ap.add_argument("--max-area", type=int, default=480 * 832, help="생성 해상도 (h*w 상한); 480*832=386048, 352*640=225280, 240*416=99840")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    from .binding_v1 import V1Binding
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    b = V1Binding(a.ckpt, t5_cpu=True); dev = b.device
    prompt = open(a.prompt_file).read().strip()
    cond = b.prepare(a.image, prompt, a.action_path, frame_num=a.frames, max_area=a.max_area)
    lat_f, lat_h, lat_w = cond["lat"]; h, w = cond["hw"]; F = a.frames
    L = a.prefix_latents; P = 4 * (L - 1) + 1; B = a.blend_latents
    V = read_video(a.prefix_video); assert len(V) >= P, (len(V), P)
    Lx = L + B if not a.sdedit_sigma else lat_f; Px = min(4 * (Lx - 1) + 1, len(V)); Lx = (Px - 1) // 4 + 1   # extended encode for blend / sdedit
    x = torch.from_numpy(V[:Px]).permute(3, 0, 1, 2).float() / 127.5 - 1.0
    if (x.shape[2], x.shape[3]) != (h, w):
        x = torch.nn.functional.interpolate(x.permute(1, 0, 2, 3), size=(h, w), mode="bicubic", align_corners=False).permute(1, 0, 2, 3)
    with torch.no_grad():
        src_lat = b.pipe.vae.encode([x.to(dev)])[0].float()           # [16, Lx, lat_h, lat_w]
    pref_lat = src_lat[:, :L]; assert pref_lat.shape[1] == L, pref_lat.shape
    B = max(0, min(B, src_lat.shape[1] - L)); blend_w = torch.tensor([1.0 - (j + 1) / (B + 1) for j in range(B)], device=dev).view(1, B, 1, 1) if B else None
    if B: print("[blend] %d latents after the prefix, weights %s" % (B, [round(float(v), 2) for v in blend_w.flatten()]), flush=True)
    if a.mode == "mask":
        msk = torch.ones(1, F, lat_h, lat_w, device=dev); msk[:, P:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]
        with torch.no_grad():
            y_lat = b.pipe.vae.encode([torch.concat([x, torch.zeros(3, F - P, h, w)], dim=1).to(dev)])[0]
        cond["y"] = torch.concat([msk, y_lat])
    print("[prefix] mode=%s latents=%d pixels=%d causal=%s" % (a.mode, L, P, os.environ.get("SPECWM_CAUSAL_FRAMES", "off")), flush=True)

    models = {"high": b.pipe.high_noise_model, "low": b.pipe.low_noise_model}
    sch = FlowUniPCMultistepScheduler(num_train_timesteps=b.pipe.num_train_timesteps, shift=1, use_dynamic_shifting=False)
    sch.set_timesteps(a.steps, device=dev, shift=a.shift); timesteps = sch.timesteps; sigmas = sch.sigmas
    seed_g = torch.Generator(device=dev); seed_g.manual_seed(a.seed)
    latent = torch.randn(cond["shape"], generator=seed_g, device=dev)
    if a.sdedit_sigma:   # start from the laundered source (all frames) noised to s0; only steps with sigma <= s0 run
        s0 = a.sdedit_sigma; assert src_lat.shape[1] == lat_f, (src_lat.shape, lat_f)
        keep = [i for i in range(len(timesteps)) if float(sigmas[i]) <= s0 + 1e-6]; i0 = keep[0]; s_start = float(sigmas[i0])
        latent = (1 - s_start) * src_lat + s_start * latent; timesteps = timesteps[i0:]; sigmas = sigmas[i0:]; print("[sdedit] start sigma %.3f (%d/%d steps)" % (s_start, len(timesteps), a.steps), flush=True)
    boundary = b.boundary_t; resident = None; t0 = time.time(); t_swap = 0.0; torch.cuda.reset_peak_memory_stats()
    if a.resident:   # both experts resident on the GPU (cloud serving): no CPU<->GPU swap in the loop
        ts = time.time(); models["high"].to(dev); models["low"].to(dev); torch.cuda.synchronize(); print("[resident] both experts on GPU (%.1fs, %.1f GiB)" % (time.time() - ts, torch.cuda.memory_allocated() / 2**30), flush=True)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=b.param_dtype):
        for i, t in enumerate(timesteps):
            s_cur = float(sigmas[i]); s_nxt = float(sigmas[i + 1]) if i + 1 < len(sigmas) else 0.0
            U = a.resample if (a.mode == "repaint" and a.resample > 1 and i < len(timesteps) - 1) else 1
            st0 = _sch_state(sch) if U > 1 else None
            for u in range(U):
                if u > 0: _sch_restore(sch, st0)
                if a.mode == "repaint":
                    eps = torch.randn_like(pref_lat)
                    latent[:, :L] = (1 - s_cur) * pref_lat + s_cur * eps            # RePaint replacement
                    if B:                                                           # soft seam: partial replacement with decaying weight
                        eb = torch.randn_like(src_lat[:, L:L + B]); rep = (1 - s_cur) * src_lat[:, L:L + B] + s_cur * eb
                        latent[:, L:L + B] = blend_w * rep + (1 - blend_w) * latent[:, L:L + B]
                need = "high" if float(t) >= boundary else "low"
                if need != resident:
                    ts = time.time()
                    if not a.resident:
                        if resident: models[resident].cpu(); torch.cuda.empty_cache()
                        models[need].to(dev)
                    resident = need; torch.cuda.synchronize(); t_swap += time.time() - ts
                m = models[need]; tt = torch.stack([t]).to(dev); zin = [latent.to(b.param_dtype)]
                fc = m(zin, t=tt, context=[cond["ctx"][0]], seq_len=cond["seq_len"], y=[cond["y"]], dit_cond_dict=cond["dit_cond"])[0]
                fu = m(zin, t=tt, context=cond["ctxn"], seq_len=cond["seq_len"], y=[cond["y"]], dit_cond_dict=cond["dit_cond"])[0]
                gs = a.guide[1] if float(t) >= boundary else a.guide[0]; pred = fu + gs * (fc - fu)
                nxt = sch.step(pred.unsqueeze(0), t, latent.unsqueeze(0), return_dict=False, generator=seed_g)[0].squeeze(0)
                if u < U - 1:
                    # jump back sigma_{i+1} -> sigma_i  (flow matching: x_s = (1-s) x0 + s eps)
                    aa = (1.0 - s_cur) / max(1e-8, 1.0 - s_nxt)
                    var = max(0.0, s_cur ** 2 - (aa * s_nxt) ** 2)
                    latent = aa * nxt + (var ** 0.5) * torch.randn_like(nxt)
                else:
                    latent = nxt
            if i % 4 == 0: print("  step %d/%d u=%d %.0fs" % (i, len(timesteps), U, time.time() - t0), flush=True)
    if a.mode == "repaint": latent[:, :L] = pref_lat
    models["high"].cpu(); models["low"].cpu(); torch.cuda.empty_cache(); gen_s = time.time() - t0; peak = torch.cuda.max_memory_allocated() / 2 ** 30
    lat = latent.detach().float().cpu()
    with torch.no_grad(): img = b.pipe.vae.decode([lat.to(dev)])[0].detach().cpu()
    v = ((img.clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy()
    np.save(os.path.join(a.out, "frames_all.npy"), (v * 255).astype("uint8")[:: (1 if a.full_frames else 4)]); torch.save(lat, os.path.join(a.out, "latent.pt"))
    try:
        import imageio; imageio.mimwrite(os.path.join(a.out, "video.mp4"), list((v * 255).astype("uint8")), fps=16, quality=8)
    except Exception as e: print("[video]", e, flush=True)
    rep = dict(model="base_prefix", mode=a.mode, resample=a.resample, blend_latents=B, sdedit_sigma=a.sdedit_sigma, causal=os.environ.get("SPECWM_CAUSAL_FRAMES", "off"), steps=a.steps, prefix_latents=L, frames=int(v.shape[0]), hw=[int(h), int(w)], seed=a.seed, swap_s=round(t_swap, 1),
               gen_s=round(gen_s, 1), peak_alloc_GiB=round(peak, 2), flicker=round(float((img[:, 1:] - img[:, :-1]).abs().mean()), 5))
    json.dump(rep, open(os.path.join(a.out, "revisit.json"), "w"), indent=1); print(json.dumps(rep), flush=True); print("[BASE_PREFIX_DONE]", flush=True)


if __name__ == "__main__":
    main()
