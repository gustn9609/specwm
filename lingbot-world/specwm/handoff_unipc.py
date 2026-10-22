"""업링크 핸드오프 v2: 재노이즈 ancestral 체인 대신 공식 UniPC ODE 를 sigma_start 부터 실행 (bench_base_official 의 generate() 루프와 동일 구성).
  x0_fast (Base VAE 인코딩) -> down(res)/quant(bits) -> up -> z = (1-s)x0 + s*eps (s = 70-스텝 UniPC 그리드에서 sigma_start 이하 첫 노드)
  -> timesteps[idx:] 를 UniPC 로 끝까지 (CFG guide=5, boundary 기준 expert 라우팅)
handoff_grid.py 의 정제 체인이 '이득 없음' 의 원인인지 검증하는 용도. 출력 형식은 handoff_grid 와 동일 (handoff_grid.json).
"""
import argparse, json, os, time, gc
import numpy as np, torch
from .handoff_grid import hf_energy, psnr, read_video, resample, quantize

GiB = 2 ** 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--scene", required=True)
    ap.add_argument("--image", required=True); ap.add_argument("--action-dir", required=True); ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--fast-video", required=True)
    ap.add_argument("--sigmas", default="0.4,0.625,0.833"); ap.add_argument("--res", default="1.0,0.5"); ap.add_argument("--bits", default="8")
    ap.add_argument("--steps", type=int, default=70, help="UniPC 전체 스텝 수 (sigma_start 이하 노드만 실행)")
    ap.add_argument("--shift", type=float, default=10.0); ap.add_argument("--guide", type=float, nargs=2, default=(5.0, 5.0)); ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rep_path = os.path.join(a.out, "handoff_grid.json")
    rep = json.load(open(rep_path)) if os.path.exists(rep_path) else {"config": vars(a), "solver": "unipc", "arms": {}}
    from .binding_v1 import V1Binding
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    t0 = time.time(); b = V1Binding(a.ckpt, t5_cpu=True); dev = b.device
    models = {"high": b.pipe.high_noise_model, "low": b.pipe.low_noise_model}; resident = {"name": None}

    def use_expert(name):
        if resident["name"] == name: return models[name]
        for k, m in models.items():
            if k != name: m.cpu()
        torch.cuda.empty_cache(); models[name].to(dev); resident["name"] = name; return models[name]

    prompt = open(a.prompt_file).read().strip(); V = read_video(a.fast_video); T_use = ((len(V) - 1) // 4) * 4 + 1
    cond = b.prepare(a.image, prompt, a.action_dir, frame_num=T_use); lat_f, H, W = cond["lat"]; hw = cond["hw"]
    x = resample(torch.from_numpy(V[:T_use]).permute(3, 0, 1, 2).float() / 127.5 - 1.0, hw)
    with torch.no_grad(): x0_fast = b.pipe.vae.encode([x.to(dev)])[0].float()
    del x; torch.cuda.empty_cache(); full_bytes = x0_fast.numel() * 2
    print("[load] %.0fs latent %s boundary t=%.0f" % (time.time() - t0, tuple(x0_fast.shape), b.boundary_t), flush=True)
    arg_c = dict(context=[cond["ctx"][0]], seq_len=cond["seq_len"], y=[cond["y"]], dit_cond_dict=cond["dit_cond"])
    arg_n = dict(context=cond["ctxn"], seq_len=cond["seq_len"], y=[cond["y"]], dit_cond_dict=cond["dit_cond"])

    @torch.no_grad()
    def unipc_from(x_init, s_start, gen):
        sch = FlowUniPCMultistepScheduler(num_train_timesteps=b.pipe.num_train_timesteps, shift=1, use_dynamic_shifting=False)
        sch.set_timesteps(a.steps, device=dev, shift=a.shift); ts = sch.timesteps; sig = sch.sigmas.float().cpu()
        idx = int(torch.nonzero(sig[:len(ts)] <= s_start + 1e-6).flatten()[0])
        s0 = float(sig[idx]); eps = torch.randn(x_init.shape, generator=gen).to(dev)
        latent = (1 - s0) * x_init + s0 * eps; steps = []
        for t in ts[idx:]:
            t1 = time.time(); m = use_expert("high" if t.item() >= b.boundary_t else "low"); g = a.guide[1] if t.item() >= b.boundary_t else a.guide[0]
            tt = torch.stack([t]).to(dev); zin = [latent.to(b.param_dtype)]
            with torch.amp.autocast("cuda", dtype=b.param_dtype):
                pc = m(zin, t=tt, **arg_c)[0]; pu = m(zin, t=tt, **arg_n)[0]
            pred = pu + g * (pc - pu)
            latent = sch.step(pred.unsqueeze(0), t, latent.unsqueeze(0), return_dict=False, generator=gen)[0].squeeze(0)
            steps.append(dict(t=round(float(t), 1), s=round(time.time() - t1, 1)))
        return latent.float(), steps, s0

    def decode_save(xl, tag, ref=None):
        with torch.no_grad(): img = b.pipe.vae.decode([xl.to(dev)])[0].detach().cpu()
        v = ((img.clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy(); d = os.path.join(a.out, tag); os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "frames_all.npy"), (v * 255).astype("uint8")[::4])
        try:
            import imageio; imageio.mimwrite(os.path.join(d, "video.mp4"), list((v * 255).astype("uint8")), fps=16, quality=8)
        except Exception as e: print("[video]", e, flush=True)
        return img, dict(frames=int(v.shape[0]), hf_energy=round(hf_energy(img), 5), flicker=round(float((img[:, 1:] - img[:, :-1]).abs().mean()), 5), psnr_vs_fast=(round(psnr(img, ref), 2) if ref is not None else 99.0))

    def done(t): return t in rep["arms"] and "error" not in rep["arms"][t]
    if not done("FAST_asis"):
        img_ref, m = decode_save(x0_fast, "FAST_asis"); rep["arms"]["FAST_asis"] = dict(**m, res=1.0, bits=16, sigma=0.0, steps=[], refine_s=0.0, payload_bytes=full_bytes, payload_ratio_vs_bf16=1.0)
    else:
        with torch.no_grad(): img_ref = b.pipe.vae.decode([x0_fast.to(dev)])[0].detach().cpu()
    json.dump(rep, open(rep_path, "w"), indent=1)
    for r in [float(x) for x in a.res.split(",")]:
        for bits in [int(x) for x in a.bits.split(",")]:
            for s in [float(x) for x in a.sigmas.split(",")]:
                tag = "unipc_r%g_b%d_s%g" % (r, bits, s)
                if done(tag): print("[%s] 이미 있음" % tag, flush=True); continue
                try:
                    gen = torch.Generator().manual_seed(a.seed); lo_hw = (max(8, int(round(H * r))), max(8, int(round(W * r))))
                    payload, nb = quantize(resample(x0_fast, lo_hw), bits); x_up = resample(payload, (H, W)).to(dev)
                    torch.cuda.reset_peak_memory_stats(); t1 = time.time(); xl, steps, s0 = unipc_from(x_up, s, gen); torch.cuda.synchronize(); rs = time.time() - t1
                    img, m = decode_save(xl, tag, ref=img_ref)
                    rep["arms"][tag] = dict(**m, res=r, bits=bits, sigma=s, sigma_node=round(s0, 4), lo_hw=list(lo_hw), steps=steps, n_steps=len(steps), refine_s=round(rs, 1),
                                            peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / GiB, 2), payload_bytes=int(nb), payload_ratio_vs_bf16=round(nb / full_bytes, 4))
                    print("[%s] %d UniPC steps %.0fs payload %.1fMB | %s" % (tag, len(steps), rs, nb / 2 ** 20, m), flush=True); del xl, img
                except Exception as e:
                    rep["arms"][tag] = {"error": str(e)[:300]}; print("[%s] 실패: %s" % (tag, str(e)[:300]), flush=True)
                json.dump(rep, open(rep_path, "w"), indent=1); gc.collect(); torch.cuda.empty_cache()
    print("[HANDOFF_UNIPC_DONE]", flush=True)


if __name__ == "__main__":
    main()
