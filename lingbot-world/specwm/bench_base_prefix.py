"""E3: Base 가 'draft 의 drift 된 prefix' 에 조건부로 나머지를 재생성한다 (공식 UniPC, CFG, expert 라우팅 그대로).

I2V 의 y = [mask(4ch), VAE(prefix frames + zeros)] 에서 mask 를 prefix 픽셀프레임까지 1 로 둔다.
(공식은 첫 프레임만 1. 다중 프레임 prefix 는 학습 분포 밖일 수 있다 — 그것 자체가 E3 의 질문.)
--prefix-video: draft 영상, --prefix-latents L: 앞 L 잠재프레임을 조건으로 (픽셀 4(L-1)+1 프레임).
출력: latent.pt(전체), frames_all.npy, video.mp4, revisit.json
"""
import argparse, json, os, time, math
import numpy as np, torch


def read_video(path):
    import cv2; cv2.setNumThreads(1)
    c = cv2.VideoCapture(path); fr = []
    while True:
        ok, f = c.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return np.stack(fr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--image", required=True)
    ap.add_argument("--action-path", required=True); ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--prefix-video", required=True); ap.add_argument("--prefix-latents", type=int, default=9)
    ap.add_argument("--frames", type=int, default=161); ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--shift", type=float, default=10.0); ap.add_argument("--guide", type=float, nargs=2, default=(5.0, 5.0))
    ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    from .binding_v1 import V1Binding
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    b = V1Binding(a.ckpt, t5_cpu=True); dev = b.device
    prompt = open(a.prompt_file).read().strip()
    # 기본 cond (첫 프레임 마스크) 를 만든 뒤 y 만 prefix 버전으로 교체
    cond = b.prepare(a.image, prompt, a.action_path, frame_num=a.frames)
    lat_f, lat_h, lat_w = cond["lat"]; h, w = cond["hw"]; F = a.frames
    L = a.prefix_latents; P = 4 * (L - 1) + 1
    V = read_video(a.prefix_video); assert len(V) >= P, (len(V), P)
    x = torch.from_numpy(V[:P]).permute(3, 0, 1, 2).float() / 127.5 - 1.0
    if (x.shape[2], x.shape[3]) != (h, w):
        x = torch.nn.functional.interpolate(x.permute(1, 0, 2, 3), size=(h, w), mode="bicubic", align_corners=False).permute(1, 0, 2, 3)
    msk = torch.ones(1, F, lat_h, lat_w, device=dev); msk[:, P:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]
    with torch.no_grad():
        y_lat = b.pipe.vae.encode([torch.concat([x, torch.zeros(3, F - P, h, w)], dim=1).to(dev)])[0]
    cond["y"] = torch.concat([msk, y_lat])
    print("[prefix] latents=%d (pixel %d), mask ones in latent frames: %d" % (L, P, int((msk[0].sum(dim=(1, 2)) > 0).sum())), flush=True)

    models = {"high": b.pipe.high_noise_model, "low": b.pipe.low_noise_model}
    sch = FlowUniPCMultistepScheduler(num_train_timesteps=b.pipe.num_train_timesteps, shift=1, use_dynamic_shifting=False)
    sch.set_timesteps(a.steps, device=dev, shift=a.shift); timesteps = sch.timesteps
    seed_g = torch.Generator(device=dev); seed_g.manual_seed(a.seed)
    latent = torch.randn(cond["shape"], generator=seed_g, device=dev)
    boundary = b.boundary_t; resident = None; t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=b.param_dtype):
        for i, t in enumerate(timesteps):
            need = "high" if float(t) >= boundary else "low"
            if need != resident:
                if resident: models[resident].cpu(); torch.cuda.empty_cache()
                models[need].to(dev); resident = need
            m = models[need]; tt = torch.stack([t]).to(dev); zin = [latent.to(b.param_dtype)]
            fc = m(zin, t=tt, context=[cond["ctx"][0]], seq_len=cond["seq_len"], y=[cond["y"]], dit_cond_dict=cond["dit_cond"])[0]
            fu = m(zin, t=tt, context=cond["ctxn"], seq_len=cond["seq_len"], y=[cond["y"]], dit_cond_dict=cond["dit_cond"])[0]
            gs = a.guide[1] if float(t) >= boundary else a.guide[0]
            pred = fu + gs * (fc - fu)
            latent = sch.step(pred.unsqueeze(0), t, latent.unsqueeze(0), return_dict=False, generator=seed_g)[0].squeeze(0)
            if i % 4 == 0: print("  step %d/%d %.0fs" % (i, len(timesteps), time.time() - t0), flush=True)
    models[resident].cpu(); torch.cuda.empty_cache(); gen_s = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    lat = latent.detach().float().cpu()
    with torch.no_grad(): img = b.pipe.vae.decode([lat.to(dev)])[0].detach().cpu()
    v = ((img.clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy()
    np.save(os.path.join(a.out, "frames_all.npy"), (v * 255).astype("uint8")[::4]); torch.save(lat, os.path.join(a.out, "latent.pt"))
    try:
        import imageio; imageio.mimwrite(os.path.join(a.out, "video.mp4"), list((v * 255).astype("uint8")), fps=16, quality=8)
    except Exception as e: print("[video]", e, flush=True)
    rep = dict(model="base_prefix", steps=a.steps, prefix_latents=L, prefix_pixels=P, frames=int(v.shape[0]), gen_s=round(gen_s, 1), peak_alloc_GiB=round(peak, 2),
               flicker=round(float((img[:, 1:] - img[:, :-1]).abs().mean()), 5))
    json.dump(rep, open(os.path.join(a.out, "revisit.json"), "w"), indent=1); print(json.dumps(rep), flush=True); print("[BASE_PREFIX_DONE]", flush=True)


if __name__ == "__main__":
    main()
