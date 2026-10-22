"""Base(v1) 저노이즈 expert 가 Fast 의 중간 상태 x_t(sigma_start) 를 받아 0 까지 디노이즈한다 (E-A 핸드오프).
hybrid_refine 의 refine() 와 같은 연산이지만, 첫 스텝에서 '노이즈를 다시 넣지 않고' 받은 x_t 를 그대로 쓴다.
--mode verify : 검사기(E-B). 55초 스트림 덤프를 3청크 창으로 돌며 Base 의 한 번 forward x0 추정과 Fast x0 의 거리를 기록.
"""
import argparse, os, json, time, numpy as np, torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--mode", choices=["handoff", "verify"], default="handoff")
    # handoff
    ap.add_argument("--xt", help="Fast 중간 상태 [16,9,H,W]"); ap.add_argument("--sigma-start", type=float, default=0.625)
    ap.add_argument("--image"); ap.add_argument("--action-dir"); ap.add_argument("--frames", type=int, default=33)
    ap.add_argument("--schedule", default="0.625,0.20,0.08", help="첫 값은 받은 x_t 의 sigma(재노이즈 안 함), 나머지는 재노이즈 후 정제")
    # verify
    ap.add_argument("--stream-dir"); ap.add_argument("--slice-root"); ap.add_argument("--scene-jpg-root")
    ap.add_argument("--slice-pattern", default="traj55_slice05_f{f0}_n33", help="slice-root 아래 디렉토리 이름 패턴")
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--guide", type=float, default=5.0); ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    from .binding_v1 import V1Binding
    b = V1Binding(a.ckpt, t5_cpu=True); dev = b.device
    models = {"high": b.pipe.high_noise_model, "low": b.pipe.low_noise_model}
    models["high"].cpu(); models["low"].to(dev); torch.cuda.empty_cache()
    prompt = open(a.prompt_file).read().strip()
    sig_all, ts_all = b.node_grid(70)
    sig_all = torch.as_tensor(sig_all).float(); ts_all = torch.as_tensor(ts_all).float()

    def nearest(s_target):
        i = int(torch.argmin((sig_all - s_target).abs())); return float(sig_all[i]), float(ts_all[i])

    @torch.no_grad()
    def low_x0(z, s, cond):
        s, t_val = nearest(s)
        assert t_val < b.boundary_t, "sigma %.3f 는 고노이즈 구간" % s
        tt = torch.full((1,), t_val, device=dev); zin = [z.to(b.param_dtype)]
        with torch.amp.autocast("cuda", dtype=b.param_dtype):
            fc = models["low"](zin, t=tt, context=[cond["ctx"][0]], seq_len=cond["seq_len"], y=[cond["y"]], dit_cond_dict=cond["dit_cond"])[0]
            fu = models["low"](zin, t=tt, context=cond["ctxn"], seq_len=cond["seq_len"], y=[cond["y"]], dit_cond_dict=cond["dit_cond"])[0]
        flow = fu + a.guide * (fc - fu)
        return (z - s * flow.float()), s, t_val

    def decode_save(x, d):
        os.makedirs(d, exist_ok=True)
        img = b.pipe.vae.decode([x.to(dev)])[0].detach().cpu()
        v = ((img.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 3, 0).numpy()
        import imageio
        imageio.mimwrite(os.path.join(d, "video.mp4"), list(v), fps=16, quality=8)
        np.save(os.path.join(d, "frames_all.npy"), v); imageio.imwrite(os.path.join(d, "first.jpg"), v[0])
        return v

    if a.mode == "handoff":
        cond = b.prepare(a.image, prompt, a.action_dir, frame_num=a.frames)
        xt = torch.load(a.xt, map_location="cpu").float().to(dev)
        assert xt.shape[1] == cond["shape"][1], (xt.shape, cond["shape"])
        sched = [float(s) for s in a.schedule.split(",")]
        g = torch.Generator().manual_seed(a.seed); steps = []; t0 = time.time(); x = xt
        with torch.no_grad():
            for k, s_req in enumerate(sched):
                if k == 0:
                    z = x                                                    # 받은 x_t 를 그대로 (핵심)
                else:
                    s_n, _ = nearest(s_req); noise = torch.randn(x.shape, generator=g, device="cpu").to(dev)
                    z = (1 - s_n) * x + s_n * noise
                x, s_used, t_used = low_x0(z, s_req, cond); steps.append(dict(sigma_req=s_req, sigma=round(s_used, 4), t=round(t_used, 1)))
                print("  step %d sigma %.3f (t=%.0f)  %.1fs" % (k, s_used, t_used, time.time() - t0), flush=True)
        torch.save(x.cpu(), os.path.join(a.out, "latent.pt")); v = decode_save(x, a.out)
        json.dump(dict(mode="handoff", sigma_start=a.sigma_start, schedule=sched, steps=steps, frames=int(len(v)), gen_s=round(time.time() - t0, 1)),
                  open(os.path.join(a.out, "handoff.json"), "w"), indent=1)
        print("[HANDOFF_DONE]", flush=True)

    else:
        xt = torch.load(os.path.join(a.stream_dir, "xt_node3.pt"), map_location="cpu").float()
        x0f = torch.load(os.path.join(a.stream_dir, "x0.pt"), map_location="cpu").float()
        N = xt.shape[1] // 3; CS = 3
        import imageio
        # 창의 첫 표시 프레임을 Fast x0 에서 디코딩해 cond 이미지로 쓴다 (디스플레이 프레임과 정렬)
        rows = []; t0 = time.time()
        for k in range(2, N, a.every):
            f0 = 0 if k == 2 else 12 * (k - 2) - 3
            sl = os.path.join(a.slice_root, a.slice_pattern.format(f0=f0))
            if not os.path.isdir(sl): print("  skip k=%d (slice missing %s)" % (k, sl), flush=True); continue
            lat_w = xt[:, (k - 2) * CS:(k + 1) * CS]; x0_w = x0f[:, (k - 2) * CS:(k + 1) * CS]
            with torch.no_grad():
                fr = b.pipe.vae.decode([x0_w[:, :1].to(dev)])[0].detach().cpu()
            img = ((fr[:, 0].clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).numpy()
            jp = os.path.join(a.out, "win_k%d_first.jpg" % k); imageio.imwrite(jp, img)
            cond = b.prepare(jp, prompt, sl, frame_num=33)
            with torch.no_grad():
                x0b, s_used, _ = low_x0(lat_w.to(dev), a.sigma_start, cond)
            xb = x0b[:, -CS:].cpu(); xf = x0_w[:, -CS:]
            d = float((xb - xf).norm() / (xf.norm() + 1e-9))
            dl = float((xb - xf).norm() / (xf.norm() + 1e-9))
            rows.append(dict(chunk=k, f0=f0, dist_rel=round(d, 4)))
            print("  k=%2d  f0=%3d  |x0_base - x0_fast|/|x0_fast| = %.4f   %.0fs" % (k, f0, d, time.time() - t0), flush=True)
            torch.save(xb, os.path.join(a.out, "x0base_k%d.pt" % k))
        json.dump(dict(mode="verify", sigma=a.sigma_start, rows=rows), open(os.path.join(a.out, "verify.json"), "w"), indent=1)
        print("[VERIFY_DONE] %d windows" % len(rows), flush=True)


if __name__ == "__main__":
    main()
