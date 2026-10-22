"""B1 ③ target(Base) diffusion loss as a verifier: 영상 → VAE latent x0 → 몇 개 σ 에서 z=(1-σ)x0+σε 재노이즈 → Base(low expert, cond) 로 x0 예측 → 프레임(잠재프레임)별 ||x0_pred - x0||² 평균.
낮을수록 Base 분포 안('있을 법함'). 카메라 조건은 rt2 poses 로 prepare (Fast/Base 모두 같은 조건). 출력 json: per-latent-frame loss (σ 별) + 평균."""
import argparse, json, os, time, numpy as np, torch


def read_video(path, hw=None):
    import imageio, cv2; fr = [np.asarray(f) for f in imageio.get_reader(path)]
    if hw is not None and fr[0].shape[:2] != tuple(hw): fr = [cv2.resize(f, (hw[1], hw[0]), interpolation=cv2.INTER_AREA) for f in fr]
    return np.stack(fr)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--video", required=True); ap.add_argument("--image", required=True); ap.add_argument("--action-dir", required=True); ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--frames", type=int, default=153); ap.add_argument("--sigmas", default="0.2,0.4,0.6"); ap.add_argument("--n-noise", type=int, default=2); ap.add_argument("--max-area", type=int, default=480 * 832); ap.add_argument("--out", required=True)
    ap.add_argument("--likephys", type=int, default=0, help="LikePhys 프로토콜: N 개 균등 timestep(70-노드 그리드), flow-MSE(=noise MSE), cond only, expert 는 t 로 선택")
    ap.add_argument("--first-frame", action="store_true", help="--image 대신 영상 첫 프레임을 조건 이미지로")
    ap.add_argument("--lat-frames", type=int, default=0, help="latent 프레임 수 제한(0=전체)")
    a = ap.parse_args(); os.makedirs(os.path.dirname(a.out), exist_ok=True)
    from .binding_v1 import V1Binding
    b = V1Binding(a.ckpt, t5_cpu=True); models = {"high": b.pipe.high_noise_model, "low": b.pipe.low_noise_model}; models["high"].cpu(); models["low"].to(b.device); torch.cuda.empty_cache()
    prompt = open(a.prompt_file).read().strip() if os.path.exists(a.prompt_file) else a.prompt_file
    img_path = a.image
    if a.first_frame:
        import imageio; f0 = next(iter(imageio.get_reader(a.video))); img_path = a.out + ".first.png"; imageio.imwrite(img_path, f0)
    cond = b.prepare(img_path, prompt, a.action_dir, frame_num=a.frames, max_area=a.max_area); sig_all, ts_all = b.node_grid(70)
    hw = tuple(cond["hw"]) if "hw" in cond else None
    V = read_video(a.video, hw)[: a.frames]; x = torch.from_numpy(V).permute(3, 0, 1, 2).float() / 127.5 - 1.0
    with torch.no_grad(): x0 = b.pipe.vae.encode([x.to(b.device)])[0].float()
    T_base = cond["shape"][1]
    if x0.shape[1] != T_base: raise SystemExit("latent T %d != cond %d" % (x0.shape[1], T_base))
    def nearest(s_target):
        i = int(torch.argmin((sig_all - s_target).abs())); return float(sig_all[i]), float(ts_all[i])
    res = {}; g = torch.Generator().manual_seed(0)
    if a.likephys:                                    # LikePhys: N 균등 timestep, loss = ||flow_pred - (eps - x0)||^2 (= noise MSE for flow matching)
        ids = [int(i * len(sig_all) / a.likephys) for i in range(a.likephys)]; fl = []; per_t = {}
        with torch.no_grad():
            for i in ids:
                s, t_val = float(sig_all[i]), float(ts_all[i]); use_high = t_val >= b.boundary_t
                mdl = models["high"] if use_high else models["low"]
                if use_high and next(models["high"].parameters()).device.type != "cuda": models["low"].cpu(); models["high"].to(b.device); torch.cuda.empty_cache()
                if not use_high and next(models["low"].parameters()).device.type != "cuda": models["high"].cpu(); models["low"].to(b.device); torch.cuda.empty_cache()
                noise = torch.randn(x0.shape, generator=g).to(b.device); z = (1 - s) * x0 + s * noise; tt = torch.full((1,), t_val, device=b.device)
                with torch.amp.autocast("cuda", dtype=b.param_dtype):
                    fc = mdl([z.to(b.param_dtype)], t=tt, context=[cond["ctx"][0]], seq_len=cond["seq_len"], y=[cond["y"]], dit_cond_dict=cond["dit_cond"])[0]
                l = ((fc.float() - (noise - x0)) ** 2).mean(dim=(0, 2, 3)).cpu().numpy(); fl.append(float(l.mean())); per_t["t%.0f" % t_val] = l.tolist()
        rep = dict(video=a.video, protocol="likephys", n_t=a.likephys, loss=float(np.mean(fl)), loss_array=fl, per_frame=per_t); json.dump(rep, open(a.out, "w"), indent=1); print("[DIFFLOSS_DONE]", a.out, "likephys loss %.4f" % rep["loss"], flush=True); return
    with torch.no_grad():
        for s_req in [float(s) for s in a.sigmas.split(",")]:
            s, t_val = nearest(s_req); mdl = models["low"] if t_val < b.boundary_t else None
            if mdl is None: models["low"].cpu(); models["high"].to(b.device); mdl = models["high"]
            per = []
            for k in range(a.n_noise):
                noise = torch.randn(x0.shape, generator=g).to(b.device); z = (1 - s) * x0 + s * noise; tt = torch.full((1,), t_val, device=b.device)
                with torch.amp.autocast("cuda", dtype=b.param_dtype):
                    fc = mdl([z.to(b.param_dtype)], t=tt, context=[cond["ctx"][0]], seq_len=cond["seq_len"], y=[cond["y"]], dit_cond_dict=cond["dit_cond"])[0]
                x0_pred = z - s * fc.float(); per.append(((x0_pred - x0) ** 2).mean(dim=(0, 2, 3)).cpu().numpy())      # per latent frame
            res["sigma_%.2f" % s] = np.mean(per, 0).tolist()
            if mdl is models["high"]: models["high"].cpu(); models["low"].to(b.device)
    rep = dict(video=a.video, frames=int(len(V)), latent_T=int(x0.shape[1]), per_frame=res, mean={k: float(np.mean(v)) for k, v in res.items()})
    json.dump(rep, open(a.out, "w"), indent=1); print("[DIFFLOSS_DONE]", a.out, rep["mean"], flush=True)


if __name__ == "__main__":
    main()
