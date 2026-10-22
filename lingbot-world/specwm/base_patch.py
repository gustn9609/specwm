"""패치 단위 수락, target = Base (아래층 진짜 시험).
Fast 의 완성 잠재값을 받아 Base 저노이즈 expert 로 한 번 다듬고(SDEdit), 8x8 잠재 패치마다
|Base - Fast| / |Fast| < tau 이면 Fast 패치 유지, 아니면 Base 패치로 교체 -> 하이브리드.
tau 를 훑어 Fast 혼자(tau=inf) ... Base 다듬기 전체(tau=0) 사이의 계단을 만든다.
"""
import argparse, os, json, time, numpy as np, torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--fast-latent", required=True)
    ap.add_argument("--image", required=True); ap.add_argument("--action-path", required=True); ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--schedule", default="0.40,0.20,0.08"); ap.add_argument("--taus", default="0.05,0.10,0.20,0.40")
    ap.add_argument("--patch", type=int, default=8); ap.add_argument("--guide", type=float, default=5.0); ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    from .binding_v1 import V1Binding
    b = V1Binding(a.ckpt, t5_cpu=True); dev = b.device
    models = {"high": b.pipe.high_noise_model, "low": b.pipe.low_noise_model}; models["high"].cpu(); models["low"].to(dev); torch.cuda.empty_cache()
    xf = torch.load(a.fast_latent, map_location="cpu").float(); T = xf.shape[1]; F = 4 * (T - 1) + 1
    prompt = open(a.prompt_file).read().strip()
    cond = b.prepare(a.image, prompt, a.action_path, frame_num=F)
    assert cond["shape"][1] == T, (cond["shape"], T)
    xf = xf.to(dev); sig_all, ts_all = b.node_grid(70); sig_all = torch.as_tensor(sig_all).float(); ts_all = torch.as_tensor(ts_all).float()
    def nearest(s): i = int(torch.argmin((sig_all - s).abs())); return float(sig_all[i]), float(ts_all[i])
    g = torch.Generator().manual_seed(a.seed); t0 = time.time()
    # ---- Base 다듬기 (hybrid_refine 과 동일) ----
    x = xf.clone()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=b.param_dtype):
        for s_req in [float(v) for v in a.schedule.split(",")]:
            s, t_val = nearest(s_req); assert t_val < b.boundary_t
            z = (1 - s) * x + s * torch.randn(x.shape, generator=g, device="cpu").to(dev)
            tt = torch.full((1,), t_val, device=dev); zin = [z.to(b.param_dtype)]
            fc = models["low"](zin, t=tt, context=[cond["ctx"][0]], seq_len=cond["seq_len"], y=[cond["y"]], dit_cond_dict=cond["dit_cond"])[0]
            fu = models["low"](zin, t=tt, context=cond["ctxn"], seq_len=cond["seq_len"], y=[cond["y"]], dit_cond_dict=cond["dit_cond"])[0]
            x = z - s * (fu + a.guide * (fc - fu)).float()
    xb = x; print("[refine] %.0fs" % (time.time() - t0), flush=True)
    # ---- 패치 거리 ----
    C, T_, H, W = xf.shape; P = a.patch; Hp, Wp = H // P, W // P
    def pv(t): return t[:, :, :Hp * P, :Wp * P].reshape(C, T_, Hp, P, Wp, P).permute(1, 2, 4, 0, 3, 5).reshape(T_, Hp, Wp, -1)
    rel = ((pv(xb) - pv(xf)).norm(dim=-1) / (pv(xf).norm(dim=-1) + 1e-6)).cpu()        # [T, Hp, Wp]
    print("[patch] rel dist: mean %.3f  p10 %.3f  p50 %.3f  p90 %.3f" % (rel.mean(), rel.quantile(0.1), rel.median(), rel.quantile(0.9)), flush=True)
    np.save(os.path.join(a.out, "rel_dist.npy"), rel.numpy())
    def stitch(keep):                       # keep: [T,Hp,Wp] bool (True=Fast)
        out = xb.clone()
        for t in range(T_):
            for i in range(Hp):
                for j in range(Wp):
                    if bool(keep[t, i, j]): out[:, t, i * P:(i + 1) * P, j * P:(j + 1) * P] = xf[:, t, i * P:(i + 1) * P, j * P:(j + 1) * P]
        return out
    def save(lat, tag):
        d = os.path.join(a.out, tag); os.makedirs(d, exist_ok=True)
        with torch.no_grad(): img = b.pipe.vae.decode([lat.to(dev)])[0].detach().cpu()
        v = ((img.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 3, 0).numpy()
        import imageio; imageio.mimwrite(os.path.join(d, "video.mp4"), list(v), fps=16, quality=8); np.save(os.path.join(d, "frames_all.npy"), v[::4]); torch.save(lat.cpu(), os.path.join(d, "latent.pt"))
    rep = dict(schedule=a.schedule, patch=P, T=T_, frames=F, arms={})
    save(xf, "fast_alone"); rep["arms"]["fast_alone"] = dict(keep=1.0)
    save(xb, "base_refined"); rep["arms"]["base_refined"] = dict(keep=0.0)
    for tau in [float(v) for v in a.taus.split(",")]:
        keep = rel < tau; tag = "hyb_t%03d" % int(round(tau * 100)); save(stitch(keep), tag)
        rep["arms"][tag] = dict(tau=tau, keep=round(float(keep.float().mean()), 4)); print("  %s keep %.2f" % (tag, keep.float().mean()), flush=True)
    json.dump(rep, open(os.path.join(a.out, "base_patch.json"), "w"), indent=1); print("[BASE_PATCH_DONE] %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
