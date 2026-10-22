"""Find a failure mode where the BIDIRECTIONAL model succeeds and the
few-step AUTOREGRESSIVE model does not -- the existence proof a verifier needs.

Candidate: REVISIT. Drive the camera forward for ~5 s, then retrace the exact
same path back to the start (the paper's Stonehenge test). The last frame
should reproduce the first. Base (bidirectional, "60 s memory") sees the whole
clip at once, so the return is trivially consistent. Fast keeps only
local_attn=18 latent frames (4.5 s): by the time the camera is back, the origin
has been EVICTED from the KV cache, so it must hallucinate the start view.

Same script serves both models (one process each -- they cannot coexist in
memory). Fast is also swept over denoising steps (4 / 16 / 40) so this doubles
as three cells of the bidirectional x many-step 2x2 that the literature has
not filled.

Metrics:
  revisit_psnr   PSNR(first frame, last frame)      -- the memory test
  hf_energy      mean high-frequency energy per frame -- fine-detail proxy
                 (LingBot admits Fast lost the low-noise expert's detail)
  flicker        mean |frame_t - frame_{t-1}|         -- temporal stability
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np, torch

GiB = 2 ** 30


def make_roundtrip(src_dir, dst_dir, half=80):
    """forward poses[0:half+1], then the same path reversed -> 2*half+1 frames."""
    os.makedirs(dst_dir, exist_ok=True)
    P = np.load(os.path.join(src_dir, "poses.npy"))
    fwd = P[:half + 1]
    rt = np.concatenate([fwd, fwd[-2::-1]], axis=0)      # ends exactly at P[0]
    np.save(os.path.join(dst_dir, "poses.npy"), rt)
    for f in ("intrinsics.npy", "action.npy", "wasd_action.npy"):
        p = os.path.join(src_dir, f)
        if os.path.exists(p):
            a = np.load(p)
            if a.shape[0] >= len(rt):
                np.save(os.path.join(dst_dir, f), a[:len(rt)])
            else:
                np.save(os.path.join(dst_dir, f), np.concatenate(
                    [a, np.repeat(a[-1:], len(rt) - a.shape[0], axis=0)]))
    return len(rt)


def psnr(a, b):
    a, b = a.float(), b.float()
    mse = float((a - b).pow(2).mean())
    rng = float(a.max() - a.min()) or 1.0
    return 99.0 if mse < 1e-12 else float(10 * torch.log10(torch.tensor(rng * rng / mse)))


def hf_energy(img):                 # img [C,T,H,W] in [-1,1]
    g = img.float().mean(0)         # [T,H,W] grayscale
    f = torch.fft.fftshift(torch.fft.fft2(g), dim=(-2, -1)).abs()
    T, H, W = g.shape
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    r = ((yy - H / 2) ** 2 + (xx - W / 2) ** 2).sqrt()
    hf = (r > min(H, W) / 4).to(f.dtype)
    return float((f * hf).sum(dim=(-2, -1)).mean() / (f.sum(dim=(-2, -1)).mean() + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["fast", "base"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scene", default="00")
    ap.add_argument("--scene-root", default="examples")
    ap.add_argument("--prompt", default="A sweeping cinematic journey along the Great Wall of "
                    "China, winding through golden autumn hills under a brilliant blue sky.")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--half", type=int, default=80)
    ap.add_argument("--local-attn", type=int, default=18)
    ap.add_argument("--sink", type=int, default=6)
    ap.add_argument("--guide", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--action-dir", default=None,
                    help="use this pose dir as-is instead of building a round trip")
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--frames", type=int, default=None,
                    help="cap frame count (e.g. 81 for a cheap sanity run)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    src = os.path.join(a.scene_root, a.scene)
    if a.prompt_file and os.path.exists(a.prompt_file):
        a.prompt = open(a.prompt_file).read().strip()
    if a.action_dir:
        rt_dir = a.action_dir
        n_frames = len(np.load(os.path.join(rt_dir, "poses.npy")))
        print(f"[action] using {rt_dir}: {n_frames} frames", flush=True)
    else:
        rt_dir = os.path.join(a.out, "roundtrip_action")
        n_frames = make_roundtrip(src, rt_dir, half=a.half)
        print(f"[roundtrip] {n_frames} frames ({n_frames/16:.1f}s): forward {a.half}, back {a.half}",
              flush=True)
    if a.frames:
        n_frames = min(n_frames, a.frames)
    image = os.path.join(src, "image.jpg")
    t0 = time.time()

    if a.model == "fast":
        from .binding_v2 import V2Binding
        idx = tuple(int(round(x)) for x in np.linspace(0, 999, a.steps + 1)[:-1]) \
            if a.steps != 4 else (0, 250, 500, 750)
        b = V2Binding(a.ckpt, target_mode="fast_bf16", quant=None, sink=a.sink,
                      local_attn=a.local_attn, swap_models=False, build_draft=False,
                      timesteps_index=idx)
        cond = b.prepare(image, a.prompt, rt_dir, frame_num=n_frames)
        N = cond["num_chunks"]; sig = b.node_sigmas()
        print(f"[load] {time.time()-t0:.0f}s, {N} chunks, {len(sig)} steps", flush=True)
        b.reset_caches(); torch.cuda.reset_peak_memory_stats()
        g = torch.Generator().manual_seed(1234); outs = []; tg = time.time()
        for ci in range(N):
            z = b.chunk_noise(ci, g); x0 = None
            for i in range(len(sig)):
                x0 = b.target_x0(z, i, ci)
                if i < len(sig) - 1:
                    s = float(sig[i + 1]); z = (1 - s) * x0 + s * b.randn_like(x0, g)
            b.commit(x0, ci); outs.append(x0.detach().float().cpu())
        lat = torch.cat(outs, dim=1); gen_s = time.time() - tg
        vae = b.pipe.vae; dev = b.device
        meta = dict(model="fast", steps=len(sig), chunks=N, local_attn=a.local_attn)

    else:
        from .binding_v1 import V1Binding
        b = V1Binding(a.ckpt, t5_cpu=True)
        cond = b.prepare(image, a.prompt, rt_dir, frame_num=n_frames)
        sigmas, timesteps = b.node_grid(a.steps)
        # ONE expert resident at a time: high-noise for early steps, low-noise
        # after the boundary. 161 frames of bidirectional attention (62k tokens)
        # will not fit next to two 14B experts.
        models = {"high": b.pipe.high_noise_model, "low": b.pipe.low_noise_model}
        models["low"].cpu(); models["high"].to(b.device); resident = "high"
        torch.cuda.empty_cache()
        x0_fn = b.make_x0_fn(models, cond, sigmas, timesteps, (a.guide, a.guide))
        print(f"[load] {time.time()-t0:.0f}s, {len(sigmas)} steps, boundary t={b.boundary_t:.0f}",
              flush=True)
        torch.cuda.reset_peak_memory_stats()
        g = torch.Generator().manual_seed(1234)
        z = torch.randn(cond["shape"], generator=g).to(b.device); tg = time.time()
        for i in range(len(sigmas)):
            need = "high" if float(timesteps[i]) >= b.boundary_t else "low"
            if need != resident:
                models[resident].cpu(); torch.cuda.empty_cache()
                models[need].to(b.device); resident = need
                print(f"[swap] -> {need} expert at step {i}", flush=True)
            x0 = x0_fn(z, i)
            if i < len(sigmas) - 1:
                s = float(sigmas[i + 1])
                z = (1 - s) * x0 + s * torch.randn(cond["shape"], generator=g).to(b.device)
            if i % 10 == 0:
                print(f"  step {i}/{len(sigmas)}  {time.time()-tg:.0f}s", flush=True)
        lat = x0.detach().float().cpu(); gen_s = time.time() - tg
        models[resident].cpu(); torch.cuda.empty_cache()
        vae = b.pipe.vae; dev = b.device
        meta = dict(model="base", steps=len(sigmas), chunks=None, local_attn=None)

    peak = torch.cuda.max_memory_allocated() / GiB
    img = vae.decode([lat.to(dev)])[0].detach().cpu()       # [C,T,H,W]
    T = img.shape[1]
    rep = dict(**meta, frames=T, gen_s=round(gen_s, 1), peak_alloc_GiB=round(peak, 2),
               revisit_psnr=round(psnr(img[:, 0], img[:, -1]), 2),
               revisit_psnr_mid=round(psnr(img[:, 0], img[:, T // 2]), 2),
               hf_energy=round(hf_energy(img), 5),
               hf_first=round(hf_energy(img[:, :1]), 5), hf_last=round(hf_energy(img[:, -1:]), 5),
               flicker=round(float((img[:, 1:] - img[:, :-1]).abs().mean()), 5))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    v = ((img.clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy()
    # save enough keyframes to compare ANY two visits later (e.g. the two 90-deg
    # views in the 0->90->0->90 trajectory, at ~1/3 and the end)
    keys = {"first": 0, "q1": T // 3, "mid": T // 2, "q3": (2 * T) // 3, "last": T - 1}
    for tag, fi in keys.items():
        plt.imsave(os.path.join(a.out, f"{tag}.jpg"), v[fi].clip(0, 1))
    np.save(os.path.join(a.out, "frames_all.npy"), (v * 255).astype("uint8")[::4])  # every 4th
    try:
        import imageio
        imageio.mimwrite(os.path.join(a.out, "video.mp4"), list((v * 255).astype("uint8")),
                         fps=16, quality=8)
    except Exception as e:
        print("[video]", e)
    torch.save(lat, os.path.join(a.out, "latent.pt"))
    json.dump(rep, open(os.path.join(a.out, "revisit.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1), flush=True)


if __name__ == "__main__":
    main()
