"""PoseGram E0 oracle probe (학습 없음): 카메라 pose 가 같은 과거 청크의 latent 를 현재 청크 초기화에 블렌딩하면 장기 일관성이 오르는가.
arm:
  ctrl        : 일반 fast_alone (alpha=0, 난수 정렬 동일 -> 비트 동일 대조)
  oracle_a{α} : 각 latent 프레임 f 에 대해 window 밖(>= --min-gap 잠재프레임 전) 과거 프레임 중 pose 거리 최소 f' 를 찾아
                (상대 위치거리 < --pos-tol * extent, 각도 < --ang-tol) 이면 노드0 예측 x0 를 (1-α)x0 + α x0[f'] 로 교체 후 계속 디노이즈
  rand_a{α}   : pose 무관 무작위 과거 프레임 (random-address baseline)
출력: <out>/<arm>/{frames_all.npy(4fps), video.mp4, latent.pt, arm.json(매칭 통계)}
"""
import argparse, json, os, time, gc
import numpy as np, torch
CS = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--action-root", required=True); ap.add_argument("--prompt-file", required=True); ap.add_argument("--scene", default="05")
    ap.add_argument("--frames", type=int, default=873); ap.add_argument("--window", type=int, default=18); ap.add_argument("--sink", type=int, default=18)
    ap.add_argument("--seeds", default="1234"); ap.add_argument("--alphas", default="0.1,0.3,0.5"); ap.add_argument("--rand-alpha", type=float, default=0.3)
    ap.add_argument("--min-gap", type=int, default=18, help="매칭 후보는 이 잠재프레임 수 이전 (=window 밖)")
    ap.add_argument("--pos-tol", type=float, default=0.05); ap.add_argument("--ang-tol", type=float, default=10.0)
    ap.add_argument("--arms", default=""); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    from .binding_v2 import V2Binding
    prompt = open(a.prompt_file).read().strip(); t0 = time.time()
    b = V2Binding(a.ckpt, target_mode="fast_bf16", quant=None, sink=a.sink, local_attn=a.window, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
    cond = b.prepare(os.path.join("examples", a.scene, "image.jpg"), prompt, a.action_root, frame_num=a.frames)
    N = cond["num_chunks"]; sig = [float(s) for s in b.node_sigmas()]; L = N * CS
    P = np.load(os.path.join(a.action_root, "poses.npy")); pos = P[:, :3, 3]; fwd = P[:, :3, :3][:, :, 2]
    ext = float(np.linalg.norm(pos.max(0) - pos.min(0)))
    lf2px = lambda f: min(4 * f, len(P) - 1)                        # 잠재프레임 -> 픽셀프레임(자세 인덱스)
    print("[load] %.0fs N=%d L=%d window=%d sink=%d extent=%.2f" % (time.time() - t0, N, L, a.window, a.sink, ext), flush=True)

    def match_for(f, rng=None):
        """잠재프레임 f 의 pose 매칭 과거 잠재프레임 f' (없으면 None)"""
        cand = np.arange(0, f - a.min_gap + 1)
        if len(cand) == 0: return None, None
        if rng is not None: return int(rng.choice(cand)), None
        pi, fi = pos[lf2px(f)], fwd[lf2px(f)]; pc, fc = pos[[lf2px(c) for c in cand]], fwd[[lf2px(c) for c in cand]]
        d = np.linalg.norm(pc - pi, axis=1) / max(ext, 1e-6); ang = np.degrees(np.arccos(np.clip((fc * fi).sum(1), -1, 1)))
        score = d + 0.01 * ang; j = int(np.argmin(score))
        if d[j] < a.pos_tol and ang[j] < a.ang_tol: return int(cand[j]), (float(d[j]), float(ang[j]))
        return None, (float(d[j]), float(ang[j]))

    def run(seed, alpha, random_addr):
        b.reset_caches(); torch.cuda.empty_cache(); g = torch.Generator().manual_seed(seed); rng = np.random.default_rng(seed) if random_addr else None
        lat = torch.zeros(16, L, b._chunk_shape[2], b._chunk_shape[3]); stats = []; t1 = time.time()
        for ci in range(N):
            z = b.chunk_noise(ci, g); x0 = None; n_m = 0
            for i in range(len(sig)):
                x0 = b.target_x0(z, i, ci)
                if i == 0 and alpha > 0:
                    for k in range(CS):
                        f = ci * CS + k; j, info = match_for(f, rng)
                        if j is not None:
                            x0[:, k] = (1 - alpha) * x0[:, k] + alpha * lat[:, j].to(x0.device, x0.dtype); n_m += 1
                if i < len(sig) - 1: z = (1 - sig[i + 1]) * x0 + sig[i + 1] * b.randn_like(x0, g)
            b.commit(x0, ci); lat[:, ci * CS:(ci + 1) * CS] = x0.detach().float().cpu(); stats.append(n_m)
            if ci % 12 == 0: print("   chunk %d/%d matched %d  %.0fs" % (ci, N, n_m, time.time() - t1), flush=True)
        with torch.no_grad(): img = b.pipe.vae.decode([lat.to(b.device)])[0].detach().cpu()
        return lat, img, time.time() - t1, stats

    arms = []
    for seed in [int(s) for s in a.seeds.split(",")]:
        arms.append(("ctrl_s%d" % seed, seed, 0.0, False))
        for al in [float(x) for x in a.alphas.split(",")]: arms.append(("oracle_a%g_s%d" % (al, seed), seed, al, False))
        arms.append(("rand_a%g_s%d" % (a.rand_alpha, seed), seed, a.rand_alpha, True))
    if a.arms: arms = [x for x in arms if x[0] in a.arms.split(",")]
    for name, seed, al, rnd in arms:
        d = os.path.join(a.out, name)
        if os.path.exists(os.path.join(d, "arm.json")): print("[%s] 이미 있음" % name, flush=True); continue
        try:
            lat, img, s, stats = run(seed, al, rnd); os.makedirs(d, exist_ok=True)
            v = ((img.clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy(); np.save(os.path.join(d, "frames_all.npy"), (v * 255).astype("uint8")[::4]); torch.save(lat, os.path.join(d, "latent.pt"))
            import imageio; imageio.mimwrite(os.path.join(d, "video.mp4"), list((v * 255).astype("uint8")), fps=16, quality=8)
            r = dict(arm=name, seed=seed, alpha=al, random_addr=rnd, window=a.window, sink=a.sink, frames=int(v.shape[0]), gen_s=round(s, 1), matched_frames=int(sum(stats)), total_frames=L,
                     matched_per_chunk=stats, flicker=round(float((img[:, 1:] - img[:, :-1]).abs().mean()), 5))
            json.dump(r, open(os.path.join(d, "arm.json"), "w"), indent=1); print("[%s] gen %.0fs matched %d/%d flicker %.4f" % (name, s, r["matched_frames"], L, r["flicker"]), flush=True)
            del lat, img
        except Exception as e:
            print("[%s] 실패: %s" % (name, str(e)[:300]), flush=True)
        gc.collect(); torch.cuda.empty_cache()
    print("[ORACLE_PROBE_DONE]", flush=True)


if __name__ == "__main__":
    main()
