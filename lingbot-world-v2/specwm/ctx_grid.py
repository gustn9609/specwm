"""다운링크 그리드 (cloud -> edge 컨텍스트, action item #4 의 반대 방향).

resync_k6 (Base 청크 0..5 를 컨텍스트로 commit, 6.. Fast 생성) 에서 cloud 가 보내는 컨텍스트를
  res   : 잠재 공간 해상도 축소 (down -> 전송 -> up)
  bits  : 잠재 양자화 (16=bf16, 8, 4)
  sigma : cloud 가 완전히 디노이즈하기 전 중간 상태(x_sigma) 를 먼저 보내고, edge(Fast) 가 그 노드부터 마무리
로 변형해 다운링크 바이트 / edge 추가 비용 / 품질(DINO 는 floor_analysis.py) 을 기록한다.
난수 소비는 resync_v4 와 동일하게 맞춰 생성 청크(6..) 의 노이즈가 resync_k6 와 같도록 한다.

arm 이름: ctx_r{res}_b{bits}_s{sigma}_s{seed}  (+ fast_alone_s{seed}, resync_k6 == ctx_r1_b16_s0)
"""
import argparse, json, os, time, gc
import numpy as np, torch
import torch.nn.functional as F
from .resync_v4 import read_video, encode_video, CS


def resample(x, hw):
    C, T, H, W = x.shape
    if (H, W) == tuple(hw): return x
    y = F.interpolate(x.permute(1, 0, 2, 3), size=tuple(hw), mode="bilinear", align_corners=False, antialias=(hw[0] < H))
    return y.permute(1, 0, 2, 3).contiguous()


def quantize(x, bits):
    if bits >= 16: return x.to(torch.bfloat16).float(), x.numel() * 2
    qmax = 2 ** (bits - 1) - 1
    s = x.abs().amax(dim=(2, 3), keepdim=True).clamp(min=1e-8) / qmax
    return (x / s).round().clamp(-qmax, qmax) * s, int(x.numel() * bits / 8 + s.numel() * 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--action-root", required=True)
    ap.add_argument("--prompt-file", required=True); ap.add_argument("--scene", default="05")
    ap.add_argument("--base-video", required=True)
    ap.add_argument("--frames", type=int, default=161); ap.add_argument("--window", type=int, default=18); ap.add_argument("--sink", type=int, default=6)
    ap.add_argument("--k", type=int, default=6, help="컨텍스트 청크 수")
    ap.add_argument("--res", default="1.0,0.75,0.5"); ap.add_argument("--bits", default="16,8,4"); ap.add_argument("--sigmas", default="0,0.625,0.833")
    ap.add_argument("--seeds", default="1234")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rep_path = os.path.join(a.out, "ctx_grid.json")
    rep = json.load(open(rep_path)) if os.path.exists(rep_path) else {"config": vars(a), "arms": {}}
    from .binding_v2 import V2Binding
    prompt = open(a.prompt_file).read().strip(); seeds = [int(s) for s in a.seeds.split(",")]
    t0 = time.time()
    b = V2Binding(a.ckpt, target_mode="fast_bf16", quant=None, sink=a.sink, local_attn=a.window,
                  swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
    cond = b.prepare(os.path.join("examples", a.scene, "image.jpg"), prompt, a.action_root, frame_num=a.frames)
    N = cond["num_chunks"]; sig = [float(s) for s in b.node_sigmas()]; shp = cond["shape"]
    hw = (int(shp[2]) * 8, int(shp[3]) * 8); H, W = int(shp[2]), int(shp[3])
    print("[load] %.0fs, %d청크, sigmas %s, latent hw %s" % (time.time() - t0, N, [round(s, 3) for s in sig], (H, W)), flush=True)
    V_base = read_video(a.base_video); torch.cuda.empty_cache()
    base_lat = encode_video(b, V_base, N, hw=hw).cpu()          # [16, N*3, H, W]
    if not os.path.exists(os.path.join(a.out, "base_frames.npy")):
        np.save(os.path.join(a.out, "base_frames.npy"), V_base[:(N * CS - 1) * 4 + 1][::4])
    n_draw = len(sig)

    def node_for(s_target):
        return int(np.argmin([abs(s - s_target) for s in sig]))

    def gen_chunk(ci, g):
        z = b.chunk_noise(ci, g); x0 = None
        for i in range(len(sig)):
            x0 = b.target_x0(z, i, ci)
            if i < len(sig) - 1: z = (1 - sig[i + 1]) * x0 + sig[i + 1] * b.randn_like(x0, g)
        return x0

    def finish_from(ci, g, x_sigma, start_node):
        """cloud 가 보낸 x_sigma(이미 노이즈 포함) 를 sigma[start_node] 노드부터 edge 가 마무리. 난수 소비 수 = n_draw."""
        b.chunk_noise(ci, g)                                    # 정렬용 소비 (cloud 의 노이즈는 이미 x_sigma 에 포함)
        for _ in range(start_node): b.randn_like(x_sigma, g)
        z = x_sigma.to(b.device).float(); x0 = None
        for i in range(start_node, len(sig)):
            x0 = b.target_x0(z, i, ci)
            if i < len(sig) - 1: z = (1 - sig[i + 1]) * x0 + sig[i + 1] * b.randn_like(x0, g)
        return x0

    def skip_draws(g):
        for _ in range(n_draw): torch.randn(b._chunk_shape, generator=g)

    def run(seed, ctx_fn):
        """ctx_fn(ci, g) -> (x0, edge_s, bytes) for ci < k, else None -> 생성"""
        b.reset_caches(); torch.cuda.empty_cache(); g = torch.Generator().manual_seed(seed)
        outs, edge_s, nbytes, t1 = [], 0.0, 0, time.time()
        for ci in range(N):
            r = ctx_fn(ci, g) if ci < a.k else None
            if r is None: x0 = gen_chunk(ci, g)
            else: x0, es, nb = r; edge_s += es; nbytes += nb
            torch.cuda.synchronize(); tc = time.time(); b.commit(x0, ci); torch.cuda.synchronize()
            if ci < a.k: edge_s += time.time() - tc                # 컨텍스트 ingest(commit) 도 edge 비용
            outs.append(x0.detach().float().cpu())
        lat = torch.cat(outs, dim=1)
        with torch.no_grad(): img = b.pipe.vae.decode([lat.to(b.device)])[0].detach().cpu()
        return lat, img, time.time() - t1, edge_s, nbytes

    def save(key, lat, img, secs, extra):
        v = ((img.clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy()
        d = os.path.join(a.out, key); os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "frames_all.npy"), (v * 255).astype("uint8")[::4]); torch.save(lat, os.path.join(d, "latent.pt"))
        try:
            import imageio; imageio.mimwrite(os.path.join(d, "video.mp4"), list((v * 255).astype("uint8")), fps=16, quality=8)
        except Exception as e: print("[video]", e, flush=True)
        r = dict(gen_s=round(secs, 1), flicker=round(float((img[:, 1:] - img[:, :-1]).abs().mean()), 5), **extra)
        rep["arms"][key] = r; json.dump(rep, open(rep_path, "w"), indent=1); print("[%s] %s" % (key, r), flush=True)

    def done(k): return k in rep["arms"] and "error" not in rep["arms"][k]
    RES = [float(r) for r in a.res.split(",")]; BITS = [int(x) for x in a.bits.split(",")]; SIG = [float(s) for s in a.sigmas.split(",")]
    full_ctx_bytes = int(16 * a.k * CS * H * W * 2)
    for seed in seeds:
        key = "fast_alone_s%d" % seed
        if not done(key):
            try:
                lat, img, s, _, _ = run(seed, lambda ci, g: None); save(key, lat, img, s, dict(seed=seed, kind="fast_alone"))
            except Exception as e: rep["arms"][key] = {"error": str(e)[:300]}; print("[%s] 실패 %s" % (key, e), flush=True)
        for r in RES:
            for bits in BITS:
                for sg in SIG:
                    key = "ctx_r%g_b%d_s%g_s%d" % (r, bits, sg, seed)
                    if done(key): print("[%s] 이미 있음" % key, flush=True); continue
                    lo_hw = (max(8, int(round(H * r))), max(8, int(round(W * r))))
                    start = node_for(sg) if sg > 0 else None
                    gq = torch.Generator().manual_seed(seed + 7)     # cloud 측 노이즈 (edge 난수와 독립)

                    def ctx_fn(ci, g, lo_hw=lo_hw, bits=bits, sg=sg, start=start, gq=gq):
                        src = base_lat[:, ci * CS:(ci + 1) * CS]
                        payload, nb = quantize(resample(src, lo_hw), bits)
                        x_up = resample(payload, (H, W))
                        if start is None:
                            skip_draws(g); return x_up.to(b.device), 0.0, nb
                        s0 = sig[start]; x_sig = (1 - s0) * x_up + s0 * torch.randn(x_up.shape, generator=gq)
                        torch.cuda.synchronize(); t1 = time.time(); x0 = finish_from(ci, g, x_sig, start); torch.cuda.synchronize()
                        return x0, time.time() - t1, nb
                    try:
                        lat, img, s, es, nb = run(seed, ctx_fn)
                        save(key, lat, img, s, dict(seed=seed, kind="ctx", res=r, bits=bits, sigma=sg, start_node=start, lo_hw=list(lo_hw),
                                                    edge_extra_s=round(es, 2), downlink_bytes=int(nb), downlink_ratio_vs_bf16=round(nb / full_ctx_bytes, 4)))
                    except Exception as e:
                        rep["arms"][key] = {"error": str(e)[:300]}; json.dump(rep, open(rep_path, "w"), indent=1); print("[%s] 실패: %s" % (key, str(e)[:300]), flush=True)
                    gc.collect(); torch.cuda.empty_cache()
    print("[CTX_GRID_DONE]", flush=True)


if __name__ == "__main__":
    main()
