"""StEvo-Bench generation daemon for LingBot-World v2 Fast (single GPU, weights loaded once).

각 task: init frame + camera_WM prompt (비어 있으면 그대로 빈 프롬프트: 원 논문 파이프라인과 동일) + 카메라 포즈(action 폴더, 253 프레임)
출력: <out>/<task_id>.mp4 (16fps) + <out>/timing.json (prepare/gen/decode 초, VAE 디코드 포함 총 시간)
arm: fast_bf16 | fast_kv4 | ours(GPTQ int6 + kv4 + tiled VAE)  --shard i/n 으로 GPU 분할
"""
import argparse, json, os, time, gc
import numpy as np, torch, torch.nn as nn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--tasks", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="fast_bf16"); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--frames", type=int, default=253); ap.add_argument("--max-area", type=int, default=480 * 832)
    ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--local-attn", type=int, default=18); ap.add_argument("--sink", type=int, default=6)
    ap.add_argument("--t5-gpu", action="store_true"); ap.add_argument("--wc", type=int, default=0, help="Fast WorldCache probe k (0=off)"); ap.add_argument("--wc-thresh", type=float, default=0.06); ap.add_argument("--wc-ret", type=float, default=0.25); ap.add_argument("--nodes", default="0,1,2,3", help="사용할 디노이즈 노드 인덱스 (few-step: \"0\"=1-step, \"0,3\"=2-step σ1.0→0.625)"); ap.add_argument("--overwrite", action="store_true"); ap.add_argument("--latent-up", default="", help="D: latent 를 H,W 픽셀 크기의 latent 격자(H/8,W/8)로 공간 보간 후 VAE 디코드 (예 464,848)"); ap.add_argument("--latent-up-mode", default="bicubic")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    si, sn = [int(x) for x in a.shard.split("/")]
    tasks = json.load(open(a.tasks)); tasks = [t for i, t in enumerate(tasks) if i % sn == si]
    from .binding_v2 import V2Binding
    from .vae_tiled import tiled_decode
    from . import kvstore2, lowbit_alloc
    gptq = a.arm in ("ours", "gptq6", "gptq6_kv4"); kv_bits = 4 if a.arm in ("ours", "fast_kv4", "gptq6_kv4") else 0; tiled = a.arm == "ours"   # ablation: gptq6 (weights only) / gptq6_kv4 / ours (+ tiled VAE)
    t0 = time.time()
    b = V2Binding(a.ckpt, target_mode="fast_bf16", quant=None, sink=a.sink, local_attn=a.local_attn, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
    if kv_bits: lowbit_alloc.patch(b.pipe, verbose=False)
    if a.t5_gpu:
        te = b.pipe.text_encoder; te.model.to(b.device)
        class _TE:
            def __call__(self, texts, device): return te(texts, b.device)
        b.pipe.text_encoder = _TE()
    print("[load] %.0fs" % (time.time() - t0), flush=True)
    tm = b.target_model; sig = None
    if a.wc: from . import worldcache_fast; worldcache_fast.install(tm, k=a.wc, thresh=a.wc_thresh, ret_ratio=a.wc_ret, steps_per_chunk=len(a.nodes.split(","))); print("[fast-wc] k=%d" % a.wc, flush=True)
    decode0 = (lambda lat: tiled_decode(b.pipe.vae, lat, tiles=(2, 2))) if tiled else (lambda lat: b.pipe.vae.decode([lat])[0])
    if a.latent_up:
        UH, UW = [int(v) // 8 for v in a.latent_up.split(",")]
        def decode(lat):   # [16, T, h, w] -> spatial resize of the latent grid, then decode (extra cost = one interpolate)
            up = torch.nn.functional.interpolate(lat.permute(1, 0, 2, 3), size=(UH, UW), mode=a.latent_up_mode, align_corners=False).permute(1, 0, 2, 3).contiguous()
            return decode0(up)
        print("[latent-up] latent grid -> %dx%d (%s), pixels %dx%d" % (UH, UW, a.latent_up_mode, UH * 8, UW * 8), flush=True)
    else: decode = decode0
    rep_path = os.path.join(a.out, "timing.json"); rep = json.load(open(rep_path)) if os.path.exists(rep_path) else {}

    NODES = [int(x) for x in a.nodes.split(",")]
    def one_chunk(ci, g):
        z = b.chunk_noise(ci, g); x0 = None
        for j, i in enumerate(NODES):
            x0 = b.target_x0(z, i, ci)
            if j < len(NODES) - 1: s = float(sig[NODES[j + 1]]); z = (1 - s) * x0 + s * b.randn_like(x0, g)
        b.commit(x0, ci); return x0

    quantized = False
    for t in tasks:
        tid = t["id"]; mp4 = os.path.join(a.out, tid + ".mp4")
        if os.path.exists(mp4) and not a.overwrite: print("[skip]", tid, flush=True); continue
        try:
            torch.cuda.synchronize(); tp = time.time()
            cond = b.prepare(t["image"], t.get("camera_WM", ""), t["action"], frame_num=a.frames, max_area=a.max_area); N = cond["num_chunks"]; sig = b.node_sigmas()
            torch.cuda.synchronize(); prep_s = time.time() - tp
            if kv_bits and not getattr(b, "_kv_installed", False): kvstore2.install(tm, bits=kv_bits, k_axis="token", v_axis="token", asym=True, verbose=False); b._kv_installed = True
            if gptq and not quantized:      # 첫 task 의 조건으로 Hessian 수집 후 int6 양자화 (latency_v2 와 동일)
                from .gptq import HessianCollector, GPTQLinear
                from .latency import parent_and_leaf
                th = time.time(); hc = HessianCollector(tm, min_features=1024); b.reset_caches(); g0 = torch.Generator().manual_seed(7)
                for ci in range(min(3, N)): one_chunk(ci, g0)
                Hs = hc.finish(); b.reset_caches()
                for n_ in [n for n, m in tm.named_modules() if isinstance(m, nn.Linear) and min(m.in_features, m.out_features) >= 1024]:
                    par, leaf = parent_and_leaf(tm, n_); lin = getattr(par, leaf); setattr(par, leaf, GPTQLinear(lin, Hs[n_], bits=6, group=128)); del lin
                del Hs; gc.collect(); torch.cuda.empty_cache(); quantized = True; print("[quant] %.0fs" % (time.time() - th), flush=True)
            b.reset_caches(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); g = torch.Generator().manual_seed(a.seed)
            torch.cuda.synchronize(); tg = time.time(); outs = []
            for ci in range(N): outs.append(one_chunk(ci, g).detach().float().cpu())
            torch.cuda.synchronize(); gen_s = time.time() - tg; lat = torch.cat(outs, dim=1)
            td = time.time()
            with torch.no_grad(): img = decode(lat.to(b.device)).detach().float().cpu()
            torch.cuda.synchronize(); dec_s = time.time() - td
            v = ((img.clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy()
            import imageio; imageio.mimwrite(mp4 + ".tmp.mp4", list((v * 255).astype("uint8")), fps=16, quality=8); os.replace(mp4 + ".tmp.mp4", mp4)
            from . import worldcache_fast as _wcf
            rep[tid] = dict(arm=a.arm, nodes=a.nodes, wc=(_wcf.stats(tm) if a.wc else None), frames=int(v.shape[0]), hw=[int(v.shape[1]), int(v.shape[2])], chunks=N, prepare_s=round(prep_s, 2), gen_s=round(gen_s, 2), dec_s=round(dec_s, 2),
                            total_s=round(prep_s + gen_s + dec_s, 2), peak_GiB=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), prompt=t.get("camera_WM", ""), pose=t["pose"])
            print("[%s] %s" % (tid, rep[tid]), flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); rep[tid] = dict(arm=a.arm, error=str(e)[:300]); torch.cuda.empty_cache()
        json.dump(rep, open(rep_path + ".%d" % si, "w"), indent=1)
    print("[STEVO_GEN_DONE] shard %s" % a.shard, flush=True)


if __name__ == "__main__":
    main()
