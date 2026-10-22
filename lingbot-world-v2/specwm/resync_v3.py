"""Resync 실험 v2 (Fast 전용, 한 프로세스에서 (window, sink) 설정별로 바인딩을 재구성).

수정/추가:
  * RNG 정렬 버그 수정: commit 청크에서도 생성과 동일하게 노이즈 4회(chunk_noise + 3 randn) 소비
  * E7 다중 시드: --seeds 1234,4321,9999
  * E4 delayed(keep): Base 0..k-1 + Fast 자기 chunk k..k+L-1 (fast_alone latent) 를 commit, k+L 부터 생성
        (rollback 변형은 상태가 resync_k 와 동일하므로 오프라인 분석)
  * plateau: (W,sink) = (36,6), (18,0), (24,6), (24,18) 에서 resync_k6 + fast_alone
        (24,18) 은 E5 '고정 앵커' = 재접지한 Base 청크 0..5(18프레임) 를 sink 로 고정, Fast 청크만 roll
  * --ctx16: Base 16-step 영상을 context 로 (싼 target 검증)
arm 이름: <kind>_s<seed>  (kind: fast_alone, ctrl_k6, resync_k3/k6/k9, every3, keep_k3_L2/L4/L8)
"""
import argparse, json, os, time, gc
import numpy as np, torch

CS = 3


def read_video(path):
    import cv2; cv2.setNumThreads(1)
    c = cv2.VideoCapture(path); fr = []
    while True:
        ok, f = c.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return np.stack(fr)


def encode_video(b, V, N, hw=None):
    T_need = (N * CS - 1) * 4 + 1
    T_use = min(len(V), T_need)
    L3 = (((T_use - 1) // 4 + 1) // CS) * CS                                   # 잠재 프레임을 청크(3) 배수로
    T_use = 4 * (L3 - 1) + 1                                                    # 4n+1 픽셀 프레임
    x = torch.from_numpy(V[:T_use]).permute(3, 0, 1, 2).float() / 127.5 - 1.0
    if hw is not None and (x.shape[2], x.shape[3]) != tuple(hw):
        x = torch.nn.functional.interpolate(x.permute(1, 0, 2, 3), size=tuple(hw), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
    with torch.no_grad():
        lat = b.pipe.vae.encode([x.to(b.device)])[0].float()
    assert lat.shape[1] % CS == 0, lat.shape
    return lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--action-root", required=True)
    ap.add_argument("--prompt-file", required=True); ap.add_argument("--scene", default="05")
    ap.add_argument("--base-video", required=True); ap.add_argument("--base16-video", default=None)
    ap.add_argument("--prefix-videos", default="", help="E3/E3' cond arms: name=path,name=path (각각 chunks 3..5 를 그 영상에서 commit)")
    ap.add_argument("--xfade", action="store_true", help="E4 rollback + latent crossfade arms (k=3, L=2/4/8)")
    ap.add_argument("--xfade-node", type=int, default=1, help="crossfade 시작 노드 (1 -> sigma[1] 에서 Fast 청크를 재노이즈)")
    ap.add_argument("--configs", default="all")
    ap.add_argument("--frames", type=int, default=161)
    ap.add_argument("--seeds", default="1234,4321,9999")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    from .binding_v2 import V2Binding
    prompt = open(a.prompt_file).read().strip()
    seeds = [int(s) for s in a.seeds.split(",")]
    PREF = [kv.split("=", 1) for kv in a.prefix_videos.split(",") if "=" in kv]

    # (window, sink) -> arm 목록.  각 항목: (name, seed, plan_fn)
    CONFIGS = [
        ((18, 6), [("fast_alone", s) for s in seeds] + [("ctrl_k6", seeds[0])]
                  + [("resync_k%d" % k, s) for k in (3, 6, 9) for s in seeds]
                  + [("every3", s) for s in seeds]
                  + [("keep_k3_L%d" % L, seeds[0]) for L in (2, 4, 8)]
                  + ([("resync_k6_ctx16", seeds[0])] if a.base16_video else [])
                  + [("cond_%s" % n, seeds[0]) for n, _ in PREF]
                  + ([("xfade_k3_L%d" % L, seeds[0]) for L in (2, 4, 8)] if a.xfade else [])),
        ((36, 6), [("fast_alone", seeds[0]), ("resync_k6", seeds[0])]),
        ((18, 0), [("fast_alone", seeds[0]), ("resync_k6", seeds[0])]),
        ((24, 6), [("fast_alone", seeds[0]), ("resync_k6", seeds[0])]),
        ((24, 18), [("fast_alone", seeds[0]), ("resync_k6", seeds[0])]),
    ]
    V_base = read_video(a.base_video); V_b16 = read_video(a.base16_video) if a.base16_video else None
    V_pref = {n: read_video(p) for n, p in PREF}
    if a.configs != "all": CONFIGS = [c for c in CONFIGS if "W%d_sink%d" % c[0] in a.configs.split(",")]
    rep_path = os.path.join(a.out, "resync_v2.json")
    rep = json.load(open(rep_path)) if os.path.exists(rep_path) else {}

    for (W, sink), arms in CONFIGS:
        cfg = "W%d_sink%d" % (W, sink)
        todo = [(n, s) for n, s in arms if "%s/%s_s%d" % (cfg, n, s) not in rep]
        if not todo:
            print("[%s] 모두 완료" % cfg, flush=True); continue
        t0 = time.time()
        b = V2Binding(a.ckpt, target_mode="fast_bf16", quant=None, sink=sink, local_attn=W,
                      swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
        cond = b.prepare(os.path.join("examples", a.scene, "image.jpg"), prompt, a.action_root, frame_num=a.frames)
        N = cond["num_chunks"]; sig = b.node_sigmas()
        print("[%s] 로드 %.0fs, %d청크, node sigmas %s" % (cfg, time.time() - t0, N, [round(float(x), 3) for x in sig]), flush=True)
        base_lat = encode_video(b, V_base, N); b16_lat = encode_video(b, V_b16, N) if V_b16 is not None else None
        pref_lat = {n: encode_video(b, V, N) for n, V in V_pref.items()}
        if not os.path.exists(os.path.join(a.out, "base_frames.npy")):
            np.save(os.path.join(a.out, "base_frames.npy"), V_base[:(N * CS - 1) * 4 + 1][::4])
        n_draw = len(sig)   # 생성 시 청크당 난수 소비: chunk_noise 1 + randn_like (len(sig)-1)

        def gen_chunk(ci, g):
            z = b.chunk_noise(ci, g); x0 = None
            for i in range(len(sig)):
                x0 = b.target_x0(z, i, ci)
                if i < len(sig) - 1:
                    s = float(sig[i + 1]); z = (1 - s) * x0 + s * b.randn_like(x0, g)
            return x0

        def skip_draws(g):
            for _ in range(n_draw): torch.randn(b._chunk_shape, generator=g)

        def gen_chunk_from(ci, g, x0_init, start_node):
            """SDEdit 식: x0_init 를 sigma[start_node] 로 재노이즈해 그 노드부터 디노이즈 (난수 소비 수는 동일하게 맞춤)"""
            s0 = float(sig[start_node]); z = (1 - s0) * x0_init.to(b.device).float() + s0 * b.chunk_noise(ci, g); x0 = None
            for _ in range(start_node): b.randn_like(z, g)          # 건너뛴 노드만큼 난수 소비 (정렬 유지)
            for i in range(start_node, len(sig)):
                x0 = b.target_x0(z, i, ci)
                if i < len(sig) - 1:
                    s = float(sig[i + 1]); z = (1 - s) * x0 + s * b.randn_like(x0, g)
            return x0

        def run(plan, seed, ctx):
            """plan[ci] in {'G', 'B', 'F'}: 생성 / Base context commit / Fast(fast_alone) context commit"""
            b.reset_caches(); torch.cuda.empty_cache()
            g = torch.Generator().manual_seed(seed); outs = []; t1 = time.time()
            for ci in range(N):
                if plan[ci] == "G":
                    x0 = gen_chunk(ci, g)
                elif plan[ci] == "X":
                    x0 = gen_chunk_from(ci, g, ctx["X"][:, ci * CS:(ci + 1) * CS], a.xfade_node)
                else:
                    skip_draws(g)
                    src = ctx["B"] if plan[ci] == "B" else ctx["F"]
                    x0 = src[:, ci * CS:(ci + 1) * CS].to(b.device).float()
                b.commit(x0, ci); outs.append(x0.detach().float().cpu())
            lat = torch.cat(outs, dim=1)
            with torch.no_grad(): img = b.pipe.vae.decode([lat.to(b.device)])[0].detach().cpu()
            return lat, img, time.time() - t1

        def save(key, lat, img, plan, secs, seed):
            v = ((img.clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy()
            d = os.path.join(a.out, key); os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, "frames_all.npy"), (v * 255).astype("uint8")[::4])
            torch.save(lat, os.path.join(d, "latent.pt"))
            try:
                import imageio; imageio.mimwrite(os.path.join(d, "video.mp4"), list((v * 255).astype("uint8")), fps=16, quality=8)
            except Exception as e: print("[video]", e, flush=True)
            r = dict(config=cfg, window=W, sink=sink, seed=seed, plan="".join(plan), gen_s=round(secs, 1),
                     flicker=round(float((img[:, 1:] - img[:, :-1]).abs().mean()), 5))
            rep[key] = r; json.dump(rep, open(rep_path, "w"), indent=1); print("[%s] %s" % (key, r), flush=True)

        fast_lat = {}
        for name, seed in todo:
            key = "%s/%s_s%d" % (cfg, name, seed)
            try:
                ctx = {"B": base_lat, "F": fast_lat.get(seed)}
                if name == "fast_alone":
                    plan = ["G"] * N; lat, img, s = run(plan, seed, ctx); fast_lat[seed] = lat
                elif name == "ctrl_k6":
                    if fast_lat.get(seed) is None: fast_lat[seed] = torch.load(os.path.join(a.out, "%s/fast_alone_s%d/latent.pt" % (cfg, seed)))
                    ctx["F"] = fast_lat[seed]; plan = ["F"] * 6 + ["G"] * (N - 6); lat, img, s = run(plan, seed, ctx)
                elif name.startswith("cond_"):
                    # E3/E3': 0..2 = Fast 자기 latent(drift 된 prefix), 3..5 = target 이 그 prefix 뒤를 재생성한 창, 6.. 생성
                    if fast_lat.get(seed) is None: fast_lat[seed] = torch.load(os.path.join(a.out, "%s/fast_alone_s%d/latent.pt" % (cfg, seed)))
                    ctx = {"B": pref_lat[name[5:]], "F": fast_lat[seed]}; plan = ["F"] * 3 + ["B"] * 3 + ["G"] * (N - 6); lat, img, s = run(plan, seed, ctx)
                elif name.startswith("xfade_k3_L"):
                    # E4 rollback + latent crossfade: Base ctx 0..2, 재생성 3..3+L-1 (미표시), 청크 3+L 은 Fast 의 해당 청크를
                    # sigma[xfade_node] 로 재노이즈한 것에서 시작(SDEdit 전이), 이후 정상 생성
                    L = int(name.split("L")[-1]); k = 3
                    if fast_lat.get(seed) is None: fast_lat[seed] = torch.load(os.path.join(a.out, "%s/fast_alone_s%d/latent.pt" % (cfg, seed)))
                    ctx = {"B": base_lat, "F": fast_lat[seed], "X": fast_lat[seed]}
                    plan = ["B"] * k + ["G"] * L + ["X"] + ["G"] * (N - k - L - 1); lat, img, s = run(plan, seed, ctx)
                elif name.startswith("resync_k6_ctx16"):
                    ctx = {"B": b16_lat, "F": None}; plan = ["B"] * 6 + ["G"] * (N - 6); lat, img, s = run(plan, seed, ctx)
                elif name.startswith("resync_k"):
                    k = int(name.split("k")[-1]); plan = ["B"] * k + ["G"] * (N - k); lat, img, s = run(plan, seed, ctx)
                elif name.startswith("keep_k3_L"):
                    L = int(name.split("L")[-1]); k = 3
                    if fast_lat.get(seed) is None: fast_lat[seed] = torch.load(os.path.join(a.out, "%s/fast_alone_s%d/latent.pt" % (cfg, seed)))
                    ctx["F"] = fast_lat[seed]; plan = ["B"] * k + ["F"] * L + ["G"] * (N - k - L); lat, img, s = run(plan, seed, ctx)
                elif name == "every3":
                    lat_parts = []; tsum = 0.0
                    for k0, k1 in [(0, 3), (3, 6), (6, 9), (9, N)]:
                        b.reset_caches(); torch.cuda.empty_cache(); g = torch.Generator().manual_seed(seed); t1 = time.time()
                        for ci in range(k1):
                            if ci < k0:
                                skip_draws(g); b.commit(base_lat[:, ci * CS:(ci + 1) * CS].to(b.device).float(), ci)
                            else:
                                x0 = gen_chunk(ci, g); b.commit(x0, ci); lat_parts.append(x0.detach().float().cpu())
                        tsum += time.time() - t1
                    lat = torch.cat(lat_parts, dim=1)
                    with torch.no_grad(): img = b.pipe.vae.decode([lat.to(b.device)])[0].detach().cpu()
                    plan = ["G"] * N; s = tsum
                else:
                    raise ValueError(name)
                save(key, lat, img, plan, s, seed)
            except Exception as e:
                rep[key] = {"error": str(e)[:300]}; json.dump(rep, open(rep_path, "w"), indent=1)
                print("[%s] 실패: %s" % (key, str(e)[:300]), flush=True)
            gc.collect(); torch.cuda.empty_cache()
        b._caches = {}; del b; gc.collect(); torch.cuda.empty_cache()
    print("[RESYNC_V2_DONE]", flush=True)


if __name__ == "__main__":
    main()
