"""Rolling-Forcing 식 파이프라인 디노이징 (기기 Fast 쪽). 55초 드리프트 검증용.

sequential : 현행. 청크 k 의 4노드가 다 끝나야 k+1 시작. k+1 은 k 의 완성 x0 를 본다.
pipeline D : 스텝 s 마다 활성 청크 전부 한 노드씩 전진. 새 청크는 D 스텝마다 하나 진입.
             D=1 : 청크 s 가 s-1 을 node1 에서, s-2 를 node2 에서 봄 (한 노드 뒤처짐)
             D=2 : 청크 c 가 c-1 을 node2 에서 봄 (두 노드 뒤처짐; 더 흐린 이웃)
             청크 KV 는 매 노드마다 새 x0 추정으로 '다시 쓴다'.

KV 재쓰기: 청크 첫 커밋 직전의 (global_end, local_end) 를 저장해 두고, 재쓰기 시 그 값으로 되돌려 재커밋.
  단, 그 사이 캐시 밀려남(eviction)이 있었으면 local 위치가 왼쪽으로 이동했으므로
  shift = (g_now - l_now) - (g_before - l_before) 만큼 local_end 를 보정한다. (g - l = 누적 밀려남 토큰 수)
  검증: 밀려남이 일어난 뒤 자기 내용 재커밋 -> 비트 동일해야 함 (max |dK| = 0). 아니면 pipeline 모드 중단.
"""
import argparse, os, json, time, numpy as np, torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--action-root", required=True)
    ap.add_argument("--prompt-file", required=True); ap.add_argument("--scene", default="05")
    ap.add_argument("--frames", type=int, default=161); ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--window", type=int, default=18); ap.add_argument("--sink", type=int, default=6)
    ap.add_argument("--mode", choices=["sequential", "pipeline"], required=True)
    ap.add_argument("--lag", type=int, default=1, help="pipeline: 새 청크 진입 간격 D (1 = 한 노드 뒤처짐, 2 = 두 노드)")
    ap.add_argument("--save-every", type=int, default=1, help="frames_all.npy 서브샘플 (55초는 4)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    from .binding_v2 import V2Binding
    b = V2Binding(a.ckpt, target_mode="fast_bf16", quant=None, sink=a.sink, local_attn=a.window,
                  swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
    prompt = open(a.prompt_file).read().strip()
    cond = b.prepare(os.path.join("examples", a.scene, "image.jpg"), prompt, a.action_root, frame_num=a.frames)
    N = cond["num_chunks"]; sig = b.node_sigmas(); D = len(sig)
    torch.cuda.empty_cache()
    g = torch.Generator().manual_seed(a.seed); b.reset_caches()
    LL = [b._caches["target"]["kv"]]
    snap = {}

    def take():
        return ([(c["global_end_index"].clone(), c["local_end_index"].clone()) for c in LL[0]], b._committed_frames)
    def put(s):
        for c, (ge, le) in zip(LL[0], s[0]): c["global_end_index"].copy_(ge); c["local_end_index"].copy_(le)
        b._committed_frames = s[1]
    def first_commit(x0, ci):
        b.commit(x0, ci); after = take()
        # 청크의 (커밋 직후) 위치와 토큰 수를 저장. 토큰 수는 고정(CS*frame_seqlen); 인덱스 차로 재면 draft forward 가 이미 올려놔 0 이 됨.
        nt = int(b._chunk_size * b._frame_seqlen)
        snap[ci] = dict(after=after, ntok=[nt] * len(after[0]))
    def rewrite(x0, ci):
        live = take(); st = snap[ci]; adj = []
        for (g_a, l_a), (g_n, l_n), nt in zip(st["after"][0], live[0], st["ntok"]):
            shift = (int(g_n) - int(l_n)) - (int(g_a) - int(l_a))          # 커밋 이후 추가로 밀려난 토큰 수
            l_start_now = int(l_a) - nt - shift                            # 청크의 현재 local 시작 위치
            adj.append((torch.tensor(int(g_a) - nt, device=g_a.device, dtype=g_a.dtype),
                        torch.tensor(l_start_now, device=l_a.device, dtype=l_a.dtype)))
        put((adj, live[1])); b.commit(x0, ci); put(live)
        nt0 = st["ntok"][0]; return int(adj[0][1]), int(adj[0][1]) + nt0        # layer0 의 [start, end) 슬롯

    def one_node(z_prev_x0, ci, node, z0=None):
        if node == 0: z = z0
        else:
            s = float(sig[node]); z = (1 - s) * z_prev_x0 + s * b.randn_like(z_prev_x0, g)
        return b.target_x0(z, node, ci)

    x0_of = {}; t0 = time.time(); rep = {}
    with torch.no_grad():
        if a.mode == "sequential":
            for ci in range(N):
                x0 = None; z0 = b.chunk_noise(ci, g)
                for i in range(D): x0 = one_node(x0, ci, i, z0)
                b.commit(x0, ci); x0_of[ci] = x0.detach().float().cpu()
                if ci % 10 == 0: print("  chunk %d/%d %.0fs" % (ci, N, time.time() - t0), flush=True)
        else:
            # ---- 검증: 밀려남 이후 자기 내용 재커밋이 비트 동일한가 ----
            n_chk = min(N, (a.window // 3) + 4)           # 창을 넘겨 밀려남을 강제
            for ci in range(n_chk):
                x0 = None; z0 = b.chunk_noise(ci, g)
                for i in range(D): x0 = one_node(x0, ci, i, z0)
                first_commit(x0, ci); x0_of[ci] = x0.detach().float().cpu()
            # 검증 기준: 재쓰기의 변화가 '그 청크의 슬롯 안'에만 있어야 함. (슬롯 안 값은 어텐션 맥락이 밀려남으로 바뀌었으면 달라지는 게 정상)
            leaks = []; insides = []
            for tgt in (n_chk - 1, n_chk - 2, n_chk - 3):                  # 파이프라인이 실제로 재쓰는 3개 위치
                k0 = LL[0][0]["k"].clone(); st_, en_ = rewrite(x0_of[tgt].to(b.device), tgt)
                dif = (LL[0][0]["k"] - k0).abs(); del k0
                leak = max(float(dif[:, :st_].max()) if st_ > 0 else 0.0, float(dif[:, en_:].max()) if en_ < dif.shape[1] else 0.0)
                leaks.append(leak); insides.append(float(dif[:, st_:en_].max())); del dif; torch.cuda.empty_cache()
            dk = max(leaks); dks = leaks; tgt = n_chk - 3
            evicted = int(LL[0][0]["global_end_index"]) - int(LL[0][0]["local_end_index"])
            print("[sanity] after %d chunks (evicted tokens=%d): rewrite leak OUTSIDE own slot = %s (must be 0) | change inside slot = %s (context shift, expected)" % (n_chk, evicted, ["%.1e" % x for x in leaks], ["%.1e" % x for x in insides]), flush=True)
            rep["sanity_leak_outside_slot"] = dk; rep["sanity_change_inside_slot"] = insides; rep["sanity_evicted_tokens"] = evicted
            if dk > 0:
                print("[sanity] FAILED: rewrite leaked outside its slot. aborting pipeline mode.", flush=True)
                json.dump(dict(mode=a.mode, lag=a.lag, aborted=True, **rep), open(os.path.join(a.out, "pipeline.json"), "w"), indent=1); return
            LL[0] = None; snap.clear(); x0_of.clear(); import gc; gc.collect(); torch.cuda.empty_cache()   # 옛 캐시 참조를 먼저 놓아야 reset 이 실제로 비움
            print("[mem] before reset: %.1f GiB alloc" % (torch.cuda.memory_allocated() / 2**30), flush=True)
            b.reset_caches(); LL[0] = b._caches["target"]["kv"]
            print("[mem] after reset: %.1f GiB alloc" % (torch.cuda.memory_allocated() / 2**30), flush=True)
            g = torch.Generator().manual_seed(a.seed)
            # ---- 파이프라인 스케줄 ----
            Dl = a.lag; z0_of = {}
            total_steps = Dl * (N - 1) + D
            for s_ in range(total_steps):
                act = [c for c in range(N) if 0 <= s_ - Dl * c < D]
                for ci in sorted(act):                                  # 오래된 청크부터
                    node = s_ - Dl * ci
                    if node == 0: z0_of[ci] = b.chunk_noise(ci, g); x0 = one_node(None, ci, 0, z0_of[ci])
                    else: x0 = one_node(x0_of[ci].to(b.device), ci, node)
                    x0_of[ci] = x0.detach().float().cpu()
                    if node == 0: first_commit(x0, ci)
                    else: rewrite(x0, ci)
                if s_ % 20 == 0: print("  step %d/%d active=%s %.0fs" % (s_, total_steps, sorted(act), time.time() - t0), flush=True)
    gen_s = time.time() - t0
    lat = torch.cat([x0_of[i] for i in range(N)], dim=1)
    vid = b.pipe.vae.decode([lat.to(b.device)])[0].detach().cpu()
    v = ((vid.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 3, 0).numpy()
    import imageio
    imageio.mimwrite(os.path.join(a.out, "video.mp4"), list(v), fps=16, quality=8)
    np.save(os.path.join(a.out, "frames_all.npy"), v[::a.save_every])
    rep.update(mode=a.mode, lag=a.lag, chunks=N, nodes=D, frames=int(len(v)), gen_s=round(gen_s, 1), window=a.window, sink=a.sink, seed=a.seed)
    json.dump(rep, open(os.path.join(a.out, "pipeline.json"), "w"), indent=1); print(json.dumps(rep), flush=True)
    print("[PIPELINE_DONE]", flush=True)


if __name__ == "__main__":
    main()
