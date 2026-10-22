"""PoseGram (비파라메트릭, 학습 없음) — pose 주소 기반 외부 기억을 Fast(causal) 생성에 붙인다.

주소 (§3.1): 잠재프레임 f 의 카메라 pose 에서 위치를 rho0*extent 격자로, 시선방향을 theta0 도 격자(방위각/고도)로 양자화한 셀 키.
테이블 (§3.2): cell -> [(latent_frame_index t, latent[16,H,W] (host 또는 GPU), conf)], write policy (§3.3): always / first / gated(latent flicker < 이동 중앙값)
읽기·융합 (§3.4, 학습 없음): 청크 k 의 각 잠재프레임 주소로 테이블 조회 (t <= f - min_gap 인 항목만) -> 매칭된 과거 청크(최대 sink/3 개) 의 latent 를
   sink 슬롯에 원래 시간 위치(RoPE)로 commit = 모델의 attention 이 '그때 그 자리'를 다시 볼 수 있게 함 (KV 주입). 계획서의 Plücker scale&shift 주입은
   latent 를 카메라 임베딩 공간으로 옮겨야 해서 학습 없이는 불가 -> KV 주입으로 대체(문서화).
oracle 변형 (E0 강화): 정확 탐색(해싱 없음) + blend_all(전 노드 x0 블렌딩) / init(σ1 에서 과거 latent 로 초기화) / kv(KV 주입)
random-address baseline: 테이블에서 무작위 셀을 읽음.
시스템 (§3.5): --table-device cpu 면 latent 를 호스트에 두고 읽을 때 전송; --prefetch 면 다음 청크 주소를 미리 계산해 비동기 복사. 전송 바이트/시간 기록.
출력: <out>/<arm>/{frames_all.npy, video.mp4, latent.pt, arm.json}
"""
import argparse, json, os, time, gc, math, random
import numpy as np, torch
CS = 3


def quant_dir(d, theta0):
    az = math.degrees(math.atan2(d[1], d[0])); el = math.degrees(math.asin(max(-1.0, min(1.0, d[2]))))
    return (int(math.floor(az / theta0)), int(math.floor(el / theta0)))


class PoseTable:
    def __init__(self, poses, extent, rho0, theta0, device, first_write, gated):
        self.pos = poses[:, :3, 3]; self.fwd = poses[:, :3, :3][:, :, 2]; self.extent = extent
        self.rho0, self.theta0, self.device = rho0, theta0, device; self.first_write, self.gated = first_write, gated
        self.table = {}; self.flick_hist = []; self.bytes_up = 0; self.n_write = 0; self.n_skip = 0

    def addr(self, px_frame):
        p = self.pos[px_frame]; d = self.fwd[px_frame]
        cell_p = tuple(int(math.floor(c / (self.rho0 * self.extent))) for c in p)
        return cell_p + quant_dir(d, self.theta0)

    def write(self, f, px_frame, lat_f, chunk_flicker):
        if self.gated:
            med = float(np.median(self.flick_hist)) if len(self.flick_hist) >= 3 else float("inf")
            self.flick_hist.append(chunk_flicker)
            if chunk_flicker > med: self.n_skip += 1; return
        key = self.addr(px_frame)
        if key in self.table and self.first_write: self.n_skip += 1; return
        v = lat_f.detach().to(self.device, non_blocking=True)
        self.table[key] = (f, v); self.n_write += 1
        if self.device == "cpu": self.bytes_up += v.numel() * v.element_size()

    def read(self, px_frame, f, min_gap, rng=None, lo=0):
        if rng is not None:
            keys = [k for k, (t, _) in self.table.items() if lo <= t <= f - min_gap]
            if not keys: return None
            return self.table[keys[rng.integers(len(keys))]]
        e = self.table.get(self.addr(px_frame))
        if e is None or e[0] > f - min_gap or e[0] < lo: return None
        return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--action-root", required=True); ap.add_argument("--prompt-file", required=True); ap.add_argument("--scene", default="05")
    ap.add_argument("--frames", type=int, default=873); ap.add_argument("--window", type=int, default=18); ap.add_argument("--sink", type=int, default=18); ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--mode", required=True, choices=["ctrl", "oracle_blend_all", "oracle_init", "oracle_kv", "posegram", "posegram_rand", "posegram_blend", "posegram_rand_blend", "anchor_blend", "recent_blend"])
    ap.add_argument("--alpha", type=float, default=0.3); ap.add_argument("--init-node", type=int, default=1)
    ap.add_argument("--theta0", type=float, default=10.0); ap.add_argument("--rho0", type=float, default=0.05)
    ap.add_argument("--write", default="always", choices=["always", "first", "gated"]); ap.add_argument("--min-gap", type=int, default=18)
    ap.add_argument("--pos-tol", type=float, default=0.05); ap.add_argument("--ang-tol", type=float, default=10.0)
    ap.add_argument("--table-device", default="cuda", choices=["cuda", "cpu"]); ap.add_argument("--prefetch", action="store_true")
    ap.add_argument("--evicted-only", action="store_true", help="매칭 후보에서 sink(첫 sink 잠재프레임) 제외 -> 실제로 attention 이 못 보는 과거만")
    ap.add_argument("--name", default=None); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    name = a.name or (a.mode if a.mode == "ctrl" else "%s_a%g_th%g_rho%g_w%s_%s%s%s_s%d" % (a.mode, a.alpha, a.theta0, a.rho0, a.write, a.table_device, "_pf" if a.prefetch else "", "_ev" if a.evicted_only else "", a.seed))
    if a.mode == "ctrl": name = "ctrl_s%d" % a.seed
    d = os.path.join(a.out, name); os.makedirs(d, exist_ok=True)
    if os.path.exists(os.path.join(d, "arm.json")): print("[%s] 이미 있음" % name); return
    from .binding_v2 import V2Binding
    prompt = open(a.prompt_file).read().strip(); t0 = time.time()
    b = V2Binding(a.ckpt, target_mode="fast_bf16", quant=None, sink=a.sink, local_attn=a.window, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
    cond = b.prepare(os.path.join("examples", a.scene, "image.jpg"), prompt, a.action_root, frame_num=a.frames)
    N = cond["num_chunks"]; sig = [float(s) for s in b.node_sigmas()]; L = N * CS; fs = b._frame_seqlen; nslot = a.sink // CS
    P = np.load(os.path.join(a.action_root, "poses.npy")); pos = P[:, :3, 3]; fwd = P[:, :3, :3][:, :, 2]; ext = float(np.linalg.norm(pos.max(0) - pos.min(0)))
    lf2px = lambda f: min(4 * f, len(P) - 1)
    print("[load] %.0fs N=%d L=%d window=%d sink=%d(%d slots) mode=%s" % (time.time() - t0, N, L, a.window, a.sink, nslot, a.mode), flush=True)
    lat = torch.zeros(16, L, b._chunk_shape[2], b._chunk_shape[3])
    table = PoseTable(P, ext, a.rho0, a.theta0, a.table_device, a.write == "first", a.write == "gated") if a.mode.startswith("posegram") else None
    rng = np.random.default_rng(a.seed) if a.mode in ("posegram_rand", "posegram_rand_blend") else None
    stats = dict(matched=0, injected_chunks=0, inject_events=0, bytes_down=0, read_s=0.0, inject_s=0.0)

    lo = a.sink if a.evicted_only else 0
    def exact_match(f):
        cand = np.arange(lo, f - a.min_gap + 1)
        if len(cand) == 0: return None
        pi, fi = pos[lf2px(f)], fwd[lf2px(f)]; pc = pos[[lf2px(c) for c in cand]]; fc = fwd[[lf2px(c) for c in cand]]
        dd = np.linalg.norm(pc - pi, axis=1) / max(ext, 1e-6); ang = np.degrees(np.arccos(np.clip((fc * fi).sum(1), -1, 1)))
        j = int(np.argmin(dd + 0.01 * ang))
        return int(cand[j]) if (dd[j] < a.pos_tol and ang[j] < a.ang_tol) else None

    def reads_for_chunk(ci):
        """청크 ci 의 각 잠재프레임에 대한 (f, 과거 잠재프레임 t, latent) 목록"""
        out = []
        for k in range(CS):
            f = ci * CS + k
            if a.mode.startswith("oracle"):
                j = exact_match(f)
                if j is not None: out.append((f, j, lat[:, j]))
            elif a.mode == "anchor_blend":                       # pose 무관: 항상 첫 청크의 같은 슬롯 프레임 (순수 앵커 대조)
                if f >= a.min_gap: out.append((f, k, lat[:, k]))
            elif a.mode == "recent_blend":                       # pose 무관: window 바로 밖의 가장 최근 과거 (recency 대조)
                if f >= a.min_gap: out.append((f, f - a.min_gap, lat[:, f - a.min_gap]))
            else:
                e = table.read(lf2px(f), f, a.min_gap, rng, lo=lo)
                if e is not None: out.append((f, e[0], e[1]))
        return out

    def inject_kv(reads):
        """매칭된 과거 청크(중복 제거, 최대 nslot) 의 latent 를 sink 슬롯에 원래 시간 위치로 commit"""
        chunks = []
        for _, t, _ in reads:
            cj = t // CS
            if cj not in chunks: chunks.append(cj)
        chunks = sorted(chunks)[-nslot:]
        if not chunks: return 0
        layers = b._caches["target"]["kv"]; saved = [(c["global_end_index"].clone(), c["local_end_index"].clone()) for c in layers]; cf = b._committed_frames
        t1 = time.time()
        for i, cj in enumerate(chunks):
            for c in layers: c["global_end_index"].fill_(cj * CS * fs); c["local_end_index"].fill_(i * CS * fs)
            x = lat[:, cj * CS:(cj + 1) * CS].to(b.device, non_blocking=True).float(); stats["bytes_down"] += x.numel() * 4 if a.table_device == "cpu" else 0
            b.commit(x, cj)
        for c, (ge, le) in zip(layers, saved): c["global_end_index"].copy_(ge); c["local_end_index"].copy_(le)
        b._committed_frames = cf; torch.cuda.synchronize(); stats["inject_s"] += time.time() - t1
        return len(chunks)

    b.reset_caches(); torch.cuda.empty_cache(); g = torch.Generator().manual_seed(a.seed); t_gen = time.time(); last_inj = None; per_chunk = []
    pre = reads_for_chunk(0) if a.mode != "ctrl" else []
    for ci in range(N):
        t1 = time.time(); reads = pre if a.mode != "ctrl" else []
        if a.prefetch and a.mode != "ctrl" and ci + 1 < N:
            nxt = reads_for_chunk(ci + 1)                              # 주소는 입력이므로 다음 청크 읽기를 미리 준비 (host->device 비동기)
            for _, _, v in nxt: v.to(b.device, non_blocking=True)
        stats["read_s"] += time.time() - t1; stats["matched"] += len(reads)
        # KV 주입 (oracle_kv / posegram): 읽기 집합이 바뀌었을 때만
        if reads and a.mode in ("oracle_kv", "posegram", "posegram_rand"):
            key = tuple(sorted({t // CS for _, t, _ in reads}))
            if key != last_inj: n = inject_kv(reads); last_inj = key; stats["injected_chunks"] += n; stats["inject_events"] += 1
        z = b.chunk_noise(ci, g); x0 = None; start = 0
        if reads and a.mode == "oracle_init":                          # σ[init_node] 에서 과거 latent 로 초기화 (프레임 단위)
            x_init = torch.zeros_like(z); m = torch.zeros(CS, dtype=torch.bool)
            for f, t, v in reads: x_init[:, f - ci * CS] = v.to(z.device, z.dtype); m[f - ci * CS] = True
            s0 = sig[a.init_node]; z_mix = (1 - s0) * x_init + s0 * z
            for k in range(CS):
                if m[k]: z[:, k] = z_mix[:, k]
            # 난수 정렬: 노드 0..init_node-1 의 randn 소비를 동일하게
            for _ in range(a.init_node): b.randn_like(z, g)
            start = a.init_node
        for i in range(start, len(sig)):
            x0 = b.target_x0(z, i, ci)
            if reads and a.mode in ("oracle_blend_all", "posegram_blend", "posegram_rand_blend", "anchor_blend", "recent_blend"):   # 전 노드 x0 블렌딩 (테이블 읽기 기반이면 E1 융합)
                for f, t, v in reads: x0[:, f - ci * CS] = (1 - a.alpha) * x0[:, f - ci * CS] + a.alpha * v.to(x0.device, x0.dtype)
            if i < len(sig) - 1: z = (1 - sig[i + 1]) * x0 + sig[i + 1] * b.randn_like(x0, g)
        b.commit(x0, ci); xc = x0.detach().float().cpu(); lat[:, ci * CS:(ci + 1) * CS] = xc
        if table is not None:
            fl = float((xc[:, 1:] - xc[:, :-1]).abs().mean())
            for k in range(CS): table.write(ci * CS + k, lf2px(ci * CS + k), xc[:, k], fl)
        per_chunk.append(dict(chunk=ci, reads=len(reads), s=round(time.time() - t1, 2)))
        if a.mode != "ctrl" and ci + 1 < N: pre = reads_for_chunk(ci + 1)
        if ci % 12 == 0: print("   chunk %d/%d reads %d inj %d  %.0fs" % (ci, N, len(reads), stats["injected_chunks"], time.time() - t_gen), flush=True)
    gen_s = time.time() - t_gen
    with torch.no_grad(): img = b.pipe.vae.decode([lat.to(b.device)])[0].detach().cpu()
    v = ((img.clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy(); np.save(os.path.join(d, "frames_all.npy"), (v * 255).astype("uint8")[::4]); torch.save(lat, os.path.join(d, "latent.pt"))
    import imageio; imageio.mimwrite(os.path.join(d, "video.mp4"), list((v * 255).astype("uint8")), fps=16, quality=8)
    r = dict(arm=name, **{k: v_ for k, v_ in vars(a).items() if k not in ("ckpt", "out", "frames")}, frames_req=a.frames, frames=int(v.shape[0]), gen_s=round(gen_s, 1), latent_frames=L,
             flicker=round(float((img[:, 1:] - img[:, :-1]).abs().mean()), 5), stats={k: (round(v_, 2) if isinstance(v_, float) else v_) for k, v_ in stats.items()},
             table=(dict(cells=len(table.table), writes=table.n_write, skipped=table.n_skip, bytes_up=table.bytes_up) if table else None), per_chunk=per_chunk)
    json.dump(r, open(os.path.join(d, "arm.json"), "w"), indent=1)
    print("[%s] gen %.0fs matched %d injected %d (events %d, inject %.0fs, read %.1fs) table %s flicker %.4f" % (name, gen_s, stats["matched"], stats["injected_chunks"], stats["inject_events"], stats["inject_s"], stats["read_s"], r["table"], r["flicker"]), flush=True)
    print("[POSEGRAM_DONE]", flush=True)


if __name__ == "__main__":
    main()
