"""패치 단위 투기적 수락 (CSD / LISA 아이디어를 우리 설정으로).

원본(l2_spec_chunk): 청크 전체 하나의 거리 d=||m_q-m_p|| 로 수락. d 는 차원 수 289,536 의 제곱근으로 커져 alpha 가 0.71/0.29/0.00 으로 붕괴.
CSD/LISA: 토큰(패치) 단위로 수락. 같은 노드에서 패치별 d_u 는 훨씬 작으므로 대부분 수락 가능.
여기서 하는 것:
  * 같은 노이즈(공유 eps)에서 draft x0 와 target x0 를 노드마다 구함 (원본과 동일)
  * 단위별 거리 d_u 와 정확 수락률 alpha_u = 2*Phi(-d_u/(2s)) 를 기록 (unit = chunk / frame / patch)
  * LISA 식 부드러운 수락: d_u/||m_p_u|| < tau 인 단위는 draft, 아니면 target 으로 채워 하이브리드 x0 구성
  * 하이브리드를 다음 노드로 재노이즈해 이어가고, 마지막 노드의 하이브리드를 commit.
정확 샘플링이 아님(SDVG 처럼 draft 쪽으로 분포 편향). 대신 "얼마나 많은 패치를 draft 에서 살릴 수 있고, 그 하이브리드가 coherent 한가" 를 잰다.
"""
import argparse, os, json, time, math, numpy as np, torch


def phi(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def unit_view(x, unit, P=8):
    """x: [C,F,H,W] -> [U, D] 단위별 벡터. unit=chunk|frame|patch. 패치는 PxP 잠재 격자."""
    C, F, H, W = x.shape
    if unit == "chunk": return x.reshape(1, -1)
    if unit == "frame": return x.permute(1, 0, 2, 3).reshape(F, -1)
    Hp, Wp = H // P, W // P
    xx = x[:, :, :Hp * P, :Wp * P].reshape(C, F, Hp, P, Wp, P).permute(1, 2, 4, 0, 3, 5)   # [F,Hp,Wp,C,P,P]
    return xx.reshape(F * Hp * Wp, -1)


def stitch(x_d, x_t, keep, unit, P=8):
    """keep: [U] bool (True = draft). 하이브리드 x0."""
    C, F, H, W = x_d.shape; out = x_t.clone()
    if unit == "chunk": return x_d if bool(keep[0]) else x_t
    if unit == "frame":
        for f in range(F):
            if bool(keep[f]): out[:, f] = x_d[:, f]
        return out
    Hp, Wp = H // P, W // P; k = keep.reshape(F, Hp, Wp)
    for f in range(F):
        for i in range(Hp):
            for j in range(Wp):
                if bool(k[f, i, j]): out[:, f, i * P:(i + 1) * P, j * P:(j + 1) * P] = x_d[:, f, i * P:(i + 1) * P, j * P:(j + 1) * P]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--target-ckpt", default=None)
    ap.add_argument("--target-mode", default="fast_bf16"); ap.add_argument("--quant", default="int8-g128", help="draft 양자화 (원본 alpha 측정과 동일: draft=Fast int8, target=Fast bf16)")
    ap.add_argument("--action-root", required=True); ap.add_argument("--prompt-file", required=True); ap.add_argument("--scene", default="05")
    ap.add_argument("--frames", type=int, default=161); ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--unit", choices=["chunk", "frame", "patch"], default="patch"); ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.5, help="상대 거리 임계. 1e9=전부 draft, 0=전부 target")
    ap.add_argument("--target-substeps", type=int, default=1)
    ap.add_argument("--kappa", type=float, default=0.0, help=">0 이면 LISA 확신도 항: target x0 가 인접 sigma 에서 상대변화 kappa 이상인 패치는 draft 불신")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    from .binding_v2 import V2Binding
    from .l2_spec_chunk import substep_x0
    b = V2Binding(a.ckpt, target_mode=a.target_mode, target_ckpt=a.target_ckpt, quant=a.quant, sink=6, local_attn=18,
                  swap_models=False, build_draft=True, share_kv=True, timesteps_index=(0, 250, 500, 750))
    prompt = open(a.prompt_file).read().strip()
    cond = b.prepare(os.path.join("examples", a.scene, "image.jpg"), prompt, a.action_root, frame_num=a.frames)
    N = cond["num_chunks"]; sig = b.node_sigmas(); n = len(sig)
    gen = torch.Generator().manual_seed(a.seed); b.reset_caches()
    stats = {u: {i: [] for i in range(n - 1)} for u in ["chunk", "frame", "patch"]}   # 정확 alpha (분석용, 세 단위 모두)
    keep_frac = {i: [] for i in range(n - 1)}; x0s = []; t0 = time.time()
    with torch.no_grad():
        for ci in range(N):
            if hasattr(b, "reset_chunk_slot"): b.reset_chunk_slot(ci)
            z = b.chunk_noise(ci, gen); x_h = None
            for i in range(n):
                x_d = b.draft_x0(z, i, ci)
                s_next = float(sig[i + 1]) if i < n - 1 else 0.0
                x_t = substep_x0(lambda zz, ss: b.target_flow(zz, ss, ci), z, float(sig[i]), s_next, a.target_substeps)
                if i < n - 1:
                    s = s_next; m_p, m_q = (1 - s) * x_d, (1 - s) * x_t
                    # 세 단위의 정확 alpha 를 전부 기록 (어느 단위가 살아남는지 보려고)
                    for u in ["chunk", "frame", "patch"]:
                        dp = unit_view(m_p, u, a.patch); dq = unit_view(m_q, u, a.patch)
                        d = (dq - dp).norm(dim=1); al = torch.tensor([2 * phi(-float(x) / (2 * s)) for x in d])
                        stats[u][i].append(float(al.mean()))
                    # 부드러운 수락 (선택한 단위)
                    dp = unit_view(m_p, a.unit, a.patch); dq = unit_view(m_q, a.unit, a.patch)
                    rel = (dq - dp).norm(dim=1) / (dp.norm(dim=1) + 1e-6)
                    keep = rel < a.tau
                    if a.kappa > 0:                                            # LISA 확신도: target 이 인접 sigma 에서 안정한가
                        s2 = min(0.999, float(sig[i]) + 0.05)
                        x_t2 = substep_x0(lambda zz, ss: b.target_flow(zz, ss, ci), z, s2, s_next, a.target_substeps)
                        dq2 = unit_view((1 - s) * x_t2, a.unit, a.patch)
                        conf = (dq2 - dq).norm(dim=1) / (dq.norm(dim=1) + 1e-6)
                        keep = keep & (conf < a.kappa)
                    keep_frac[i].append(float(keep.float().mean()))
                    x_h = stitch(x_d, x_t, keep, a.unit, a.patch)
                    z = (1 - s) * x_h + s * b.randn_like(x_h, gen)
                else:
                    # 마지막 노드: 같은 규칙으로 하이브리드 (s=0 이므로 상대 거리만)
                    dp = unit_view(x_d, a.unit, a.patch); dq = unit_view(x_t, a.unit, a.patch)
                    rel = (dq - dp).norm(dim=1) / (dp.norm(dim=1) + 1e-6); keep = rel < a.tau
                    x_h = stitch(x_d, x_t, keep, a.unit, a.patch)
            b.commit(x_h, ci); x0s.append(x_h.detach().float().cpu())
            if ci % 4 == 0:
                print("  chunk %d/%d  exact alpha chunk/frame/patch @node1 = %.3f/%.3f/%.3f  keep@node1 %.2f  %.0fs" % (
                    ci, N, np.mean(stats["chunk"][1]), np.mean(stats["frame"][1]), np.mean(stats["patch"][1]), np.mean(keep_frac[1]), time.time() - t0), flush=True)
    lat = torch.cat(x0s, dim=1); vid = b.pipe.vae.decode([lat.to(b.device)])[0].detach().cpu()
    v = ((vid.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 3, 0).numpy()
    import imageio
    imageio.mimwrite(os.path.join(a.out, "video.mp4"), list(v), fps=16, quality=8); np.save(os.path.join(a.out, "frames_all.npy"), v[::4])
    rep = dict(unit=a.unit, tau=a.tau, kappa=a.kappa, patch=a.patch, seed=a.seed, target_mode=a.target_mode, draft_quant=a.quant, chunks=N, gen_s=round(time.time() - t0, 1),
               exact_alpha={u: {str(i): round(float(np.mean(v_)), 4) for i, v_ in d_.items()} for u, d_ in stats.items()},
               keep_frac={str(i): round(float(np.mean(v_)), 4) for i, v_ in keep_frac.items()})
    json.dump(rep, open(os.path.join(a.out, "patch_spec.json"), "w"), indent=1); print(json.dumps(rep), flush=True)
    print("[PATCH_SPEC_DONE]", flush=True)


if __name__ == "__main__":
    main()
