"""Fast(v2) 의 중간 상태 덤프.
--phase prefix : 청크 0..2 를 생성하며 마지막 노드(sigma=0.625)에 들어가는 x_t 와 그 직전 노드(0.833)의 x_t,
                 그리고 표시용 x0 를 저장. 핸드오프 실험(E-A)의 입력.
--phase stream : 청크 0..N-1 전체를 생성하며 매 청크의 x_t(node3) 와 x0 를 저장. 검사기 실험(E-B)의 입력.
"""
import argparse, os, json, time, numpy as np, torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--action-root", required=True)
    ap.add_argument("--prompt-file", required=True); ap.add_argument("--scene", default="05")
    ap.add_argument("--frames", type=int, default=161); ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--phase", choices=["prefix", "stream"], required=True)
    ap.add_argument("--window", type=int, default=18); ap.add_argument("--sink", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    from .binding_v2 import V2Binding
    b = V2Binding(a.ckpt, target_mode="fast_bf16", quant=None, sink=a.sink, local_attn=a.window,
                  swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
    prompt = open(a.prompt_file).read().strip()
    cond = b.prepare(os.path.join("examples", a.scene, "image.jpg"), prompt, a.action_root, frame_num=a.frames)
    N = cond["num_chunks"]; sig = b.node_sigmas(); CS = 3
    print("[dump] phase=%s chunks=%d sigmas=%s" % (a.phase, N, [round(float(s), 3) for s in sig]), flush=True)
    n_gen = 3 if a.phase == "prefix" else N
    g = torch.Generator().manual_seed(a.seed); b.reset_caches()
    XT3, XT2, X0 = [], [], []; t0 = time.time()
    with torch.no_grad():
        for ci in range(n_gen):
            z = b.chunk_noise(ci, g); x0 = None; xt_last = None; xt_prev = None
            for i in range(len(sig)):
                if i == len(sig) - 1: xt_last = z.detach().float().cpu()        # sigma = 0.625 에서의 x_t
                if i == len(sig) - 2: xt_prev = z.detach().float().cpu()        # sigma = 0.833 에서의 x_t
                x0 = b.target_x0(z, i, ci)
                if i < len(sig) - 1:
                    s = float(sig[i + 1]); z = (1 - s) * x0 + s * b.randn_like(x0, g)
            b.commit(x0, ci)
            XT3.append(xt_last); XT2.append(xt_prev); X0.append(x0.detach().float().cpu())
            if ci % 10 == 0: print("  chunk %d/%d  %.0fs" % (ci, n_gen, time.time() - t0), flush=True)
    xt3 = torch.cat(XT3, dim=1); xt2 = torch.cat(XT2, dim=1); x0 = torch.cat(X0, dim=1)
    torch.save(xt3, os.path.join(a.out, "xt_node3.pt")); torch.save(xt2, os.path.join(a.out, "xt_node2.pt"))
    torch.save(x0, os.path.join(a.out, "x0.pt"))
    print("[dump] saved xt_node3 %s  x0 %s" % (list(xt3.shape), list(x0.shape)), flush=True)
    if a.phase == "prefix":
        vid = b.pipe.vae.decode([x0.to(b.device)])[0].detach().cpu()
        v = ((vid.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 3, 0).numpy()
        import imageio
        imageio.imwrite(os.path.join(a.out, "first.jpg"), v[0])
        imageio.mimwrite(os.path.join(a.out, "video.mp4"), list(v), fps=16, quality=8)
        np.save(os.path.join(a.out, "frames_all.npy"), v)
        print("[dump] prefix video %d frames" % len(v), flush=True)
    json.dump(dict(phase=a.phase, chunks=n_gen, sigmas=[float(s) for s in sig], seed=a.seed, gen_s=round(time.time() - t0, 1)),
              open(os.path.join(a.out, "dump.json"), "w"), indent=1)
    print("[FAST_DUMP_DONE]", flush=True)


if __name__ == "__main__":
    main()
