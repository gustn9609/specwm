"""3-way video comparison on lingbot-world-v2 (same seed, same actions):
   target-only (reference)  |  draft-only (degraded)  |  speculative (exact)
Saves mp4 for each + a side-by-side grid + per-mode metrics.
"""
import argparse, json, os, time, torch
from .core import SpecStats
from .l2_spec_chunk import ChunkSpec


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--action", required=True)
    ap.add_argument("--prompt", default="A serene lakeside scene with a lone tree.")
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--chunk-size", type=int, default=3)
    ap.add_argument("--quant", default="int8")
    ap.add_argument("--local-attn", type=int, default=12)
    ap.add_argument("--sink", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="specwm_out/compare")
    return ap.parse_args()


@torch.no_grad()
def plain_chunks(b, n_chunks, which, seed):
    """Standard 4-node renoise chain using ONE model (target or draft)."""
    sig = b.node_sigmas(); n = len(sig)
    gen = torch.Generator().manual_seed(seed)
    outs = []
    for c in range(n_chunks):
        z = b.chunk_noise(c, gen)
        x0 = None
        for i in range(n):
            x0 = (b.target_x0(z, i, c) if which == "target"
                  else b.draft_x0(z, i, c))
            if i < n - 1:
                s = float(sig[i + 1])
                z = (1 - s) * x0 + s * b.randn_like(x0, gen)
        b.commit(x0, c, which=which)
        outs.append(x0)
    return outs


def main():
    a = get_args(); os.makedirs(a.out, exist_ok=True)
    from .binding_v2 import V2Binding
    b = V2Binding(a.ckpt, target_mode="fast_bf16", quant=a.quant,
                  sink=a.sink, local_attn=a.local_attn)
    cond = b.prepare(a.image, a.prompt, a.action, frame_num=a.frames,
                     chunk_size=a.chunk_size)
    N = cond["num_chunks"]
    report = {"num_chunks": N, "quant": a.quant, "seed": a.seed, "modes": {}}
    lat = {}

    for mode in ["target", "draft", "spec"]:
        b.reset_caches()
        t0 = time.time()
        if mode == "spec":
            st = SpecStats()
            spec = ChunkSpec(b, b.node_sigmas(), st, target_substeps=1)
            gen = torch.Generator().manual_seed(a.seed)
            chunks = [spec.run_chunk(c, cond, gen) for c in range(N)]
            report["modes"]["spec"] = st.summary()
        else:
            chunks = plain_chunks(b, N, mode, a.seed)
            report["modes"][mode] = {}
        lat[mode] = time.time() - t0
        report["modes"][mode]["wallclock"] = lat[mode]
        lat_t = torch.cat(chunks, dim=1)
        torch.save(lat_t.cpu(), os.path.join(a.out, f"latent_{mode}.pt"))
        vid = b.pipe.vae.decode([lat_t.to(b.device)])[0]
        import imageio
        v = ((vid.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 3, 0).cpu().numpy()
        imageio.mimwrite(os.path.join(a.out, f"{mode}.mp4"), list(v), fps=16,
                         quality=8)
        print(f"[{mode}] {lat[mode]:.1f}s -> {mode}.mp4")

    # side-by-side grid (target | draft | spec)
    try:
        import imageio, numpy as np
        vids = [imageio.mimread(os.path.join(a.out, f"{m}.mp4"), memtest=False)
                for m in ["target", "draft", "spec"]]
        T = min(len(v) for v in vids)
        grid = [np.concatenate([v[t] for v in vids], axis=1) for t in range(T)]
        imageio.mimwrite(os.path.join(a.out, "sidebyside.mp4"), grid, fps=16, quality=8)
        print("saved sidebyside.mp4 (target | draft | spec)")
    except Exception as e:
        print("grid failed:", e)

    # latent-space distance of each mode vs target (does spec match target?)
    import itertools
    L = {m: torch.load(os.path.join(a.out, f"latent_{m}.pt")) for m in ["target", "draft", "spec"]}
    report["latent_l2_vs_target"] = {
        m: float((L[m] - L["target"]).norm()) for m in ["draft", "spec"]}
    with open(os.path.join(a.out, "compare.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
