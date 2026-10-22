"""B: draft-quality -> acceptance scaling law.
Protocol (apples-to-apples): run the bf16 TARGET ancestral chain once, record
states z_i and target x0_q(z_i). Then for each quantization spec, load a fresh
draft and evaluate x0_p(z_i) at the SAME states -> d_i, alpha_i(tile).
This isolates draft quality as an intrinsic property (no trajectory drift).
Also stores per-frame alpha for axis C.
"""
import argparse, copy, json, math, os, torch
from .quantize import fake_quantize_family


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--action", default=None)
    ap.add_argument("--prompt", default="A serene lakeside scene with a lone tree.")
    ap.add_argument("--nodes", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--specs", nargs="*",
                    default=["int4", "int4-g128", "int8", "int8-g128", "nf4", "nf4-g64"])
    ap.add_argument("--out", default="specwm_out/draft_sweep")
    return ap.parse_args()


def main():
    a = get_args(); os.makedirs(a.out, exist_ok=True)
    from .binding_v1 import V1Binding
    b = V1Binding(a.ckpt, t5_cpu=True)
    cond = b.prepare(a.image, a.prompt, a.action, frame_num=a.frames)
    sigmas, timesteps = b.node_grid(a.nodes)
    C, F, H, W = cond["shape"]

    # pristine CPU copies for re-quantization
    sd = {k: {n: p.detach().cpu().clone() for n, p in m.state_dict().items()}
          for k, m in [("low", b.pipe.low_noise_model), ("high", b.pipe.high_noise_model)]}

    def fresh(spec):
        for k, m in [("low", b.pipe.low_noise_model), ("high", b.pipe.high_noise_model)]:
            m.load_state_dict(sd[k])
        models = {"low": b.pipe.low_noise_model.to(b.device),
                  "high": b.pipe.high_noise_model.to(b.device)}
        if spec != "none":
            for k in models:
                nq = fake_quantize_family(models[k], spec)
            print(f"[quantize] {spec}: {nq} Linear layers fake-quantized per expert")
        return models

    # ---- reference pass: bf16 target trajectory + x0_q ----
    tgt = fresh("none")
    x0_tgt_fn = b.make_x0_fn(tgt, cond, sigmas, timesteps)
    traj = []
    for sdi in range(a.seeds):
        g = torch.Generator().manual_seed(555 + sdi)
        z = torch.randn(cond["shape"], generator=g).to(b.device)
        rec = []
        for i in range(a.nodes):
            x0q = x0_tgt_fn(z, i)
            rec.append((z.cpu().clone(), x0q.cpu().clone()))
            if i < a.nodes - 1:
                s = float(sigmas[i + 1])
                z = (1 - s) * x0q + s * torch.randn(cond["shape"], generator=g).to(b.device)
        traj.append(rec)
        print(f"[ref] seed {sdi} done")

    def alphas(diff, s):
        d_tot = float(diff.norm())
        d_frame = diff.pow(2).sum(dim=(0, 2, 3)).sqrt()
        hh, ww = H // 2, W // 2
        d_quad = diff.reshape(C, F, 2, hh, 2, ww).pow(2).sum(dim=(0, 3, 5)).sqrt()
        A = lambda d: 2 * 0.5 * (1 + math.erf((-float(d) / (2 * s)) / math.sqrt(2)))
        return dict(whole=A(d_tot),
                    frame=[A(x) for x in d_frame],
                    quad_mean=sum(A(x) for x in d_quad.flatten()) / d_quad.numel())

    results = {"sigmas": [float(s) for s in sigmas], "frames": F, "specs": {}}
    for spec in a.specs:
        drf = fresh(spec)
        x0_d_fn = b.make_x0_fn(drf, cond, sigmas, timesteps)
        per_node = []
        for sdi, rec in enumerate(traj):
            for i, (z, x0q) in enumerate(rec):
                if i >= a.nodes - 1: continue
                s = float(sigmas[i + 1])
                x0p = x0_d_fn(z.to(b.device), i)
                diff = ((1 - s) * (x0q.to(b.device) - x0p)).float().cpu()
                r = alphas(diff, s); r.update(node=i, seed=sdi, s=s)
                per_node.append(r)
        results["specs"][spec] = per_node
        mw = sum(r["whole"] for r in per_node) / len(per_node)
        mq = sum(r["quad_mean"] for r in per_node) / len(per_node)
        print(f"[{spec}] mean alpha: whole={mw:.4f} quad={mq:.4f}")
        with open(os.path.join(a.out, "draft_sweep.json"), "w") as f:
            json.dump(results, f)
    print("saved draft_sweep.json")


if __name__ == "__main__":
    main()
