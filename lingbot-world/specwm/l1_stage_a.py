"""L1 Stage A — two-pass acceptance study (memory-safe: ONE model resident).

Pass 1: DRAFT (quantized Base) runs the re-noise ancestral chain, recording all
        visited states z_i and its own x0 estimates. -> saved to disk
Pass 2: TARGET (bf16 Base) replays the recorded states (no sequential
        dependence -> pure batch), computes x0_q, then per-node:
        empirical accept prob (MC over seeds) + CLOSED-FORM 2*Phi(-d/2s).

Run (drop specwm/ into lingbot-world repo root):
  python -m specwm.l1_stage_a --ckpt ./lingbot-world-base-cam \
      --image examples/00/image.jpg --action examples/00 --pass 1
  python -m specwm.l1_stage_a ... --pass 2
"""
import argparse, json, os, time
import torch
from .core import shifted_sigmas, expected_accept  # noqa
from .quantize import quantize_draft


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--action", default=None)
    ap.add_argument("--prompt", default="A serene lakeside scene with a lone tree.")
    ap.add_argument("--pass", dest="stage", type=int, required=True, choices=[1, 2])
    ap.add_argument("--nodes", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--quant", default="int8", choices=["int8", "int4-ao", "none"])
    ap.add_argument("--out", default="specwm_out/l1_stage_a")
    ap.add_argument("--guide", type=float, default=5.0)
    return ap.parse_args()


def main():
    a = get_args()
    os.makedirs(a.out, exist_ok=True)
    from .binding_v1 import V1Binding
    b = V1Binding(a.ckpt, t5_cpu=True)
    cond = b.prepare(a.image, a.prompt, a.action, frame_num=a.frames)
    sigmas, timesteps = b.node_grid(a.nodes)
    gs = (a.guide, a.guide)

    if a.stage == 1:
        models = {'low': b.pipe.low_noise_model, 'high': b.pipe.high_noise_model}
        for k in models:
            models[k] = quantize_draft(models[k].to(b.device), a.quant)
        x0_fn = b.make_x0_fn(models, cond, sigmas, timesteps, gs)
        recs = []
        for sd in range(a.seeds):
            g = torch.Generator(device="cpu").manual_seed(1234 + sd)
            z = torch.randn(cond['shape'], generator=g).to(b.device)
            traj = {"z": [], "x0d": [], "sd": sd}
            t0 = time.time()
            for i in range(a.nodes):
                x0 = x0_fn(z, i)
                traj["z"].append(z.cpu().clone()); traj["x0d"].append(x0.cpu().clone())
                if i < a.nodes - 1:
                    s = float(sigmas[i + 1])
                    eps = torch.randn(cond['shape'], generator=g).to(b.device)
                    z = (1 - s) * x0 + s * eps
            print(f"[pass1] seed {sd} done in {time.time()-t0:.1f}s")
            recs.append(traj)
        torch.save({"recs": recs, "sigmas": sigmas, "timesteps": timesteps,
                    "meta": vars(a)}, os.path.join(a.out, "draft_traj.pt"))
        print("saved draft_traj.pt")
        return

    # ---- pass 2: target replay ----
    blob = torch.load(os.path.join(a.out, "draft_traj.pt"), map_location="cpu")
    sigmas, timesteps = blob["sigmas"], blob["timesteps"]
    models = {'low': b.pipe.low_noise_model.to(b.device),
              'high': b.pipe.high_noise_model.to(b.device)}
    x0_fn = b.make_x0_fn(models, cond, sigmas, timesteps, gs)
    per_node = {}
    for traj in blob["recs"]:
        for i, (z, x0d) in enumerate(zip(traj["z"], traj["x0d"])):
            if i >= len(sigmas) - 1:
                continue
            z = z.to(b.device); x0d = x0d.to(b.device)
            x0q = x0_fn(z, i)
            s = float(sigmas[i + 1])
            m_p, m_q = (1 - s) * x0d, (1 - s) * x0q
            pa = expected_accept(m_p, m_q, s)
            per_node.setdefault(i, []).append(pa)
            print(f"seed {traj['sd']} node {i}: E[accept]={pa:.4f}  "
                  f"|dx0|={ (x0q-x0d).norm().item():.3f}")
    res = {int(k): sum(v)/len(v) for k, v in sorted(per_node.items())}
    overall = sum(res.values()) / len(res)
    out = dict(per_node_expected_accept=res, overall_expected_accept=overall,
               sigmas=[float(s) for s in sigmas], meta=blob["meta"])
    with open(os.path.join(a.out, "acceptance.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
