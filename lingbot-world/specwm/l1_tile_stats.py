"""Tile-level design data for delta-lag verification (path 2).
Reuses draft_traj.pt from stage A. For each recorded (z, node):
  - x0_q via bf16 target
  - per-tile d = ||m_q - m_p||_tile for tiles = latent frames (21) and
    frame-quadrants (84); predicted per-tile acceptance 2*Phi(-d/2s)
  - Lipschitz probe on 3 nodes: perturb one frame-tile by residual-scale
    noise, re-forward, measure downstream x0 shift ratio (propagation gain)
Output: specwm_out/l1_stage_a/tile_stats.json
"""
import json, math, os, torch
from .core import expected_accept

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--action", default=None)
    ap.add_argument("--prompt", default="A serene lakeside scene with a lone tree.")
    ap.add_argument("--out", default="specwm_out/l1_stage_a")
    ap.add_argument("--probe-nodes", type=int, nargs="*", default=[2, 8, 15])
    a = ap.parse_args()
    from .binding_v1 import V1Binding
    blob = torch.load(os.path.join(a.out, "draft_traj.pt"), map_location="cpu")
    sigmas, timesteps = blob["sigmas"], blob["timesteps"]
    b = V1Binding(a.ckpt, t5_cpu=True)
    cond = b.prepare(a.image, a.prompt, a.action,
                     frame_num=blob["meta"]["frames"])
    models = {'low': b.pipe.low_noise_model.to(b.device),
              'high': b.pipe.high_noise_model.to(b.device)}
    x0_fn = b.make_x0_fn(models, cond, sigmas, timesteps)

    C, F, H, W = cond['shape']
    def tile_d(diff, mode):
        if mode == "frame":
            return diff.pow(2).sum(dim=(0, 2, 3)).sqrt()          # (F,)
        hh, ww = H // 2, W // 2
        q = diff.reshape(C, F, 2, hh, 2, ww).pow(2).sum(dim=(0, 3, 5))
        return q.reshape(F, 4).sqrt()                              # (F,4)

    out = {"sigmas": [float(s) for s in sigmas], "frames": F,
           "per_node": [], "probe": []}
    n = len(sigmas)
    for traj in blob["recs"][:4]:
        for i, (z, x0d) in enumerate(zip(traj["z"], traj["x0d"])):
            if i >= n - 1: continue
            z = z.to(b.device); x0d = x0d.to(b.device)
            x0q = x0_fn(z, i)
            s = float(sigmas[i + 1])
            diff = ((1 - s) * (x0q - x0d)).float().cpu()
            df = tile_d(diff, "frame"); dq = tile_d(diff, "quad")
            rec = dict(seed=traj["sd"], node=i, s=s,
                       d_total=float(diff.norm()),
                       d_frame=[float(x) for x in df],
                       acc_frame=[2*0.5*(1+math.erf((-float(x)/(2*s))/math.sqrt(2))) for x in df],
                       acc_quad_mean=float(torch.tensor(
                           [2*0.5*(1+math.erf((-float(x)/(2*s))/math.sqrt(2)))
                            for x in dq.flatten()]).mean()))
            out["per_node"].append(rec)
            # Lipschitz probe (seed 0 only)
            if traj["sd"] == 0 and i in a.probe_nodes:
                pert = torch.zeros_like(z)
                mid = F // 2
                pert[:, mid] = s * torch.randn(C, H, W, device=b.device) * 0.5
                x0q_p = x0_fn(z + pert, i)
                shift = (x0q_p - x0q).float().cpu()
                own = float(shift[:, mid].norm()); others = float(
                    (shift.norm()**2 - shift[:, mid].norm()**2)**0.5)
                out["probe"].append(dict(
                    node=i, s=s, pert_norm=float(pert.norm()),
                    shift_own_frame=own, shift_other_frames=others,
                    gain=others / max(float(pert.norm()), 1e-9)))
                print(f"[probe] node {i}: gain={out['probe'][-1]['gain']:.4f}")
            print(f"seed {traj['sd']} node {i}: acc_frame_mean="
                  f"{sum(rec['acc_frame'])/F:.3f} acc_quad={rec['acc_quad_mean']:.3f}")
    with open(os.path.join(a.out, "tile_stats.json"), "w") as f:
        json.dump(out, f)
    print("saved tile_stats.json")

if __name__ == "__main__":
    main()
