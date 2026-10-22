"""L1 Stage B — TRUE exact speculative sampling with both models resident.
draft = quantized Base (int8 default), target = bf16 Base.
Outputs: exact-speculative samples + target-only + draft-only references,
latency accounting, latent-space MMD (optional DINO MMD on decoded frames).

VRAM budget (GH200 96GB, 480p, 81f): target bf16 2 experts ~56G + draft int8
~28G exceeds -> default keeps draft experts on CPU and streams the ACTIVE
expert to GPU per node group (draft is called sequentially anyway).
Use --draft-on-gpu if you quantize to int4.
"""
import argparse, json, os, time
import torch
from .core import SpecStats, ancestral_chain, speculative_chain
from .quantize import quantize_draft


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--action", default=None)
    ap.add_argument("--prompt", default="A serene lakeside scene with a lone tree.")
    ap.add_argument("--nodes", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--quant", default="int8")
    ap.add_argument("--draft-on-gpu", action="store_true")
    ap.add_argument("--decode", action="store_true", help="decode & save videos")
    ap.add_argument("--out", default="specwm_out/l1_stage_b")
    return ap.parse_args()


def main():
    a = get_args()
    os.makedirs(a.out, exist_ok=True)
    from .binding_v1 import V1Binding
    import copy
    b = V1Binding(a.ckpt, t5_cpu=True)
    cond = b.prepare(a.image, a.prompt, a.action, frame_num=a.frames)
    sigmas, timesteps = b.node_grid(a.nodes)

    # target = the loaded bf16 experts (GPU)
    tgt = {'low': b.pipe.low_noise_model.to(b.device),
           'high': b.pipe.high_noise_model.to(b.device)}
    # draft = quantized deep-copy (CPU staging unless --draft-on-gpu)
    drf = {k: quantize_draft(copy.deepcopy(m).cpu(), a.quant) for k, m in tgt.items()}
    if a.draft_on_gpu:
        drf = {k: m.to(b.device) for k, m in drf.items()}

    def draft_models_on(node_t):
        if a.draft_on_gpu:
            return drf
        # stream only the needed expert
        need = 'high' if node_t >= b.boundary_t else 'low'
        drf[need] = drf[need].to(b.device)
        other = 'low' if need == 'high' else 'high'
        drf[other] = drf[other].cpu()
        return drf

    x0_tgt = b.make_x0_fn(tgt, cond, sigmas, timesteps)
    def x0_drf(z, i):
        draft_models_on(float(timesteps[i]))
        return b.make_x0_fn(drf, cond, sigmas, timesteps)(z, i)

    def target_batch(z_stack, idxs):
        outs = []
        for k, i in enumerate(idxs):          # loop (batch opt possible)
            outs.append(x0_tgt(z_stack[k], i))
        return torch.stack(outs)

    stats = SpecStats()
    lat, spec_x0, tgt_x0, drf_x0 = {}, [], [], []
    for sd in range(a.seeds):
        g = torch.Generator(device="cpu").manual_seed(7000 + sd)
        gg = lambda: torch.Generator(device="cpu").manual_seed(7000 + sd)
        t0 = time.time()
        spec_x0.append(speculative_chain(
            x0_drf, target_batch, cond['shape'], sigmas, stats, gen=gg(),
            device=b.device).cpu())
        lat.setdefault("spec", []).append(time.time() - t0)
        t0 = time.time()
        tgt_x0.append(ancestral_chain(x0_tgt, cond['shape'], sigmas, gen=gg(),
                                      device=b.device).cpu())
        lat.setdefault("target", []).append(time.time() - t0)
        t0 = time.time()
        drf_x0.append(ancestral_chain(x0_drf, cond['shape'], sigmas, gen=gg(),
                                      device=b.device).cpu())
        lat.setdefault("draft", []).append(time.time() - t0)
        print(f"[seed {sd}] spec {lat['spec'][-1]:.1f}s  target {lat['target'][-1]:.1f}s")

    from .metrics import latent_mmd
    res = dict(
        stats=stats.summary(),
        wallclock={k: sum(v)/len(v) for k, v in lat.items()},
        latent_mmd=dict(
            spec_vs_target=latent_mmd(spec_x0, tgt_x0),
            draft_vs_target=latent_mmd(drf_x0, tgt_x0)),
        nodes=a.nodes, seeds=a.seeds, quant=a.quant)
    if a.decode:
        from .metrics import save_videos
        save_videos(b, {"spec": spec_x0, "target": tgt_x0, "draft": drf_x0}, a.out)
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
