"""L2 — chunk-axis exact speculative sampling for lingbot-world-v2 (causal).

Draft  = v2 causal_fast (4 nodes/chunk: timesteps_index=[0,250,500,750]),
         quantized (int8 default).
Target = --target fast_bf16        : bf16 causal_fast, same 4-node grid (L1',
                                     RUNNABLE TODAY)
         --target causal_pretrain  : v2 40-step causal teacher via
                                     --target-substeps K deterministic Euler
                                     refinements between the shared 4 nodes
                                     (guarded: needs the not-yet-released ckpt)

Key structural facts exploited (read from v2 wan/image2video.py):
  * fast loop re-noises with FRESH Gaussian between nodes  -> exact Gaussian
    kernels N((1-s')x0_hat, s'^2 I) at 3 nodes/chunk (+ chunk-init noise)
  * KV commit is a SEPARATE forward at t=0 after the chunk  -> commit-on-accept
    gives rollback for free (B=1 chunk blocks; B>1 documented as extension)

Exactness caveat (documented): "target" is the teacher model UNDER THE SAME
re-noise node grid; substeps only refine x0_hat deterministically, so kernels
stay Gaussian and verification stays exact w.r.t. that sampler.
"""
import argparse, copy, json, os, time
import torch
from .core import SpecStats, log_ratio, expected_accept, sample_residual
from .quantize import quantize_draft


def substep_x0(model_call, z, sigma_hi, sigma_lo, k):
    """Deterministic Euler refinement inside [sigma_hi -> sigma_lo] in
    x0-parameterization; returns final x0_hat. k=1 reduces to single call."""
    z_cur, s_cur = z, sigma_hi
    x0 = None
    for j in range(k):
        flow = model_call(z_cur, s_cur)
        x0 = z_cur - s_cur * flow
        if j < k - 1:
            s_next = sigma_hi + (sigma_lo - sigma_hi) * (j + 1) / k
            eps_implied = (z_cur - (1 - s_cur) * x0) / max(s_cur, 1e-8)
            z_cur, s_cur = (1 - s_next) * x0 + s_next * eps_implied, s_next
    return x0


class ChunkSpec:
    def __init__(self, binding, sigmas, stats: SpecStats, target_substeps=1):
        self.b, self.sigmas, self.stats = binding, sigmas, stats
        self.k = target_substeps

    @torch.no_grad()
    def run_chunk(self, chunk_id, cond, gen):
        """Exact speculative sampling of ONE chunk. Returns accepted x0 chunk.
        Draft KV/target KV are only committed AFTER acceptance (rollback-free)."""
        b, sig, st = self.b, self.sigmas, self.stats
        n = len(sig)
        z = b.chunk_noise(chunk_id, gen)
        i_cur, rounds = 0, 0
        while True:
            rounds += 1
            if hasattr(b, "reset_chunk_slot"):
                b.reset_chunk_slot(chunk_id)
            zs, x0d, idxs = [], [], []
            zc = z
            for i in range(i_cur, n):
                x0 = b.draft_x0(zc, i, chunk_id)          # uses draft KV (committed prefix)
                st.draft_calls += 1
                zs.append(zc); x0d.append(x0); idxs.append(i)
                if i < n - 1:
                    s = float(sig[i + 1])
                    zc = (1 - s) * x0 + s * b.randn_like(x0, gen)
            # one batched target involvement
            x0q = [substep_x0(lambda zz, ss: b.target_flow(zz, ss, chunk_id),
                              zs[k], float(sig[idxs[k]]),
                              float(sig[idxs[k] + 1]) if idxs[k] < n - 1 else 0.0,
                              self.k)
                   for k in range(len(idxs))]
            st.target_passes += 1; st.target_evals += len(idxs) * self.k
            rej = -1
            for k, i in enumerate(idxs):
                if i == n - 1:
                    break
                s = float(sig[i + 1])
                m_p, m_q = (1 - s) * x0d[k], (1 - s) * x0q[k]
                z_prop = zs[k + 1] if k + 1 < len(zs) else zc
                st.exp_accept.append((i, expected_accept(m_p, m_q, s)))
                ok = torch.rand((), generator=gen).item() < \
                    torch.exp(log_ratio(z_prop, m_p, m_q, s).clamp(max=0.0)).item()
                st.accept_events.append((i, 1 if ok else 0))
                if not ok:
                    y, np_, fb = sample_residual(m_p, m_q, s, gen)
                    st.residual_props += np_; st.residual_fallbacks += int(fb)
                    z, i_cur, rej = y, i + 1, k
                    break
            if rej == -1:
                st.rounds_per_sample.append(rounds)
                x0_final = x0q[-1]
                b.commit(x0_final, chunk_id)              # KV commit AFTER accept
                return x0_final


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--action", default=None)
    ap.add_argument("--prompt", default="A serene lakeside scene.")
    ap.add_argument("--target", default="fast_bf16",
                    choices=["fast_bf16", "causal_pretrain"])
    ap.add_argument("--target-ckpt", default=None,
                    help="causal-pretrain dir (when released)")
    ap.add_argument("--target-substeps", type=int, default=1)
    ap.add_argument("--frames", type=int, default=161)
    ap.add_argument("--chunk-size", type=int, default=3)
    ap.add_argument("--quant", default="int8")
    ap.add_argument("--sink", type=int, default=6)
    ap.add_argument("--local-attn", type=int, default=18)
    ap.add_argument("--out", default="specwm_out/l2")
    return ap.parse_args()


def main():
    a = get_args()
    os.makedirs(a.out, exist_ok=True)
    if a.target == "causal_pretrain" and not (a.target_ckpt and os.path.isdir(a.target_ckpt)):
        raise SystemExit("[L2] causal-pretrain weights not found. This mode is "
                         "scaffolded for the announced-but-unreleased checkpoint. "
                         "Use --target fast_bf16 (runnable today) meanwhile.")
    from .binding_v2 import V2Binding
    stats = SpecStats()
    b = V2Binding(a.ckpt, target_mode=a.target, target_ckpt=a.target_ckpt,
                  quant=a.quant, sink=a.sink, local_attn=a.local_attn)
    cond = b.prepare(a.image, a.prompt, a.action, frame_num=a.frames,
                     chunk_size=a.chunk_size)
    spec = ChunkSpec(b, b.node_sigmas(), stats, a.target_substeps)
    gen = torch.Generator(device="cpu").manual_seed(1234)
    t0 = time.time(); chunks = []
    for c in range(cond["num_chunks"]):
        chunks.append(spec.run_chunk(c, cond, gen))
        print(f"[chunk {c}] mean_rounds so far "
              f"{sum(stats.rounds_per_sample)/len(stats.rounds_per_sample):.2f}")
    res = dict(stats=stats.summary(), wallclock=time.time() - t0,
               num_chunks=cond["num_chunks"], target=a.target,
               target_substeps=a.target_substeps)
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    b.decode_and_save(chunks, a.out)


if __name__ == "__main__":
    main()
