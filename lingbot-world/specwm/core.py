"""
specwm.core — Exact speculative sampling core for rectified-flow video models.

Math (all kernels are equal-variance Gaussians, so everything is closed-form):
  Wan2.2 flow parameterization (verified against lingbot v2 `_convert_flow_pred_to_x0`):
      flow_pred = noise - x0 ;  x_sigma = (1-sigma)*x0 + sigma*noise
      =>  x0_hat = x_sigma - sigma * flow_pred          (exact identity)
  Re-noise ancestral sampler (the sampler lingbot-v2 causal_fast ALREADY uses):
      at node sigma_i:  x0_hat = f_model(z_i, sigma_i)
      z_{i+1} = (1 - s') * x0_hat + s' * eps,  eps ~ N(0, I),  s' = sigma_{i+1}
      =>  transition kernel  N( (1-s') * x0_hat , s'^2 I )
  Draft kernel p, target kernel q share s' => per-node:
      log q/p = ( ||z-m_p||^2 - ||z-m_q||^2 ) / (2 s'^2)
      accept prob        = min(1, q/p)
      E_p[min(1, q/p)]   = 2 * Phi( -||m_q - m_p|| / (2 s') )   (equal-var overlap)
      residual r ∝ max(0, q - p): propose y~q, accept w.p. 1 - min(1, p/q)
Exactness: accepted-or-residual output is distributed exactly as the target
chain (Leviathan et al. 2023; De Bortoli et al. ICML'25 continuous case).
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional
import torch


def shifted_sigmas(n_nodes: int, shift: float, device="cpu") -> torch.Tensor:
    """sigma grid used by Wan/FlowUniPC: sigma(t) = shift*t / (1 + (shift-1)*t),
    t uniform in (0, 1].  Descending, last node sigma=0 appended implicitly by
    the caller taking x0 at the end. Matches FlowUniPCMultistepScheduler with
    use_dynamic_shifting=False (verified formula; scheduler object can be
    passed instead via `sigmas_override` in samplers for bit-exactness)."""
    t = torch.linspace(1.0, 1.0 / n_nodes, n_nodes, device=device)
    return shift * t / (1 + (shift - 1) * t)


def log_ratio(z: torch.Tensor, m_p: torch.Tensor, m_q: torch.Tensor, s: float) -> torch.Tensor:
    """log q(z)/p(z) for equal-variance Gaussians. Sums over all dims."""
    return ((z - m_p).pow(2).sum() - (z - m_q).pow(2).sum()) / (2.0 * s * s)


def expected_accept(m_p: torch.Tensor, m_q: torch.Tensor, s: float) -> float:
    """Closed form E_{z~p}[min(1, q/p)] = 2*Phi(-d/(2s)), d = ||m_q-m_p||."""
    d = (m_q - m_p).float().norm().item()
    return 2.0 * 0.5 * (1.0 + math.erf((-d / (2.0 * s)) / math.sqrt(2.0)))


def sample_residual(m_p: torch.Tensor, m_q: torch.Tensor, s: float,
                    gen: Optional[torch.Generator] = None, max_iter: int = 1000):
    """Sample from r ∝ max(0, q - p). Zero model calls. Returns (y, n_props, fb)."""
    for i in range(max_iter):
        eps = torch.randn(m_q.shape, generator=gen).to(m_q.device, m_q.dtype)
        y = m_q + s * eps
        lp_minus_lq = ((y - m_q).pow(2).sum() - (y - m_p).pow(2).sum()) / (2 * s * s)
        # accept w.p. 1 - min(1, p/q); log p/q = lp_minus_lq
        if torch.rand((), generator=gen).item() < \
                1.0 - math.exp(min(0.0, lp_minus_lq.item())):
            return y, i + 1, False
    return m_q + s * torch.randn(m_q.shape, generator=gen).to(m_q.device, m_q.dtype), max_iter, True


@dataclass
class SpecStats:
    draft_calls: int = 0
    target_passes: int = 0          # sequential target involvements (latency proxy)
    target_evals: int = 0           # total states evaluated by target
    rounds_per_sample: List[int] = field(default_factory=list)
    accept_events: List[tuple] = field(default_factory=list)   # (node_idx, 0/1)
    exp_accept: List[tuple] = field(default_factory=list)      # (node_idx, prob)
    residual_props: int = 0
    residual_fallbacks: int = 0

    def summary(self) -> dict:
        per_node_emp, per_node_exp = {}, {}
        for i, ok in self.accept_events:
            per_node_emp.setdefault(int(i), []).append(ok)
        for i, p in self.exp_accept:
            per_node_exp.setdefault(int(i), []).append(p)
        return dict(
            draft_calls=self.draft_calls,
            target_passes=self.target_passes,
            target_evals=self.target_evals,
            mean_rounds=(sum(self.rounds_per_sample) / max(1, len(self.rounds_per_sample))),
            overall_accept=(sum(o for _, o in self.accept_events) /
                            max(1, len(self.accept_events))),
            per_node_accept_emp={k: sum(v) / len(v) for k, v in sorted(per_node_emp.items())},
            per_node_accept_closed={k: sum(v) / len(v) for k, v in sorted(per_node_exp.items())},
            residual_props=self.residual_props,
            residual_fallbacks=self.residual_fallbacks,
        )


@torch.no_grad()
def ancestral_chain(x0_fn: Callable, shape, sigmas: torch.Tensor,
                    gen: Optional[torch.Generator] = None, device="cpu",
                    dtype=torch.float32):
    """Baseline sampler: sequential x0-renoise ancestral chain for ONE model."""
    z = torch.randn(shape, generator=gen).to(device, dtype)
    n = len(sigmas)
    x0 = None
    for i in range(n):
        x0 = x0_fn(z, i)
        if i < n - 1:
            s = float(sigmas[i + 1])
            z = (1 - s) * x0 + s * torch.randn(shape, generator=gen).to(device, dtype)
    return x0


@torch.no_grad()
def speculative_chain(draft_x0_fn: Callable, target_x0_fn_batch: Callable,
                      shape, sigmas: torch.Tensor, stats: SpecStats,
                      gen: Optional[torch.Generator] = None, device="cpu",
                      dtype=torch.float32, record_expected: bool = True):
    """
    Exact speculative sampling of the TARGET chain using DRAFT proposals.

    draft_x0_fn(z, node_idx) -> x0_hat                (called sequentially)
    target_x0_fn_batch(z_stack, node_idx_list) -> x0_stack   (ONE batched pass)

    Returns final x0 (exact target-chain sample).
    """
    n = len(sigmas)
    z = torch.randn(shape, generator=gen).to(device, dtype)
    i_cur = 0
    rounds = 0
    while True:
        rounds += 1
        # ---- draft rollout over remaining nodes ----
        zs, x0s_d, idxs = [], [], []
        zc = z
        for i in range(i_cur, n):
            x0_d = draft_x0_fn(zc, i)
            stats.draft_calls += 1
            zs.append(zc); x0s_d.append(x0_d); idxs.append(i)
            if i < n - 1:
                s = float(sigmas[i + 1])
                zc = (1 - s) * x0_d + s * torch.randn(shape, generator=gen).to(device, dtype)
                zs_next = zc  # noqa (kept for clarity)
        # ---- ONE batched target pass over all visited states ----
        x0s_q = target_x0_fn_batch(torch.stack(zs), idxs)
        stats.target_passes += 1
        stats.target_evals += len(idxs)
        # ---- sequential verification ----
        rej_pos = -1
        z_run = None
        for k, i in enumerate(idxs):
            if i == n - 1:
                break  # final node handled after loop
            s = float(sigmas[i + 1])
            m_p = (1 - s) * x0s_d[k]
            m_q = (1 - s) * x0s_q[k]
            # the z that was actually proposed for node i+1:
            z_prop = zs[k + 1] if k + 1 < len(zs) else zc
            if record_expected:
                stats.exp_accept.append((i, expected_accept(m_p, m_q, s)))
            la = float(log_ratio(z_prop, m_p, m_q, s).clamp(max=0.0))
            ok = math.log(torch.rand((), generator=gen).item() + 1e-30) < la
            stats.accept_events.append((i, 1 if ok else 0))
            if not ok:
                y, np_, fb = sample_residual(m_p, m_q, s, gen)
                stats.residual_props += np_
                stats.residual_fallbacks += int(fb)
                z = y
                i_cur = i + 1
                rej_pos = k
                break
        if rej_pos == -1:
            # all intermediate nodes accepted -> final x0 must come from TARGET
            stats.rounds_per_sample.append(rounds)
            return x0s_q[-1]


# ===================== Path (2): tile-factorized OPV =========================
@torch.no_grad()
def tiled_speculative_chain(draft_x0_fn, target_x0_fn_batch, shape, sigmas,
                            tile_ids, stats: SpecStats, gen=None,
                            device="cpu", dtype=torch.float32, max_rounds=64):
    """
    Optimistic Patch-and-Verify (OPV).
    Kernels are diagonal Gaussians => they factorize over tiles. Run Leviathan
    speculative sampling PER TILE with fixed per-(node,tile) uniforms; iterate
    Jacobi sweeps (each sweep = ONE batched target pass) patching rejected
    tiles from per-tile residuals until a rejection-free sweep.

    tile_ids: LongTensor broadcastable to `shape`, mapping each element -> tile.
    Returns (x0_final, n_sweeps).
    Sequential target depth = n_sweeps (the quantity we want << n_nodes).
    """
    n = len(sigmas)
    T = int(tile_ids.max().item()) + 1
    flat_tid = tile_ids.reshape(-1)
    z0 = torch.randn(shape, generator=gen).to(device, dtype)

    # round 0: draft rollout
    Z = [None] * n              # Z[i] = state entering node i
    Mprop = [None] * (n - 1)    # mean that actually generated Z[i+1]
    Z[0] = z0
    for i in range(n - 1):
        x0_d = draft_x0_fn(Z[i], i); stats.draft_calls += 1
        s = float(sigmas[i + 1])
        m = (1 - s) * x0_d
        eps = torch.randn(shape, generator=gen).to(device, dtype)
        Z[i + 1] = m + s * eps
        Mprop[i] = m.clone()
    # fixed uniforms per (node, tile)
    U = torch.rand(n - 1, T, generator=gen)

    def tile_sums(v):  # sum of v over dims, per tile -> (T,)
        return torch.zeros(T, dtype=torch.float64).scatter_add_(
            0, flat_tid, v.reshape(-1).double())

    sweeps = 0
    x0_q_last = None
    while sweeps < max_rounds:
        sweeps += 1
        x0_q = target_x0_fn_batch(torch.stack(Z), list(range(n)))
        stats.target_passes += 1; stats.target_evals += n
        x0_q_last = x0_q[n - 1]
        any_rej = False
        for i in range(n - 1):
            s = float(sigmas[i + 1])
            m_q = (1 - s) * x0_q[i]
            zP = Z[i + 1]
            # per-tile log ratio
            lr = tile_sums((zP - Mprop[i]).pow(2) - (zP - m_q).pow(2)) / (2 * s * s)
            rej = (U[i].double().log() >= lr.clamp(max=0.0))
            if not bool(rej.any()):
                continue
            any_rej = True
            mask = rej[flat_tid].reshape(shape)
            # residual resample per rejected tile: propose from q_tile,
            # accept w.p. 1 - min(1, p/q); vectorized loop with cap
            cur = zP.clone()
            undecided = mask.clone()
            for _ in range(200):
                if not bool(undecided.any()):
                    break
                eps = torch.randn(shape, generator=gen).to(device, dtype)
                y = m_q + s * eps
                lpq = tile_sums(((y - m_q).pow(2) - (y - Mprop[i]).pow(2)) *
                                undecided.reshape(shape)) / (2 * s * s)
                a = torch.rand(T, generator=gen).double()
                acc_t = (a < (1.0 - lpq.clamp(max=0.0).exp())) & rej
                if bool(acc_t.any()):
                    am = acc_t[flat_tid].reshape(shape) & undecided
                    cur = torch.where(am, y, cur)
                    undecided = undecided & ~am
                stats.residual_props += 1
            if bool(undecided.any()):   # cap fallback: take q draw
                stats.residual_fallbacks += 1
                eps = torch.randn(shape, generator=gen).to(device, dtype)
                cur = torch.where(undecided, m_q + s * eps, cur)
            Z[i + 1] = cur
            Mprop[i] = torch.where(mask, m_q, Mprop[i])
            # NOTE: downstream Z stay (stale) — next sweep re-verifies them
            for t in torch.nonzero(rej).flatten().tolist():
                stats.accept_events.append((i, 0))
        if not any_rej:
            break
    stats.rounds_per_sample.append(sweeps)
    return x0_q_last, sweeps
