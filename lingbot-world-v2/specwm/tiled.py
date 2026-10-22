"""Tile-factorized verification for the chunk axis, with an optional delta-lag.

WHY TILES.  The accept probability is alpha = 2*Phi(-||mu_q-mu_p|| / 2 sigma)
and the norm is taken over EVERY dimension being verified, so it grows like
sqrt(D).  Measured on the 14B backbone: whole clip (D=524160) -> alpha 0.09,
per frame (D=24960) -> 0.44, per quadrant (D=6240) -> 0.70.

TWO REGIMES (be precise about which one you are running):

  lag = 0   EXACT.  Kernels are diagonal Gaussians so they factorize over
            tiles; per-tile maximal coupling gives an exact draw from the
            target kernel.  BUT a correction at node i invalidates the draft's
            already-rolled suffix, so we must resume from node i.  Since
            P(all tiles accept) = prod(alpha_t) is tiny for many tiles, this
            does NOT reduce sequential depth -- it is the honest exact
            baseline, and it does reduce DRAFT compute (one rollout per round).

  lag = k   APPROXIMATE with a bounded error.  We allow the draft's suffix to
            stay for up to k nodes after a correction instead of re-rolling.
            The induced error per carried node is bounded by
                ||delta||  *  G
            where delta is the corrected mass and G is the measured correction
            propagation gain (this session: G ~= 0.8-1.7, i.e. ~1, so the error
            accumulates LINEARLY in k, not exponentially).  This is the dial
            between SDVG (no guarantee) and exact (depth n).
"""
from __future__ import annotations
import math
import torch
from .core import SpecStats, expected_accept


def make_tiles(shape, mode="quad"):
    """shape = (C, F, H, W) latent chunk -> LongTensor tile ids of that shape."""
    C, F, H, W = shape
    ids = torch.zeros(shape, dtype=torch.long)
    if mode == "whole":
        return ids
    if mode == "frame":
        for f in range(F):
            ids[:, f] = f
        return ids
    if mode == "quad":
        hh, ww = H // 2, W // 2
        t = 0
        for f in range(F):
            for a in range(2):
                for b in range(2):
                    ids[:, f, a * hh:(a + 1) * hh if a == 0 else H,
                        b * ww:(b + 1) * ww if b == 0 else W] = t
                    t += 1
        return ids
    raise ValueError(mode)


class TiledChunkSpec:
    """Chunk-axis speculative sampling with per-tile accept/reject."""

    def __init__(self, binding, sigmas, stats: SpecStats, tile_mode="quad",
                 lag=0, target_substeps=1, max_rounds=32):
        self.b, self.sig, self.st = binding, sigmas, stats
        self.tile_mode, self.lag = tile_mode, lag
        self.k, self.max_rounds = target_substeps, max_rounds
        self._tid = None

    def _tiles(self, shape):
        if self._tid is None or self._tid.shape != shape:
            self._tid = make_tiles(shape, self.tile_mode).to(
                self.b.device if hasattr(self.b, "device") else "cpu")
        return self._tid

    @torch.no_grad()
    def run_chunk(self, chunk_id, cond, gen):
        b, sig, st = self.b, self.sig, self.st
        n = len(sig)
        z = b.chunk_noise(chunk_id, gen)
        tid = self._tiles(z.shape).to(z.device)
        T = int(tid.max().item()) + 1
        flat = tid.reshape(-1)
        i_cur, rounds, carried = 0, 0, 0

        def tile_sum(v):
            return torch.zeros(T, dtype=torch.float64, device=v.device).scatter_add_(
                0, flat, v.reshape(-1).double())

        while True:
            rounds += 1
            if rounds > self.max_rounds:
                st.residual_fallbacks += 1
                break
            if hasattr(b, "reset_chunk_slot"):
                b.reset_chunk_slot(chunk_id)
            # ---- draft rolls the remaining nodes ----
            zs, x0d, idxs = [], [], []
            zc = z
            for i in range(i_cur, n):
                x0 = b.draft_x0(zc, i, chunk_id); st.draft_calls += 1
                zs.append(zc); x0d.append(x0); idxs.append(i)
                if i < n - 1:
                    s = float(sig[i + 1])
                    zc = (1 - s) * x0 + s * b.randn_like(x0, gen)
            # ---- ONE batched target involvement ----
            x0q = [b.target_x0(zs[j], idxs[j], chunk_id) for j in range(len(idxs))]
            st.target_passes += 1; st.target_evals += len(idxs)
            # ---- per-tile verification ----
            rej_at = -1
            for j, i in enumerate(idxs):
                if i == n - 1:
                    break
                s = float(sig[i + 1])
                m_p, m_q = (1 - s) * x0d[j], (1 - s) * x0q[j]
                z_prop = zs[j + 1] if j + 1 < len(zs) else zc
                st.exp_accept.append((i, expected_accept(m_p, m_q, s)))
                lr = tile_sum((z_prop - m_p).pow(2) - (z_prop - m_q).pow(2)) / (2 * s * s)
                u = torch.rand(T, generator=gen).double().to(lr.device)
                rej = u.log() >= lr.clamp(max=0.0)
                st.accept_events.append((i, int((~rej).sum().item() == T)))
                if not bool(rej.any()):
                    continue
                # per-tile residual: propose from q, accept w.p. 1 - min(1, p/q)
                mask = rej[flat].reshape(z_prop.shape)
                cur, undecided = z_prop.clone(), mask.clone()
                for _ in range(400):
                    if not bool(undecided.any()):
                        break
                    y = m_q + s * b.randn_like(m_q, gen)
                    lpq = tile_sum(((y - m_q).pow(2) - (y - m_p).pow(2))
                                   * undecided.to(y.dtype)) / (2 * s * s)
                    a = torch.rand(T, generator=gen).double().to(lpq.device)
                    acc = (a < (1.0 - lpq.clamp(max=0.0).exp())) & rej
                    if bool(acc.any()):
                        am = acc[flat].reshape(y.shape) & undecided
                        cur = torch.where(am, y, cur)
                        undecided &= ~am
                    st.residual_props += 1
                if bool(undecided.any()):
                    st.residual_fallbacks += 1
                    cur = torch.where(undecided, m_q + s * b.randn_like(m_q, gen), cur)
                zs[j + 1] = cur
                if carried < self.lag:
                    # delta-lag: keep the (now stale) suffix, error bounded by
                    # ||corrected mass|| * propagation gain
                    carried += 1
                    continue
                z, i_cur, rej_at = cur, i + 1, j
                carried = 0
                break
            if rej_at == -1:
                st.rounds_per_sample.append(rounds)
                x0_final = x0q[-1]
                b.commit(x0_final, chunk_id)
                return x0_final
