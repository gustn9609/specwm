"""Tree speculation: the draft PRE-ROLLS K branches so an accepted candidate
always has a valid continuation.

Motivation (measured, not assumed): raising the multi-candidate acceptance
alpha(K) does NOT by itself reduce sequential depth. In a batched scheme the
draft rolls ONE trajectory ahead; if we accept a candidate other than the one
it rolled from, the suffix is stale and we must restart the round anyway. The
probability of keeping the pre-rolled suffix stays alpha, whatever K is.

So the candidates must be PRE-ROLLED. At the first node of each round the
draft expands K branches and rolls each to the end; the target scores every
branch in ONE batched pass. Whichever candidate is accepted already has a
valid suffix, so verification continues down that branch.

Cost: K x draft compute and K x target FLOPs inside a single pass -- but the
number of SEQUENTIAL target passes (the metric that matters for latency) can
drop. Exactness is preserved: acceptance uses the exact multi-candidate
residual recursion from multidraft.py.
"""
from __future__ import annotations
import torch
from .core import SpecStats, expected_accept
from .multidraft import Residual1D


@torch.no_grad()
def _verify_multi(z_cands, m_p, m_q, s, gen, st, node_i):
    """Exact accept among pre-rolled candidates. Returns (idx, sample).

    idx >= 0 : candidate idx accepted (its pre-rolled suffix stays valid)
    idx == -1: all rejected, `sample` is an exact residual draw
    """
    diff = (m_p - m_q).double()
    d = float(diff.norm())
    if d < 1e-12:
        return 0, z_cands[0]
    u = diff / d
    res = Residual1D(d, s)
    for j, y in enumerate(z_cands):
        t = float(((y - m_q).double() * u).sum())
        if float(torch.rand(1, generator=gen)) < min(1.0, res.ratio_at(t)):
            return j, y
        res.update_after_reject()
        st.residual_props += 1
        if res.mass <= 1e-300:
            break
    t_new = res.sample(gen)
    xi = torch.randn(m_q.shape, generator=gen).to(m_q).double()
    perp = xi - (xi * u).sum() * u
    return -1, (m_q.double() + t_new * u + s * perp).to(m_q.dtype)


class TreeChunkSpec:
    def __init__(self, binding, sigmas, stats: SpecStats, K=4, max_rounds=32):
        self.b, self.sig, self.st = binding, sigmas, stats
        self.K, self.max_rounds = K, max_rounds

    @torch.no_grad()
    def run_chunk(self, chunk_id, cond, gen):
        b, sig, st, K = self.b, self.sig, self.st, self.K
        n = len(sig)
        z = b.chunk_noise(chunk_id, gen)
        i_cur, rounds = 0, 0

        while True:
            rounds += 1
            if rounds > self.max_rounds:
                st.residual_fallbacks += 1
                break
            if hasattr(b, "reset_chunk_slot"):
                b.reset_chunk_slot(chunk_id)

            # ---- expand K branches at the current node, roll each to the end
            x0_here = b.draft_x0(z, i_cur, chunk_id); st.draft_calls += 1
            if i_cur == n - 1:
                x0q = b.target_x0(z, i_cur, chunk_id)
                st.target_passes += 1; st.target_evals += 1
                st.rounds_per_sample.append(rounds)
                b.commit(x0q, chunk_id)
                return x0q
            s_next = float(sig[i_cur + 1])
            m_q = (1 - s_next) * x0_here
            branches = []
            for _ in range(K):
                zc = m_q + s_next * b.randn_like(m_q, gen)
                zs, x0d = [zc], []
                cur = zc
                for i in range(i_cur + 1, n):
                    xx = b.draft_x0(cur, i, chunk_id); st.draft_calls += 1
                    x0d.append(xx)
                    if i < n - 1:
                        ss = float(sig[i + 1])
                        cur = (1 - ss) * xx + ss * b.randn_like(xx, gen)
                        zs.append(cur)
                branches.append((zs, x0d))

            # ---- ONE batched target pass scores the whole tree
            m_p = (1 - s_next) * b.target_x0(z, i_cur, chunk_id)
            per_branch_q = []
            for zs, _ in branches:
                per_branch_q.append([b.target_x0(zs[k], i_cur + 1 + k, chunk_id)
                                     for k in range(len(zs))])
            st.target_passes += 1
            st.target_evals += 1 + sum(len(v) for v in per_branch_q)
            st.exp_accept.append((i_cur, expected_accept(m_p, m_q, s_next)))

            j, z_new = _verify_multi([br[0][0] for br in branches], m_p, m_q,
                                     s_next, gen, st, i_cur)
            st.accept_events.append((i_cur, int(j >= 0)))
            if j < 0:
                z, i_cur = z_new, i_cur + 1
                continue

            # ---- accepted branch j: its suffix is valid, keep verifying it
            zs, x0d = branches[j]
            x0q = per_branch_q[j]
            i, adv = i_cur + 1, True
            while i < n - 1 and adv:
                k = i - (i_cur + 1)
                ss = float(sig[i + 1])
                mp, mq = (1 - ss) * x0q[k], (1 - ss) * x0d[k]
                zp = zs[k + 1]
                st.exp_accept.append((i, expected_accept(mp, mq, ss)))
                lr = float((((zp - mp).pow(2) - (zp - mq).pow(2)).sum()) / (2 * ss * ss))
                ok = float(torch.rand(1, generator=gen).log()) < min(lr, 0.0)
                st.accept_events.append((i, int(ok)))
                if ok:
                    i += 1
                else:
                    jj, zn = _verify_multi([zp], mp, mq, ss, gen, st, i)
                    z, i_cur, adv = zn, i + 1, False
            if adv:
                st.rounds_per_sample.append(rounds)
                out = x0q[-1]
                b.commit(out, chunk_id)
                return out
