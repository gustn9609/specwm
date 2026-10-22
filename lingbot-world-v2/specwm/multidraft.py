"""EXACT multi-candidate speculative sampling for the Gaussian denoise chain.

THE REDUCTION
-------------
Both kernels share the same covariance, p = N(mu_p, s^2 I), q = N(mu_q, s^2 I),
so the likelihood ratio depends on y ONLY through its projection on the mean
difference:

    log p(y)/q(y) = ( 2 y.(mu_p-mu_q) + |mu_q|^2 - |mu_p|^2 ) / 2 s^2

Write u = (mu_p-mu_q)/d with d = |mu_p-mu_q|, and t = u.(y - mu_q). Then

    p(y) = p1(t) * g(y_perp),      q(y) = q1(t) * g(y_perp)

with the SAME orthogonal factor g, and 1-D marginals q1 = N(0,s^2),
p1 = N(d,s^2). Every accept/reject decision, and every residual
(p - q)_+ / mass, therefore lives in ONE dimension.

WHY THIS MATTERS.  Single-candidate acceptance is alpha = 2*Phi(-d/2s) and
collapses as the verified dimension grows. The standard fix is to try several
candidates with a recursively-updated residual, which raises acceptance to
1 - prod_k (rejection prob). That recursion needs the residual DENSITY, which
is intractable in 524160-D -- but after the reduction above it is a 1-D
function we can hold on a grid and integrate exactly.

Crucially the K candidates cost NO extra draft forward passes: one draft call
gives mu_q, and the candidates are just K noise draws from N(mu_q, s^2 I).
"""
from __future__ import annotations
import math
import torch


def _norm_pdf(t, mu, s):
    return torch.exp(-0.5 * ((t - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))


class Residual1D:
    """Normalised residual density on a grid, updated by r <- (r - q1)_+ ."""

    def __init__(self, d, s, n_grid=4001, span=9.0):
        lo = min(0.0, d) - span * s
        hi = max(0.0, d) + span * s
        self.t = torch.linspace(lo, hi, n_grid, dtype=torch.float64)
        self.dt = float(self.t[1] - self.t[0])
        self.q1 = _norm_pdf(self.t, 0.0, s)
        self.r = _norm_pdf(self.t, d, s)          # r_0 = p1
        self.d, self.s = d, s
        self.mass = float((self.r * self.dt).sum())

    def ratio_at(self, tv):
        """r(t)/q1(t) by linear interpolation on the grid."""
        i = torch.clamp(((tv - self.t[0]) / self.dt).long(), 0, len(self.t) - 2)
        w = (tv - self.t[i]) / self.dt
        r = (1 - w) * self.r[i] + w * self.r[i + 1]
        q = (1 - w) * self.q1[i] + w * self.q1[i + 1]
        return float(r / max(float(q), 1e-300))

    def accept_prob_mass(self):
        """int min(r, q1) -- the probability the NEXT candidate is accepted."""
        return float((torch.minimum(self.r, self.q1) * self.dt).sum())

    def update_after_reject(self):
        self.r = torch.clamp(self.r - self.q1, min=0.0)
        m = float((self.r * self.dt).sum())
        self.mass = m
        if m > 1e-300:
            self.r = self.r / m

    def sample(self, gen):
        """Inverse-CDF draw from the current residual."""
        c = torch.cumsum(self.r * self.dt, 0)
        c = c / c[-1]
        u = torch.rand(1, generator=gen, dtype=torch.float64)
        j = int(torch.searchsorted(c, u).clamp(0, len(self.t) - 1))
        return float(self.t[j])


@torch.no_grad()
def multi_draft_sample(mu_p, mu_q, s, K, gen, cands=None):
    """Draw EXACTLY from N(mu_p, s^2 I) using up to K candidates from q.

    Returns (y, k_used, accepted) where k_used is how many candidates were
    consumed and `accepted` says whether one of them was taken (False means we
    fell back to the residual, which is still an exact draw from p).
    """
    diff = (mu_p - mu_q).double()
    d = float(diff.norm())
    if d < 1e-12:
        y = mu_q + s * torch.randn(mu_q.shape, generator=gen).to(mu_q)
        return y, 1, True
    u = (diff / d)

    res = Residual1D(d, s)
    for k in range(1, K + 1):
        if cands is not None and k <= len(cands):
            y = cands[k - 1]
        else:
            y = mu_q + s * torch.randn(mu_q.shape, generator=gen).to(mu_q)
        t = float(((y - mu_q).double() * u).sum())
        a = min(1.0, res.ratio_at(t))
        if float(torch.rand(1, generator=gen)) < a:
            return y, k, True
        res.update_after_reject()
        if res.mass <= 1e-300:
            break

    # all candidates rejected -> exact draw from the remaining residual:
    # sample the 1-D coordinate from r_K and the orthogonal part from q
    t_new = res.sample(gen)
    xi = torch.randn(mu_q.shape, generator=gen).to(mu_q).double()
    xi_par = (xi * u).sum()
    perp = xi - xi_par * u                       # orthogonal component
    y = mu_q.double() + t_new * u + s * perp
    return y.to(mu_q.dtype), K, False


def expected_accept_multi(mu_p, mu_q, s, K):
    """Closed-form-ish acceptance after K candidates (1-D integration)."""
    d = float((mu_p - mu_q).double().norm())
    if d < 1e-12:
        return 1.0
    res = Residual1D(d, s)
    p_rej = 1.0
    total = 0.0
    for _ in range(K):
        a = res.accept_prob_mass()
        total += p_rej * a
        p_rej *= (1 - a)
        res.update_after_reject()
        if res.mass <= 1e-300:
            break
    return total
