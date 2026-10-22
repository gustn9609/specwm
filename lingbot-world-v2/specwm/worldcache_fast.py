"""WorldCache (official decision logic) for the causal Fast model (WanModelFast, 4-step per chunk, single stream, KV cache written inside blocks).
블록별 래퍼: probe 블록 0..k-1 은 항상 계산; 블록 k-1 출력에서 drift 판정(누적 상대 L1, saliency 가중, motion α, decay) → skip 이면 블록 k..N-2 는 통과, 마지막 블록이 캐시 residual(OSI 외삽) 을 더한다.
청크마다(σ=1.0 첫 호출) 상태 리셋; commit 호출(t=0) 은 절대 스킵하지 않음(KV 캐시가 블록 안에서 갱신되므로).
"""
import torch, torch.nn as nn


class FastWCCtl:
    def __init__(self, k, n_blocks, thresh=0.06, ret_ratio=0.25, alpha=0.2, beta_s=0.12, osi=True, decay=True, steps_per_chunk=4):
        self.k, self.N, self.thresh, self.ret, self.alpha, self.beta_s, self.osi, self.decay, self.spc = k, n_blocks, thresh, ret_ratio, alpha, beta_s, osi, decay, steps_per_chunk
        self.reset_chunk(); self.n_full = self.n_skip = self.n_commit = 0; self.step = 0; self.commit_call = False; self.log = []

    def reset_chunk(self):
        self.prev_in = None; self.prev_state = None; self.acc = 0.0; self.res_win = []; self.pres_win = []; self.skip = False; self.x_k = None; self.x_in = None; self.step = 0

    def decide(self, x_k):
        eps = 1e-8; self.skip = False; self.x_k = x_k
        if self.commit_call or self.step < max(int(round(self.ret * self.spc)), 1) or self.prev_state is None or len(self.res_win) < 1:
            self.prev_state = x_k; return
        delta_x = float((self.x_in - self.prev_in).abs().mean() / (self.prev_in.abs().mean() + eps)) if self.prev_in is not None else 0.0
        ps_abs = float(self.prev_state.abs().mean() + eps); diff = (x_k - self.prev_state).abs()
        if self.beta_s > 0:
            sal = self.prev_state.float().var(-1, keepdim=True); sal = (sal - sal.amin(1, keepdim=True)) / (sal.amax(1, keepdim=True) - sal.amin(1, keepdim=True) + eps)
            delta_y = float((diff.mean(-1, keepdim=True).float() * (1.0 + self.beta_s * sal)).mean() / ps_abs)
        else: delta_y = float(diff.mean() / ps_abs)
        self.acc += delta_y; thr = self.thresh / (1.0 + self.alpha * delta_x)
        if self.decay: thr *= (1.0 + 4.0 * self.step / max(self.spc - 1, 1))
        self.skip = self.acc < thr
        self.log.append((self.step, round(self.acc, 5), round(thr, 5), int(self.skip)))
        if not self.skip: self.acc = 0.0; self.prev_state = x_k
        else: self.prev_state = x_k

    def r_hat(self):
        rw, pw = self.res_win, self.pres_win
        if len(rw) >= 2 and self.osi:
            ri = (self.x_k - self.x_in); dt = (ri - pw[-2]).float().flatten(); ds = (pw[-1] - pw[-2]).float().flatten()
            g = float(torch.dot(dt, ds) / (torch.dot(ds, ds) + 1e-6)); g = min(max(g, 0.0), 2.0); return rw[-2] + g * (rw[-1] - rw[-2])
        return rw[-1]

    def record(self, x_out):
        res = (x_out - self.x_in).to(torch.bfloat16); pres = (self.x_k - self.x_in).to(torch.bfloat16)
        self.res_win = (self.res_win + [res])[-2:]; self.pres_win = (self.pres_win + [pres])[-2:]; self.prev_in = self.x_in


class _Gate(nn.Module):
    def __init__(self, block, i, ctl):
        super().__init__(); self.block = block; self.i = i; self.ctl = ctl

    def forward(self, x, **kw):
        c = self.ctl
        if self.i == 0: c.x_in = x
        if self.i < c.k:
            out = self.block(x, **kw)
            if self.i == c.k - 1: c.decide(out)
            return out
        if c.skip:
            if self.i == c.N - 1: c.n_skip += 1; return x + c.r_hat().to(x.dtype)
            return x
        out = self.block(x, **kw)
        if self.i == c.N - 1:
            if not c.commit_call: c.record(out); c.n_full += 1
            else: c.n_commit += 1
        return out


def install(model, k=3, thresh=0.06, ret_ratio=0.25, alpha=0.2, beta_s=0.12, osi=True, decay=True, steps_per_chunk=4, t_max=1000.0):
    if getattr(model, "_worldcache_fast", None) is not None: return model._worldcache_fast
    N = len(model.blocks); ctl = FastWCCtl(k, N, thresh, ret_ratio, alpha, beta_s, osi, decay, steps_per_chunk)
    model.blocks = nn.ModuleList([_Gate(b, i, ctl) for i, b in enumerate(model.blocks)])

    def pre(mod, args, kwargs):
        t = kwargs.get("t", args[1] if len(args) > 1 else None)
        if t is None: return
        tv = float(t.flatten()[0]); ctl.commit_call = tv <= 1e-6
        if not ctl.commit_call:
            if tv >= t_max - 1: ctl.reset_chunk()           # σ=1.0: new chunk
            else: ctl.step += 1
    model.register_forward_pre_hook(pre, with_kwargs=True); model._worldcache_fast = ctl
    return ctl


def stats(model):
    c = getattr(model, "_worldcache_fast", None)
    if c is None: return None
    tot = c.n_full + c.n_skip
    import numpy as _np
    accs = _np.array([l[1] for l in c.log]) if c.log else _np.zeros(0); thrs = _np.array([l[2] for l in c.log]) if c.log else _np.zeros(0)
    return dict(k=c.k, full=c.n_full, skip=c.n_skip, commit=c.n_commit, skip_rate=round(c.n_skip / max(tot, 1), 3), thresh=c.thresh, ret=c.ret,
                acc_pcts=([round(float(x), 4) for x in _np.percentile(accs, [5, 25, 50, 75, 95])] if len(accs) else None), thr_pcts=([round(float(x), 4) for x in _np.percentile(thrs, [5, 25, 50, 75, 95])] if len(thrs) else None),
                acc_by_step={int(st): round(float(_np.mean([l[1] for l in c.log if l[0] == st])), 4) for st in sorted(set(l[0] for l in c.log))}, log_tail=c.log[-12:])
