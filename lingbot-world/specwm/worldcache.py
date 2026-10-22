"""WorldCache-style training-free feature caching for the Wan DiT block loop (arXiv 2603.22286, 재구현/축약판).

블록 0..k-1(probe)만 매 스텝 계산하고, probe 출력의 변화량 δ(saliency 가중) 가 임계값 τ(t) 아래면 깊은 블록 k..N-1 을 건너뛰고
캐시된 깊은 residual r = x_N - x_k 를 최소자승 외삽(OSI: γ* = <Δtgt,Δsrc>/|Δsrc|², [0,2] 클램프)으로 재사용한다.
  CFC: τ = τ0 / (1 + α·v),  v = |x_in(t) - x_in(t-2)|₁ / |x_in(t-2)|₁     (t-2 앵커: t-1 이 캐시 근사일 수 있어서)
  SWD: δ = mean_token( mean_c|Δprobe| · (1 + β_s·ŝ) ) / mean|probe_prev|,  ŝ = 토큰별 채널 분산을 [0,1] 정규화
  ATS: τ(t) = τ_CFC · (1 + β_d · t/T)   (t/T = 디노이즈 진행률, 타임스텝 1000→0 에서 1 - t/1000)
  생략: latent 상관 기반 motion-compensated warping (논문의 선택 항목). CFG cond/uncond 두 스트림과 MoE 전문가별로 캐시를 분리.
사용: install(model, k=4, ...) -> model.blocks 를 래핑. stats() 로 스킵률 확인.
"""
import torch, torch.nn as nn


class _Ctl:
    def __init__(self, k, tau0, alpha, beta_s, beta_d, min_frac, n_streams, gmax=2.0):
        self.k, self.tau0, self.alpha, self.beta_s, self.beta_d, self.min_frac, self.n, self.gmax = k, tau0, alpha, beta_s, beta_d, min_frac, n_streams, gmax
        self.call = 0; self.frac = 0.0; self.st = [dict(probe_prev=None, in_hist=[], r_hist=[], pr_hist=[]) for _ in range(n_streams)]
        self.n_full = 0; self.n_skip = 0; self.log = []

    def reset(self):
        self.__init__(self.k, self.tau0, self.alpha, self.beta_s, self.beta_d, self.min_frac, self.n, self.gmax)


class CachedBlocks(nn.Module):
    def __init__(self, blocks, ctl):
        super().__init__(); self.inner = blocks; self.ctl = ctl

    def __len__(self): return 1

    def __iter__(self):
        yield self._run

    @torch.no_grad()
    def _run(self, x, **kw):
        c = self.ctl; s = c.st[c.call % c.n]; c.call += 1; eps = 1e-6
        x_in = x
        v = None
        if len(s["in_hist"]) >= 2:
            a = s["in_hist"][-2]; v = float((x_in - a).abs().mean() / (a.abs().mean() + eps))
        for b in self.inner[:c.k]: x = b(x, **kw)
        xp = x; skip = False; delta = None; tau = None
        if s["probe_prev"] is not None and len(s["r_hist"]) >= 1 and c.frac >= c.min_frac:
            d = (xp - s["probe_prev"]).abs().mean(-1)                                   # [B, L]
            if c.beta_s > 0:
                sal = xp.float().var(-1); sal = (sal - sal.amin(-1, keepdim=True)) / (sal.amax(-1, keepdim=True) - sal.amin(-1, keepdim=True) + eps)
                d = d.float() * (1 + c.beta_s * sal)
            delta = float(d.mean() / (s["probe_prev"].abs().mean() + eps))
            tau = c.tau0 / (1 + c.alpha * (v if v is not None else 0.0)) * (1 + c.beta_d * c.frac)
            skip = delta < tau
        pr = (xp - x_in)
        if skip:
            rh, ph = s["r_hist"], s["pr_hist"]
            if len(rh) >= 2 and len(ph) >= 2:
                dt = (pr - ph[-2]).float().flatten(); ds = (ph[-1] - ph[-2]).float().flatten()
                g = float(torch.dot(dt, ds) / (torch.dot(ds, ds) + eps)); g = min(max(g, 0.0), c.gmax)
                r_hat = rh[-2] + g * (rh[-1] - rh[-2])
            else: r_hat = rh[-1]
            x = xp + r_hat.to(xp.dtype); c.n_skip += 1
        else:
            for b in self.inner[c.k:]: x = b(x, **kw)
            s["r_hist"] = (s["r_hist"] + [(x - xp).to(torch.bfloat16)])[-2:]; c.n_full += 1
        s["pr_hist"] = (s["pr_hist"] + [pr.to(torch.bfloat16)])[-2:]; s["probe_prev"] = xp; s["in_hist"] = (s["in_hist"] + [x_in])[-2:]
        c.log.append((round(c.frac, 3), None if delta is None else round(delta, 5), None if tau is None else round(tau, 5), int(skip)))
        return x


def install(model, k=4, tau0=0.08, alpha=2.0, beta_s=0.12, beta_d=4.0, min_frac=0.1, n_streams=2, t_max=1000.0, gmax=2.0):
    """model.blocks 를 캐시 래퍼로 교체. 타임스텝은 forward pre-hook 으로 읽어 진행률로 변환."""
    if isinstance(model.blocks, CachedBlocks): return model.blocks.ctl
    ctl = _Ctl(k, tau0, alpha, beta_s, beta_d, min_frac, n_streams, gmax)
    model.blocks = CachedBlocks(model.blocks, ctl)

    def pre(mod, args, kwargs):
        t = kwargs.get("t", args[1] if len(args) > 1 else None)
        if t is not None: ctl.frac = 1.0 - float(t.flatten()[0]) / t_max
    model.register_forward_pre_hook(pre, with_kwargs=True)
    model._worldcache = ctl
    return ctl


def stats(model):
    c = getattr(model, "_worldcache", None)
    if c is None: return None
    tot = c.n_full + c.n_skip
    return dict(k=c.k, full=c.n_full, skip=c.n_skip, skip_rate=round(c.n_skip / max(tot, 1), 3))


# ---------------------------------------------------------------------------------------------
# 공식 구현(github.com/umair1221/WorldCache, Models/WAN2.1/infer_i2v_worldcache.py worldcache_forward) 의 판정 논리를 블록 루프에 이식.
# 차이: 공식은 WanModel.forward 전체를 교체(WAN2.1 전용 시그니처)하지만 우리 Base 는 cam-plucker/MoE 가 있어 블록 루프만 교체한다.
#   - drift 를 스텝마다 **누적**(accumulated_rel_l1_distance)하고, 캐시 미스 때 0 으로 리셋 (DiCache 방식)
#   - ret_ratio: 초기 20% 스텝은 항상 전체 계산  - 기본값: probe_depth 3, rel_l1_thresh 0.06, motion α 5.0, saliency β 0.12, OSI, dynamic decay ×(1+4·진행률)
#   - AdUC: 진행률 > aduc_start(0.5) 이면 uncond 스트림은 이전 출력 재사용  - flow warp 는 공식 기본값도 off
#   스텝 진행률은 공식의 cnt/num_steps 대신 타임스텝(1 - t/1000) 을 사용 (MoE 전문가별로 호출 수가 달라 cnt 가 정의되지 않음).
# ---------------------------------------------------------------------------------------------
class _OCtl:
    def __init__(self, k, thresh, ret_ratio, alpha, beta_s, osi, decay, aduc, aduc_start, n_streams):
        self.k, self.thresh, self.ret, self.alpha, self.beta_s, self.osi, self.decay, self.aduc, self.aduc_start, self.n = k, thresh, ret_ratio, alpha, beta_s, osi, decay, aduc, aduc_start, n_streams
        self.call = 0; self.frac = 0.0; self.n_full = 0; self.n_skip = 0; self.n_aduc = 0; self.log = []
        self.st = [dict(prev_in=None, prev_state=None, acc=0.0, resume=False, res_cache=None, res_win=[], pres_win=[], prev_out=None) for _ in range(n_streams)]


class OfficialCachedBlocks(nn.Module):
    def __init__(self, blocks, ctl):
        super().__init__(); self.inner = blocks; self.ctl = ctl

    def __len__(self): return 1

    def __iter__(self):
        yield self._run

    @torch.no_grad()
    def _run(self, x, **kw):
        c = self.ctl; idx = c.call % c.n; c.call += 1; s = c.st[idx]; ori = x; skip = False; test_x = None; eps = 1e-8
        if c.frac >= c.ret:
            test_x = x
            for b in self.inner[:c.k]: test_x = b(test_x, **kw)
            if s["prev_in"] is not None and s["prev_state"] is not None:
                delta_x = float((x - s["prev_in"]).abs().mean() / (s["prev_in"].abs().mean() + eps))
                ps_abs = float(s["prev_state"].abs().mean() + eps)
                diff = (test_x - s["prev_state"]).abs()
                if c.beta_s > 0:
                    sal = s["prev_state"].float().var(-1, keepdim=True); sal = (sal - sal.amin(1, keepdim=True)) / (sal.amax(1, keepdim=True) - sal.amin(1, keepdim=True) + eps)
                    delta_y = float((diff.mean(-1, keepdim=True).float() * (1.0 + c.beta_s * sal)).mean() / ps_abs)
                else: delta_y = float(diff.mean() / ps_abs)
                s["acc"] += delta_y
                if c.aduc and idx == 1 and c.frac > c.aduc_start and s["prev_out"] is not None:
                    c.n_aduc += 1; c.log.append((round(c.frac, 3), "aduc")); return s["prev_out"]
                thr = c.thresh / (1.0 + c.alpha * delta_x)
                if c.decay: thr *= (1.0 + 4.0 * c.frac)
                if s["acc"] < thr:
                    skip = True; s["resume"] = False
                else:
                    s["resume"] = True; s["acc"] = 0.0; s["prev_state"] = test_x
                c.log.append((round(c.frac, 3), round(s["acc"] if not skip else s["acc"], 5), round(thr, 5), int(skip)))
        if skip:
            rw, pw = s["res_win"], s["pres_win"]
            if len(rw) >= 2:
                ri = test_x - x
                if c.osi:
                    dt = (ri - pw[-2]).float().flatten(); ds = (pw[-1] - pw[-2]).float().flatten()
                    g = float(torch.dot(dt, ds) / (torch.dot(ds, ds) + 1e-6)); g = min(max(g, 0.0), 2.0)
                else:
                    num = float((ri - pw[-2]).abs().mean()); den = float((pw[-1] - pw[-2]).abs().mean()); g = 1.0 if den < 1e-6 else min(max(num / den, 1.0), 2.0)
                x = x + (rw[-2] + g * (rw[-1] - rw[-2])).to(x.dtype)
            else: x = x + s["res_cache"].to(x.dtype)
            s["prev_state"] = test_x; s["prev_in"] = ori; c.n_skip += 1
        else:
            if s["resume"] and test_x is not None: x = test_x; blocks = self.inner[c.k:]
            else: blocks = self.inner
            for i, b in enumerate(blocks):
                x = b(x, **kw)
                real = i + (c.k if (s["resume"] and test_x is not None) else 0)
                if real == c.k - 1: s["prev_state"] = (test_x if (c.frac >= c.ret and test_x is not None) else x)
            res = x - ori; s["res_cache"] = res; pres = s["prev_state"] - ori; s["prev_in"] = ori; c.n_full += 1
            if len(s["res_win"]) <= 2: s["res_win"].append(res.to(torch.bfloat16)); s["pres_win"].append(pres.to(torch.bfloat16))
            else: s["res_win"][-2] = s["res_win"][-1]; s["res_win"][-1] = res.to(torch.bfloat16); s["pres_win"][-2] = s["pres_win"][-1]; s["pres_win"][-1] = pres.to(torch.bfloat16)
        s["prev_out"] = x
        return x


class StepCounter:
    """공식 구현처럼 스텝 인덱스 기반 진행률 (cnt / num_steps). MoE 두 expert 가 공유 — 한 스텝 = n_streams 회 forward."""
    def __init__(self, total_steps, n_streams=2): self.n = 0; self.total = total_steps; self.k = n_streams
    def frac(self): return min((self.n // self.k) / max(self.total, 1), 1.0)


def install_official(model, k=3, thresh=0.06, ret_ratio=0.2, alpha=0.2, beta_s=0.12, osi=True, decay=True, aduc=False, aduc_start=0.5, n_streams=2, t_max=1000.0, step_counter=None):   # 공식 WAN2.1 i2v-14B 권장 플래그: motion_sensitivity 0.2, osi, saliency, dynamic_decay (AdUC/flow 미사용)
    if isinstance(model.blocks, (CachedBlocks, OfficialCachedBlocks)): return model.blocks.ctl
    ctl = _OCtl(k, thresh, ret_ratio, alpha, beta_s, osi, decay, aduc, aduc_start, n_streams)
    model.blocks = OfficialCachedBlocks(model.blocks, ctl)

    def pre(mod, args, kwargs):
        if step_counter is not None:
            ctl.frac = step_counter.frac(); step_counter.n += 1          # 공식: cnt / num_steps (스텝 인덱스)
        else:
            t = kwargs.get("t", args[1] if len(args) > 1 else None)
            if t is not None: ctl.frac = 1.0 - float(t.flatten()[0]) / t_max   # 타임스텝 진행률 (이전 이식판)
    model.register_forward_pre_hook(pre, with_kwargs=True); model._worldcache = ctl
    return ctl


_stats_old = stats
def stats(model):
    c = getattr(model, "_worldcache", None)
    if c is None: return None
    tot = c.n_full + c.n_skip + getattr(c, "n_aduc", 0)
    return dict(k=c.k, full=c.n_full, skip=c.n_skip, aduc=getattr(c, "n_aduc", 0), skip_rate=round((c.n_skip + getattr(c, "n_aduc", 0)) / max(tot, 1), 3))
