"""Token-wise early exit for the 14B causal-fast (Speculative Forcing with our learned head), as a reusable module (used by spec_early.py and
spec_pipe.py). EarlyExit(B, eps) wraps a V2Binding whose target runs the 4-step schedule (timesteps_index (0, 250, 500, 750)) with fast_patch
(dedup_t + no_kv_cat + ring_kv). gen_chunk(c, z, g [, x0_1]) denoises one chunk: after step k < 4 the head predicts each token's gap to the 4-step
output from the last block's hidden state (standardized, random-projected, 2x2 pooled, 9x9 spatial + temporal smoothing); tokens with prediction
<= eps are committed (x0 frozen, K/V kept) and only the rest are forwarded (fast_patch.ACTIVE). If x0_1 is given, step 1 was already run by the
caller (e.g. the agree verifier) and its hidden state is already in the buffer. Returns (x0, mean_steps, hist)."""
import os, torch, torch.nn.functional as F
from specwm import fast_patch as FP
S = "/home/hk28965/specwm"
class Head(torch.nn.Module):
    def __init__(s, cin=1536):
        super().__init__(); s.ln = torch.nn.LayerNorm(cin); s.c1 = torch.nn.Conv3d(cin, 96, 3, padding=1); s.c2 = torch.nn.Conv3d(96, 48, 3, padding=1); s.c3 = torch.nn.Conv3d(48, 1, 1)
    def forward(s, h): x = s.ln(h.float()).permute(0, 4, 1, 2, 3); x = F.gelu(s.c1(x)); x = F.gelu(s.c2(x)); return s.c3(x).squeeze(1)
def gauss1d(n, sd): x = torch.arange(n, dtype=torch.float32) - n // 2; g = torch.exp(-x ** 2 / (2 * sd ** 2)); return g / g.sum()
def patch(x):   # [16, F, H, W] -> [F*(H/2)*(W/2), 64]
    c, f, h, w = x.shape; return x.view(c, f, 1, h // 2, 2, w // 2, 2).permute(1, 3, 5, 2, 4, 6, 0).reshape(f * (h // 2) * (w // 2), 64)
def unpatch(t, shape):
    c, f, h, w = shape; return t.view(f, h // 2, w // 2, 1, 2, 2, c).permute(6, 0, 3, 1, 4, 2, 5).reshape(c, f, h, w)
class EarlyExit:
    def __init__(self, B, eps=0.24, head=S + "/lingbot-world-v2/specwm_out/head14b_v3/head.pt", data=S + "/lingbot-world-v2/specwm_out/gap_dump_v3",
                 norm=S + "/lingbot-world-v2/specwm_out/chan_norm_14b.pt", bucket=128, smooth=True, device="cuda:0", min_exit=1, refresh_last=False, chunk_frac=0.0):
        self.B = B; self.eps = eps; self.min_exit = min_exit; self.refresh_last = refresh_last; self.chunk_frac = chunk_frac   # chunk_frac > 0: chunk-wise stop (all tokens) once >= this fraction of tokens is predicted converged (no K/V freezing)
        self.bucket = bucket; self.do_smooth = smooth; self.dev = device; self.K = len(B.node_sigmas()); assert self.K == 4, "early exit expects the 4-step schedule"
        self.head = None
        if head and os.path.exists(head):   # oracle-only diagnostics need no head
            self.head = Head().to(device).eval(); self.head.load_state_dict(torch.load(head, map_location=device)["state"])
            self.proj = torch.load(data + "/proj.pt").to(device, torch.bfloat16); n = torch.load(norm); self.mu, self.sd = n["mu"].to(device), n["sd"].to(device)
        self.G9 = (gauss1d(9, 1.0)[:, None] * gauss1d(9, 1.0)[None, :]).to(device).view(1, 1, 9, 9); self.hid = None
        if self.head is not None: B.target_model.blocks[-1].register_forward_hook(self._hook)
    def _hook(self, m, inp, out):
        h = out[0] if isinstance(out, tuple) else out; act = FP.ACTIVE["a"]; f = (((h[0].float() - self.mu) / self.sd).to(torch.bfloat16) @ self.proj)
        if act is None: self.hid = f
        else: self.hid[act["idx"][:act["n"]]] = f[:act["n"]]
    def smooth(self, p):
        f_ = p.shape[0]; q = F.conv2d(F.pad(p.unsqueeze(1), (4, 4, 4, 4), mode="reflect"), self.G9).squeeze(1)
        wt = torch.stack([gauss1d(2 * f_ - 1, 1.0)[f_ - 1 - i:2 * f_ - 1 - i] for i in range(f_)]).to(self.dev); wt = wt / wt.sum(1, keepdim=True)
        return torch.einsum("ij,jhw->ihw", wt, q)
    def predict(self, f_, h2, w2):
        hp = F.avg_pool2d(self.hid.view(f_, h2, w2, -1).permute(0, 3, 1, 2).float(), 2, ceil_mode=True).permute(0, 2, 3, 1)
        pred = self.head(hp.unsqueeze(0))[0]; pred = F.interpolate(pred.unsqueeze(0), size=(h2, w2), mode="nearest")[0]
        return self.smooth(pred) if self.do_smooth else pred
    @torch.no_grad()
    @torch.no_grad()
    def dense_trajectory(self, c, z, g):
        """all tokens, all K steps (no freezing): returns the per-step x0 list; consumes the generator like gen_chunk does"""
        B = self.B; sig = [float(s) for s in B.node_sigmas()]; xs = []
        for i in range(self.K):
            x = B.target_x0(z, i, c).float(); xs.append(x)
            if i < self.K - 1: s_ = sig[i + 1]; z = (1 - s_) * x + s_ * B.randn_like(x, g)
        return xs
    def oracle_masks(self, xs):
        """from a dense trajectory: per step k < K the tokens whose gap to the final output is <= eps (the oracle stop rule)"""
        tK = patch(xs[-1]); sg = float(tK.norm(dim=1).median()); return [((patch(xs[k]) - tK).norm(dim=1) / sg) <= self.eps for k in range(self.K - 1)]
    def gen_chunk(self, c, z, g, x0_1=None, masks=None, sparse=True, after=None):
        """masks: None = learned head decides; else a list (per step k < K) of bool tensors [L] = tokens allowed to stop at that step (oracle).
        sparse: True = only active tokens are forwarded (K/V of committed tokens frozen); False = diagnostic: every token is forwarded at every
        step (K/V recomputed as in dense sampling) but committed tokens keep their frozen x0."""
        B = self.B; sig = [float(s) for s in B.node_sigmas()]; shape = tuple(z.shape); f_, h2, w2 = shape[1], shape[2] // 2, shape[3] // 2; L = f_ * h2 * w2
        x0_tok = torch.zeros(L, 64, device=self.dev); committed = torch.zeros(L, dtype=torch.bool, device=self.dev); step = torch.full((L,), self.K, device=self.dev); FP.ACTIVE["a"] = None
        try:
            for i in range(self.K):
                if i == 0 and x0_1 is not None: x0 = x0_1.float()
                else:
                    if i > 0:
                        idx = (~committed).nonzero().squeeze(1); n = int(idx.numel())
                        if n == 0: break
                        if sparse and not (self.refresh_last and i == self.K - 1): pad = (-n) % self.bucket; idx = torch.cat([idx, idx[-1:].expand(pad)]) if pad else idx; FP.ACTIVE["a"] = dict(idx=idx, n=n, L=L)
                    x0 = B.target_x0(z, i, c).float()
                if after is not None: after(i)   # pipeline hook (commit offload: pull the clean K/V after step 1)
                xt = patch(x0)
                if i == 0: x0_tok = xt
                else: x0_tok[idx[:n]] = xt[idx[:n]]
                if i < self.K - 1:
                    if i + 1 >= self.min_exit:   # min_exit: no token may stop before this step (stale-K/V diagnostic)
                        dec = masks[i] if masks is not None else (self.predict(f_, h2, w2).flatten() <= self.eps)
                        if self.chunk_frac > 0: dec = torch.full_like(dec, bool(dec.float().mean() >= self.chunk_frac))   # all-or-nothing per chunk
                        acc = dec & ~committed; step[acc] = i + 1; committed |= acc
                    x0_full = unpatch(x0_tok, shape); s_ = sig[i + 1]; z = (1 - s_) * x0_full + s_ * B.randn_like(x0_full, g)
        finally: FP.ACTIVE["a"] = None
        return unpatch(x0_tok, shape), float(step.float().mean()), [int((step == k).sum()) for k in range(1, self.K + 1)]
