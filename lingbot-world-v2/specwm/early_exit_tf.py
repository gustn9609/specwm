"""Training-free verifier for token-wise early exit (Speculative Forcing, arXiv 2610.22347 Sec. 3.3.1 / App. A) on the LingBot causal-fast models,
as a subclass of early_exit.EarlyExit (early_exit.py is not modified; used by spec_early_tf.py --mode tf).

Token definition: one DiT token = the 16 latent channels of a 1x2x2 latent patch = a 64-dim vector, i.e. the chunk's clean estimate x0 [16, F, H, W]
seen as patch(x0) = [F*(H/2)*(W/2), 64] (early_exit.patch). Every distance below is the L2 norm of such 64-dim vectors, divided by
sigma = the median token norm of the current draft x0^(k) over the chunk (norm="draft"; the paper's oracle rel_k divides by the median token norm of
the full-step output, which does not exist at verification time, and the published thresholds 0.375 / 0.300 are of the size of eps = 0.24, i.e.
normalized units; norm="none" = raw latent units, kept for the calibration diagnostic).
  step 1 (k = 1):  s_red(i) = min_{h in H} ||x0^(1)(i) - h||_2 / sigma_1, H = the clean (committed) tokens of the previous chunks that are inside the
                   attention window of chunk c (ring_kv layout: the sink chunks + the local slots -> for sink 6 / window 18 / chunk 3 that is chunks
                   {0, 1, c-3, c-2, c-1}), at ALL spatial positions, plus the reference frame = the VAE latent of the conditioning image (the y input;
                   image_ref=True; it is the only bank entry for chunk 0). Accept if S[s_red](i) <= tau_red.
  steps k = 2..K-1: s_stab(i) = ||x0^(k)(i) - x0^(k-1)(i)||_2 / sigma_k for the active tokens (a frozen token did not change: its score is exactly 0).
                   Accept if S[s_stab](i) <= tau_stab.
  S = 9x9 Gaussian spatial smoothing (sd 1 token, reflect padding, per latent frame) followed by a temporal Gaussian (sd 1 latent frame) over the
      CURRENT AND FOLLOWING latent frames of the chunk (App. A; EarlyExit.smooth is symmetric in time, so it is overridden here).
Accepted tokens are frozen (x0 and the K/V of their acceptance step), only the active tokens are forwarded at later steps (fast_patch.ACTIVE), and
the chunk's clean t=0 commit (done by the caller, B.commit) refreshes all K/V as usual. Published LingBot thresholds: tau_red 0.375, tau_stab 0.300
(calibrated by the authors against rel_k <= 0.24 with 4 latent frames per chunk; we run our 3-frame chunks)."""
import time, torch, torch.nn.functional as F
from specwm import fast_patch as FP
from specwm.early_exit import EarlyExit, patch, unpatch


class EarlyExitTF(EarlyExit):
    def __init__(self, B, tau_red=0.375, tau_stab=0.300, norm="draft", image_ref=True, bucket=128, smooth=True, device="cuda:0", eps=0.24):
        super().__init__(B, eps=eps, head="", bucket=bucket, smooth=smooth, device=device)   # head="" -> no learned head, no hook
        self.tau_red, self.tau_stab, self.norm, self.image_ref = float(tau_red), float(tau_stab), norm, image_ref
        self.hist = {}; self.img_tok = None; self.t_verify = 0.0; self.last = []
        self.sink_f = int(getattr(B.pipe, "sink_size", 6)); self.window_f = int(getattr(B.pipe, "local_attn_size", 18))

    # ---------------- clean token bank ----------------
    def reset(self):
        """new clip: drop the history bank and cache the reference frame (VAE latent of the conditioning image = y[4:, 0] of chunk 0)"""
        self.hist = {}; self.img_tok = None; self.t_verify = 0.0; self.last = []
        if self.image_ref: self.img_tok = patch(self.B._y_chunks[0][4:, :1].float().to(self.dev)).contiguous()   # [(H/2)(W/2), 64]

    def window_chunks(self, c):
        """ids of the previous chunks whose K/V are inside chunk c's attention window (ring_kv: sink chunks + the local slots)"""
        B = self.B; fs = B._frame_seqlen; St = B._chunk_size * fs; sink = self.sink_f * fs; sink_c = sink // St; slots = max(1, (B._kv_size - sink) // St)
        return [j for j in range(c) if j < sink_c or j >= c - slots + 1]

    def bank(self, c):
        parts = [self.hist[j] for j in self.window_chunks(c) if j in self.hist]
        if self.img_tok is not None: parts.append(self.img_tok)
        return torch.cat(parts) if parts else None

    @staticmethod
    def nn_dist(q, bank, rows=2048):
        """min_h ||q_i - h||_2 over the bank, row-chunked cdist (fp32)"""
        out = torch.empty(q.shape[0], device=q.device)
        for s in range(0, q.shape[0], rows): out[s:s + rows] = torch.cdist(q[s:s + rows].float(), bank.float()).amin(dim=1)
        return out

    def sigma(self, tok): return float(tok.norm(dim=1).median()) if self.norm == "draft" else 1.0

    def smooth(self, p):   # p [F, h, w]: 9x9 spatial Gaussian (reflect) per frame, then Gaussian over the current and FOLLOWING frames (sd 1)
        f_ = p.shape[0]; q = F.conv2d(F.pad(p.unsqueeze(1), (4, 4, 4, 4), mode="reflect"), self.G9).squeeze(1)
        wt = torch.zeros(f_, f_, device=p.device)
        for i in range(f_): w = torch.exp(-(torch.arange(0, f_ - i, device=p.device).float() ** 2) / 2.0); wt[i, i:] = w / w.sum()
        return torch.einsum("ij,jhw->ihw", wt, q)

    # ---------------- generation ----------------
    def gen_chunk(self, c, z, g, x0_1=None, masks=None, sparse=True, after=None):
        """denoise chunk c with the training-free verifier; returns (x0 [16,F,H,W], mean steps per token, [#tokens stopped at step 1..K]).
        masks is ignored (kept for the EarlyExit signature). sparse=False: every token is forwarded at every step (diagnostic), frozen x0 kept."""
        B = self.B; sig = [float(s) for s in B.node_sigmas()]; shape = tuple(z.shape); f_, h2, w2 = shape[1], shape[2] // 2, shape[3] // 2; L = f_ * h2 * w2
        x0_tok = torch.zeros(L, 64, device=self.dev); committed = torch.zeros(L, dtype=torch.bool, device=self.dev); step = torch.full((L,), self.K, device=self.dev)
        FP.ACTIVE["a"] = None; self.last = []; bank = self.bank(c); idx = None; n = L; sync = torch.cuda.synchronize if str(self.dev).startswith("cuda") else (lambda: None)
        try:
            for i in range(self.K):
                if i == 0 and x0_1 is not None: x0 = x0_1.float()
                else:
                    if i > 0:
                        idx = (~committed).nonzero().squeeze(1); n = int(idx.numel())
                        if n == 0: break
                        if sparse: pad = (-n) % self.bucket; idx = torch.cat([idx, idx[-1:].expand(pad)]) if pad else idx; FP.ACTIVE["a"] = dict(idx=idx, n=n, L=L)
                    x0 = B.target_x0(z, i, c).float()
                if after is not None: after(i)
                xt = patch(x0)
                if i < self.K - 1: sync(); tv = time.time()
                if i == 0: s_raw = self.nn_dist(xt, bank) if bank is not None else torch.full((L,), float("inf"), device=self.dev); x0_tok = xt
                else:
                    act = idx[:n]; s_raw = torch.zeros(L, device=self.dev); s_raw[act] = (xt[act] - x0_tok[act]).norm(dim=1); x0_tok[act] = xt[act]
                if i < self.K - 1:
                    sg = self.sigma(x0_tok); s = s_raw / sg; sm = self.smooth(s.view(f_, h2, w2)).flatten() if self.do_smooth else s
                    tau = self.tau_red if i == 0 else self.tau_stab; dec = sm <= tau
                    acc = dec & ~committed; step[acc] = i + 1; committed |= acc
                    sync(); self.t_verify += time.time() - tv
                    self.last.append(dict(k=i + 1, sigma=sg, raw=s_raw.detach(), smooth=sm.detach(), acc=acc.detach(), n_active=n))
                    x0_full = unpatch(x0_tok, shape); s_ = sig[i + 1]; z = (1 - s_) * x0_full + s_ * B.randn_like(x0_full, g)
        finally: FP.ACTIVE["a"] = None
        self.hist[c] = x0_tok.clone()
        for j in [j for j in self.hist if j not in self.window_chunks(c + 1)]: del self.hist[j]   # keep only what the next chunk's window can see
        return unpatch(x0_tok, shape), float(step.float().mean()), [int((step == k).sum()) for k in range(1, self.K + 1)]

    # ---------------- calibration diagnostic ----------------
    @torch.no_grad()
    def calib_scores(self, c, z, g):
        """dense 4-step trajectory of chunk c (consumes the generator like gen_chunk; caller restores it): oracle gap rel_k = ||x0^(k) - x0^(K)|| / sigma_K
        and the training-free scores computed on the dense drafts (nothing frozen), smoothed, for both normalizations. Returns (xs, dict of CPU tensors):
        gap [K-1, L], red_draft / red_none [L], stab_draft / stab_none [K-2, L] (steps 2..K-1), sigma_draft [K-1], sigma_K."""
        xs = self.dense_trajectory(c, z, g); shape = tuple(xs[0].shape); f_, h2, w2 = shape[1], shape[2] // 2, shape[3] // 2
        tK = patch(xs[-1]); sK = float(tK.norm(dim=1).median()); toks = [patch(x) for x in xs]
        gap = torch.stack([(toks[k] - tK).norm(dim=1) / sK for k in range(self.K - 1)])
        bank = self.bank(c); red = self.nn_dist(toks[0], bank) if bank is not None else torch.full((toks[0].shape[0],), float("inf"), device=self.dev)
        stab = torch.stack([(toks[k] - toks[k - 1]).norm(dim=1) for k in range(1, self.K - 1)])
        sd = [float(t.norm(dim=1).median()) for t in toks[:self.K - 1]]; S = lambda v: self.smooth(v.view(f_, h2, w2)).flatten()
        out = dict(gap=gap.cpu(), sigma_draft=torch.tensor(sd), sigma_K=sK, red_none=S(red).cpu(), red_draft=S(red / sd[0]).cpu(),
                   stab_none=torch.stack([S(stab[j]) for j in range(stab.shape[0])]).cpu(), stab_draft=torch.stack([S(stab[j] / sd[j + 1]) for j in range(stab.shape[0])]).cpu())
        return xs, out
