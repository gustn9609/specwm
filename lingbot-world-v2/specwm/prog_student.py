"""[10/22] Loader + sampler for the PROGRESSIVE 2-step student trained by specwm/prog_dmd14.py (half-res step 1 with its own pooled KV, full-res step 2).
load(tm, ck, dev)        -> attaches the 'lo' / 'hi' LoRAs (gated: set_step(lo=True/False) before each forward) or loads the full fine-tuned DiT state
ProgSampler(binding, ck) -> prepare(task) builds the HI / LO binding states (multires_pilot mechanics); chunk(c, g) runs step 1 at the half grid,
                            up-samples, re-noises to node 2 and runs step 2 at the full grid; commit(x, c) commits to both KVs.
                            hidden(layer) returns the layer-20 hidden of the LAST half-res pass (the verifier input, step 2 of the plan).
Used by specwm/prog_eval.py (solo rollouts for the frozen sets) and, later, by two_machine --student-prog (device role)."""
import os, math, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
from specwm import lora_util as LU
class _Gated(nn.Module):
    def __init__(s, base, r, g_lo, g_hi, dev):
        super().__init__(); fi, fo = base.in_features, base.out_features; s.base = base; s.g_lo = g_lo; s.g_hi = g_hi
        s.A = nn.Parameter(torch.zeros(r, fi, device=dev), requires_grad=False); s.B = nn.Parameter(torch.zeros(fo, r, device=dev), requires_grad=False)
        s.A3 = nn.Parameter(torch.zeros(r, fi, device=dev), requires_grad=False); s.B3 = nn.Parameter(torch.zeros(fo, r, device=dev), requires_grad=False)
    def forward(s, x):
        xf = x.float(); return s.base(x) + (s.g_lo * ((xf @ s.A.t()) @ s.B.t()) + s.g_hi * ((xf @ s.A3.t()) @ s.B3.t())).to(x.dtype)
def load(tm, ck, dev):
    """ck = torch.load(student_*.pt). Returns (gates (g_lo, g_hi) or None, info dict)."""
    if ck["adapt"] == "lora":
        g_lo = torch.zeros((), device=dev); g_hi = torch.ones((), device=dev); LOR = []
        for blk in tm.blocks:
            for nm in ("q", "k", "v", "o"): l = _Gated(getattr(blk.self_attn, nm), ck["rank"], g_lo, g_hi, dev); setattr(blk.self_attn, nm, l); LOR.append(l)
            if ck.get("ffn", 1):
                for i in (0, 2): l = _Gated(blk.ffn[i], ck["rank"], g_lo, g_hi, dev); blk.ffn[i] = l; LOR.append(l)
        for l, A_, B_, A3_, B3_ in zip(LOR, ck["lo"]["A"], ck["lo"]["B"], ck["hi"]["A"], ck["hi"]["B"]): l.A.data.copy_(A_.to(dev)); l.B.data.copy_(B_.to(dev)); l.A3.data.copy_(A3_.to(dev)); l.B3.data.copy_(B3_.to(dev))
        return (g_lo, g_hi), dict(adapt="lora", rank=ck["rank"], n_layers=len(LOR), step=ck.get("step"))
    sd = {k_.replace(".base.", "."): v for k_, v in ck["state"].items()}   # TriLoRA wrappers of the trainer -> plain linears
    missing, unexpected = tm.load_state_dict(sd, strict=False); return None, dict(adapt="full", step=ck.get("step"), missing=len(missing), unexpected=len(unexpected))
def set_step(gates, lo):
    if gates is not None: gates[0].fill_(1.0 if lo else 0.0); gates[1].fill_(0.0 if lo else 1.0)
STATE_KEYS = ["_y_chunks", "_emb_chunks", "_ctx", "_seq_len", "_chunk_shape", "_frame_seqlen", "_chunk_size", "_kv_size", "_mk_cache", "_caches", "_cross_init", "_committed_frames"]
def area_for(image, lh, lw):
    from PIL import Image; im = Image.open(image); asp = im.height / im.width; nom = 64 * lh * lw
    for A in range(int(nom * 0.6), int(nom * 1.7), 32):
        if (round(np.sqrt(A * asp) // 8 // 2 * 2), round(np.sqrt(A / asp) // 8 // 2 * 2)) == (lh, lw): return A
    raise RuntimeError("no max_area gives grid %dx%d for aspect %.4f" % (lh, lw, asp))
class ProgSampler:
    def __init__(s, b, gates, prog=1, hook_layer=20):
        s.b = b; s.gates = gates; s.prog = prog; s.SIG = [float(v) for v in b.node_sigmas()]; s.HI = s.LO = None; s.HW = s.LW = None; s.hid = {}
        def _h(m_, i, o): s.hid["h"] = (o[0] if isinstance(o, tuple) else o)
        b.target_model.blocks[hook_layer - 1].register_forward_hook(_h)
    def _snap(s): return {k: getattr(s.b, k) for k in STATE_KEYS}
    def _use(s, st):
        for k, v in st.items(): setattr(s.b, k, v)
    def prepare(s, image, prompt, action, frames, area):
        b = s.b; b._caches = {}; torch.cuda.empty_cache()
        cond = b.prepare(image, prompt, action, frame_num=frames, max_area=area, chunk_size=3); s.HI = s._snap(); s.HI["_caches"] = None; s.HW = tuple(cond["shape"][2:]); N = cond["num_chunks"]
        if s.prog:
            s.LW = (round(s.HW[0] / 2 / 2) * 2, round(s.HW[1] / 2 / 2) * 2); b._caches = {}
            cond2 = b.prepare(image, prompt, action, frame_num=frames, max_area=area_for(image, *s.LW), chunk_size=3); assert tuple(cond2["shape"][2:]) == s.LW and cond2["num_chunks"] == N
            s.LO = s._snap(); s.LO["_caches"] = None; s.PH, s.PW = s._pool(s.HW[0], s.LW[0]), s._pool(s.HW[1], s.LW[1])
        s.reset(); return N
    def _pool(s, Hf, hl):
        bb = np.round(np.linspace(0, Hf, hl + 1)).astype(int); P = torch.zeros(hl, Hf)
        for i in range(hl): P[i, bb[i]:bb[i + 1]] = 1.0 / np.sqrt(bb[i + 1] - bb[i])
        return P.to(s.b.device)
    def reset(s):
        for st in ((s.HI, s.LO) if s.prog else (s.HI,)): st["_caches"] = None
        for st in ((s.HI, s.LO) if s.prog else (s.HI,)): s._use(st); s.b.reset_caches(); st.update(s._snap())
    @staticmethod
    def up(x, hw): return Fn.interpolate(x.permute(1, 0, 2, 3), size=tuple(hw), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
    @staticmethod
    def down(x, hw): return Fn.interpolate(x.permute(1, 0, 2, 3), size=tuple(hw), mode="bilinear", align_corners=False, antialias=True).permute(1, 0, 2, 3)
    @torch.no_grad()
    def chunk(s, c, g, timing=None):
        """progressive 2-step chunk: returns x0 [16,3,h,w] at the full grid; timing dict gets lo_s / hi_s (CUDA-synced)"""
        b = s.b; e0 = torch.randn(s.HI["_chunk_shape"], generator=g).to(b.device); t_ = torch.cuda.synchronize
        if s.prog:
            s._use(s.LO); set_step(s.gates, True); z = torch.einsum("ah,cthw,bw->ctab", s.PH, e0.float(), s.PW); t_(); t0 = __import__("time").time()
            x_lo = b.target_x0(z, 0, c).float(); t_(); s.hid["lo"] = s.hid.get("h"); x_full = s.up(x_lo, s.HW)
            if timing is not None: timing["lo_s"] = __import__("time").time() - t0
        else:
            s._use(s.HI); set_step(s.gates, False); t_(); t0 = __import__("time").time(); x_full = b.target_x0(e0, 0, c).float(); t_()
            if timing is not None: timing["lo_s"] = __import__("time").time() - t0
        s._use(s.HI); set_step(s.gates, False); sg = s.SIG[1]; z = (1 - sg) * x_full + sg * torch.randn(x_full.shape, generator=g).to(b.device)
        t_(); t0 = __import__("time").time(); x = b.target_x0(z, 1, c).float(); t_()
        if timing is not None: timing["hi_s"] = __import__("time").time() - t0
        return x
    @torch.no_grad()
    def commit(s, x, c):
        s._use(s.HI); set_step(s.gates, False); s.b.commit(x, c)
        if s.prog: s._use(s.LO); set_step(s.gates, True); s.b.commit(s.down(x, s.LW), c)
    def hidden_lo(s): return s.hid.get("lo")   # layer-20 output of the last half-res pass [1, tokens, C] -> verifier input
