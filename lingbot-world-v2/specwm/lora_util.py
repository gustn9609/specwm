"""[10/16] LoRA helpers shared by the trainers (specwm/lora14_train.py) and the runtime (specwm/two_machine.py --student / --corr).
attach(model, rank, ffn, dev) wraps q,k,v,o[,ffn0,ffn2] of every block with a gated LoRA (GATE tensor: 1 = active, 0 = base only) and returns
(LOR list in block order, GATE). save/load use {rank, ffn, A: [..], B: [..], skip: [...], mode} with A/B in the same order."""
import math, torch, torch.nn as nn
class LoRA(nn.Module):
    def __init__(s, base, r, gate, dev):
        super().__init__(); fi = getattr(base, "in_features", None) or base.inf; fo = getattr(base, "out_features", None) or base.outf   # nn.Linear or fast_patch.FP8Linear
        s.base = base; s.gate = gate; s.A = nn.Parameter(torch.randn(r, fi, device=dev) / math.sqrt(fi)); s.B = nn.Parameter(torch.zeros(fo, r, device=dev))
    def forward(s, x): return s.base(x) + (s.gate * ((x.float() @ s.A.t()) @ s.B.t())).to(x.dtype)
def attach(model, rank, ffn, dev, gate=None):
    gate = gate if gate is not None else torch.ones((), device=dev); LOR = []
    for blk in model.blocks:
        for nm in ("q", "k", "v", "o"): l = LoRA(getattr(blk.self_attn, nm), rank, gate, dev); setattr(blk.self_attn, nm, l); LOR.append(l)
        if ffn:
            for i in (0, 2): l = LoRA(blk.ffn[i], rank, gate, dev); blk.ffn[i] = l; LOR.append(l)
    return LOR, gate
def state(LOR, **extra): return dict(A=[l.A.detach().cpu() for l in LOR], B=[l.B.detach().cpu() for l in LOR], **extra)
def load(model, ck, dev, gate=None):
    """ck = torch.load(...) dict; returns (LOR, GATE); applies ck['skip'] (identity blocks) when present"""
    for i in ck.get("skip", []): model.blocks[i].forward = (lambda x, **kw: x)
    LOR, gate = attach(model, ck["rank"], ck["ffn"], dev, gate)
    for l, A, Bm in zip(LOR, ck["A"], ck["B"]): l.A.data.copy_(A.to(dev)); l.B.data.copy_(Bm.to(dev))
    return LOR, gate
