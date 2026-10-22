import os
# 패치 검증: 작은 텐서로 SDPA 폴백이 수학적으로 맞는지 (마스크 없는 경우 = 표준 어텐션, k_lens 패딩 케이스)
import torch, sys
sys.path.insert(0, os.environ["SCRATCH"]+"/SolarWM/src")
from solarwm.backends.wan22.runtime.modeling import attention as A
assert not (A.FLASH_ATTN_2_AVAILABLE or A.FLASH_ATTN_3_AVAILABLE), "flash 가 있으면 이 테스트는 폴백을 안 탐"
dev = "cuda"; torch.manual_seed(0)
B, Lq, Lk, H, C = 2, 7, 11, 4, 32
q = torch.randn(B, Lq, H, C, device=dev, dtype=torch.bfloat16); k = torch.randn(B, Lk, H, C, device=dev, dtype=torch.bfloat16); v = torch.randn(B, Lk, H, C, device=dev, dtype=torch.bfloat16)
def ref(q, k, v, klens=None, causal=False):
    qf, kf, vf = q.float(), k.float(), v.float(); out = torch.zeros_like(qf)
    for bi in range(B):
        L = int(klens[bi]) if klens is not None else Lk
        s = torch.einsum("qhc,khc->hqk", qf[bi], kf[bi, :L]) / C ** 0.5
        if causal: s = s.masked_fill(~torch.ones(Lq, L, dtype=torch.bool, device=dev).tril(), float("-inf"))
        out[bi] = torch.einsum("hqk,khc->qhc", s.softmax(-1), vf[bi, :L])
    return out
o1 = A.flash_attention(q, k, v).float(); r1 = ref(q, k, v)
print("no-mask  max|diff| = %.2e" % (o1 - r1).abs().max())
kl = torch.tensor([11, 5], dtype=torch.int32, device=dev)
o2 = A.flash_attention(q, k, v, k_lens=kl).float(); r2 = ref(q, k, v, kl)
print("k_lens   max|diff| = %.2e" % (o2 - r2).abs().max())
o3 = A.flash_attention(q, k[:, :Lq], v[:, :Lq], causal=True).float(); r3 = ref(q, k[:, :Lq], v[:, :Lq], causal=True)
print("causal   max|diff| = %.2e" % (o3 - r3).abs().max())
ok = all(((a-b).abs().max() < 2e-2) for a, b in [(o1, r1), (o2, r2), (o3, r3)])
print("[ATTN_TEST_%s]" % ("PASS" if ok else "FAIL"))
