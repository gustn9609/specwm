"""v1 sdpa_patch 에 chunk-causal 마스크 옵션을 추가한다 (E6).
환경변수 SPECWM_CAUSAL_FRAMES=N (잠재프레임 단위 창), SPECWM_LAT_F=잠재프레임 수.
self-attention(q,k 길이 동일 = lat_f * frame_tokens) 에만 적용. cross-attn 은 건드리지 않는다.
청크(3잠재프레임) 단위: 쿼리 청크 c 는 키 청크 [c-N/3+1, c] 만 본다 (청크 내부는 전체).
"""
p = os.environ["SCRATCH"]+"/lingbot-world/specwm/sdpa_patch.py"
s = open(p).read()
if "SPECWM_CAUSAL_FRAMES" not in s:
    s = s.replace('''import torch
''', '''import torch
import os

_MASK_CACHE = {}


def _causal_mask(L, device):
    """[hskim99 E6] chunk-causal mask for Base self-attention, or None."""
    N = int(os.environ.get("SPECWM_CAUSAL_FRAMES", "0") or 0)
    lat_f = int(os.environ.get("SPECWM_LAT_F", "0") or 0)
    if N <= 0 or lat_f <= 0 or L % lat_f != 0:
        return None
    key = (L, N, lat_f, str(device))
    if key not in _MASK_CACHE:
        ft = L // lat_f
        f = torch.arange(L, device=device) // ft          # latent frame index per token
        c = f // 3                                        # chunk index
        nc = max(1, N // 3)                               # chunks visible incl. own
        cq = c[:, None]; ck = c[None, :]
        allowed = (ck <= cq) & (ck >= cq - (nc - 1))
        _MASK_CACHE[key] = allowed
        print("[sdpa_patch] chunk-causal mask: L=%d lat_f=%d N=%d -> %d chunks visible" % (L, lat_f, N, nc), flush=True)
    return _MASK_CACHE[key]
''', 1)
    s = s.replace('''    qq, kk, vv = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
''', '''    qq, kk, vv = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    mask = _causal_mask(q.shape[1], q.device) if q.shape[1] == k.shape[1] else None
    if mask is not None:
        assert not causal
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                                             torch.nn.attention.SDPBackend.MATH]):
            out = torch.nn.functional.scaled_dot_product_attention(
                qq, kk, vv, attn_mask=mask, dropout_p=dropout_p, scale=softmax_scale)
        return out.transpose(1, 2).contiguous()
''', 1)
    open(p, "w").write(s)
import py_compile; py_compile.compile(p, doraise=True); print("sdpa_patch causal 옵션 추가 + 컴파일 OK")
