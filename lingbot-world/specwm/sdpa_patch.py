"""Monkeypatch wan.modules.attention.flash_attention -> PyTorch SDPA.
Activate BEFORE importing wan models:  import specwm.sdpa_patch as _p; _p.apply()
Needed when flash-attn is unavailable (e.g., quick venv on GH200 without
compiling flash-attn). Handles the b=1 fixed-length path used in inference."""
import torch
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


def _sdpa_flash_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
                          softmax_scale=None, q_scale=None, causal=False,
                          window_size=(-1, -1), deterministic=False,
                          dtype=torch.bfloat16, version=None):
    # inputs: [B, L, N, D] (wan convention) -> SDPA wants [B, N, L, D]
    half = lambda x: x if x.dtype in (torch.float16, torch.bfloat16) else x.to(dtype)
    q, k, v = half(q), half(k), half(v)
    if q_scale is not None:
        q = q * q_scale
    qq, kk, vv = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    mask = _causal_mask(q.shape[1], q.device) if q.shape[1] == k.shape[1] else None
    if mask is not None:
        assert not causal
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                                             torch.nn.attention.SDPBackend.MATH]):
            out = torch.nn.functional.scaled_dot_product_attention(
                qq, kk, vv, attn_mask=mask, dropout_p=dropout_p, scale=softmax_scale)
        return out.transpose(1, 2).contiguous()
    # Force a memory-efficient backend: the default math backend materializes
    # the full [B,H,Lq,Lk] score matrix, which dominates activation memory for
    # long KV windows (this is what caused OOM at local_attn>=12).
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION,
                          SDPBackend.EFFICIENT_ATTENTION,
                          SDPBackend.MATH]):
            out = torch.nn.functional.scaled_dot_product_attention(
                qq, kk, vv, is_causal=causal, dropout_p=dropout_p,
                scale=softmax_scale)
    except Exception:
        with torch.backends.cuda.sdp_kernel(enable_flash=True,
                                            enable_mem_efficient=True,
                                            enable_math=False):
            out = torch.nn.functional.scaled_dot_product_attention(
                qq, kk, vv, is_causal=causal, dropout_p=dropout_p,
                scale=softmax_scale)
    return out.transpose(1, 2).contiguous()


def apply():
    import sys
    import wan.modules.attention as A
    A.flash_attention = _sdpa_flash_attention
    # rebind stale references in already-imported wan.* modules
    # (model_fast/model_causal do `from .attention import flash_attention`)
    n = 0
    for name, mod in list(sys.modules.items()):
        if name.startswith("wan") and mod is not None \
                and getattr(mod, "flash_attention", None) is not None \
                and mod is not A:
            mod.flash_attention = _sdpa_flash_attention
            n += 1
    print(f"[sdpa_patch] rebound {n} stale flash_attention refs")
    if hasattr(A, "attention"):
        # wan.modules.attention.attention dispatches to flash_attention; keep it.
        pass
    print("[sdpa_patch] wan.modules.attention.flash_attention -> SDPA fallback")
