import os
p = os.environ["SCRATCH"]+"/SolarWM/src/solarwm/backends/wan22/runtime/modeling/attention.py"
s = open(p).read()
anchor = '''    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype
'''
assert anchor in s
fallback = anchor + '''
    # [hskim99 patch] GH200/aarch64: no flash-attn wheel. Exact SDPA fallback with the
    # same semantics (padding masks from q_lens/k_lens, causal flag). window_size is
    # only ever (-1, -1) at the call sites in this repo.
    if not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
        assert tuple(window_size) == (-1, -1), "SDPA fallback does not implement sliding window"
        qh = q.to(dtype).transpose(1, 2)            # [B, Nq, Lq, C]
        kh = k.to(dtype).transpose(1, 2)            # [B, Nk, Lk, C]
        vh = v.to(dtype).transpose(1, 2)
        if kh.size(1) != qh.size(1):                # GQA: repeat kv heads
            rep = qh.size(1) // kh.size(1)
            kh = kh.repeat_interleave(rep, dim=1); vh = vh.repeat_interleave(rep, dim=1)
        if q_scale is not None:
            qh = qh * q_scale
        attn_mask = None
        if k_lens is not None or q_lens is not None:
            kl = k_lens if k_lens is not None else torch.full((b,), lk, device=q.device)
            ql = q_lens if q_lens is not None else torch.full((b,), lq, device=q.device)
            kmask = torch.arange(lk, device=q.device)[None, :] < kl.to(q.device)[:, None]   # [B, Lk]
            qmask = torch.arange(lq, device=q.device)[None, :] < ql.to(q.device)[:, None]   # [B, Lq]
            attn_mask = (qmask[:, :, None] & kmask[:, None, :])[:, None]                    # [B,1,Lq,Lk]
            if causal:
                attn_mask = attn_mask & torch.ones(lq, lk, dtype=torch.bool, device=q.device).tril()[None, None]
            # fully-masked query rows (padded queries) would give NaN; allow them to attend to key 0
            dead = ~attn_mask.any(dim=-1, keepdim=True)
            attn_mask = attn_mask | (dead & (torch.arange(lk, device=q.device) == 0)[None, None, None, :])
        out = torch.nn.functional.scaled_dot_product_attention(
            qh, kh, vh, attn_mask=attn_mask, is_causal=(causal and attn_mask is None),
            dropout_p=dropout_p, scale=softmax_scale)
        return out.transpose(1, 2).contiguous().type(out_dtype)   # [B, Lq, Nq, C]
'''
s = s.replace(anchor, fallback)
open(p, "w").write(s)
import py_compile; py_compile.compile(p, doraise=True); print("attention.py 패치 + 컴파일 OK")
