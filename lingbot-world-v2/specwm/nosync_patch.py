"""Remove the .item() host syncs from the local-attention KV cache path.

WHY. Kernel profiling showed elementwise ops are 33.9% of GPU time -- the
largest fusable target. torch.compile could not touch them: it produced 41
graph breaks, ALL of them from this path:

    24x  Unsupported Tensor.item() call with capture_scalar_outputs=False
    17x  Data dependent operator: aten._local_scalar_dense.default

The cache indices are read back with .item() ~630x per chunk. Each read is a
host sync AND a hard graph break, fragmenting exactly the region we want fused.

The values are deterministic given current_start and the eviction rule, so we
mirror them as Python ints on the host. The upstream code already does this in
the local_attn_size == -1 branch, where the authors noted "All Python ints --
no .item() syncs"; the local-attention branch simply never got the same
treatment.

The GPU tensors are still written (with async fill_) so any other reader keeps
working. Output must be bit-identical -- the caller verifies this.
"""
from __future__ import annotations
import math
import torch


def _host_idx(kv_cache, key, mirror):
    """Python-int view of a cache index, seeded once from the tensor."""
    k = f"_host_{key}"
    if k not in mirror:
        mirror[k] = int(kv_cache[key].item())      # one sync per cache, not per call
    return mirror[k]


def apply(model, verbose=True):
    """Patch every CausalWanSelfAttention.forward on `model` in place."""
    # Match by class NAME, not isinstance: sdpa_patch re-imports the wan
    # modules, so the class object the model was built from is no longer the
    # one `import` returns here. The first attempt matched 0 modules.
    # Match by DUCK TYPING. Two earlier attempts matched 0 modules: isinstance
    # failed because sdpa_patch re-imports the wan modules (so the class object
    # differs from the one the model was built with), and the class name is not
    # what the released source suggests either. What we actually depend on is
    # the attribute surface, so test for that and report what we saw.
    import sys
    NEED = ("q", "k", "v", "o", "norm_q", "norm_k",
            "local_attn_size", "sink_size", "num_heads", "head_dim")
    seen = {}
    n = 0
    for m in model.modules():
        cn = type(m).__name__
        if not all(hasattr(m, x) for x in NEED):
            if "ttention" in cn or "ttn" in cn:
                seen[cn] = seen.get(cn, 0) + 1
            continue
        mod = sys.modules[type(m).__module__]
        if not (hasattr(mod, "causal_rope_apply") and hasattr(mod, "attention")):
            seen[cn + " (module lacks rope/attention)"] = 1
            continue
        causal_rope_apply = mod.causal_rope_apply
        attention = mod.attention
        if getattr(m, "_nosync", False):
            continue
        m._mirror = {}
        n += 1

        def fwd(self, x, seq_lens, grid_sizes, freqs, kv_cache=None,
                current_start=0, max_attention_size=1_000_000,
                frame_seqlen=None, seq_lens_int=None, _m=m):
            del seq_lens_int
            b, s, nh, d = *x.shape[:2], self.num_heads, self.head_dim
            q = self.norm_q(self.q(x)).view(b, s, nh, d)
            k = self.norm_k(self.k(x)).view(b, s, nh, d)
            v = self.v(x).view(b, s, nh, d)

            if frame_seqlen is None:
                frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            csf = current_start // frame_seqlen
            roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=csf).type_as(v)
            roped_key = causal_rope_apply(k, grid_sizes, freqs, start_frame=csf).type_as(v)

            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]

            mir = _m._mirror
            # a fresh cache object resets the mirror
            if mir.get("_cache_id") != id(kv_cache):
                mir.clear(); mir["_cache_id"] = id(kv_cache)
            gei = _host_idx(kv_cache, "global_end_index", mir)
            lei = _host_idx(kv_cache, "local_end_index", mir)

            if self.local_attn_size == -1:
                local_end_index = current_end
                local_start_index = current_start
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            elif (current_end > gei) and (num_new_tokens + lei > kv_cache_size):
                num_evicted = num_new_tokens + lei - kv_cache_size
                num_rolled = lei - num_evicted - sink_tokens
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled] = \
                    kv_cache["k"][:, sink_tokens + num_evicted:
                                  sink_tokens + num_evicted + num_rolled].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled] = \
                    kv_cache["v"][:, sink_tokens + num_evicted:
                                  sink_tokens + num_evicted + num_rolled].clone()
                local_end_index = lei + current_end - gei - num_evicted
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            else:
                local_end_index = lei + current_end - gei
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v

            lo = max(0, local_end_index - max_attention_size)
            k_cache = kv_cache["k"][:, lo:local_end_index]
            v_cache = kv_cache["v"][:, lo:local_end_index]
            out = attention(roped_query, k_cache, v_cache)

            # keep host mirror and device tensors in sync (fill_ is async)
            mir["_host_global_end_index"] = current_end
            mir["_host_local_end_index"] = local_end_index
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

            return self.o(out.flatten(2))

        m.forward = fwd.__get__(m, type(m))
        m._nosync = True
    if verbose:
        print(f"[nosync] patched {n} attention modules", flush=True)
        if n == 0:
            print(f"[nosync] no match. attention-like classes seen: {seen}", flush=True)
            allc = {}
            for m in model.modules():
                allc[type(m).__name__] = allc.get(type(m).__name__, 0) + 1
            print(f"[nosync] module census: "
                  f"{sorted(allc.items(), key=lambda x:-x[1])[:12]}", flush=True)
    return n
