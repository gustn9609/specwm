"""Phase x module profiler for LingBot-World-Fast.

Fills the DYNAMIC half of the 4090 budget map. The static half (weights, KV,
umT5) is computed offline in static_budget.py; here we measure what only a run
can tell us: activation peaks, VAE decode peak, per-module time, HBM traffic.

PHASES
  P0 load            weights resident, per module
  P1 text encode     umT5 resident + one-off cost
  P2 context encode  KV growth + prefill cost
  P3 denoise         activation peak, per-step DiT time   <- the hot loop
  P4 VAE decode      decode activation peak + time

MEASUREMENT DISCIPLINE (learned the hard way this session)
  * 2 warmup iterations discarded; cold and warm reported separately
    (a cold first run inflated our timing by 5.4x and nearly cost us the
     whole wallclock story)
  * torch.cuda.synchronize() around every timed region
  * both max_memory_allocated (real) and max_memory_reserved (allocator);
    the 24 GB verdict uses RESERVED
  * chunk 0 has an empty cache and is cheap -- steady state is reported
    separately once the local_attn window is full
  * per-module time is collected with CUDA events, not wall time
"""
from __future__ import annotations
import argparse, json, os, time
from collections import defaultdict
import torch


GiB = 2 ** 30


# ────────────────────────────── timing ──────────────────────────────
class ModuleTimer:
    """CUDA-event timer attached to named submodules.

    Falls back to a wall clock when CUDA is unavailable so the hook wiring can
    be unit-tested on CPU; GPU numbers always come from CUDA events, since wall
    time around an async kernel launch measures nothing.
    """

    def __init__(self):
        self.ev = defaultdict(list)          # name -> [[start, end], ...]
        self.handles = []
        self._on = True
        self.cuda = torch.cuda.is_available()

    def _mark(self):
        if self.cuda:
            e = torch.cuda.Event(enable_timing=True); e.record(); return e
        return time.perf_counter()

    def attach(self, name, module):
        def pre(_m, _i):
            if not self._on:
                return
            self.ev[name].append([self._mark(), None])
        def post(_m, _i, _o):
            if not self._on or not self.ev[name] or self.ev[name][-1][1] is not None:
                return
            self.ev[name][-1][1] = self._mark()
        self.handles.append(module.register_forward_pre_hook(pre))
        self.handles.append(module.register_forward_hook(post))

    def enable(self, on):  self._on = on
    def clear(self):       self.ev = defaultdict(list)

    def totals(self):
        if self.cuda:
            torch.cuda.synchronize()
        out = {}
        for k, pairs in self.ev.items():
            ms = [(s.elapsed_time(e) if self.cuda else (e - s) * 1e3)
                  for s, e in pairs if e is not None]
            if ms:
                out[k] = dict(total_ms=round(sum(ms), 2), calls=len(ms),
                              mean_ms=round(sum(ms) / len(ms), 4))
        return out

    def remove(self):
        for h in self.handles: h.remove()
        self.handles = []


def attach_blocks(timer, model, max_blocks=None):
    """Hook the DiT block internals.

    CausalWanAttentionBlock (wan/modules/model_fast.py) exposes:
      .self_attn {q,k,v,o}   .cross_attn   .ffn
      .cam_injector_layer1/2 .cam_scale_layer .cam_shift_layer
    Hooking every block would swamp us with events, so by default we
    instrument a few representative blocks and scale by 40.
    """
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        print("[warn] model has no .blocks"); return 0
    n = len(blocks)
    idx = range(n) if max_blocks is None else \
        sorted({0, n // 2, n - 1} | set(range(min(max_blocks, n))))
    for i in idx:
        b = blocks[i]
        if hasattr(b, "self_attn"):
            timer.attach("self_attn.total", b.self_attn)
            for p in ("q", "k", "v", "o"):
                if hasattr(b.self_attn, p):
                    timer.attach(f"self_attn.{p}_proj", getattr(b.self_attn, p))
        if hasattr(b, "cross_attn"): timer.attach("cross_attn", b.cross_attn)
        if hasattr(b, "ffn"):        timer.attach("ffn", b.ffn)
        for nm in ("cam_injector_layer1", "cam_injector_layer2",
                   "cam_scale_layer", "cam_shift_layer"):
            if hasattr(b, nm): timer.attach("cam_inject", getattr(b, nm))
    return len(list(idx))


# ────────────────────────────── memory ──────────────────────────────
class Phase:
    """with Phase('P3 denoise', rec): ...  -> peak alloc/reserved + wall time."""
    def __init__(self, name, rec, reset=True):
        self.name, self.rec, self.reset = name, rec, reset
    def __enter__(self):
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            torch.cuda.synchronize()
            if self.reset: torch.cuda.reset_peak_memory_stats()
        self.base = torch.cuda.memory_allocated() if self.cuda else 0
        self.t = time.perf_counter(); return self
    def __exit__(self, *a):
        if self.cuda:
            torch.cuda.synchronize()
            ma, mr, cur = (torch.cuda.max_memory_allocated(),
                           torch.cuda.max_memory_reserved(),
                           torch.cuda.memory_allocated())
        else:
            ma = mr = cur = 0
        self.rec[self.name] = dict(
            wall_s=round(time.perf_counter() - self.t, 3),
            peak_alloc_GiB=round(ma / GiB, 2),
            peak_reserved_GiB=round(mr / GiB, 2),
            resident_after_GiB=round(cur / GiB, 2),
            delta_GiB=round((cur - self.base) / GiB, 2))


def module_weights(model):
    """Resident weight bytes grouped by role (matches the static accounting)."""
    g = defaultdict(int)
    for n, p in model.named_parameters():
        b = p.numel() * p.element_size()
        if   "cam_" in n or "c2ws" in n or "wancamctrl" in n: g["action/camera adapter"] += b
        elif ".ffn" in n:                                     g["ffn"] += b
        elif "self_attn" in n:                                g["self_attn"] += b
        elif "cross_attn" in n:                               g["cross_attn"] += b
        elif "patch_embedding" in n:                          g["patch_embed"] += b
        elif "text_embedding" in n:                           g["text_embed"] += b
        elif "time_" in n:                                    g["time_embed"] += b
        elif "head" in n:                                     g["head"] += b
        else:                                                 g["norms/other"] += b
    return {k: round(v / GiB, 3) for k, v in sorted(g.items(), key=lambda x: -x[1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--action", required=True)
    ap.add_argument("--prompt", default="A serene lakeside scene with a lone tree.")
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--local-attn", type=int, default=18)
    ap.add_argument("--sink", type=int, default=6)
    ap.add_argument("--warmup-chunks", type=int, default=2)
    ap.add_argument("--steady-from", type=int, default=3,
                    help="chunk index from which the attention window is full")
    ap.add_argument("--hook-blocks", type=int, default=3)
    ap.add_argument("--kernel-trace", action="store_true",
                    help="also run torch.profiler for HBM traffic (slow)")
    ap.add_argument("--out", default="specwm_out/profile")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from .binding_v2 import V2Binding
    rec, report = {}, {}

    # ── P0: load ──────────────────────────────────────────────
    with Phase("P0_load", rec):
        b = V2Binding(a.ckpt, target_mode="fast_bf16", quant=None,
                      sink=a.sink, local_attn=a.local_attn,
                      swap_models=False, build_draft=False)
    tm = b.target_model
    report["P0_weights_by_module_GiB"] = module_weights(tm)
    report["P0_total_params_B"] = round(sum(p.numel() for p in tm.parameters()) / 1e9, 3)

    # umT5 residency (the budget's hidden cost)
    te = getattr(getattr(b, "pipe", None), "text_encoder", None)
    if te is not None:
        mdl = getattr(te, "model", te)
        try:
            nb = sum(p.numel() * p.element_size() for p in mdl.parameters())
            report["P1_umT5_resident_GiB"] = round(nb / GiB, 2)
            report["P1_umT5_params_B"] = round(sum(p.numel() for p in mdl.parameters()) / 1e9, 3)
            report["P1_umT5_device"] = str(next(mdl.parameters()).device)
        except Exception as e:
            report["P1_umT5_error"] = str(e)

    # ── P1+P2: text + context encode ──────────────────────────
    with Phase("P1_P2_prepare", rec):
        cond = b.prepare(a.image, a.prompt, a.action, frame_num=a.frames)
    N = cond["num_chunks"]
    report["num_chunks"] = N
    report["chunk_shape"] = list(cond["shape"])

    sig = b.node_sigmas()
    gen = torch.Generator().manual_seed(1234)

    def one_chunk(c, timed):
        z = b.chunk_noise(c, gen); x0 = None
        for i in range(len(sig)):
            x0 = b.target_x0(z, i, c)
            if i < len(sig) - 1:
                s = float(sig[i + 1]); z = (1 - s) * x0 + s * b.randn_like(x0, gen)
        b.commit(x0, c)
        return x0

    # ── warmup (discarded) ────────────────────────────────────
    for c in range(min(a.warmup_chunks, N)):
        one_chunk(c, False)
    torch.cuda.synchronize()
    report["note"] = (f"first {a.warmup_chunks} chunks discarded as warmup; "
                      f"steady state = chunk >= {a.steady_from}")

    # ── P3: denoise, per chunk, with module hooks ─────────────
    timer = ModuleTimer()
    n_hooked = attach_blocks(timer, tm, a.hook_blocks)
    report["hooked_blocks"] = n_hooked
    report["total_blocks"] = len(getattr(tm, "blocks", []))

    per_chunk, outs = [], []
    for c in range(a.warmup_chunks, N):
        timer.clear()
        with Phase(f"P3_chunk{c}", rec):
            outs.append(one_chunk(c, True))
        e = rec[f"P3_chunk{c}"]
        per_chunk.append(dict(chunk=c, **e, modules=timer.totals()))
    timer.enable(False); timer.remove()

    steady = [p for p in per_chunk if p["chunk"] >= a.steady_from]
    if steady:
        report["P3_steady"] = dict(
            chunks=[p["chunk"] for p in steady],
            wall_s_mean=round(sum(p["wall_s"] for p in steady) / len(steady), 3),
            peak_alloc_GiB=max(p["peak_alloc_GiB"] for p in steady),
            peak_reserved_GiB=max(p["peak_reserved_GiB"] for p in steady))
        agg = defaultdict(float); calls = defaultdict(int)
        for p in steady:
            for k, v in p["modules"].items():
                agg[k] += v["total_ms"]; calls[k] += v["calls"]
        scale = (report["total_blocks"] / n_hooked) if n_hooked else 1.0
        tot = sum(v for k, v in agg.items() if k != "self_attn.total")
        report["P3_module_ms_per_chunk"] = {
            k: dict(measured_ms=round(v / len(steady), 3),
                    est_all_blocks_ms=round(v / len(steady) * scale, 3),
                    calls=calls[k])
            for k, v in sorted(agg.items(), key=lambda x: -x[1])}
        report["P3_note"] = ("self_attn.total includes the attention kernel; "
                             "kernel time = self_attn.total - sum(q,k,v,o proj)")

    report["P3_per_chunk"] = per_chunk

    # ── P4: VAE decode ────────────────────────────────────────
    try:
        lat = torch.cat(outs, dim=1).to(b.device)
        with Phase("P4_vae_decode", rec):
            vid = b.pipe.vae.decode([lat])[0]
        report["P4_frames"] = int(vid.shape[1])
        del vid
    except Exception as e:
        report["P4_error"] = f"{type(e).__name__}: {e}"

    # ── optional kernel trace for HBM traffic ─────────────────
    if a.kernel_trace:
        from torch.profiler import profile, ProfilerActivity
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True, profile_memory=True) as prof:
            one_chunk(min(a.steady_from, N - 1), False)
            torch.cuda.synchronize()
        ka = prof.key_averages()
        rows = sorted(ka, key=lambda x: -x.self_device_time_total)[:25]
        report["kernels_top25"] = [
            dict(name=r.key[:70], self_ms=round(r.self_device_time_total / 1e3, 3),
                 calls=r.count,
                 mem_MB=round(getattr(r, "self_device_memory_usage", 0) / 2**20, 2))
            for r in rows]
        prof.export_chrome_trace(os.path.join(a.out, "trace.json"))

    report["phases"] = rec
    with open(os.path.join(a.out, "profile.json"), "w") as f:
        json.dump(report, f, indent=1)

    # ── console summary ───────────────────────────────────────
    print("\n" + "=" * 78); print("PHASE 요약  (peak는 reserved 기준으로 판정)"); print("=" * 78)
    print(f"{'phase':22}{'wall s':>9}{'peak alloc':>13}{'peak resv':>12}{'resident':>11}")
    print("-" * 78)
    for k, v in rec.items():
        if k.startswith("P3_chunk") and k != f"P3_chunk{a.warmup_chunks}":
            continue
        print(f"{k:22}{v['wall_s']:>9.3f}{v['peak_alloc_GiB']:>12.2f}G"
              f"{v['peak_reserved_GiB']:>11.2f}G{v['resident_after_GiB']:>10.2f}G")
    if "P3_steady" in report:
        s = report["P3_steady"]
        print(f"\n정상상태(chunk>={a.steady_from}): {s['wall_s_mean']:.3f}s/chunk, "
              f"peak {s['peak_alloc_GiB']:.2f}G alloc / {s['peak_reserved_GiB']:.2f}G reserved")
        print("\n" + "=" * 78); print("모듈별 시간 (40블록 환산)"); print("=" * 78)
        for k, v in report["P3_module_ms_per_chunk"].items():
            print(f"  {k:<24}{v['est_all_blocks_ms']:>10.2f} ms   ({v['calls']} calls)")
    print("\n" + "=" * 78); print("P0 weight 분해"); print("=" * 78)
    for k, v in report["P0_weights_by_module_GiB"].items():
        print(f"  {k:<24}{v:>8.2f} GiB")
    print(f"\n  총 파라미터: {report['P0_total_params_B']} B")
    if "P1_umT5_resident_GiB" in report:
        print(f"  umT5 상주:   {report['P1_umT5_resident_GiB']} GiB "
              f"({report.get('P1_umT5_params_B')} B, {report.get('P1_umT5_device')})")
    print(f"\nsaved {os.path.join(a.out, 'profile.json')}")


if __name__ == "__main__":
    main()
