"""Which layers' INT4 hurts most? Then build mixed precision from the answer.

Weight precision, not KV, turned out to be the quality bottleneck: with INT4
weights every KV improvement is masked (tail sweep: 16.2-16.5 dB regardless),
while INT8 weights alone reach 21.2 dB. But INT8 everywhere breaks 24 GB.

So quantize ONE group of Linear layers to INT4 at a time, leave everything
else bf16, and measure the PSNR drop. That is the sensitivity of that group.
Then keep the most sensitive groups at INT8, everything else INT4, and check
whether the mix fits the budget while recovering quality.

Groups: 8 depth bands of 5 blocks, plus the camera adapter (22.9% of params,
runs in every block) and the FFN (largest single sub-module).
Every arm restores the exact bf16 Linear modules before the next.
"""
from __future__ import annotations
import argparse, json, os, time, re
import torch, torch.nn as nn

GiB = 2 ** 30


def wbytes(m):
    return (sum(p.numel() * p.element_size() for p in m.parameters())
            + sum(b.numel() * b.element_size() for b in m.buffers()))


def psnr(a, b):
    a, b = a.float(), b.float()
    mse = float((a - b).pow(2).mean())
    rng = float(a.max() - a.min()) or 1.0
    return 99.0 if mse < 1e-12 else float(10 * torch.log10(torch.tensor(rng * rng / mse)))


def parent_and_leaf(model, name):
    parts = name.split(".")
    p = model
    for q in parts[:-1]:
        p = getattr(p, q)
    return p, parts[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--action", required=True)
    ap.add_argument("--prompt", default="A serene lakeside scene with a lone tree.")
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--local-attn", type=int, default=18)
    ap.add_argument("--sink", type=int, default=6)
    ap.add_argument("--band", type=int, default=5)
    ap.add_argument("--out", default="specwm_out/sensitivity")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from .binding_v2 import V2Binding
    from .quantize import PackedInt4Linear, GroupQuantLinear

    rep = {"groups": {}, "mixed": {}}
    t0 = time.time()
    b = V2Binding(a.ckpt, target_mode="fast_bf16", quant=None, sink=a.sink,
                  local_attn=a.local_attn, swap_models=False, build_draft=False)
    cond = b.prepare(a.image, a.prompt, a.action, frame_num=a.frames)
    N = cond["num_chunks"]; sig = b.node_sigmas(); tm = b.target_model
    print(f"[load] {time.time()-t0:.0f}s", flush=True)

    # ---- catalogue every quantizable Linear, keep the ORIGINAL module ----
    orig = {}
    for name, m in tm.named_modules():
        if isinstance(m, nn.Linear) and m.in_features >= 1024:
            orig[name] = m
    print(f"[cat] {len(orig)} Linear layers", flush=True)

    def blk_of(name):
        m = re.match(r"blocks\.(\d+)\.", name)
        return int(m.group(1)) if m else None

    def is_cam(name):  return "cam_" in name or "c2ws" in name
    def is_ffn(name):  return ".ffn." in name
    def is_attn(name): return "self_attn" in name or "cross_attn" in name

    nblk = len(tm.blocks)
    groups = {}
    for lo in range(0, nblk, a.band):
        hi = min(lo + a.band, nblk)
        groups[f"blocks_{lo:02d}-{hi-1:02d}"] = [
            n for n in orig if blk_of(n) is not None and lo <= blk_of(n) < hi]
    groups["cam_adapter(all)"] = [n for n in orig if is_cam(n)]
    groups["ffn(all)"] = [n for n in orig if is_ffn(n) and not is_cam(n)]
    groups["attn(all)"] = [n for n in orig if is_attn(n)]
    groups["embed/head"] = [n for n in orig if blk_of(n) is None and not is_cam(n)]

    def restore():
        for name, m in orig.items():
            p, leaf = parent_and_leaf(tm, name)
            setattr(p, leaf, m)
        torch.cuda.empty_cache()

    def apply(names, bits):
        for name in names:
            p, leaf = parent_and_leaf(tm, name)
            lin = orig[name]
            if bits == 4:
                setattr(p, leaf, PackedInt4Linear(lin, group=128, search=False))
            elif bits == 8:
                setattr(p, leaf, GroupQuantLinear(lin, bits=8, group=128, kind="int"))
        torch.cuda.empty_cache()

    def gen():
        b.reset_caches()
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        g = torch.Generator().manual_seed(1234)
        outs = []
        for ci in range(N):
            z = b.chunk_noise(ci, g); x0 = None
            for i in range(len(sig)):
                x0 = b.target_x0(z, i, ci)
                if i < len(sig) - 1:
                    s = float(sig[i + 1]); z = (1 - s) * x0 + s * b.randn_like(x0, g)
            b.commit(x0, ci)
            outs.append(x0.detach().float().cpu())
        return outs

    def dec(lat):
        z = torch.cat(lat, dim=1).to(b.device)
        img = b.pipe.vae.decode([z])[0].detach().cpu()
        del z; return img

    ref = dec(gen())

    # ---- sensitivity: one group INT4 at a time ----
    print("\n=== 그룹별 민감도 (해당 그룹만 INT4) ===", flush=True)
    for gname, names in groups.items():
        if not names:
            continue
        restore(); apply(names, 4)
        p = round(psnr(dec(gen()), ref), 2)
        nparam = sum(orig[n].weight.numel() for n in names)
        rep["groups"][gname] = dict(psnr=p, n_layers=len(names),
                                    params_B=round(nparam / 1e9, 3))
        print(f"  {gname:22} {len(names):4} layers {nparam/1e9:6.2f}B  PSNR {p}", flush=True)
    restore()

    # ---- all INT4 reference point ----
    apply(list(orig), 4)
    p_all4 = round(psnr(dec(gen()), ref), 2)
    rep["all_int4_psnr"] = p_all4
    print(f"\n  전부 INT4: PSNR {p_all4}", flush=True)
    restore()

    # ---- mixed precision from the ranking ----
    # sensitivity = PSNR drop; lower PSNR == more sensitive
    band_rank = sorted([(k, v["psnr"]) for k, v in rep["groups"].items()
                        if k.startswith("blocks_")], key=lambda x: x[1])
    print("\n=== 혼합 정밀도 (민감한 그룹만 INT8, 나머지 INT4) ===", flush=True)
    cands = {
        "cam@8": groups["cam_adapter(all)"],
        "top1band@8": groups[band_rank[0][0]],
        "top2band@8": groups[band_rank[0][0]] + groups[band_rank[1][0]],
        "cam+top1band@8": list(set(groups["cam_adapter(all)"] + groups[band_rank[0][0]])),
        "attn@8": groups["attn(all)"],
    }
    for cname, keep8 in cands.items():
        restore()
        keep8 = set(keep8)
        apply([n for n in orig if n not in keep8], 4)
        apply(list(keep8), 8)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        outs = gen()
        peak = round(torch.cuda.max_memory_allocated() / GiB, 2)
        p = round(psnr(dec(outs), ref), 2)
        w = round(wbytes(tm) / GiB, 2)
        rep["mixed"][cname] = dict(psnr=p, weight_GiB=w, peak_alloc_GiB=peak,
                                   n_int8=len(keep8))
        print(f"  {cname:18} INT8 {len(keep8):3} layers  weight {w:5.2f}G  "
              f"peak {peak:5.2f}G  PSNR {p}", flush=True)
    restore()

    with open(os.path.join(a.out, "sensitivity.json"), "w") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    print("\n기준: 노이즈 플로어 13.73 / 전부 INT4 {} / 전부 INT8 21.20".format(p_all4))


if __name__ == "__main__":
    main()
