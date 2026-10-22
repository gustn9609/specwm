"""Device-fit report for the 14B alone (specwm.fit24 runs in lingbot-world-v2/specwm_out/fit24): memory table, fit verdict at 24 / 32 GB,
speed, and the quality check of the quantized variants against the FP8-weights / bf16-KV reference (same seed): mean PSNR to the reference
clip, and the VLM clip judge (quality ↑ / drift ↓) from fit24_vlm.json if present. English markdown.
usage: python scripts/fit24_report.py [--json paper/meeting_1012/data/fit24.json]"""
import sys, os, glob, json, numpy as np, cv2
S = "/home/hk28965/specwm"; O = S + "/lingbot-world-v2/specwm_out/fit24"; CTX = 1.0   # GiB outside the allocator (CUDA context, cuBLAS / compile workspaces)
CFG = [("704x384, FP8 weights, bf16 KV (the realtime mode as run)", "a704_fp8_bf16"), ("704x384, FP8 weights, FP8 KV", "a704_fp8_fp8kv"),
       ("704x384, INT4 weights, bf16 KV", "a704_int4_bf16"), ("704x384, INT4 weights, FP8 KV", "a704_int4_fp8kv"), ("704x384, AWQ-INT4 weights, FP8 KV", "a704_awq_fp8kv"),
       ("832x480, FP8 weights, bf16 KV (the quality mode as run)", "a832_fp8_bf16"), ("832x480, FP8 weights, FP8 KV", "a832_fp8_fp8kv"),
       ("832x480, INT4 weights, FP8 KV", "a832_int4_fp8kv"), ("832x480, AWQ-INT4 weights, FP8 KV", "a832_awq_fp8kv"),
       ("704x384, GPTQ-4 weights (Triton dequant), FP8 KV", "q43_a704_gptq4_fp8kv"), ("704x384, FP8 weights, FP8 KV, window 12", "q43_a704w12_fp8_fp8kv"), ("832x480, FP8 weights, FP8 KV, window 12", "q43_a832w12_fp8_fp8kv"), ("704x384, GPTQ-4 weights, FP8 KV, window 12", "q43_a704w12_gptq4_fp8kv")]
def load(k):
    p = O + "/" + k + "/fit.json"; return json.load(open(p)) if os.path.exists(p) else None
def frames(p):
    cap = cv2.VideoCapture(p); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    return fr
def psnr_to(ref, p):
    A, B = frames(ref), frames(p); n = min(len(A), len(B))
    if n == 0: return None, None
    ps = []
    for i in range(n):
        b = B[i] if B[i].shape == A[i].shape else cv2.resize(B[i], (A[i].shape[1], A[i].shape[0]))
        mse = np.mean((A[i].astype(np.float32) - b.astype(np.float32)) ** 2); ps.append(10 * np.log10(255 ** 2 / max(mse, 1e-6)))
    ps = np.array(ps); first = int(np.argmax(ps < 30)) if (ps < 30).any() else None
    return float(ps.mean()), first
out = {}
print("| config (14B alone, 2-step, fused + partial commit, 161 frames) | weights (+ CLIP image encoder) ↓ | KV window (18 latent frames) ↓ | resident after prepare ↓ | peak, generation ↓ | peak, streamed decode ↓ | peak reserved ↓ | fits 24 GB (≤ 23 GiB reserved) | fits 32 GB | steady fps ↑ |")
print("|---|---|---|---|---|---|---|---|---|---|")
for lab, k in CFG:
    r = load(k)
    if not r: print("| %s | – | – | – | – | – | – | – | – | – |" % lab); continue
    for kk in ("alloc_after_awq_GiB", "alloc_after_gptq_GiB"):
        if kk in r: r["alloc_after_prepare_GiB"] = round(r[kk], 2)   # AWQ is quantized after prepare (the bf16 model was still resident at prepare time)
    pr = r.get("peak_reserved_GiB"); f24 = "yes" if pr is not None and pr + CTX <= 24 else "no"; f32 = "yes" if pr is not None and pr + CTX <= 32 else "no"
    if r.get("oom_stage"): f24 = f32 = "OOM at " + r["oom_stage"]
    r["w_GiB"] = round(r["alloc_after_load_GiB"] - r["vae_GiB"], 2)   # FP8Linear keeps w8 as plain attributes, so the parameter sum misses it: use the resident total minus the VAE
    out[k] = dict(label=lab, w_GiB=r["w_GiB"], **{kk: r.get(kk) for kk in ("dit_GiB", "kv_GiB", "alloc_after_prepare_GiB", "peak_gen_GiB", "peak_decode_GiB", "peak_reserved_GiB", "fps", "steady_chunk_s", "oom_stage", "cap")}, fits24=f24, fits32=f32)
    print("| %s | %.1f GiB | %.1f GiB | %.1f GiB | %.1f GiB | %.1f GiB | %.1f GiB | %s | %s | %.1f |" % (lab, r["w_GiB"], r["kv_GiB"], r["alloc_after_prepare_GiB"], r.get("peak_gen_GiB", float("nan")), r.get("peak_decode_GiB", float("nan")), pr if pr is not None else float("nan"), f24, f32, r.get("fps", float("nan"))))
caps = sorted(glob.glob(O + "/cap*/fit.json")) + sorted(glob.glob(O + "/tiled*/fit.json"))
if caps:
    print("\nAllocator-cap runs (device emulation: the CUDA caching allocator may not grow past the cap):"); print("| run (tiled* = spatial 2x2 tiled chunk decode instead of per-frame streamed decode) | cap | result | peak reserved ↓ | steady fps ↑ | decode s/chunk ↓ |"); print("|---|---|---|---|---|---|")
    for p in caps:
        r = json.load(open(p)); print("| %s | %.0f GB | %s | %s | %s | %s |" % (os.path.basename(os.path.dirname(p)), r["cap"], ("OOM at " + r["oom_stage"]) if r.get("oom_stage") else "ran to the end", ("%.1f GiB" % r["peak_reserved_GiB"]) if r.get("peak_reserved_GiB") else "–", ("%.1f" % r["fps"]) if r.get("fps") else "–", ("%.2f" % r["decode_chunk_s"][-1]) if r.get("decode_chunk_s") else "–"))
        out["cap_" + os.path.basename(os.path.dirname(p))] = dict(cap=r["cap"], oom_stage=r.get("oom_stage"), peak_reserved_GiB=r.get("peak_reserved_GiB"), fps=r.get("fps"))
VL = json.load(open(O + "/fit24_vlm.json")) if os.path.exists(O + "/fit24_vlm.json") else {}
VL = {(k if k.startswith("/") else S + "/" + k): v for k, v in VL.items()}
def vids(k):
    d = O + "/" + k; return sorted([d + "/spec.mp4"] + glob.glob(d + "/*/spec.mp4"))
# No pixel comparison to the reference: a rerun of the SAME config and seed already diverges (easy_lake 704: 34 dB at frame 0, 21 dB at frame 12,
# 11.5 dB at frame 80 vs the bench-43 clip), so PSNR to a reference clip measures AR chaos, not quantization loss. Only the VLM judge is used.
print("\nQuality check, seed 1234, 8 bench-43 prompts (easy_lake + 7 physics), paired with the FP8 weights / bf16 KV run at the same resolution:")
print("| config | clips | VLM quality ↑ | VLM drift ↓ | Δ quality vs reference (paired) ↑ | Δ drift vs reference (paired) ↓ | quality: better / equal / worse |"); print("|---|---|---|---|---|---|---|")
for lab, k in CFG:
    ref = "a704_fp8_bf16" if k.startswith("a704") else "a832_fp8_bf16"; V = vids(k)
    if len(V) < 8: continue   # a704_awq_fp8kv hung after 2 clips (killed); its 43-prompt run below covers it
    q, d, dq, dd = [], [], [], []
    for p in V:
        rp = p.replace("/" + k + "/", "/" + ref + "/")
        if p in VL:
            q.append(VL[p]["quality"]); d.append(VL[p]["drift"])
            if rp in VL and k != ref: dq.append(VL[p]["quality"] - VL[rp]["quality"]); dd.append(VL[p]["drift"] - VL[rp]["drift"])
    se = lambda v: np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0; dq_ = np.array(dq)
    out.setdefault(k, {}).update(clips=len(V), vlm_q=float(np.mean(q)) if q else None, vlm_d=float(np.mean(d)) if d else None, dq=float(np.mean(dq)) if dq else None, dq_se=float(se(dq)) if dq else None, dd=float(np.mean(dd)) if dd else None, dd_se=float(se(dd)) if dd else None)
    print("| %s | %d | %s | %s | %s | %s | %s |" % (lab, len(V), "%.2f ± %.2f" % (np.mean(q), se(q)) if q else "–", "%.1f ± %.1f" % (np.mean(d), se(d)) if d else "–", "%+.2f ± %.2f" % (np.mean(dq), se(dq)) if dq else "(reference)", "%+.1f ± %.1f" % (np.mean(dd), se(dd)) if dd else "(reference)", "%d / %d / %d" % ((dq_ > 0).sum(), (dq_ == 0).sum(), (dq_ < 0).sum()) if dq else "–"))
if "--json" in sys.argv: json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
# ---- 43-prompt quality check: q43_* runs (seed 1234, all bench-43 prompts) paired with the bench-43 clips of the same config without
# KV / weight quantization (pc1_704_s1234 / pc1_832_s1234, FP8 weights + bf16 KV, scores in paper/1012/vlm_scores_bench43.json)
B = S + "/lingbot-world-v2/specwm_out/bench43"; VB = json.load(open(S + "/paper/1012/vlm_scores_bench43.json")) if os.path.exists(S + "/paper/1012/vlm_scores_bench43.json") else {}
VB = {(k if k.startswith("/") else S + "/" + k): v for k, v in VB.items()}
Q43 = [("704x384, FP8 weights, FP8 KV", "q43_a704_fp8_fp8kv", "pc1_704"), ("704x384, INT4 weights, FP8 KV", "q43_a704_int4_fp8kv", "pc1_704"), ("704x384, AWQ-INT4 weights, FP8 KV", "q43_a704_awq_fp8kv", "pc1_704"),
       ("832x480, FP8 weights, FP8 KV", "q43_a832_fp8_fp8kv", "pc1_832"), ("704x384, GPTQ-4 weights, FP8 KV", "q43_a704_gptq4_fp8kv", "pc1_704"),
       ("704x384, FP8 weights, FP8 KV, KV window 12", "q43_a704w12_fp8_fp8kv", "pc1_704"), ("832x480, FP8 weights, FP8 KV, KV window 12", "q43_a832w12_fp8_fp8kv", "pc1_832"),
       ("704x384, FP8 weights, FP8 KV, seed 4321", "q43s4321_a704_fp8_fp8kv", "pc1_704"), ("832x480, GPTQ-4 weights, FP8 KV", "q43_a832_gptq4_fp8kv", "pc1_832"), ("704x384, GPTQ-4 weights, FP8 KV, KV window 12 (24 GB candidate)", "q43_a704w12_gptq4_fp8kv", "pc1_704")]
print("\n43-prompt quality check (seed 1234), paired with the bench-43 clip of the same mode without quantization (FP8 weights, bf16 KV):")
print("| config | pairs | VLM quality ↑ | VLM drift ↓ | reference quality ↑ | reference drift ↓ | Δ quality (paired) ↑ | Δ drift (paired) ↓ | quality: better / equal / worse |"); print("|---|---|---|---|---|---|---|---|---|")
for lab, k, refarm in Q43:
    V = vids(k)
    if not V: continue
    q, d, rq, rd = [], [], [], []
    for p in V:
        t = os.path.basename(os.path.dirname(p)); t = "easy_lake" if t == k else t
        rp = "%s/%s/%s_s%d/cloud/spec.mp4" % (B, t, refarm, 4321 if "s4321" in k else 1234)
        if p in VL and rp in VB: q.append(VL[p]["quality"]); d.append(VL[p]["drift"]); rq.append(VB[rp]["quality"]); rd.append(VB[rp]["drift"])
    if not q: continue
    q, d, rq, rd = map(np.array, (q, d, rq, rd)); se = lambda v: np.std(v, ddof=1) / np.sqrt(len(v)); dq = q - rq; dd = d - rd
    out[k] = dict(label=lab, pairs=len(q), vlm_q=float(q.mean()), vlm_d=float(d.mean()), ref_q=float(rq.mean()), ref_d=float(rd.mean()), dq=float(dq.mean()), dq_se=float(se(dq)), dd=float(dd.mean()), dd_se=float(se(dd)))
    print("| %s | %d | %.2f ± %.2f | %.1f ± %.1f | %.2f | %.1f | %+.2f ± %.2f | %+.1f ± %.1f | %d / %d / %d |" % (lab, len(q), q.mean(), se(q), d.mean(), se(d), rq.mean(), rd.mean(), dq.mean(), se(dq), dd.mean(), se(dd), (dq > 0).sum(), (dq == 0).sum(), (dq < 0).sum()))
if "--json" in sys.argv: json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
