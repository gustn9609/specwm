#!/usr/bin/env python3
"""노이즈 플로어 + 모든 유사도 수치에 플로어를 붙여 보고 (action item #2).

지표 = 우리 metric: DINOv2-vitb14 (518px) 프레임별 코사인 거리 1-cos, 4fps 프레임(frames_all.npy) 기준, 청크(3프레임) 평균.
  floor_base  : 같은 프롬프트, Base 70-step 시드쌍 거리 (모든 시드 쌍, 씬별)  -> 'plausible but different' 의 크기
  floor_fast  : 같은 프롬프트, Fast 시드쌍 거리
  cross       : Fast(seed i) vs Base(seed j) 전 쌍 (모델 간 거리; 컨텍스트 없는 fast_alone 의 기대 거리)
  arms        : handoff_grid / ctx_grid / resync 계열 arm 의 Base 참조 대비 거리. 참조는 seed 1234 하나가 아니라
                Base 시드 전체 평균(mean over refs) 과 최소(min over refs) 를 같이 보고, 플로어 대비 z = (d - floor)/floor_std 도 붙인다.
구간: all / early(청크 0-5) / late(청크 6+). 픽셀 PSNR 도 보조로 기록.
출력: specwm_out/floor/floor.json, floor.md
"""
import os, sys, json, glob, itertools, time, re
import numpy as np, torch

S = os.environ.get("SCRATCH", "/home/hk28965/specwm"); R = S + "/lingbot-world-v2/specwm_out"; OUT = R + "/floor"
os.makedirs(OUT, exist_ok=True); torch.hub.set_dir(S + "/.torch_hub")
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval().to(dev)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(dev); std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(dev)
_cache = {}


def feats(path):
    if path in _cache: return _cache[path]
    V = np.load(path); V = V[::4] if len(V) > 100 else V                      # 16fps 저장분은 4fps 로
    X = torch.from_numpy(np.ascontiguousarray(V)).permute(0, 3, 1, 2).float().to(dev) / 255
    X = torch.nn.functional.interpolate(X, size=(518, 518), mode="bilinear"); X = (X - mean) / std; o = []
    with torch.no_grad():
        for i in range(0, len(X), 8): o.append(torch.nn.functional.normalize(m(X[i:i + 8]).flatten(1), dim=-1).cpu())
    _cache[path] = (torch.cat(o), V); return _cache[path]


def dist(pa, pb):
    (FA, VA), (FB, VB) = feats(pa), feats(pb); n = min(len(FA), len(FB))
    d = (1 - (FA[:n] * FB[:n]).sum(-1)).numpy()
    if VA.shape[1:3] != VB.shape[1:3]:                                   # 해상도 다른 소스(H3 464x832 vs S00 512x768): PSNR 용으로 B 를 A 크기로 리사이즈
        import cv2; VB = np.stack([cv2.resize(f, (VA.shape[2], VA.shape[1]), interpolation=cv2.INTER_AREA) for f in VB[:n]])
    a, b = VA[:n].astype(np.float32) / 255, VB[:n].astype(np.float32) / 255
    mse = float(((a - b) ** 2).mean()); ps = 99.0 if mse < 1e-12 else 10 * np.log10(1.0 / mse)
    pc = [float(d[c * 3:(c + 1) * 3].mean()) for c in range(n // 3)]
    return dict(all=float(d.mean()), early=float(np.mean(pc[:6])) if len(pc) >= 6 else float(d.mean()), late=float(np.mean(pc[6:])) if len(pc) > 6 else None,
                per_chunk=[round(x, 4) for x in pc], psnr=round(float(ps), 2), n=int(n))


def stats(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return dict(mean=round(float(np.mean(v)), 4), std=round(float(np.std(v)), 4), min=round(float(np.min(v)), 4), max=round(float(np.max(v)), 4), n=len(v)) if v else None


def scene_of(path):
    import re; mm = re.search(r"[/_]S(\d\d)[/_]", path) or re.search(r"/S(\d\d)/", path)
    if mm: return mm.group(1)
    return "hands" if "hands" in path else None


out = {"metric": "DINOv2-vitb14 518px 1-cos, 4fps, chunk(3-frame) mean", "scenes": {}}
# ---- 참조 수집: Base 70-step 시드들 (floor/base/S{sc}_s{seed}/frames_all.npy + 기존 revisit/base_official_S{sc}) ----
base = {}
for p in glob.glob(R + "/floor/base/S*_s*/frames_all.npy") + glob.glob(R + "/revisit/base_official_S??/frames_all.npy"):
    sc = scene_of(p); seed = int(p.split("_s")[-1].split("/")[0]) if "/floor/base/" in p else 1234
    rj = os.path.join(os.path.dirname(p), "revisit.json")
    if os.path.exists(rj) and json.load(open(rj)).get("steps", 70) != 70: continue
    base.setdefault(sc, {})[seed] = p
fast = {}
for p in glob.glob(R + "/floor/fast/S*/W18_sink6/fast_alone_s*/frames_all.npy"):
    sc = re.search(r"/floor/fast/S(\d\d|hands)/", p).group(1); seed = int(p.split("fast_alone_s")[1].split("/")[0]); fast.setdefault(sc, {})[seed] = p
h3 = {}
for p in glob.glob(R + "/h3/floor_S*_s*/frames_all.npy"):
    mm = re.search(r"floor_S(\d\d|hands)_s(\d+)/", p); h3.setdefault(mm.group(1), {})[int(mm.group(2))] = p
print("[refs] base:", {k: sorted(v) for k, v in base.items()}, "fast:", {k: sorted(v) for k, v in fast.items()}, "h3:", {k: sorted(v) for k, v in h3.items()}, flush=True)

t0 = time.time()
for sc in sorted(set(base) | set(fast) | set(h3)):
    o = {"base_seeds": sorted(base.get(sc, {})), "fast_seeds": sorted(fast.get(sc, {}))}
    pairs = [dist(base[sc][i], base[sc][j]) for i, j in itertools.combinations(sorted(base.get(sc, {})), 2)]
    o["floor_base"] = {k: stats(pairs, k) for k in ("all", "early", "late", "psnr")} if pairs else None
    pairs = [dist(fast[sc][i], fast[sc][j]) for i, j in itertools.combinations(sorted(fast.get(sc, {})), 2)]
    o["floor_fast"] = {k: stats(pairs, k) for k in ("all", "early", "late", "psnr")} if pairs else None
    pairs = [dist(fast[sc][i], base[sc][j]) for i in fast.get(sc, {}) for j in base.get(sc, {})]
    o["cross_fast_vs_base"] = {k: stats(pairs, k) for k in ("all", "early", "late", "psnr")} if pairs else None
    if 1234 in fast.get(sc, {}) and 1234 in base.get(sc, {}): o["fast1234_vs_base1234"] = dist(fast[sc][1234], base[sc][1234])
    pairs = [dist(h3[sc][i], h3[sc][j]) for i, j in itertools.combinations(sorted(h3.get(sc, {})), 2)]
    o["floor_h3"] = {k: stats(pairs, k) for k in ("all", "early", "late", "psnr")} if pairs else None
    pairs = [dist(h3[sc][i], base[sc][j]) for i in h3.get(sc, {}) for j in base.get(sc, {})]
    o["cross_h3_vs_base"] = {k: stats(pairs, k) for k in ("all", "early", "late", "psnr")} if pairs else None
    pairs = [dist(h3[sc][i], fast[sc][j]) for i in h3.get(sc, {}) for j in fast.get(sc, {})]
    o["cross_h3_vs_fast"] = {k: stats(pairs, k) for k in ("all", "early", "late", "psnr")} if pairs else None
    # 실험 arm 들 (이 씬의 Base 참조 전체에 대해)
    arms = {}
    for p in glob.glob(R + "/handoff_grid/S%s*/*/frames_all.npy" % sc) + glob.glob(R + "/ctx_grid/S%s/*/frames_all.npy" % sc) + glob.glob(R + "/resync*/S%s*/*/*/frames_all.npy" % sc) + (glob.glob(R + "/handoff/*/frames_all.npy") + glob.glob(R + "/handoff/*/corr/frames_all.npy") if sc == "05" else []):
        tag = os.path.relpath(p, R).replace("/frames_all.npy", "")
        ds = [dist(p, bp) for bp in base.get(sc, {}).values()]
        if not ds: continue
        ds_ex = [dist(p, bp) for sd, bp in base.get(sc, {}).items() if sd != 1234]   # 컨텍스트 제공자(1234) 제외 -> lock-on 편향 제거
        row = {k: dict(mean=round(float(np.mean([d[k] for d in ds if d[k] is not None])), 4), min=round(float(np.min([d[k] for d in ds if d[k] is not None])), 4)) for k in ("all", "early", "late") if any(d[k] is not None for d in ds)}
        row["psnr_mean"] = round(float(np.mean([d["psnr"] for d in ds])), 2)
        if o.get("floor_base"):
            for k in ("all", "early", "late"):
                fb = o["floor_base"].get(k)
                if fb and k in row: row[k]["minus_floor"] = round(row[k]["mean"] - fb["mean"], 4); row[k]["z"] = (round((row[k]["mean"] - fb["mean"]) / fb["std"], 2) if fb["n"] >= 2 and fb["std"] > 0 else None)
        if 1234 in base.get(sc, {}): row["vs_base1234"] = {k: dist(p, base[sc][1234])[k] for k in ("all", "early", "late")}
        if ds_ex: row["excl1234"] = {k: round(float(np.mean([d[k] for d in ds_ex if d[k] is not None])), 4) for k in ("all", "early", "late") if any(d[k] is not None for d in ds_ex)}
        arms[tag] = row
    o["arms"] = arms; out["scenes"][sc] = o
    fb = o.get("floor_base"); ff = o.get("floor_fast"); cx = o.get("cross_fast_vs_base")
    print("[S%s] floor_base %s | floor_fast %s | cross %s | arms %d  (%.0fs)" % (sc, fb and fb["all"], ff and ff["all"], cx and cx["all"], len(arms), time.time() - t0), flush=True)

json.dump(out, open(OUT + "/floor.json", "w"), indent=1)
md = ["# Noise floor (%s)" % out["metric"], "", "| scene | floor Base (all / early / late) | floor Fast | cross Fast-vs-Base | fast1234 vs base1234 | floor H3 | H3-vs-Base | H3-vs-Fast |", "|---|---|---|---|---|---|---|---|"]
f = lambda s: "-" if not s else "%.3f±%.3f (n=%d)" % (s["mean"], s["std"], s["n"])
for sc, o in out["scenes"].items():
    fb, ff, cx = o.get("floor_base") or {}, o.get("floor_fast") or {}, o.get("cross_fast_vs_base") or {}
    fh, ch, cf2 = o.get("floor_h3") or {}, o.get("cross_h3_vs_base") or {}, o.get("cross_h3_vs_fast") or {}
    md.append("| S%s | %s / %s / %s | %s | %s | %s | %s | %s | %s |" % (sc, f(fb.get("all")), f(fb.get("early")), f(fb.get("late")), f(ff.get("all")), f(cx.get("all")),
                                                        ("%.3f" % o["fast1234_vs_base1234"]["all"]) if "fast1234_vs_base1234" in o else "-", f(fh.get("all")), f(ch.get("all")), f(cf2.get("all"))))
md += ["", "## Arms (distance to Base refs; mean over refs, then minus floor, z = (d-floor)/floor_std)", "",
       "| scene | arm | all | −floor | z | early | late | −floor(late) | late excl.1234 | psnr |", "|---|---|---|---|---|---|---|---|---|---|"]
for sc, o in out["scenes"].items():
    for tag, r in sorted(o["arms"].items()):
        g = lambda k, kk: r.get(k, {}).get(kk, "-")
        md.append("| S%s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (sc, tag, g("all", "mean"), g("all", "minus_floor"), g("all", "z"), g("early", "mean"), g("late", "mean"), g("late", "minus_floor"), r.get("excl1234", {}).get("late", "-"), r.get("psnr_mean", "-")))
open(OUT + "/floor.md", "w").write("\n".join(md)); print("\n".join(md[:12])); print("[FLOOR_ANALYSIS_DONE]", flush=True)
