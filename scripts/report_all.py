#!/usr/bin/env python3
"""모든 결과를 action item 순서로 묶어 specwm_out/REPORT.md 생성 (부분 결과라도 있는 만큼 작성)."""
import os, json, glob, re
import numpy as np
S = os.environ.get("SCRATCH", "/home/hk28965/specwm"); R = S + "/lingbot-world-v2/specwm_out"
J = lambda p: json.load(open(p)) if os.path.exists(p) else None
md = ["# spi0 experiment report (auto-generated)", ""]

# 1. VisionReward
vr = J(R + "/vr/vr_scores.json"); md += ["## 1. VisionReward-Video (quality metric)", ""]
if vr and vr["videos"]:
    groups = {}
    for tag, e in vr["videos"].items():
        if e.get("status") != "done": continue
        g = re.sub(r"_s\d{4,5}$", "", tag.split("/")[-1]); grp = "/".join(tag.split("/")[:-1]) + "/" + g
        groups.setdefault(grp, []).append(e["mean_score"])
    md += ["| group | n | VR mean | std (noise floor when n>1 seeds) | min |", "|---|---|---|---|---|"]
    for g, v in sorted(groups.items()):
        md.append("| %s | %d | %.4f | %s | %.4f |" % (g, len(v), np.mean(v), ("%.4f" % np.std(v)) if len(v) > 1 else "-", np.min(v)))
    md += ["", "Base 70-step 시드 그룹의 std 가 VR 지표의 노이즈 플로어. 창(6s) 단위 점수는 vr_scores.json 참조.", ""]
else: md += ["(vr_scores.json 없음)", ""]

# 2. noise floor
md += ["## 2. Noise floor (DINO metric)", ""]
fm = R + "/floor/floor.md"
md += ([l for l in open(fm).read().split("\n")[1:]] if os.path.exists(fm) else ["(floor.md 없음 - scripts/floor_analysis.py 실행)"]) + [""]

# 3. H3
md += ["## 3. MiniMax-H3 fit", ""]
h3 = R + "/h3_fit/h3_fit.md"; md += ([l for l in open(h3).read().split("\n")[1:]] if os.path.exists(h3) else ["(h3_fit.md 없음)"]) + [""]

# 4. Pyramid-Flow style uplink / downlink
md += ["## 4. Noisier / lower-res frames to the cloud (Pyramid-Flow style)", "", "### 4a. Uplink: Fast(edge) -> Base(cloud) refine grid", ""]
fl = J(R + "/floor/floor.json") or {"scenes": {}}
for p in sorted(glob.glob(R + "/handoff_grid/S*/handoff_grid.json")):
    dname = os.path.basename(os.path.dirname(p)); sc = dname[1:3]; h = json.load(open(p)); arms = fl["scenes"].get(sc, {}).get("arms", {})
    fb = (fl["scenes"].get(sc, {}).get("floor_base") or {}).get("all")
    md += ["**%s** (floor Base all = %s)" % (dname, ("%.3f±%.3f" % (fb["mean"], fb["std"])) if fb else "n/a"), "",
           "| arm | res | bits | sigma | payload MB | ratio vs bf16 latent | refine s | DINO→Base (−floor) | psnr vs Fast | flicker |", "|---|---|---|---|---|---|---|---|---|---|"]
    for tag, v in sorted(h["arms"].items(), key=lambda kv: (kv[1].get("res", 0), kv[1].get("bits", 0), kv[1].get("sigma", 0)) if "error" not in kv[1] else (9, 9, 9)):
        if "error" in v: md.append("| %s | ERROR |||||||||" % tag); continue
        d = arms.get("handoff_grid/%s/%s" % (dname, tag), {}).get("all", {})
        md.append("| %s | %g | %d | %g | %.1f | %.3f | %.0f | %s | %.2f | %.4f |" % (tag, v["res"], v["bits"], v["sigma"], v["payload_bytes"] / 2 ** 20, v["payload_ratio_vs_bf16"], v["refine_s"],
                                                                                  ("%.3f (%+.3f)" % (d["mean"], d["minus_floor"])) if "minus_floor" in d else (("%.3f" % d["mean"]) if d else "-"), v["psnr_vs_fast"], v["flicker"]))
    md.append("")
md += ["### 4b. Downlink: Base(cloud) context -> Fast(edge) grid", ""]
for p in sorted(glob.glob(R + "/ctx_grid/S*/ctx_grid.json")):
    sc = p.split("/S")[-1][:2]; h = json.load(open(p)); arms = fl["scenes"].get(sc, {}).get("arms", {})
    fb = (fl["scenes"].get(sc, {}).get("floor_base") or {}).get("late")
    md += ["**S%s** (floor Base late = %s)" % (sc, ("%.3f±%.3f" % (fb["mean"], fb["std"])) if fb else "n/a"), "",
           "| arm | res | bits | sigma | downlink MB | ratio | edge extra s | DINO→Base late (−floor) | flicker |", "|---|---|---|---|---|---|---|---|---|"]
    for tag, v in sorted(h["arms"].items()):
        if "error" in v: md.append("| %s | ERROR ||||||||" % tag); continue
        d = arms.get("ctx_grid/S%s/%s" % (sc, tag), {}).get("late", {})
        ds = ("%.3f (%+.3f)" % (d["mean"], d["minus_floor"])) if "minus_floor" in d else (("%.3f" % d["mean"]) if d else "-")
        if v.get("kind") == "ctx":
            md.append("| %s | %g | %d | %g | %.1f | %.3f | %.2f | %s | %.4f |" % (tag, v["res"], v["bits"], v["sigma"], v["downlink_bytes"] / 2 ** 20, v["downlink_ratio_vs_bf16"], v["edge_extra_s"], ds, v["flicker"]))
        else: md.append("| %s | - | - | - | 0 | 0 | 0 | %s | %.4f |" % (tag, ds, v["flicker"]))
    md.append("")

# 5. edge
md += ["## 5. Edge GPU (memory-cap verdict + roofline projection; no real edge GPU on spi0)", ""]
es = J(R + "/edge_sim_v2/edge_sim.json") or J(R + "/edge_sim/edge_sim.json")
if es:
    md += ["GPU: %s (%.0f GiB). calib GEMM %.0f TFLOPS, copy %.0f GB/s. compressed: INT4 weights %.2f GiB + KV%d store %.2f GiB, psnr vs bf16 %.2f" % (
        es["gpu"], es["total_GiB"], es["calib"]["gemm_bf16_tflops"], es["calib"]["copy_bw_gbs"], es["compressed"]["weight_GiB"], es["compressed"]["kv_bits"], es["compressed"].get("kv_store_GiB", 0), es["compressed"]["psnr_vs_bf16"]), "",
           "| cap | verdict | gen peak reserved | chunk s | decode peak |", "|---|---|---|---|---|"]
    pk = max(es["compressed"]["gen_uncapped"]["peak_reserved_GiB"], es["compressed"]["decode_uncapped"]["peak_reserved_GiB"])
    for k, v in es["caps"].items():
        g = v.get("gen", {}); d = v.get("decode", {}); cap = float(v["cap_GiB"])
        steady = "FITS (steady peak %.1f < %g)" % (pk, cap) if pk < cap else "OVER (steady peak %.1f > %g)" % (pk, cap)
        md.append("| %s | runtime cap: %s / by measured steady peak: %s | %s | %s | %s |" % (k, v["verdict"], steady, g.get("peak_reserved_GiB", "-"), g.get("chunk_s_median", "-"), d.get("peak_reserved_GiB", "-")))
    md += ["", "_runtime cap 판정 주의: v1 은 reset_caches() 의 bf16 KV 과도 할당(~21GiB)이 캡에 걸림, v2(청크1 이후 캡)는 kvstore 의 id(kv_cache) 재사용 버그로 실패 -> 하드 캡 테스트는 후속 과제. 현재 판정은 측정된 정상상태 peak_reserved(%.1f GiB) 기준._" % pk]
    md += ["", "| device | TFLOPS | GB/s | mem | chunk s (projected) | fps (projected) | bound | fits |", "|---|---|---|---|---|---|---|---|"]
    for k, v in es["projection"].items():
        md.append("| %s | %.0f | %.0f | %.0f | %.3f | %.2f | %s | %s |" % (k, v["tflops"], v["bw_gbs"], v["mem_gb"], v["chunk_s_projected"], v["fps_projected"], v["bound"], v["fits_memory"]))
    fp = ((fl.get("scenes", {}).get("05", {}).get("floor_fast") or {}).get("psnr") or {})
    es3 = J(R + "/edge_sim_v3/edge_sim.json")
    md += ["", "_projected = roofline 비율 투영 (측정 아님): %s_" % es["projection_model"]["note"],
           "", "**PSNR 해석 주의**: 압축 구성의 bf16 대비 PSNR %.1f dB (int4 scale-search 시 %s dB) 는 Fast 시드간 PSNR 플로어 %s dB (S05, n=%s) 와 같은 수준 -> '다른 그럴듯한 영상' 이지 화질 붕괴가 아님. 화질 판정은 VR/DINO 로." % (
               es["compressed"]["psnr_vs_bf16"], es3["compressed"]["psnr_vs_bf16"] if es3 else "-", ("%.1f±%.1f" % (fp["mean"], fp["std"])) if fp else "-", fp.get("n", "-")), ""]
else: md += ["(edge_sim.json 없음)", ""]

# 6. latency
md += ["## 6. Time-to-first-frame vs steady-state chunk latency", ""]
lt = J(R + "/latency_v2/latency_v2.json")
if lt:
    md += ["| arm@area | load s | TTFF cold (prep/gen0/dec0) | TTFF warm (prep/gen0/dec0) | early gen | steady gen (p90) | steady dec | fps | ingest 6 ctx s | peak GiB |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in lt["arms"].items():
        if "error" in v: md.append("| %s | ERROR |||||||||" % k); continue
        md.append("| %s | %.0f | %.2f (%.2f/%.2f/%.2f) | %.2f (%.2f/%.2f/%.2f) | %.3f | %.3f (%.3f) | %.3f | %.2f | %.2f | %.1f |" % (
            k, v["load_s"], v["ttff_cold_s"], v["prepare_cold_s"], v["gen0_cold_s"], v["dec0_cold_s"], v["ttff_warm_s"], v["prepare_warm_s"], v["gen0_warm_s"], v["dec0_warm_s"],
            v["early_gen_s"], v["steady_gen_s"], v["steady_gen_p90_s"], v["steady_dec_s"], v["throughput_fps"], v["ingest_ctx_s"], v["peak_alloc_GiB"]))
    md.append("")
else: md += ["(latency_v2.json 없음)", ""]
pf = J(R + "/profile/profile.json")
if pf and "P3_steady" in pf:
    md += ["Profile (81f, steady chunk>=6): %.3f s/chunk, peak %.2f GiB reserved; module ms/chunk (40-block est.): %s" % (
        pf["P3_steady"]["wall_s_mean"], pf["P3_steady"]["peak_reserved_GiB"], ", ".join("%s %.0f" % (k, v["est_all_blocks_ms"]) for k, v in list(pf["P3_module_ms_per_chunk"].items())[:6])), ""]
cl = R + "/clip/clip_adherence.md"
md += ["## 7. Prompt adherence (CLIP) and true-x_t handoff", ""] + ([l for l in open(cl).read().split("\n")[1:]] if os.path.exists(cl) else ["(clip_adherence.md 없음)"]) + [""]
vj = J(R + "/handoff/verify/verify.json")
if vj: md += ["E-B verifier (Base one-step x0 vs Fast x0, sigma %s): " % vj.get("sigma") + ", ".join("k%d %.3f" % (r["chunk"], r["dist_rel"]) for r in vj["rows"]), ""]
for t in ("ho3", "ho3_1step", "ho2"):
    hj = J(R + "/handoff/%s/handoff.json" % t)
    if hj: md.append("E-A %s: sigma_start %s schedule %s gen %ss (DINO vs Base refs: floor.md arms 'handoff/%s')" % (t, hj["sigma_start"], hj["schedule"], hj["gen_s"], t))
md.append("")
open(R + "/REPORT.md", "w").write("\n".join(md)); print("\n".join(md[:40])); print("... ->", R + "/REPORT.md")
