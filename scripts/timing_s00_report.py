#!/usr/bin/env python3
"""S00 seed 9999 timing matrix (task 1): 모든 모델/구성의 생성 시간(VAE 디코드 포함) 표. 출력: specwm_out/timing_s00/timing_s00.md"""
import json, os, glob
R = "/home/hk28965/specwm/lingbot-world-v2/specwm_out"; T = R + "/timing_s00"; rows = []
def add(model, cfg, frames, hw, ttff, steady, fps, total, peak, note=""): rows.append((model, cfg, frames, hw, ttff, steady, fps, total, peak, note))
# Fast arms (latency_v2)
for d in ["fast_t5gpu", "fast_t5cpu", "fast_cs1", "fast_extra"]:
    p = T + "/%s/latency_v2.json" % d
    if not os.path.exists(p): continue
    for k, v in json.load(open(p))["arms"].items():
        if "error" in v: add("LingBot v2 Fast 14B", k, "-", "-", "-", "-", "-", "ERROR", "-", v["error"][:60]); continue
        area = v["max_area"]; hw = {399360: "480x832", 184320: "360x512", 122880: "256x464"}.get(area, str(area))
        tot = v["prepare_warm_s"] + sum(v["per_chunk_gen_s"]) + sum(v["per_chunk_dec_s"])
        add("LingBot v2 Fast 14B", k.split("@")[0] + ("" if v["chunk_size"] == 3 else " cs%d" % v["chunk_size"]) + (" T5cpu" if not v["t5_gpu"] else ""), 161, hw, "%.1f" % v["ttff_warm_s"], "%.2f+%.2f" % (v["steady_gen_s"], v["steady_dec_s"]), "%.2f" % v["throughput_fps"], "%.0f" % tot, "%.1f" % v["peak_alloc_GiB"], "gptq" if v.get("gptq") else ("wq=%s" % v["wq"] if v.get("wq") else "") + (" kv%d" % v["kv_bits"] if v.get("kv_bits") else ""))
# Base / bid arms (bench_base_official)
for d in sorted(glob.glob(T + "/base_*") + glob.glob(T + "/v2bid_*")):
    p = d + "/revisit.json"
    if not os.path.exists(p): continue
    v = json.load(open(p)); wc = v.get("worldcache"); hw = {399360: "480x832", 184320: "360x512"}.get(v.get("max_area"), "?")
    add("LingBot v1 Base (A14B MoE)" if "base_" in d else "LingBot v2 Bid 14B", os.path.basename(d).split("_", 1)[1] + (" [wc k=%d skip low %.0f%%/high %.0f%%]" % (wc["k"], 100 * wc["low"]["skip_rate"], 100 * wc["high"]["skip_rate"]) if wc else ""), v["frames"], hw, "= total", "-", "%.2f" % (v["frames"] / v["gen_s"]), "%.0f" % v["gen_s"], "%.1f" % v["peak_alloc_GiB"], "70-step UniPC CFG5, offload")
b = R + "/floor/base/S00_s9999/revisit.json"
if os.path.exists(b): v = json.load(open(b)); add("LingBot v1 Base (A14B MoE)", "bf16_480x832 (floor run)", v["frames"], "480x832", "= total", "-", "%.2f" % (v["frames"] / v["gen_s"]), "%.0f" % v["gen_s"], "%.1f" % v["peak_alloc_GiB"], "70-step UniPC CFG5, offload")
# H3 (sglang) from h3_log.json
h = R + "/h3/h3_log.json"
if os.path.exists(h):
    for k, v in json.load(open(h)).items():
        if "S00" in k and "9999" in k: add("MiniMax-H3 33B (sglang, 4 GPU Ulysses)", "fl2va 768p 10 s 24 fps", 240, "768p", "= total", "-", "%.2f" % (240 / v["wall_s"]), "%.0f" % v["wall_s"], "4x~90", "50 steps, no camera input")
for d, name in [("h3_s00", "MiniMax-H3 33B (DiffSynth, 1 GPU)"), ("h3world_s00", "H3-World (H3 + LoRA, 1 GPU)")]:
    for p in glob.glob(T + "/%s/timing.json*" % d):
        for k, v in json.load(open(p)).items():
            if "error" in v: add(name, k, "-", "-", "-", "-", "-", "ERROR", "-", v["error"][:60]); continue
            add(name, k, v["frames"], "%dx%d" % tuple(v["hw"]), "= total", "-", "%.2f" % (v["frames"] / v["gen_s"]), "%.0f" % v["gen_s"], "%.1f" % v["peak_GiB"], "%d steps" % v["steps"])
# v2 1.3B / causal-pretrain: generate.py 로그의 "Generating video" -> "Saving generated video" 시각 차 (VAE 디코드 포함, 모델 로드 제외)
import re, datetime
for lg in sorted(glob.glob("/home/hk28965/specwm/logs/q79_v2.*.out")):
    txt = open(lg, errors="ignore").read()
    for m in re.finditer(r"########## V2 (\w+) S00 timing.*?\n(.*?)(?=########## V2|\Z)", txt, re.S):
        var, body = m.group(1), m.group(2)
        ts = re.findall(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+\] INFO: (Generating video|Saving generated video)", body)
        if len(ts) >= 2 and ts[0][1].startswith("Generating") and ts[1][1].startswith("Saving"):
            f = lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"); sec = (f(ts[1][0]) - f(ts[0][0])).total_seconds()
            name = {"1p3b": "LingBot v2 1.3B causal-fast", "pretrain": "LingBot v2 14B causal-pretrain (40-step CFG)"}[var]
            add(name, "bf16, official generate.py, chunk 4" + (", window 18/sink 6, T5 CPU, offload" if var == "pretrain" else ", window 18/sink 6"), 161, "480x832", "= total", "-", "%.2f" % (161 / sec), "%.0f" % sec, "-", "1 GPU; excludes model load")
        elif "OutOfMemoryError" in body: add({"1p3b": "LingBot v2 1.3B causal-fast", "pretrain": "LingBot v2 14B causal-pretrain"}[var], "bf16 480x832", 161, "480x832", "-", "-", "-", "OOM", ">95", "full-attention KV cache for cond+uncond")
out = ["# S00 (Great Wall) seed 9999 — generation time matrix (VAE decode included)", "", "| model | config | frames | res | TTFF warm (s) | steady chunk gen+dec (s) | fps | total gen (s) | peak GiB | notes |", "|---|---|---|---|---|---|---|---|---|---|"]
out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
open(T + "/timing_s00.md", "w").write("\n".join(out)); print("\n".join(out))
