"""Per-frame latency of the world-model pipelines, in the format of sdv2_latency (StreamDiffusionV2 demo measurement).
Definitions (wall clock, one host):
  request(c)  first moment any process starts work on chunk c (target chunk start; for the 2-GPU pipeline also the draft's first attempt),
              i.e. when the chunk's action is consumed
  ready(f)    the decoded pixel frame f is in host memory (VAE decode + D2H copy finished; streamed per latent frame)
  server_ms   ready(f) - request(chunk of f); output JPEG encoding / network / display are not included (as in sdv2_latency)
  segment     chunk index (chunk 0 = 9 frames, others 12)
Deadline for miss_rate: 750 ms = one chunk of video at 16 fps (12 frames), i.e. real-time playback; 1000 ms reported too.
Per config: <out>/<label>/frames.csv (frame_id, segment, server_ms), summary.json; plus a KDE plot of all configs (sdv2_latency.plot).
usage: python scripts/wm_latency.py OUT label=glob_of_clip_dirs [label=...]   (clip dirs with seed 1 = warm-up are skipped)"""
import sys, os, glob, json, csv, numpy as np
sys.path.insert(0, "/home/hk28965/specwm/sdv2_latency"); import sdv2_latency as SL
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); ALL = {}
for item in sys.argv[2:]:
    label, _, g = item.partition("="); rows = []; clips = []; fid = 0
    for d in sorted(glob.glob(g)):
        if d.endswith("_s1") or d.endswith("_warm") or not os.path.exists(d + "/loop.json"): continue
        L = json.load(open(d + "/loop.json")); DT = json.load(open(d + "/draft_times.json")) if os.path.exists(d + "/draft_times.json") else {}
        DJ = json.load(open(d + "/decode.json")) if os.path.exists(d + "/decode.json") else {}
        ready = L.get("frame_ready") or DT.get("frame_ready") or DJ.get("frame_ready")
        if not ready: print("no frame_ready in", d); continue
        req = list(L["chunk_start"])
        for c, t in (DT.get("chunk_start_draft") or {}).items(): req[int(c)] = min(req[int(c)], t)
        lat = []; rt = []
        for f, c, t in sorted(ready):
            ms = 1000 * (t - req[c]); lat.append(ms); rt.append(t); rows.append(dict(frame_id=fid, segment=c, server_ms=round(ms, 2))); fid += 1
        rt = np.array(sorted(rt)); gaps = np.diff(rt) * 1000
        clips.append(dict(dir=d, frames=len(lat), first_frame_ms=round(1000 * (rt[0] - L.get("t_gen_start", req[0])), 1), prepare_s=L.get("prepare_s"),
                          fps=round((len(rt) - 1) / (rt[-1] - rt[0]), 2), chunk_period_ms=round(1000 * (rt[-1] - rt[0]) / (L["chunks"] - 1), 1),
                          gen_s=L.get("gen_s"), accepted=L.get("accepted"), max_gap_ms=round(float(gaps.max()), 1)))
    os.makedirs("%s/%s" % (OUT, label), exist_ok=True)
    with open("%s/%s/frames.csv" % (OUT, label), "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["frame_id", "segment", "server_ms"]); w.writeheader(); [w.writerow(r) for r in rows]
    vals = [r["server_ms"] for r in rows]; st = SL.latency_stats(vals, 750.0); st["miss_rate_1000ms"] = float(np.mean(np.array(vals) > 1000)) if vals else None
    summ = dict(label=label, clips=len(clips), frames=len(rows), server_ms=st, fps_mean=float(np.mean([c["fps"] for c in clips])), chunk_period_ms_mean=float(np.mean([c["chunk_period_ms"] for c in clips])),
                first_frame_ms_mean=float(np.mean([c["first_frame_ms"] for c in clips])), prepare_s_mean=float(np.mean([c["prepare_s"] or 0 for c in clips])),
                gen_s_mean=float(np.mean([c["gen_s"] or 0 for c in clips])), per_clip=clips)
    json.dump(summ, open("%s/%s/summary.json" % (OUT, label), "w"), indent=1); ALL[label] = summ
print("| config | clips | frames | mean ↓ | p50 ↓ | p90 ↓ | p99 ↓ | max ↓ | miss >750 ms ↓ | jitter mean ↓ | output fps ↑ | chunk period ↓ | first frame ↓ |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for k, s in ALL.items():
    m = s["server_ms"]
    print("| %s | %d | %d | %.0f | %.0f | %.0f | %.0f | %.0f | %.1f %% | %.0f | %.1f | %.0f | %.0f |" % (k, s["clips"], s["frames"], m["mean"], m["p50"], m["p90"], m["p99"], m["max"], 100 * m["miss_rate"],
          m.get("jitter_mean", 0), s["fps_mean"], s["chunk_period_ms_mean"], s["first_frame_ms_mean"]))
json.dump({k: {kk: vv for kk, vv in v.items() if kk != "per_clip"} for k, v in ALL.items()}, open(OUT + "/all_summary.json", "w"), indent=1)
