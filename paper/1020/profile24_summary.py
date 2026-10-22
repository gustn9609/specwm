"""[10/20] Device-tier PROFILE summary (advisor's ask: fps / throughput / latency / memory of the 24 GB device). Reads the idle-GPU re-measure runs
baselines/runs/ours_<arm>_<set>_idle5/<task>/timing.json (arms rt2 = original FP8 weights + bf16 KV window 18; mx32 = FP8 KV window 12 cap 32; mx24 = GPTQ-4 +
FP8 KV window 12 cap 24; sets neutral_832 832x480, mg2 640x352; 6 clips each, GPU 5 held exclusively) and writes paper/1020/data/profile24.md (English, ↑/↓).
usage: python3 -I /home/hk28965/specwm/paper/1020/profile24_summary.py"""
import json, glob, os, statistics as st
B = "/home/hk28965/specwm/baselines/runs"; OUT = "/home/hk28965/specwm/paper/1020/data/profile24.md"
ARMS = [("rt2", "original (FP8 w, bf16 KV, window 18)"), ("mx32", "32 GB class (FP8 w, FP8 KV, window 12)"), ("mx24", "24 GB class (GPTQ-4 w, FP8 KV, window 12)")]
SETS = [("neutral_832", "832x480"), ("mg2", "640x352")]
def med(xs): xs = [x for x in xs if x is not None]; return st.median(xs) if xs else None
def f(x, d=2): return "n/a" if x is None else ("%.*f" % (d, x))
rows = []
for s, res in SETS:
    for a, lab in ARMS:
        T = [json.load(open(p)) for p in sorted(glob.glob("%s/ours_%s_%s_idle5/*/timing.json" % (B, a, s)))]
        if not T: rows.append((res, lab, 0, None, None, None, None, None, None)); continue
        chunk = med([med(t["chunk_s"][3:]) for t in T if isinstance(t.get("chunk_s"), list)])
        rows.append((res, lab, len(T), med([t.get("steady_fps_gen") for t in T]), chunk, med([t.get("ttff_s") for t in T]), med([t.get("steady_fps_e2e") for t in T]),
                     med([t.get("decode_s") for t in T]), med([t.get("peak_reserved_GiB") for t in T])))
L = ["# Device-tier profile on an idle GPU (10/20, GPU 5 held exclusively, 6 clips of 10 s per cell, medians over clips)", "",
     "steady fps (gen) ↑ = 12 frames / median per-chunk DiT wall (chunks ≥ 3, CUDA-synced, decode excluded); chunk_s ↓; TTFF ↓ = prepare + first decoded frame; steady fps (e2e) ↑ includes the side-stream VAE decode; peak = torch reserved GiB ↓ (the capped arms decode in bf16 2x2 tiles and have no timed side-stream decode -> TTFF / e2e n/a).", "",
     "| res | device tier | n | steady fps gen ↑ | chunk_s ↓ | TTFF s ↓ | steady fps e2e ↑ | decode_s ↓ | peak GiB ↓ |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
for r in rows: L.append("| %s | %s | %d | %s | %s | %s | %s | %s | %s |" % (r[0], r[1], r[2], f(r[3]), f(r[4], 3), f(r[5]), f(r[6]), f(r[7]), f(r[8], 1)))
os.makedirs(os.path.dirname(OUT), exist_ok=True); open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
