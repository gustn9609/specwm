"""World-model latency metrics (10/5 meeting: time-to-first-frame, steady-state throughput, steady-state action latency) for spec_ar_loop runs
(SDVG / latent / physics / low-res arms), from the per-chunk wall-clock stamps chunk_t in loop.json (chunk c runs from chunk_t[c] to chunk_t[c+1]).
The loop decodes once at the end, so frames are attributed as if decoded per chunk: decode share of chunk c = decode_s * frames_c / 153.
  TTFF            = prepare (11.3 s fixed for this loop at 832x480, measured elsewhere) + chunk 0 generation + chunk 0 decode share
  steady fps      = 12 frames / median chunk period over chunks 1..N-1 (video frames per second of wall clock, 16 fps = real time)
  action latency  = median chunk period + per-chunk decode share (action consumed at chunk start -> its 12 frames visible), plus p99 over chunks
  e2e             = prepare + gen + decode (the number reported before)
usage: python scripts/loop_latency.py label=glob [label=glob ...]"""
import sys, glob, json, numpy as np
PREP = 11.3
print("| arm | clips | TTFF ↓ (s) | steady fps ↑ | action latency p50 ↓ (s) | p99 ↓ (s) | chunk period draft-accepted / target (s) | e2e ↓ (s) | accepted ↑ |")
print("|---|---|---|---|---|---|---|---|---|")
for item in sys.argv[1:]:
    label, _, g = item.partition("="); ttff = []; fps = []; lat = []; lat99 = []; e2e = []; acc = []; pd = []; pt = []
    for d in sorted(glob.glob(g)):
        try: L = json.load(open(d + "/loop.json"))
        except Exception: continue
        ct = L.get("chunk_t")
        if not ct or len(ct) < L["chunks"] + 1: continue
        N = L["chunks"]; per = np.diff(ct); dec = L.get("decode_s") or 0.0; fr = np.array([9] + [12] * (N - 1)); ds = dec * fr / fr.sum()
        ttff.append(PREP + per[0] + ds[0]); st = per[1:]; fps.append(12 / np.median(st)); l = st + ds[1:]; lat.append(np.median(l)); lat99.append(np.percentile(l, 99))
        e2e.append(PREP + L["gen_s"] + dec); acc.append(L["plan"][1:].count("D"))
        for c in range(1, N): (pd if L["plan"][c] == "D" else pt).append(per[c])
    if not ttff: print("| %s | 0 (no chunk_t) | | | | | | | |" % label); continue
    print("| %s | %d | %.1f | %.2f | %.2f | %.2f | %s / %s | %.1f | %.1f |" % (label, len(ttff), np.mean(ttff), np.mean(fps), np.mean(lat), np.mean(lat99),
          "%.2f" % np.mean(pd) if pd else "-", "%.2f" % np.mean(pt) if pt else "-", np.mean(e2e), np.mean(acc)))
