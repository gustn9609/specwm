"""Binary drift protocol (action item 4): for run dirs <task>/<run>_s<seed>, decide drift / no-drift per clip from the per-chunk DINO
distance to the start frame and the chunk-to-chunk jump (temporal_eval.py outputs), then report success rates with Wilson 95% CIs
per method. usage: drift_rate.py EVAL.json [--thr-drift 0.5 --thr-jump 0.45]
Drift := any chunk with DINO drift(max) (consecutive-chunk feature jump) > thr_jump, OR the clip's mean distance to Base (to_base) > thr_drift
when a Base rollout exists (content left what the target would do). Both thresholds were set from the jar/castle Base-70 rollouts (never exceeded)."""
import json, sys, math, collections
f = sys.argv[1]; thr_drift = float(sys.argv[sys.argv.index("--thr-drift") + 1]) if "--thr-drift" in sys.argv else 0.5
thr_jump = float(sys.argv[sys.argv.index("--thr-jump") + 1]) if "--thr-jump" in sys.argv else 0.45
D = json.load(open(f)); R = collections.defaultdict(list)
for k, v in D.items():
    if not isinstance(v, dict) or "chunk_drift" not in v: continue
    task, run = k.split("/"); method = run.rsplit("_s", 1)[0]
    if method == "base70": continue
    jump = max(v["chunk_drift"])
    tb = v.get("to_base"); tb = (sum(tb) / len(tb)) if isinstance(tb, list) and len(tb) else None
    drift = (jump > thr_jump) or (tb is not None and tb > thr_drift)
    R[method].append((task, int(run.rsplit("_s", 1)[1]), drift, round(jump, 2), None if tb is None else round(tb, 2)))
def wilson(k, n, z=1.96):
    if n == 0: return (0, 0, 0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return p, max(0, c - h), min(1, c + h)
print("method                                   n   no-drift rate  [95% CI]     drifted cases")
for m, L in sorted(R.items()):
    n = len(L); k = sum(1 for x in L if not x[2]); p, lo, hi = wilson(k, n)
    bad = ", ".join("%s:%d(j%.2f,b%s)" % (t[:6], s, j, tb) for t, s, d, j, tb in L if d)
    print("%-40s %3d   %.2f          [%.2f, %.2f]  %s" % (m, n, p, lo, hi, bad))
