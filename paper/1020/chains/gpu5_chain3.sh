#!/bin/bash
# [10/21 00:30] GPTQ-4 file repair + mx24 profile: gptq_save.py finished (560 layers, 0 non-finite, 9.41 GB) but its reload check failed only because
# META["torch"] was a torch.TorchVersion object (not allowed by weights_only=True). (1) CPU: load the file we wrote ourselves (weights_only=False), cast
# every non-primitive meta value to str, re-save atomically, verify with weights_only=True; (2) run the 310 chunk copy on GPU 5 (its reuse path = "exists and
# loads" -> sentinel gptq4_save.ok) and move the queue's 310 to done/; (3) profile24.sh pass 2 (mx24 @832 + @640 on GPU 5 under the lock); (4) summary.
S=/home/hk28965/specwm; C=/home/hk28965/specwm/paper/1020/chains; Q=/home/hk28965/specwm/baselines/gen/queue; L=$C/gpu5_chain3.log; OUT=$S/specwm_out/gptq4/14b_g128_832.pt
echo "[$(date +%F_%H:%M)] sanitize meta" >> $L
CUDA_VISIBLE_DEVICES= $S/scripts/envrun.sh python -I - "$OUT" >> $L 2>&1 <<'EOF_PY'
import sys, os, torch
p = sys.argv[1]; d = torch.load(p, map_location="cpu", weights_only=False)
PRIM = (int, float, str, bool, type(None))
def san(v):
    if isinstance(v, torch.Tensor) or isinstance(v, PRIM) and type(v) in PRIM: return v
    if isinstance(v, dict): return {str(k): san(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [san(x) for x in v]
    return str(v)
bad = [k for k, v in d["meta"].items() if type(v) not in PRIM and not isinstance(v, (dict, list))]
print("meta keys with non-primitive values:", bad, [type(d["meta"][k]).__name__ for k in bad])
d = {"meta": san(d["meta"]), "layers": d["layers"]}
torch.save(d, p + ".tmp"); os.replace(p + ".tmp", p)
d2 = torch.load(p, map_location="cpu", weights_only=True); print("weights_only reload ok:", len(d2["layers"]), "layers", os.path.getsize(p) / 1e9, "GB")
EOF_PY
rc=$?; echo "[$(date +%F_%H:%M)] sanitize rc $rc" >> $L; [ $rc -eq 0 ] || { echo GPTQ4_REPAIR_FAIL >> $L; exit 1; }
bash $C/310_gptq4_save_gpu5.sh 5 >> $L 2>&1; rc=$?; echo "[$(date +%F_%H:%M)] 310 reuse-path rc $rc" >> $L
if [ $rc -eq 0 ] && [ -f $S/baselines/gen/sentinels/gptq4_save.ok ]; then mv $Q/8/310_gptq4_save.sh $Q/done/310_gptq4_save.sh.gpu5; echo GPTQ4_SAVE_OK >> $L; else echo GPTQ4_SAVE_FAIL >> $L; exit 1; fi
bash /tmp/claude-1741758895/-home-hk28965/76e7a8ef-ae2e-455b-8c73-052f5bde2a30/scratchpad/profile24/profile24.sh >> $L 2>&1; echo "[$(date +%F_%H:%M)] profile pass 2 rc $? ; PROFILE_MX24_DONE" >> $L
python3 -I $S/paper/1020/profile24_summary.py >> $L 2>&1; echo "[$(date +%F_%H:%M)] summary rc $? ; PROFILE_SUMMARY_DONE" >> $L
