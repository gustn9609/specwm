#!/bin/bash
S=$SCRATCH
scancel 966230 2>/dev/null; sleep 3
# 실패 기록 삭제 (재시도 대상이 되도록)
python3 - << 'PY'
import json, os
p = "$SCRATCH/lingbot-world-v2/specwm_out/fill_ours/fill_ours.json"
if os.path.exists(p):
    d = json.load(open(p))
    bad = [k for k, v in d["arms"].items() if "error" in v]
    for k in bad: del d["arms"][k]
    json.dump(d, open(p, "w"), indent=1)
    print("실패기록 %d개 삭제, 성공 %d개 유지" % (len(bad), len(d["arms"])))
else:
    print("아직 결과 파일 없음")
PY
sbatch $S/fill_ours.slurm | tail -1
squeue -u $USER -h -o "%.9i %.10j %.9T %.8M"
