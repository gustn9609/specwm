#!/bin/bash
S=$SCRATCH; cd $S
echo "== examples (v2) =="; for d in lingbot-world-v2/examples/*/; do echo "$d: $(ls $d | tr '\n' ' ')"; done
echo "== action dirs / prompts =="; ls -d rt2_action_* traj*_action_* 2>/dev/null | tr '\n' ' '; echo; ls p_*.txt 2>/dev/null | tr '\n' ' '; echo
for p in p_*.txt; do echo "$p: $(head -c 120 $p)"; done
echo "== traj60 vs rt2 =="; module load gcc cuda python3 >/dev/null 2>&1; source $S/specwm-venv/bin/activate
python - << 'PY'
import numpy as np, os
S="$SCRATCH"
for f in ["traj60_idx.npy","traj60_idx_05.npy","revisit_rt_idx_05.npy","revisit_rt_idx_01.npy","traj20_idx_05.npy"]:
    p=os.path.join(S,f); print(f, np.load(p).shape if os.path.exists(p) else "없음")
r=np.load(S+"/revisit_rt_idx_05.npy"); t60=[f for f in ["traj60_idx_05.npy","traj60_idx.npy"] if os.path.exists(S+"/"+f)]
if t60:
    a=np.load(S+"/"+t60[0]); print("traj60 first161 == rt2_05:", np.array_equal(a[:161], r)); 
    P=np.load(S+"/traj60_action_05/poses.npy") if os.path.exists(S+"/traj60_action_05/poses.npy") else None
    if P is not None:
        F=P[:,:3,2]; ang=np.degrees(np.arccos(np.clip((F[0]*F).sum(1),-1,1))); 
        import numpy as np
        homes=[i for i in range(20,len(ang)-1) if ang[i]<8 and ang[i]<=ang[i-1] and ang[i]<=ang[i+1]]
        print("traj60 frames", len(P), "home-view candidates (angle<8deg local min) at s:", [round(i/16,1) for i in homes][:20])
PY
echo "== binding_v2 commit / kvstore =="; grep -nE "def commit|def reset_caches|sink|def chunk_noise" lingbot-world-v2/specwm/binding_v2.py | head -12
grep -nE "class |def (put|write|commit|append|evict|__init__)" lingbot-world-v2/specwm/kvstore2.py 2>/dev/null | head -12
echo "== bench_base_prefix2 args =="; grep -nE "add_argument" lingbot-world/specwm/bench_base_prefix2.py | cut -c1-150
echo "== hybrid_refine args =="; grep -nE "add_argument" lingbot-world/specwm/hybrid_refine.py | cut -c1-150
echo "== revisit S01 setup =="; grep -E "action|prompt|image" lingbot-world-v2/specwm_out/revisit/base_official_S01/revisit.json 2>/dev/null | head -5
ls lingbot-world-v2/specwm_out/revisit/ | head -40 | tr '\n' ' '
