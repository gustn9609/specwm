#!/bin/bash
# after the planrc queue: score every plan candidate with the v2-bid one-step verifier (③) at plan res, build blind sheets for the renders.
cd /home/hk28965/specwm; source venv/bin/activate; export SPECWM_NO_FLASH=1
until [ "$(grep -l WORKER_DONE exp/queue/wplanrc_{0,1,2,3,4,5,6,7}.log 2>/dev/null | wc -l)" -ge 8 ]; do sleep 60; done; date
SEEDS=1234,4321,9999,101,102,103,104,105,106,107,108
python3 - <<'PY'
import json, os
V0="/home/hk28965/specwm/lingbot-world-v2/specwm_out/spec_v0"; mp={}
for t in ("sand_filling_up_jar","sandcastle_wave_impact"):
    for s in (1234,4321,9999,*range(101,109)):
        for arm in ("planv2_l4c","planv2c1_l4c","planv2c2_l4c","planv2c3_l4c"):
            if os.path.exists(f"{V0}/{t}/{arm}_s{s}/video.mp4"): mp[f"{t}|{arm}|{s}"]=dict(task=t,seed=s,arm=arm)
json.dump(mp,open("/home/hk28965/specwm/exp/kveval/planrc_map.json","w")); print("plans",len(mp))
PY
(cd lingbot-world; SPECWM_BASE_CKPT=/home/hk28965/specwm/lingbot-world-v2/bid_as_v1_cam PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=4 python -m specwm.base_verify /home/hk28965/specwm/exp/kveval/planrc_bv_s0.json --mapping /home/hk28965/specwm/exp/kveval/planrc_map.json --area 184320 --frames 153 --shard 0/2 > /home/hk28965/specwm/exp/queue/planrc_bv0.log 2>&1) &
(cd lingbot-world; SPECWM_BASE_CKPT=/home/hk28965/specwm/lingbot-world-v2/bid_as_v1_cam PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=5 python -m specwm.base_verify /home/hk28965/specwm/exp/kveval/planrc_bv_s1.json --mapping /home/hk28965/specwm/exp/kveval/planrc_map.json --area 184320 --frames 153 --shard 1/2 > /home/hk28965/specwm/exp/queue/planrc_bv1.log 2>&1) &
python3 scripts/blind_sheets.py paper/overnight/blind_planrc sand_filling_up_jar,sandcastle_wave_impact $SEEDS plhv2_plan6_ck400_l4c_st0_never,plhv2c1_plan6_ck400_l4c_st0_never,plhv2c2_plan6_ck400_l4c_st0_never,plhv2c3_plan6_ck400_l4c_st0_never
wait; echo "[PLANRC_POST_DONE] $(date)"
