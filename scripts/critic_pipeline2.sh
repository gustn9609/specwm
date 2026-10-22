#!/bin/bash
# second half of the critic pipeline (GPUs 5,6): train critics, calibrate, blind-eval, run the e3/critic loop sweeps, then sheets once ALL loop sweeps are done.
cd /home/hk28965/specwm; source venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=exp/queue; C=lingbot-world-v2/specwm_out/critic; mkdir -p $C; PL=$L/critic_pipeline.log
CUDA_VISIBLE_DEVICES=5 python scripts/critic_train.py $C/main --epochs 30 --dim 256 --layers 6 --holdout-extra 8 > $L/critic_main.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python scripts/critic_train.py $C/small --epochs 30 --dim 128 --layers 4 --holdout-extra 8 > $L/critic_small.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python scripts/critic_train.py $C/noprev --epochs 30 --dim 256 --layers 6 --holdout-extra 8 --no-prev > $L/critic_noprev.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python scripts/critic_train.py $C/nocam --epochs 30 --dim 256 --layers 6 --holdout-extra 8 --no-cam > $L/critic_nocam.log 2>&1 &
wait; echo "[pipeline2] training done ($(date))" >> $PL; grep -h "CRITIC_DONE" $L/critic_main.log $L/critic_noprev.log $L/critic_nocam.log $L/critic_small.log | cut -c1-300 >> $PL
CUDA_VISIBLE_DEVICES=5 python scripts/critic_calibrate.py $C/main exp/kveval/critic_calib.json --accept 0.3,0.5,0.7 > $L/critic_calib.log 2>&1; grep -v Warning $L/critic_calib.log >> $PL
CUDA_VISIBLE_DEVICES=6 python scripts/critic_eval_blind.py $C/main/critic.pt exp/kveval/critic_blind_main.json > $L/critic_blind_main.log 2>&1 &
read TC50 TC70 TE50 TE70 <<< $(python3 -c "import json;d=json.load(open('exp/kveval/critic_calib.json'));print(d['critic']['0.50']['tau'],d['critic']['0.70']['tau'],d['e3']['0.50']['tau'],d['e3']['0.70']['tau'])")
echo "[pipeline2] tau critic50=$TC50 critic70=$TC70 e3_50=$TE50 e3_70=$TE70" >> $PL
[ -z "$TC50" ] && { echo "[pipeline2] calibration failed, abort" >> $PL; exit 1; }
TASKS=sand_filling_up_jar,sandcastle_wave_impact; SEEDS=1234,4321,9999,101,102,103,104,105,106,107,108; CP=/home/hk28965/specwm/$C/main/critic.pt
run_loop() { (cd lingbot-world-v2; PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=$1 python -m specwm.spec_ar_loop --tasks $TASKS --seeds $SEEDS --verifiers "$2" --out /home/hk28965/specwm/lingbot-world-v2/specwm_out/loop > /home/hk28965/specwm/$L/loop_$1.log 2>&1); }
run_loop 5 "e3:$TE50,e3:$TE70" & run_loop 6 "critic:$CP:$TC50,critic:$CP:$TC70" &
wait; until ! pgrep -f "spec_ar_loo[p]" > /dev/null; do sleep 60; done
echo "[pipeline2] all loops done ($(date))" >> $PL; grep -h "LOOP_FAIL\|Traceback" $L/loop_*.log | head -5 >> $PL
ARMS=$(ls -d lingbot-world-v2/specwm_out/loop/sand_filling_up_jar/*_s1234 | xargs -n1 basename | sed 's/_s1234$//' | paste -sd,)
BLIND_ROOT=/home/hk28965/specwm/lingbot-world-v2/specwm_out/loop python3 scripts/blind_sheets.py paper/overnight/blind_loop $TASKS $SEEDS $ARMS >> $PL 2>&1
echo "[PIPELINE_DONE] $(date)" >> $PL
