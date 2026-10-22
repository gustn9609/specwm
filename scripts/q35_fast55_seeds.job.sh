#!/bin/bash
# GPU4: Fast55 W18 sink18 시드 4321/9999 (55s retention 의 시드 분산) -> geo scan
true
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out; CK=$PWD/lingbot-world-v2-14b-causal-fast
for SD in 4321 9999; do D=$R/fast55/S05_W18_sink18_s$SD; [ -f $D/frames_all.npy ] && continue
  echo "########## FAST55 W18 seed $SD ($(date)) ##########"
  python -m specwm.e2e_stage --ckpt $CK --action-root $SCRATCH/traj55_action_05 --prompt-file $SCRATCH/p_05.txt --scene 05 --frames 873 --seed $SD \
    --window 18 --sink 18 --state $D --gen-until 73 --anchor fixed --decode || echo "!!! seed $SD FAILED"
done
python $SCRATCH/scripts/geo55.py $R/geo55.json --scan
touch $SCRATCH/exp/Q35_DONE; echo "[Q35_DONE] $(date)"
