#!/bin/bash
# GPU5: (1) PoseGram 영상 VisionReward 채점 (품질 확인) (2) S05 pose 블렌딩 α0.4 2시드 (ghosting 경계) -> geo
true
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "########## VR posegram ($(date)) ##########"
$SCRATCH/vr-venv/bin/python $SCRATCH/scripts/vr_daemon.py --poll 30 --max-hours 1 --roots posegram,fast55,base55,oracle
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out; CK=$PWD/lingbot-world-v2-14b-causal-fast
for SD in 1234 4321; do echo "########## PG S05 blend a0.4 s$SD ($(date)) ##########"
  python -m specwm.posegram --ckpt $CK --action-root $SCRATCH/traj55_action_05 --prompt-file $SCRATCH/p_05.txt --scene 05 --frames 873 --window 18 --sink 18 --mode posegram_blend --alpha 0.4 --write first --seed $SD --out $R/posegram/S05_W18 || echo "!!! FAILED"; done
python $SCRATCH/scripts/geo55.py $R/geo55.json --scan; touch $SCRATCH/exp/Q48_DONE; echo "[Q48_DONE] $(date)"
