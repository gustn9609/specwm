#!/bin/bash
# GPU7 (Q13 후): 데모 프롬프트 5개 추가 (urban night/rain, lake winter, stonehenge night, hands fast) -> Fast/Base16/resync/갤러리 -> ctx_grid S02 다중 시드
true
for i in $(seq 1 480); do [ -f $SCRATCH/exp/Q13_DONE ] && break; sleep 60; done
export LIST=$SCRATCH/exp/demo_list2.json MARK=Q16_DEMO_DONE
bash $SCRATCH/scripts/demo_run.job.sh
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out; CK=$PWD/lingbot-world-v2-14b-causal-fast
echo "########## CTX GRID S02 seeds 4321,9999 ($(date)) ##########"
python -m specwm.ctx_grid --ckpt $CK --action-root $SCRATCH/rt2_action_02 --prompt-file $SCRATCH/p_02.txt --scene 02 --frames 161 \
  --base-video $R/floor/base/S02_s1234/video.mp4 --res 1.0,0.75,0.5 --bits 16,8,4 --sigmas 0,0.625,0.833 --seeds 4321,9999 --out $R/ctx_grid/S02
touch $SCRATCH/exp/Q16_DONE; echo "[Q16_DONE] $(date)"
