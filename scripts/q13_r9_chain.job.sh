#!/bin/bash
# GPU7 (Q11 후): 진짜 x_t 업링크 핸드오프 (E-A) + 스트림 검사기 (E-B) = 기존 r9handoff 파이프라인 재실행 -> hands Fast 플로어 + ctx 그리드
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
R=$SCRATCH/lingbot-world-v2/specwm_out
for i in $(seq 1 300); do [ -f $SCRATCH/exp/Q11_DONE ] && break; sleep 60; done
echo "########## R9 HANDOFF (true x_t, E-A/E-B) ($(date)) ##########"
bash $SCRATCH/scripts/r9handoff.job.sh
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; CK=$PWD/lingbot-world-v2-14b-causal-fast
echo "########## FAST floor hands ($(date)) ##########"
python -m specwm.resync_v4 --ckpt $CK --action-root $SCRATCH/hands_action --prompt-file $SCRATCH/p_hands1.txt --scene hands --frames 81 \
  --arms fast_alone --configs W18_sink6 --seeds 1234,4321,9999 --out $R/floor/fast/Shands
echo "########## CTX GRID hands (k=3) ($(date)) ##########"
python -m specwm.ctx_grid --ckpt $CK --action-root $SCRATCH/hands_action --prompt-file $SCRATCH/p_hands1.txt --scene hands --frames 81 --k 3 \
  --base-video $R/floor/base/Shands_s1234/video.mp4 --res 1.0,0.5 --bits 16,8,4 --sigmas 0,0.625,0.833 --seeds 1234,4321 --out $R/ctx_grid/Shands
touch $SCRATCH/exp/Q13_DONE; echo "[Q13_DONE] $(date)"
