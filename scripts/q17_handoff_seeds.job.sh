#!/bin/bash
# GPU6 (Q14 후): 업링크 그리드 Fast 시드 4321/9999 (S05 부분집합) -> 업링크 수치에도 시드 분산 부착; 이후 Base 플로어 추가 시드(777) S05/S01
true
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for i in $(seq 1 480); do [ -f $SCRATCH/exp/Q14_DONE ] && break; sleep 60; done
cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out
for SEED in 4321 9999; do
  echo "########## HANDOFF GRID S05 fast seed $SEED ($(date)) ##########"
  python -m specwm.handoff_grid --ckpt $PWD/lingbot-world-base-cam --scene 05 --image $SCRATCH/lingbot-world-v2/examples/05/image.jpg \
    --action-dir $SCRATCH/rt2_action_05 --prompt-file $SCRATCH/p_05.txt --fast-video $R/floor/fast/S05/W18_sink6/fast_alone_s$SEED/video.mp4 \
    --sigmas 0.4,0.625,0.833 --res 1.0,0.5 --bits 8 --steps 8 --seed $SEED --out $R/handoff_grid/S05_fs$SEED
done
F=$R/floor/base
for SC in 05 01; do D=$F/S${SC}_s777; [ -f $D/revisit.json ] && continue
  echo "########## BASE 70-step S$SC seed 777 ($(date)) ##########"
  python -m specwm.bench_base_official --ckpt $PWD/lingbot-world-base-cam --image $SCRATCH/lingbot-world-v2/examples/$SC/image.jpg \
    --action-path $SCRATCH/rt2_action_$SC --prompt-file $SCRATCH/p_$SC.txt --frames 161 --seed 777 --out $D
done
touch $SCRATCH/exp/Q17_DONE; echo "[Q17_DONE] $(date)"
