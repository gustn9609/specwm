#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD
for SC in 05 01; do
  echo "########## BASE official, scene $SC ##########"
  python -m specwm.bench_base_official --ckpt $PWD/lingbot-world-base-cam \
    --image $SCRATCH/lingbot-world-v2/examples/$SC/image.jpg \
    --action-path $SCRATCH/rt2_action_$SC --prompt-file $SCRATCH/p_$SC.txt --frames 161 \
    --out $SCRATCH/lingbot-world-v2/specwm_out/revisit/base_official_S$SC
done
echo "########## BASE official, hands (static camera control) ##########"
python -m specwm.bench_base_official --ckpt $PWD/lingbot-world-base-cam \
  --image $SCRATCH/lingbot-world-v2/examples/hands/image.jpg \
  --action-path $SCRATCH/hands_action --prompt-file $SCRATCH/p_hands1.txt --frames 81 \
  --out $SCRATCH/lingbot-world-v2/specwm_out/hands/base_official_p1
