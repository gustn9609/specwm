true
# usage: ARMS="name:ckpt:area:extra,..." ./run.sh <gpu> scripts/q83_arm.job.sh tag   (Great Wall S00 seed 9999, 161 frames)
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out/timing_s00
A="--image $SCRATCH/lingbot-world-v2/examples/00/image.jpg --action-path $SCRATCH/rt2_action_00 --prompt-file $SCRATCH/p_00.txt --frames 161 --seed 9999"
IFS=',' read -ra L <<< "$ARMS"
for spec in "${L[@]}"; do IFS=':' read -r NAME CK AREA EXTRA <<< "$spec"
  [ "$CK" = base ] && CKP=$PWD/lingbot-world-base-cam || CKP=$SCRATCH/lingbot-world-v2/bid_as_v1_cam
  [ -f $R/$NAME/video.mp4 ] && { echo "[skip] $NAME"; continue; }
  echo "########## RERUN $NAME ($(date)) ##########"; python -m specwm.bench_base_official --ckpt $CKP $A --max-area $AREA $EXTRA --out $R/$NAME
done; echo "[Q83_DONE $ARMS] $(date)"
