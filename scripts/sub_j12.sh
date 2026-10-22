#!/bin/bash
S=$SCRATCH; module load gcc cuda python3 >/dev/null 2>&1; source $S/specwm-venv/bin/activate
python $S/sdpa_causal_patch.py
cd $S/lingbot-world-v2 && python -c "import ast; ast.parse(open('specwm/resync_v2.py').read()); print('resync_v2 서버 문법 OK')"
cd $S/lingbot-world && python -c "import ast; ast.parse(open('specwm/bench_base_prefix.py').read()); print('bench_base_prefix 서버 문법 OK')"
R=$S/lingbot-world-v2/specwm_out
# ---------- J1: Fast 전용 ----------
cat > $S/j1.slurm << 'SEOF'
#!/bin/bash
true
export SPECWM_NO_FLASH=1; export OMP_NUM_THREADS=4; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out
python -m specwm.resync_v2 --ckpt $PWD/lingbot-world-v2-14b-causal-fast --action-root $SCRATCH/rt2_action_05 \
  --prompt-file $SCRATCH/p_05.txt --scene 05 --frames 161 --seeds 1234,4321,9999 \
  --base-video $R/revisit/base_official_S05/video.mp4 --base16-video $R/revisit/base_official_S05_s16/video.mp4 \
  --out $R/resync_v2
SEOF
# ---------- J2: Base 전용 (E1, E6, E3) ----------
cat > $S/j2.slurm << 'SEOF'
#!/bin/bash
true
export SPECWM_NO_FLASH=1
cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out; CK=$PWD/lingbot-world-base-cam
echo "########## E1 noise floor: Base 16-step seeds 4321, 9999 (S05) ##########"
for SD in 4321 9999; do
  python -m specwm.bench_base_official --ckpt $CK --image $SCRATCH/lingbot-world-v2/examples/05/image.jpg --action-path $SCRATCH/rt2_action_05 \
    --prompt-file $SCRATCH/p_05.txt --frames 161 --steps 16 --seed $SD --out $R/revisit/base_official_S05_s16_seed$SD
done
echo "########## E6 causal mask: hands 16-step N=3/6/12 ##########"
export SPECWM_LAT_F=21
for N in 3 6 12; do
  SPECWM_CAUSAL_FRAMES=$N python -m specwm.bench_base_official --ckpt $CK --image $SCRATCH/lingbot-world-v2/examples/hands/image.jpg --action-path $SCRATCH/hands_action \
    --prompt-file $SCRATCH/p_hands1.txt --frames 81 --steps 16 --out $R/hands/base_causal_N$N
done
unset SPECWM_CAUSAL_FRAMES SPECWM_LAT_F
echo "########## E3 prefix-conditioned Base 16-step (prefix = Fast chunks 0..2 = 9 latents) ##########"
python -m specwm.bench_base_prefix --ckpt $CK --image $SCRATCH/lingbot-world-v2/examples/05/image.jpg --action-path $SCRATCH/rt2_action_05 \
  --prompt-file $SCRATCH/p_05.txt --prefix-video $R/resync_s05/fast_alone/video.mp4 --prefix-latents 9 --frames 161 --steps 16 \
  --out $R/revisit/base_prefix_S05_L9
echo "[J2_DONE]"
SEOF
sbatch $S/j1.slurm | tail -1; sbatch $S/j2.slurm | tail -1
squeue -u $USER -h -o "%.9i %.10j %.9T %.8M"
