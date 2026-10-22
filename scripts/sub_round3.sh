#!/bin/bash
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out
true
python $S/mkact20.py
cd $S/lingbot-world && python -c "import ast; ast.parse(open('specwm/bench_base_prefix2.py').read()); print('prefix2 서버 OK')"
cd $S/lingbot-world-v2 && python -c "import ast; ast.parse(open('specwm/resync_v3.py').read()); print('resync_v3 서버 OK')"
# ---------------- J-A (v1, gh-dev): E3 원인분리 + E3'-a/b/c ----------------
cat > $S/ja.slurm << 'SEOF'
#!/bin/bash
true
export SPECWM_NO_FLASH=1; cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD
R=$SCRATCH/lingbot-world-v2/specwm_out; CK=$PWD/lingbot-world-base-cam; IMG=$SCRATCH/lingbot-world-v2/examples/05/image.jpg; ACT=$SCRATCH/rt2_action_05; PR=$SCRATCH/p_05.txt
FAST=$R/resync_v2/W18_sink6/fast_alone_s1234/video.mp4
run() { python -m specwm.bench_base_prefix2 --ckpt $CK --image $IMG --action-path $ACT --prompt-file $PR --frames 161 --steps 16 --prefix-latents 9 "$@"; }
echo "##### E3-sep-a: Base16 on its OWN prefix (seed 1234), mask #####";      run --mode mask --prefix-video $R/revisit/base_official_S05_s16/video.mp4 --out $R/revisit/e3sep_self_mask
echo "##### E3-sep-b: Base16 on OTHER-seed Base prefix (4321), mask #####";  run --mode mask --prefix-video $R/revisit/base_official_S05_s16_seed4321/video.mp4 --out $R/revisit/e3sep_otherseed_mask
echo "##### E3'-a: RePaint replacement, Fast prefix #####";                   run --mode repaint --prefix-video $FAST --out $R/revisit/e3a_repaint_fast
echo "##### E3'-a control: RePaint, Base own prefix #####";                   run --mode repaint --prefix-video $R/revisit/base_official_S05_s16/video.mp4 --out $R/revisit/e3a_repaint_self
echo "##### E3'-b: Base-causal N=12 + RePaint, Fast prefix #####"
SPECWM_CAUSAL_FRAMES=12 SPECWM_LAT_F=41 run --mode repaint --prefix-video $FAST --out $R/revisit/e3b_causal12_repaint_fast
echo "##### E3'-c: launder Fast prefix with low-noise expert (SDEdit 0.40,0.20,0.08) #####"
python -m specwm.hybrid_refine --ckpt $CK --scene 05 --action-dir $ACT --prompt-file $PR --fast-latent $R/resync_v2/W18_sink6/fast_alone_s1234/latent.pt --frames 153 --schedules "0.40,0.20,0.08" --out $R/hybrid/s05_launder
LAUND=$(ls $R/hybrid/s05_launder/REFINE*/video.mp4 | head -1); echo "laundered: $LAUND"
echo "##### E3'-c: RePaint with laundered Fast prefix #####";                run --mode repaint --prefix-video $LAUND --out $R/revisit/e3c_repaint_laundered
echo "[JA_DONE]"
SEOF
# ---------------- J-B (v1, gh): Base16 20s x2 seeds ----------------
cat > $S/jb.slurm << 'SEOF'
#!/bin/bash
true
export SPECWM_NO_FLASH=1; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD
R=$SCRATCH/lingbot-world-v2/specwm_out
for SD in 1234 4321; do
  echo "##### Base16 20 s, seed $SD #####"
  python -m specwm.bench_base_official --ckpt $PWD/lingbot-world-base-cam --image $SCRATCH/lingbot-world-v2/examples/05/image.jpg --action-path $SCRATCH/traj20_action_05 \
    --prompt-file $SCRATCH/p_05.txt --frames 321 --steps 16 --seed $SD --out $R/revisit/base20_S05_s16_seed$SD
done
echo "[JB_DONE]"
SEOF
# ---------------- J-C2 (v2, gh): 20s Fast arms ----------------
cat > $S/jc2.slurm << 'SEOF'
#!/bin/bash
true
export SPECWM_NO_FLASH=1; export OMP_NUM_THREADS=4; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out
python -m specwm.resync_v3 --ckpt $PWD/lingbot-world-v2-14b-causal-fast --action-root $SCRATCH/traj20_action_05 --prompt-file $SCRATCH/p_05.txt --scene 05 --frames 321 \
  --seeds 1234 --base-video $R/revisit/base_official_S05/video.mp4 --configs W18_sink6,W24_sink18 --out $R/resync20
SEOF
sbatch $S/ja.slurm | tail -1; sbatch $S/jb.slurm | tail -1; sbatch $S/jc2.slurm | tail -1
squeue -u $USER -h -o "%.9i %.10j %.9T %.8M %.12R"
