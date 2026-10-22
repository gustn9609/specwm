#!/bin/bash
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out
true
cd $S/lingbot-world-v2 && python -c "import ast; ast.parse(open('specwm/resync_v4.py').read()); print('v4 OK')"; cd $S/lingbot-world && python -c "import ast; ast.parse(open('specwm/bench_base_prefix2.py').read()); print('prefix2 OK')"; python -c "import ast; ast.parse(open('$S/analysis_r4.py').read()); print('analysis OK')"
cat > $S/r4F2.slurm << 'SEOF'
#!/bin/bash
true
export SPECWM_NO_FLASH=1; export OMP_NUM_THREADS=4; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V1=$S/lingbot-world; V2=$S/lingbot-world-v2; CK=$V1/lingbot-world-base-cam; B16=$R/revisit/base_official_S05_s16/video.mp4
v2() { cd $V2; PYTHONPATH=$V2 python -m specwm.resync_v4 --ckpt $V2/lingbot-world-v2-14b-causal-fast "$@"; }
echo "##### F2a: pre-encode base latent (161 f) #####"
v2 --action-root $S/rt2_action_05 --prompt-file $S/p_05.txt --scene 05 --frames 161 --seeds 1234 --base-video $B16 --configs W18_sink6 --arms fast_alone --encode-only $R/resync55/base16_lat.pt --out $R/tmp_enc
echo "##### F2b: sink capacity sweep 55 s #####"
v2 --action-root $S/traj55_action_05 --prompt-file $S/p_05.txt --scene 05 --frames 873 --seeds 1234 --base-video $B16 --base-latent $R/resync55/base16_lat.pt --configs none --extra-configs "W12_sink6:fast_alone+resync_k2,W18_sink12:fast_alone+resync_k4,W42_sink36:fast_alone+resync_k12" --out $R/resync55
echo "##### F2c: stale anchors 55 s rerun #####"
v2 --action-root $S/traj55_action_05 --prompt-file $S/p_05.txt --scene 05 --frames 873 --seeds 1234 --base-video $B16 --base-latent $R/resync55/base16_lat.pt --configs W24_sink18 --arms stale_d15_k20,stale_d30_k40,stale_d45_k60,ctrl_stale_k20 --stale-inject d15=20,d30=40,d45=60 --out $R/resync55
echo "##### F2d: target cost with swap time measured #####"
LAU=$R/hybrid/s05_launder/REFINE_0.40_0.20_0.08/video.mp4
for CFG in "386048 16" "386048 8" "225280 16" "225280 8" "99840 16" "99840 8"; do set -- $CFG
  cd $V1; PYTHONPATH=$V1 python -m specwm.bench_base_prefix2 --ckpt $CK --image $V2/examples/05/image.jpg --action-path $S/rt2_action_05 --prompt-file $S/p_05.txt --frames 73 --steps $2 --max-area $1 --prefix-latents 9 --mode repaint --seed 1234 --prefix-video $LAU --out $R/targetcost_res/area$1_s$2
done
echo "[R4F2_DONE]"
SEOF
JA=$(sbatch --parsable $S/anr4.slurm | tail -1); JG=$(sbatch --parsable --dependency=afterany:$JA $S/r4G_e2e.slurm | tail -1); JF=$(sbatch --parsable --dependency=afterany:$JG $S/r4F2.slurm | tail -1)
echo "anr4=$JA G=$JG F2=$JF"; squeue -u $USER -h -o "%.9i %.8j %.9T %.9M %.14R"
