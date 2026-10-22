#!/bin/bash
S=$SCRATCH
scancel 971376 971377 971378 971379
mk() { # $1 name, $2.. windows
cat > $S/epicdev_$1.slurm << SEOF
#!/bin/bash
true
export SPECWM_NO_FLASH=1; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd \$SCRATCH/lingbot-world; export PYTHONPATH=\$PWD
for W in ${@:2}; do
  E=\$SCRATCH/epic/\$W; echo "##### \$W #####"
  python -m specwm.bench_base_official --ckpt \$PWD/lingbot-world-base-cam --image \$E/image.jpg --action-path \$E/action --prompt-file \$E/prompt.txt --frames 161 --steps 70 --seed 1234 --out \$SCRATCH/lingbot-world-v2/specwm_out/epic/\${W}_base70
  echo "[EPIC_BASE_DONE] \$W"
done
SEOF
}
mk 1 F_cut_tomato G_peel_garlic; mk 2 D_pickup_scraps C_cupboard_garlic
J1=$(sbatch $S/epicdev_1.slurm | awk '{print $4}'); echo "job1 $J1"
J2=$(sbatch --dependency=afterany:$J1 $S/epicdev_2.slurm | awk '{print $4}'); echo "job2 $J2"
sleep 20; squeue -u $USER -h -o "%.9i %.8j %.9T %.8M %.12R"
