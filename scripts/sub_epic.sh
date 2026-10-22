#!/bin/bash
S=$SCRATCH
for W in F_cut_tomato G_peel_garlic D_pickup_scraps C_cupboard_garlic E_tomato_slices_on_bread; do
cat > $S/epic_$W.slurm << SEOF
#!/bin/bash
true
export SPECWM_NO_FLASH=1; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd \$SCRATCH/lingbot-world; export PYTHONPATH=\$PWD
E=\$SCRATCH/epic/$W
python -m specwm.bench_base_official --ckpt \$PWD/lingbot-world-base-cam --image \$E/image.jpg --action-path \$E/action --prompt-file \$E/prompt.txt --frames 161 --steps 70 --seed 1234 --out \$SCRATCH/lingbot-world-v2/specwm_out/epic/${W}_base70
echo "[EPIC_BASE_DONE] $W"
SEOF
sbatch $S/epic_$W.slurm | tail -1
done
squeue -u $USER -h -o "%.9i %.10j %.9T %.8M %.12R"
