#!/bin/bash
S=$SCRATCH; scancel 971384
cat > $S/epicdev_3.slurm << 'SEOF'
#!/bin/bash
true
export SPECWM_NO_FLASH=1; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD
# 5 s (81 frames) x 2 seeds x {G, F}; 첫 오류가 4-6 s 에 나오므로 짧은 클립 + 시드 다양화
for W in G_peel_garlic F_cut_tomato; do for SD in 7 42; do
  E=$SCRATCH/epic/$W; echo "##### $W 81f seed $SD #####"
  python -m specwm.bench_base_official --ckpt $PWD/lingbot-world-base-cam --image $E/image.jpg --action-path $E/action --prompt-file $E/prompt.txt --frames 81 --steps 70 --seed $SD --out $SCRATCH/lingbot-world-v2/specwm_out/epic/${W}_base70_81f_s$SD
  echo "[EPIC_BASE_DONE] $W 81f s$SD"
done; done
SEOF
sbatch --parsable $S/epicdev_3.slurm | tail -1; sleep 15; squeue -u $USER -h -o "%.9i %.8j %.9T %.8M %.12R"
