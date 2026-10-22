true
# Fast 14B few-step rollout on S00 seed 9999 (480x832 budget): 1-step (node 0), 2-step (nodes 0,3: σ 1.0→0.625), 3-step (0,1,3); 4-step reference = lowres_sr/fast14b_399360
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out/fewstep_s00; mkdir -p $R; CK=$PWD/lingbot-world-v2-14b-causal-fast
for N in "0" "0,3" "0,1,3" "0,2"; do T=$(echo $N | tr -d ','); echo "########## FAST few-step nodes=$N ($(date)) ##########"
  python -m specwm.stevo_gen --ckpt $CK --tasks $SCRATCH/data/stevo/tasks_s00.json --out $R/fast14b_nodes$T --arm fast_bf16 --frames 161 --max-area 399360 --seed 9999 --t5-gpu --nodes $N; done
touch $SCRATCH/exp/Q92_DONE; echo "[Q92_DONE] $(date)"
