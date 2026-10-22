true
# Fast extra quantization arms on S00 (int4 packed + kv4, AWQ int4 + kv4, GPTQ int6 weight-only, window 36) — q72 shard 0 종료 후 GPU 3
for i in $(seq 1 600); do grep -q "Q72_DONE shard 0/3" $SCRATCH/logs/q72_fast_0.*.out 2>/dev/null && break; sleep 60; done
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out; CK=$PWD/lingbot-world-v2-14b-causal-fast
echo "########## TIMING S00 extra quant arms ($(date)) ##########"
python -m specwm.latency_v2 --ckpt $CK --action-root $SCRATCH/rt2_action_00 --prompt-file $SCRATCH/p_00.txt --scene 00 --frames 161 \
  --arms int4,awq,gptq6_only,fast_w36 --areas 399360,184320 --t5-gpu --steady-from 6 --k 6 --out $R/timing_s00/fast_extra
touch $SCRATCH/exp/Q74_DONE; echo "[Q74_DONE] $(date)"
