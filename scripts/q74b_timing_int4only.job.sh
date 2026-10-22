true
for i in $(seq 1 900); do [ -f $SCRATCH/exp/Q76_DONE ] && break; sleep 60; done
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out; CK=$PWD/lingbot-world-v2-14b-causal-fast
python -m specwm.latency_v2 --ckpt $CK --action-root $SCRATCH/rt2_action_00 --prompt-file $SCRATCH/p_00.txt --scene 00 --frames 161 --arms int4_only,awq_only --areas 399360,184320 --t5-gpu --steady-from 6 --k 6 --out $R/timing_s00/fast_extra
touch $SCRATCH/exp/Q74B_DONE; echo "[Q74B_DONE] $(date)"
