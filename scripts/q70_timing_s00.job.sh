true
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out; CK=$PWD/lingbot-world-v2-14b-causal-fast
echo "########## TIMING S00 fast arms, T5 GPU, 3 resolutions ($(date)) ##########"
python -m specwm.latency_v2 --ckpt $CK --action-root $SCRATCH/rt2_action_00 --prompt-file $SCRATCH/p_00.txt --scene 00 --frames 161 \
  --arms fast_bf16,fast_kv4,ours --areas 399360,184320,122880 --t5-gpu --steady-from 6 --k 6 --out $R/timing_s00/fast_t5gpu
echo "########## TIMING S00 fast bf16, T5 CPU (official default) ($(date)) ##########"
python -m specwm.latency_v2 --ckpt $CK --action-root $SCRATCH/rt2_action_00 --prompt-file $SCRATCH/p_00.txt --scene 00 --frames 161 \
  --arms fast_bf16 --areas 399360 --steady-from 6 --k 6 --out $R/timing_s00/fast_t5cpu
echo "########## TIMING S00 chunk_size=1 ($(date)) ##########"
python -m specwm.latency_v2 --ckpt $CK --action-root $SCRATCH/rt2_action_00 --prompt-file $SCRATCH/p_00.txt --scene 00 --frames 161 \
  --arms fast_bf16,ours --areas 399360,184320 --t5-gpu --chunk-size 1 --steady-from 6 --k 6 --out $R/timing_s00/fast_cs1
touch $SCRATCH/exp/Q70_DONE; echo "[Q70_DONE] $(date)"
