true
# A3-Fast: 단일 체크포인트 self-speculative 스택 (S00 seed 9999, 768x512, 161 f): bf16 4-step(ref) -> 2-step -> 2-step+int6 -> 2-step+int6+kv4 -> 2-step+int6+kv4 @360x512
true
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out/a3_fast; mkdir -p $R; CK=$PWD/lingbot-world-v2-14b-causal-fast
g() { [ -f $R/$1/S00_s9999.mp4 ] && return; echo "########## A3 FAST $1 ($(date)) ##########"; python -m specwm.stevo_gen --ckpt $CK --tasks $SCRATCH/data/stevo/tasks_s00.json --out $R/$1 --frames 161 --seed 9999 --t5-gpu "${@:2}"; }
g s2_gptq6 --arm gptq6 --max-area 399360 --nodes 0,3
g s2_gptq6_kv4 --arm gptq6_kv4 --max-area 399360 --nodes 0,3
g s2_ours_360 --arm ours --max-area 184320 --nodes 0,3
g s2_bf16_360 --arm fast_bf16 --max-area 184320 --nodes 0,3
touch $SCRATCH/exp/Q97_DONE; echo "[Q97_DONE] $(date)"
