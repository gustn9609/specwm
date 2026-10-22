true
# LikePhys benchmark with LingBot Base as scorer (GPU 2, after B1). 11 scenarios x 10 subgroups x (1 valid + all invalid) ≈ 822 videos @ ~25 s → ~5.5 h; --per-sub 3 로 시작 (≈400 videos, ~3 h)
true
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD:$SCRATCH/scripts; mkdir -p $SCRATCH/lingbot-world-v2/specwm_out/likephys
python $SCRATCH/scripts/likephys_run.py --ckpt $PWD/lingbot-world-base-cam --data $SCRATCH/likephys/data/data --action-dir $SCRATCH/data/likephys_static_action --per-sub 3 --out $SCRATCH/lingbot-world-v2/specwm_out/likephys/base_ppe.json 2>&1 | grep -E "^\[|PPE|LIKEPHYS_DONE|Error|Traceback"
touch $SCRATCH/exp/Q107_DONE; echo "[Q107_DONE] $(date)"
