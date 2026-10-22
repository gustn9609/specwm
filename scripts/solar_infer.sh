#!/bin/bash
S=$SCRATCH
# Stage2 스모크: 커스텀 샘플 1개, 카메라 길이만큼 롤아웃
cat > $S/solar_s2.slurm << 'SEOF'
#!/bin/bash
true
source $SCRATCH/SolarWM/.venv-solar/bin/activate
cd $SCRATCH/SolarWM
export PYTHONUNBUFFERED=1
torchrun --standalone --nproc-per-node=1 -m solarwm infer \
  --config configs/examples/wan22_ti2v_5b/infer_stage2_sgf_camera_length.yaml \
  --set model.base_path=$SCRATCH/SolarWM-models/SolarWM-5B-base \
  --set checkpoint.path=$SCRATCH/SolarWM-models/SolarWM-5B-sgf-stage2-81f \
  --set data.index_root=$SCRATCH/solar_custom \
  --set data.transport.root=$SCRATCH/solar_custom \
  --set data.test_index=recipes/custom/raw-wds/test-index.jsonl.gz \
  --set data.train_index=recipes/custom/raw-wds/test-index.jsonl.gz \
  --set data.num_workers=0 \
  --set validation.sample_count=1 \
  --set inference.run_id=lingbot_s05_stage2 \
  --set runtime.output_dir=$SCRATCH/solar_out/stage2_s05
echo "[SOLAR_S2_DONE] exit=$?"
SEOF
sbatch $S/solar_s2.slurm | tail -1
squeue -u $USER -h -o "%.9i %.10j %.9T %.8M"
echo "=== latency smoke ==="; grep -E "^\[fast_bf16\]|LATENCY_DONE|실패|Error" $S/lat_smoke.968298.out 2>/dev/null | tail -4
echo "=== base official ==="; grep -E "^\[load\]|BASE official|BASE_OFFICIAL_DONE|Error|Traceback" $S/base_official.968288.out 2>/dev/null | tail -4
