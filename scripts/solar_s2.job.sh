#!/bin/bash
true
source $SCRATCH/SolarWM/.venv-solar/bin/activate
cd $SCRATCH/SolarWM
export PYTHONUNBUFFERED=1
export PATH=$SCRATCH/bin:$PATH
torchrun --standalone --nproc-per-node=1 -m solarwm infer \
  --config configs/examples/wan22_ti2v_5b/infer_stage2_sgf_camera_length.yaml \
  --set model.base_path=$SCRATCH/SolarWM-models/SolarWM-5B-base \
  --set checkpoint.path=$SCRATCH/SolarWM-models/SolarWM-5B-sgf-stage2-81f \
  --set data.index_root=$SCRATCH/solar_custom \
  --set data.transport.root=$SCRATCH/solar_custom \
  --set data.test_index=recipes/custom/raw-wds/test-index.jsonl.gz \
  --set data.train_index=recipes/custom/raw-wds/test-index.jsonl.gz \
  --set data.num_workers=1 \
  --set validation.sample_count=1 \
  --set inference.run_id=lingbot_s05_stage2 \
  --set runtime.output_dir=$SCRATCH/solar_out/stage2_s05
echo "[SOLAR_S2_DONE] exit=$?"
