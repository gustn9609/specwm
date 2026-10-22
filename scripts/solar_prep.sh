#!/bin/bash
S=$SCRATCH
true
source $S/SolarWM/.venv-solar/bin/activate
python -c "import imageio, imageio_ffmpeg, decord; print('solar venv: imageio', imageio.__version__, 'decord', decord.__version__)" 2>&1 | tail -1
cd $S && python solar_adapter.py --image $S/lingbot-world-v2/examples/05/image.jpg --poses $S/rt2_action_05/poses.npy \
  --caption-file $S/p_05.txt --out $S/solar_custom --sample-id lingbot_s05 --pixel-frames 161 2>&1 | tail -12
echo "=== stage1 config: data.encoding / validation 전체 ==="
grep -nE "^\s+encoding:|^validation:" -A9 $S/SolarWM/configs/examples/wan22_ti2v_5b/infer_stage1_tf_anyflow_v1_5_81f.yaml | head -30
echo "=== stage2 81f config validation ==="
grep -nA6 "^validation:" $S/SolarWM/configs/examples/wan22_ti2v_5b/infer_stage2_sgf_81f.yaml
echo "=== 다운로드 ==="; du -sh $S/SolarWM-models/SolarWM-5B-tf-stage1-81f/
echo "=== vbench ==="; tail -8 $S/vbench_install3.log 2>/dev/null
