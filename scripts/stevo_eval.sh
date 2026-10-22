#!/bin/bash
# usage: bash scripts/stevo_eval.sh <outputs_dir> <run_dir> [camera_controlled 0|1]   (judge: local Qwen3-VL via OpenAI-compatible API on :30100)
cd /home/hk28965/specwm/stevo_bench_code; source ../venv/bin/activate
export OPENAI_API_KEY=local OPENAI_BASE_URL=http://127.0.0.1:${JUDGE_PORT:-30100}/v1
OUT=$1; RUN=$2; M=qwen3-vl-32b   # camera-controlled 여부는 outputs 폴더명(lingbot 등)으로 자동 감지
IFS=";" read -ra SPECS <<< "${CRITERIA:-control majority;physics majority;state majority}"   # 공개 코드에는 artifact/coherence judge 가 없고 physics 로 통합됨
for spec in "${SPECS[@]}"; do set -- $spec
  python -m eval.eval_cli --outputs $OUT --task_root ../data/stevo/tasks --run_dir $RUN --$1 --ensemble_size 3 --ensemble_mode $2 --workers 4 \
    --vlm_provider openai --control_judge_model $M --physics_judge_model $M --se_judge_model $M
done
python -m eval.summarize_results --run_dir $RUN/$(basename $OUT) --print_llm
