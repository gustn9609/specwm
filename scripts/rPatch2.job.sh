#!/bin/bash
true
export SPECWM_NO_FLASH=1; export OMP_NUM_THREADS=4; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V2=$S/lingbot-world-v2
CKF=$V2/lingbot-world-v2-14b-causal-fast; PR=$S/p_05.txt; ACT=$S/rt2_action_05; OUT=$R/patchspec; mkdir -p $OUT
run() { D=$OUT/$1; [ -f $D/patch_spec.json ] && { echo "##### skip $1 #####"; return; }; echo "##### $1 #####"
  cd $V2; PYTHONPATH=$V2 python -m specwm.patch_spec --ckpt $CKF --target-mode fast_bf16 --quant int8-g128 --action-root $ACT --prompt-file $PR --scene 05 --frames 161 --seed 1234 --out $D "${@:2}" || echo "!!! $1 FAILED"; }
# 정확 alpha 를 세 단위(chunk/frame/patch)로 동시에 기록하므로 한 실행이 분석용. tau 는 하이브리드 구성만 좌우.
run all_draft   --unit patch --tau 1e9          # = Fast 혼자 (기준선, alpha 기록용)
run all_target  --unit patch --tau 0            # = Fast bf16 혼자 (target 기준)
run patch_t002  --unit patch --tau 0.02
run patch_t005  --unit patch --tau 0.05
run patch_t010  --unit patch --tau 0.10
run patch_t001  --unit patch --tau 0.01
echo "===== SUMMARY ====="; for d in $OUT/*/; do echo "$(basename $d): $(cat $d/patch_spec.json 2>/dev/null | tr -d '\n ' | cut -c1-200)"; done; echo "[RPATCH_DONE]"
