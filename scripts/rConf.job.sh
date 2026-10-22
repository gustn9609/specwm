#!/bin/bash
true
export SPECWM_NO_FLASH=1; export OMP_NUM_THREADS=4; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V2=$S/lingbot-world-v2; CKF=$V2/lingbot-world-v2-14b-causal-fast; PR=$S/p_05.txt; ACT=$S/rt2_action_05; OUT=$R/patchspec; mkdir -p $OUT
run() { D=$OUT/$1; [ -f $D/patch_spec.json ] && { echo "##### skip $1 #####"; return; }; echo "##### $1 #####"
  cd $V2; PYTHONPATH=$V2 python -m specwm.patch_spec --ckpt $CKF --target-mode fast_bf16 --quant int8-g128 --action-root $ACT --prompt-file $PR --scene 05 --frames 161 --seed 1234 --out $D "${@:2}" || echo "!!! $1 FAILED"; }
run conf_t002_k005 --unit patch --tau 0.02 --kappa 0.05
run conf_t002_k010 --unit patch --tau 0.02 --kappa 0.10
run conf_t005_k005 --unit patch --tau 0.05 --kappa 0.05
run conf_t005_k010 --unit patch --tau 0.05 --kappa 0.10
echo "===== SUMMARY ====="; for d in $OUT/conf_*/; do echo "$(basename $d): $(cat $d/patch_spec.json 2>/dev/null | tr -d '\n ' | cut -c1-200)"; done; echo "[RCONF_DONE]"
