#!/bin/bash
true
export SPECWM_NO_FLASH=1; export OMP_NUM_THREADS=4; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V1=$S/lingbot-world; V2=$S/lingbot-world-v2; CKB=$V1/lingbot-world-base-cam; OUT=$R/basepatch; mkdir -p $OUT
FL=$R/repro/S05/W18_sink6/fast_alone_s1234/latent.pt; [ -f $FL ] || FL=$R/resync_v2/W18_sink6/fast_alone_s1234/latent.pt
[ -f $FL ] || { echo "!!! fast latent missing"; find $R -name latent.pt -path "*fast_alone_s1234*" | head; exit 1; }
echo "[input] $FL"
run() { D=$OUT/$1; [ -f $D/base_patch.json ] && { echo "##### skip $1 #####"; return; }; echo "##### $1 (schedule $2) #####"
  cd $V1; PYTHONPATH=$V1 python -m specwm.base_patch --ckpt $CKB --fast-latent $FL --image $V2/examples/05/image.jpg --action-path $S/rt2_action_05 --prompt-file $S/p_05.txt --schedule "$2" --taus "0.05,0.10,0.20,0.40" --out $D || echo "!!! $1 FAILED"; }
run sched40 "0.40,0.20,0.08"      # 현행 다듬기 강도
run sched60 "0.60,0.30,0.12"      # 더 강하게 (Base 쪽으로 더 끌어당김)
echo "===== SUMMARY ====="; for d in $OUT/*/; do echo "$(basename $d): $(cat $d/base_patch.json 2>/dev/null | tr -d '\n ' | cut -c1-240)"; done; echo "[RBASE_DONE]"
