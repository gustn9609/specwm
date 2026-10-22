#!/bin/bash
true
export SPECWM_NO_FLASH=1; export OMP_NUM_THREADS=4; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V2=$S/lingbot-world-v2
ACT=$S/traj55_action_05; PR=$S/p_05.txt; B16=$R/revisit/base_official_S05_s16/video.mp4; OUT=$R/anchor_once; mkdir -p $OUT
# 질문: Base16 첫 4.5초를 한 번만 박으면 55초를 버티나 (시드 3개). resync_k6 (seed1234, 8-9/10) 의 재현 + 시드 확장.
for SD in 1234 4321 9999; do
  D=$OUT/s$SD; [ -f $D/frames_all.npy ] && { echo "##### skip s$SD #####"; continue; }
  SRC=$([ $SD = 1234 ] && echo $R/lat55/d0 || echo $R/lat55_s$SD/d0)
  echo "##### anchor-once seed $SD (state from $SRC) #####"
  cd $V2; python - << PY
import torch, json, os
src="$SRC"; dst="$D"; os.makedirs(dst, exist_ok=True); CS=3
torch.save(torch.load(os.path.join(src,"display_lat.pt"))[:, :6*CS].clone(), os.path.join(dst,"display_lat.pt"))
torch.save(torch.load(os.path.join(src,"kv_lat.pt"))[:, :6*CS].clone(), os.path.join(dst,"kv_lat.pt"))
m=json.load(open(os.path.join(src,"meta.json"))); m.update(done=6, corrections=[], anchor_chunks=None); json.dump(m, open(os.path.join(dst,"meta.json"),"w"), indent=1); print("state cut at chunk 6")
PY
  PYTHONPATH=$V2 python -m specwm.e2e_stage --ckpt $V2/lingbot-world-v2-14b-causal-fast --action-root $ACT --prompt-file $PR --scene 05 --frames 873 --seed $SD \
    --window 36 --sink 18 --state $D --gen-until 73 --anchor fixed --correction $B16:0 --decode || echo "!!! s$SD FAILED"
done
echo "===== SUMMARY ====="; for SD in 1234 4321 9999; do echo "s$SD: $([ -f $OUT/s$SD/frames_all.npy ] && echo OK || echo MISSING)"; done; echo "[RANC_DONE]"
