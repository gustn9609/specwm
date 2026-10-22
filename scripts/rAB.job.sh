#!/bin/bash
true
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V1=$S/lingbot-world; V2=$S/lingbot-world-v2
CKB=$V1/lingbot-world-base-cam; CKF=$V2/lingbot-world-v2-14b-causal-fast; PR=$S/p_05.txt
ACT=$S/rt2_action_05; OUT=$R/ideasAB; mkdir -p $OUT

# ===== A. 파이프라인 디노이징: 품질 (sequential vs pipeline, 같은 시드) + 배치 forward 시간 =====
for M in sequential pipeline; do
  if [ ! -f $OUT/A_$M/frames_all.npy ]; then
    echo "##### A $M #####"
    cd $V2; PYTHONPATH=$V2 python -m specwm.fast_pipeline --ckpt $CKF --action-root $ACT --prompt-file $PR --scene 05 \
      --frames 161 --seed 1234 --window 36 --sink 18 --mode $M $([ $M = pipeline ] && echo --time-batch) --out $OUT/A_$M || echo "!!! A $M FAILED"
  fi
done

# ===== B. 병렬 롤아웃 + Base 선택: 시드 3개 스트림 덤프 -> Base 가 창마다 채점 =====
for SD in 1234 4321 9999; do
  if [ ! -f $OUT/B_stream_s$SD/xt_node3.pt ]; then
    echo "##### B dump seed $SD #####"
    cd $V2; PYTHONPATH=$V2 python -m specwm.fast_dump --ckpt $CKF --action-root $ACT --prompt-file $PR --scene 05 \
      --frames 161 --seed $SD --phase stream --window 36 --sink 18 --out $OUT/B_stream_s$SD || echo "!!! B dump $SD FAILED"
    # 덤프의 x0 를 디코딩해 프레임도 저장 (DINO 정답용)
    cd $V2; PYTHONPATH=$V2 python - << PY
import torch, numpy as np, os, sys
sys.path.insert(0, "$V2")
from specwm.binding_v2 import V2Binding
b = V2Binding("$CKF", target_mode="fast_bf16", quant=None, sink=18, local_attn=36, swap_models=False, build_draft=False, timesteps_index=(0,250,500,750))
x0 = torch.load("$OUT/B_stream_s$SD/x0.pt", map_location="cpu").float()
with torch.no_grad(): vid = b.pipe.vae.decode([x0.to(b.device)])[0].detach().cpu()
v = ((vid.clamp(-1,1)+1)/2*255).byte().permute(1,2,3,0).numpy(); np.save("$OUT/B_stream_s$SD/frames_all.npy", v); print("decoded", v.shape)
PY
  fi
done
for SD in 1234 4321 9999; do
  if [ -f $OUT/B_stream_s$SD/xt_node3.pt ] && [ ! -f $OUT/B_score_s$SD/verify.json ]; then
    echo "##### B score seed $SD #####"
    cd $V1; PYTHONPATH=$V1 python -m specwm.base_handoff --ckpt $CKB --prompt-file $PR --mode verify \
      --stream-dir $OUT/B_stream_s$SD --slice-root $S --slice-pattern "rt2_slice05_f{f0}_n33" --sigma-start 0.625 --every 3 --out $OUT/B_score_s$SD || echo "!!! B score $SD FAILED"
  fi
done
echo "===== SUMMARY ====="
for d in A_sequential A_pipeline; do echo "$d: $([ -f $OUT/$d/frames_all.npy ] && echo OK || echo MISSING) $(cat $OUT/$d/pipeline.json 2>/dev/null | tr -d '\n ' | cut -c1-200)"; done
for SD in 1234 4321 9999; do echo "B s$SD: dump=$([ -f $OUT/B_stream_s$SD/frames_all.npy ] && echo OK || echo MISSING) score=$([ -f $OUT/B_score_s$SD/verify.json ] && echo OK || echo MISSING)"; done
echo "[RAB_DONE]"
