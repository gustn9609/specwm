#!/bin/bash
true
export SPECWM_NO_FLASH=1; export OMP_NUM_THREADS=4; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V1=$S/lingbot-world; V2=$S/lingbot-world-v2; CK=$V1/lingbot-world-base-cam
ACT=$S/traj55_action_05; PR=$S/p_05.txt; SL=$S/traj55_slice_f0_n73
SEEDS="$1"
# BR 의 뒤 2.25초: Base16 앞 33프레임을 prefix 로 RePaint (시드 무관, 한 번만)
BR=$R/decomp_shared/BR_corr
if [ ! -f $BR/video.mp4 ]; then
  echo "##### BR: RePaint on Base16's own prefix #####"; mkdir -p $R/decomp_shared
  cd $V2; python -c "
import imageio,numpy as np,os; v=np.stack(imageio.mimread('$R/revisit/base_official_S05_s16/video.mp4',memtest=False))[:33,:,:,:3]
os.makedirs('$R/decomp_shared',exist_ok=True); imageio.mimwrite('$R/decomp_shared/base16_prefix33.mp4',list(v),fps=16,quality=9); imageio.imwrite('$R/decomp_shared/base16_first.jpg',v[0],quality=97)"
  cd $V1; PYTHONPATH=$V1 python -m specwm.bench_base_prefix2 --ckpt $CK --image $R/decomp_shared/base16_first.jpg --action-path $SL --prompt-file $PR \
    --frames 73 --steps 16 --prefix-latents 9 --mode repaint --seed 1234 --prefix-video $R/decomp_shared/base16_prefix33.mp4 --out $BR || echo "!!! BR repaint FAILED"
fi
stage() { cd $V2; PYTHONPATH=$V2 python -m specwm.e2e_stage --ckpt $V2/lingbot-world-v2-14b-causal-fast --action-root $ACT --prompt-file $PR --scene 05 --frames 873 --seed $4 --window 36 --sink 18 --state $1 --gen-until $2 --anchor fixed --correction $3 --decode; }
for SD in $SEEDS; do
  D=$R/decomp_s$SD
  [ -f $D/prov_LL.mp4 ] || { cd $V2; python $S/mk_decomp.py $SD || { echo "!!! mk_decomp $SD FAILED"; continue; }; }
  for ARM in LL LB BR; do
    if [ -f $D/$ARM/frames_all.npy ]; then echo "##### skip $ARM s$SD #####"; continue; fi
    case $ARM in LL) PV=$D/prov_LL.mp4;; LB) PV=$D/prov_LB.mp4;; BR) PV=$BR/video.mp4;; esac
    [ -f $PV ] || { echo "!!! provider missing $PV"; continue; }
    echo "##### $ARM seed $SD  <- $PV #####"
    stage $D/$ARM 73 $PV:0 $SD || echo "!!! $ARM s$SD FAILED"
  done
done
echo "===== SUMMARY ====="; for SD in $SEEDS; do for ARM in LL LB BR; do echo "s$SD $ARM: $([ -f $R/decomp_s$SD/$ARM/frames_all.npy ] && echo OK || echo MISSING)"; done; done
echo "[RD_DONE]"
