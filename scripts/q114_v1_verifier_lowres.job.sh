#!/bin/bash
# V1: verifier 를 128p / 256p 입력으로 — VGGT(프레임 다운샘플) + Base diffloss(max_area 28672=128x224 / 122880=256x464) → AUC 비교 (b1_auc.py _r128/_r256)
S=$SCRATCH; O=$S/lingbot-world-v2/specwm_out; B=$O/b1
CLIPS=""
for sc in 05 01 00 02; do CLIPS="$CLIPS base_S$sc=$O/floor/base/S${sc}_s1234/video.mp4 fast_S$sc=$O/floor/fast/S$sc/W18_sink6/fast_alone_s1234/video.mp4 ctrl_S${sc}_shuffle=$B/controls/S${sc}_shuffle.mp4 ctrl_S${sc}_teleport=$B/controls/S${sc}_teleport.mp4 ctrl_S${sc}_freeze=$B/controls/S${sc}_freeze.mp4"; done
for t in falling_glass pouring_water_into_cup rocket_launch inflating_balloon; do CLIPS="$CLIPS stevo_base_$t=$O/stevo/lingbot_base/$t/video.mp4 stevo_fast_$t=$O/stevo/lingbot_fast/$t.mp4"; done
source $S/b1-venv/bin/activate
for RES in 256 128; do for kv in $CLIPS; do n=${kv%%=*}; v=${kv#*=}; P=""; case $n in base_S*|fast_S*|ctrl_S*) sc=$(echo $n | grep -o "S0[0-9]" | head -1); P="--poses $S/rt2_action_${sc#S}/poses.npy";; esac
  [ -f $B/${n}_vggt_r$RES.json ] || python $S/scripts/b1_vggt_pose.py --video $v --out $B/${n}_vggt_r$RES.json --resize $RES $P 2>&1 | grep -E "VGGT_DONE|Error|Traceback"; done; done
source $S/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd $S/lingbot-world; export PYTHONPATH=$PWD
for RA in "256 122880" "128 28672"; do set -- $RA; RES=$1; AREA=$2; for kv in $CLIPS; do n=${kv%%=*}; v=${kv#*=}; case $n in stevo_*) continue;; esac; sc=$(echo $n | grep -o "S0[0-9]" | head -1); sc=${sc#S}
  [ -f $B/${n}_diffloss_r$RES.json ] || python -m specwm.base_diffloss --ckpt $PWD/lingbot-world-base-cam --video $v --image $S/lingbot-world-v2/examples/$sc/image.jpg --action-dir $S/rt2_action_$sc --prompt-file $S/p_$sc.txt --frames 153 --max-area $AREA --out $B/${n}_diffloss_r$RES.json 2>&1 | grep -E "DIFFLOSS_DONE|Error|Traceback|SystemExit"; done; done
for suf in "" _r256 _r128; do echo "## suffix '$suf'"; python $S/scripts/b1_auc.py $suf; done > $B/b1_auc_lowres.md; cat $B/b1_auc_lowres.md
touch $S/exp/Q114_DONE; echo "[Q114_DONE] $(date)"
