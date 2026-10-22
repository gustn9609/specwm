#!/bin/bash
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out
head -n $(grep -n "^cd \$SCRATCH" $S/vbench2.slurm | cut -d: -f1) $S/vbench2.slurm | sed -e 's/-J [^ ]*/-J vb6/' -e 's#vbench2.%j.out#vb6.%j.out#' -e 's/-p gh-dev/-p gh/' > $S/vb6.slurm
V=""
for t in e3sep_self_mask e3sep_otherseed_mask e3a_repaint_fast e3a_repaint_self e3b_causal12_repaint_fast e3c_repaint_laundered; do V="$V $t=$R/revisit/$t/video.mp4"; done
V="$V laundered_fast_prefix=$R/hybrid/s05_launder/REFINE_0.40_0.20_0.08/video.mp4"
for a in cond_sep_self cond_sep_other cond_repaint_fast cond_repaint_self cond_causal12 cond_laundered xfade_k3_L2 xfade_k3_L4 xfade_k3_L8; do V="$V ${a}=$R/resync_v2/W18_sink6/${a}_s1234/video.mp4"; done
for L in 2 4 8; do V="$V xfade_k3_L${L}_node3=$R/resync_v2_xf3/W18_sink6/xfade_k3_L${L}_s1234/video.mp4"; done
V="$V base20_s4321=$R/revisit/base20_S05_s16_seed4321/video.mp4"
for a in fast_alone resync_k6 resync_k3 every3; do V="$V t20_${a}=$R/resync20/W18_sink6/${a}_s1234/video.mp4"; done
V="$V t20_W24s18_fast_alone=$R/resync20/W24_sink18/fast_alone_s1234/video.mp4 t20_W24s18_resync_k6=$R/resync20/W24_sink18/resync_k6_s1234/video.mp4"
echo "python vbench_eval.py --stage eval --out \$SCRATCH/vbench_out6 --videos $V" >> $S/vb6.slurm
echo 'echo "[VBENCH_JOB_DONE] exit=$?"' >> $S/vb6.slurm
grep -c "=" $S/vb6.slurm; grep -E "^#SBATCH" $S/vb6.slurm | tr "\n" " "; echo
# 존재하지 않는 영상 확인 (xf3/t20 는 아직 진행 중일 수 있음)
for p in $(echo $V | tr " " "\n" | cut -d= -f2); do [ -f $p ] || echo "아직 없음: $p"; done
