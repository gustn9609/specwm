#!/bin/bash
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out
head -n $(grep -n "^cd \$SCRATCH" $S/vbench2.slurm | cut -d: -f1) $S/vbench2.slurm | sed -e 's/-J [^ ]*/-J vb7/' -e 's#vbench2.%j.out#vb7.%j.out#' -e 's/-p gh-dev/-p gh-dev/' > $S/vb7.slurm
V=""
for a in control rolling fixed; do V="$V e2e_$a=$R/e2e55/$a/video.mp4"; done
for a in W36_sink18/fast_alone W36_sink18/resync_k6 W24_sink18/stale_d15_k20 W24_sink18/stale_d30_k40 W24_sink18/stale_d45_k60 W12_sink6/fast_alone W12_sink6/resync_k2 W18_sink12/fast_alone W18_sink12/resync_k4; do n=$(echo $a | tr '/' '_'); V="$V t55_$n=$R/resync55/${a}_s1234/video.mp4"; done
for a in tc386048_s16 tc386048_s8 tc225280_s16 tc225280_s8 tc99840_s16 tc99840_s8; do V="$V kv_$a=$R/targetcost_kv/W18_sink6/kvrw_${a}_s1234/video.mp4"; done
V="$V e3d_mask_refined=$R/revisit/e3d_mask_refined/video.mp4 lockon_maskref=$R/lockon/W18_sink6/kvrw_maskref_s1234/video.mp4 lockon_repaintref=$R/lockon/W18_sink6/kvrw_repaintref_s1234/video.mp4"
for W in E_spatula_to_plate B2_pour_liquid_pan D_wash_tomato; do V="$V ariaGT_$W=$S/aria/deliver/GT_${W}_5s.mp4 ariaGEN_$W=$S/aria/deliver/GEN_base70_${W}_5s_s1234.mp4"; done
echo "python vbench_eval.py --stage eval --out \$SCRATCH/vbench_out7 --videos $V" >> $S/vb7.slurm
echo 'echo "[VBENCH_JOB_DONE] exit=$?"' >> $S/vb7.slurm
n=0; for p in $(echo $V | tr " " "\n" | cut -d= -f2); do [ -f $p ] || { echo "없음: $p"; n=$((n+1)); }; done; echo "missing=$n videos=$(echo $V | wc -w)"
sbatch --parsable $S/vb7.slurm | tail -1; squeue -u $USER -h -o "%.9i %.8j %.9T %.9M %.14R"
