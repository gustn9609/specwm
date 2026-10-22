#!/bin/bash
true
export SPECWM_NO_FLASH=1; export OMP_NUM_THREADS=4; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V2=$S/lingbot-world-v2; B16=$R/revisit/base_official_S05_s16/video.mp4
v2() { cd $V2; PYTHONPATH=$V2 python -m specwm.resync_v4 --ckpt $V2/lingbot-world-v2-14b-causal-fast "$@"; }
echo "##### F1a: low-res / 8-step targets as KV providers (S05 s1234) #####"
KV=""; for A in 386048 225280 99840; do for ST in 16 8; do KV="$KV,tc${A}_s${ST}=$R/targetcost/area${A}_s${ST}/video.mp4"; done; done; KV=${KV#,}
v2 --action-root $S/rt2_action_05 --prompt-file $S/p_05.txt --scene 05 --frames 161 --seeds 1234 --configs W18_sink6 --arms kvrw_tc386048_s16,kvrw_tc386048_s8,kvrw_tc225280_s16,kvrw_tc225280_s8,kvrw_tc99840_s16,kvrw_tc99840_s8 --kv-providers "$KV" --out $R/targetcost_kv
echo "##### F1b: sink capacity sweep 55 s #####"
v2 --action-root $S/traj55_action_05 --prompt-file $S/p_05.txt --scene 05 --frames 873 --seeds 1234 --base-video $B16 --configs none --extra-configs "W12_sink6:fast_alone+resync_k2,W18_sink12:fast_alone+resync_k4,W42_sink36:fast_alone+resync_k12" --out $R/resync55
echo "##### F1d: stale anchors 55 s rerun (sink overwrite) #####"
v2 --action-root $S/traj55_action_05 --prompt-file $S/p_05.txt --scene 05 --frames 873 --seeds 1234 --base-video $B16 --configs W24_sink18 --arms stale_d15_k20,stale_d30_k40,stale_d45_k60,ctrl_stale_k20 --stale-inject d15=20,d30=40,d45=60 --out $R/resync55
echo "##### F1c: target cost with both experts resident on GPU #####"
V1=$S/lingbot-world; CK=$V1/lingbot-world-base-cam; LAU=$R/hybrid/s05_launder/REFINE_0.40_0.20_0.08/video.mp4
for CFG in "386048 16" "386048 8" "225280 16" "225280 8" "99840 16" "99840 8"; do set -- $CFG
  cd $V1; PYTHONPATH=$V1 python -m specwm.bench_base_prefix2 --ckpt $CK --image $V2/examples/05/image.jpg --action-path $S/rt2_action_05 --prompt-file $S/p_05.txt --frames 73 --steps $2 --max-area $1 --prefix-latents 9 --mode repaint --seed 1234 --prefix-video $LAU --resident --out $R/targetcost_res/area$1_s$2
done
echo "[R4F1_DONE]"
