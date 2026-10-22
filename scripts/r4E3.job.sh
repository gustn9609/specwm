#!/bin/bash
true
export SPECWM_NO_FLASH=1; export OMP_NUM_THREADS=4; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V1=$S/lingbot-world; V2=$S/lingbot-world-v2; CK=$V1/lingbot-world-base-cam
B16=$R/revisit/base_official_S05_s16/video.mp4
v2() { cd $V2; PYTHONPATH=$V2 python -m specwm.resync_v4 --ckpt $V2/lingbot-world-v2-14b-causal-fast "$@"; }
launder() { cd $V1; PYTHONPATH=$V1 python -m specwm.hybrid_refine --ckpt $CK --scene $1 --action-dir $S/rt2_action_$1 --prompt-file $S/p_$1.txt --fast-latent $R/repro/S$1/W18_sink6/fast_alone_s$2/latent.pt --frames 153 --schedules "0.40,0.20,0.08" --out $R/repro/S$1/launder_s$2; }
repaint() { cd $V1; PYTHONPATH=$V1 python -m specwm.bench_base_prefix2 --ckpt $CK --image $V2/examples/$1/image.jpg --action-path $S/rt2_action_$1 --prompt-file $S/p_$1.txt --frames 161 --steps 16 --prefix-latents 9 --mode repaint --seed $2 --prefix-video $R/repro/S$1/launder_s$2/REFINE_0.40_0.20_0.08/video.mp4 --out $R/repro/S$1/target_s$2; }
base16() { [ -f $R/repro/S$1/base16_s$2/video.mp4 ] && return; cd $V1; PYTHONPATH=$V1 python -m specwm.bench_base_official --ckpt $CK --image $V2/examples/$1/image.jpg --action-path $S/rt2_action_$1 --prompt-file $S/p_$1.txt --frames 161 --steps 16 --seed $2 --out $R/repro/S$1/base16_s$2; }
v1trio() { echo "##### v1 trio S$1 s$2 #####"; launder $1 $2; repaint $1 $2; base16 $1 $2; }

v1trio 02 1234; v1trio 02 4321; v1trio 02 9999
echo "##### mask + Base-refined prefix (S05 s1234): target #####"
cd $V1; PYTHONPATH=$V1 python -m specwm.bench_base_prefix2 --ckpt $CK --image $V2/examples/05/image.jpg --action-path $S/rt2_action_05 --prompt-file $S/p_05.txt --frames 161 --steps 16 --prefix-latents 9 --mode mask --seed 1234 --prefix-video $R/hybrid/s05_launder/REFINE_0.40_0.20_0.08/video.mp4 --out $R/revisit/e3d_mask_refined
echo "##### kvrw arms: repro #####"
for SC in 05 01 02; do v2 --action-root $S/rt2_action_$SC --prompt-file $S/p_$SC.txt --scene $SC --frames 161 --seeds 1234,4321,9999 --configs W18_sink6 --arms kvrw_laundered,kvrw_ideal --kv-providers "laundered=$R/repro/S$SC/target_s{seed}/video.mp4,ideal=$R/repro/S$SC/base16_s{seed}/video.mp4" --out $R/repro/S$SC; done
echo "##### kvrw arms: lock-on sigma sweep #####"
KV="l15=$R/lockon/launder/REFINE_0.15_0.07_0.03/video.mp4,l25=$R/lockon/launder/REFINE_0.25_0.12_0.05/video.mp4,l40=$R/hybrid/s05_launder/REFINE_0.40_0.20_0.08/video.mp4,l60=$R/lockon/launder/REFINE_0.60_0.30_0.12/video.mp4,l80=$R/lockon/launder/REFINE_0.80_0.40_0.16/video.mp4,raw=$R/resync_v2/W18_sink6/fast_alone_s1234/video.mp4,maskref=$R/revisit/e3d_mask_refined/video.mp4,repaintref=$R/revisit/e3c_repaint_laundered/video.mp4"
v2 --action-root $S/rt2_action_05 --prompt-file $S/p_05.txt --scene 05 --frames 161 --seeds 1234 --configs W18_sink6 --arms kvrw_l15,kvrw_l25,kvrw_l40,kvrw_l60,kvrw_l80,kvrw_raw,kvrw_maskref,kvrw_repaintref --kv-providers "$KV" --out $R/lockon
echo "[R4E3_DONE]"
