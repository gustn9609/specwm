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

echo "##### E1a: fast_alone all scenes #####"
for SC in 05 01 02; do v2 --action-root $S/rt2_action_$SC --prompt-file $S/p_$SC.txt --scene $SC --frames 161 --seeds 1234,4321,9999 --configs W18_sink6 --arms fast_alone --out $R/repro/S$SC; done
echo "##### E1b: stale anchors 55 s (sink overwrite) #####"
v2 --action-root $S/traj55_action_05 --prompt-file $S/p_05.txt --scene 05 --frames 873 --seeds 1234 --base-video $B16 --configs W24_sink18 --arms stale_d15_k20,stale_d30_k40,stale_d45_k60,ctrl_stale_k20 --stale-inject d15=20,d30=40,d45=60 --out $R/resync55
echo "[R4E1_DONE]"
