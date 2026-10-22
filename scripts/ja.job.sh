#!/bin/bash
true
export SPECWM_NO_FLASH=1; cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD
R=$SCRATCH/lingbot-world-v2/specwm_out; CK=$PWD/lingbot-world-base-cam; IMG=$SCRATCH/lingbot-world-v2/examples/05/image.jpg; ACT=$SCRATCH/rt2_action_05; PR=$SCRATCH/p_05.txt
FAST=$R/resync_v2/W18_sink6/fast_alone_s1234/video.mp4
run() { python -m specwm.bench_base_prefix2 --ckpt $CK --image $IMG --action-path $ACT --prompt-file $PR --frames 161 --steps 16 --prefix-latents 9 "$@"; }
echo "##### E3-sep-a: Base16 on its OWN prefix (seed 1234), mask #####";      run --mode mask --prefix-video $R/revisit/base_official_S05_s16/video.mp4 --out $R/revisit/e3sep_self_mask
echo "##### E3-sep-b: Base16 on OTHER-seed Base prefix (4321), mask #####";  run --mode mask --prefix-video $R/revisit/base_official_S05_s16_seed4321/video.mp4 --out $R/revisit/e3sep_otherseed_mask
echo "##### E3'-a: RePaint replacement, Fast prefix #####";                   run --mode repaint --prefix-video $FAST --out $R/revisit/e3a_repaint_fast
echo "##### E3'-a control: RePaint, Base own prefix #####";                   run --mode repaint --prefix-video $R/revisit/base_official_S05_s16/video.mp4 --out $R/revisit/e3a_repaint_self
echo "##### E3'-b: Base-causal N=12 + RePaint, Fast prefix #####"
SPECWM_CAUSAL_FRAMES=12 SPECWM_LAT_F=41 run --mode repaint --prefix-video $FAST --out $R/revisit/e3b_causal12_repaint_fast
echo "##### E3'-c: launder Fast prefix with low-noise expert (SDEdit 0.40,0.20,0.08) #####"
python -m specwm.hybrid_refine --ckpt $CK --scene 05 --action-dir $ACT --prompt-file $PR --fast-latent $R/resync_v2/W18_sink6/fast_alone_s1234/latent.pt --frames 153 --schedules "0.40,0.20,0.08" --out $R/hybrid/s05_launder
LAUND=$(ls $R/hybrid/s05_launder/REFINE*/video.mp4 | head -1); echo "laundered: $LAUND"
echo "##### E3'-c: RePaint with laundered Fast prefix #####";                run --mode repaint --prefix-video $LAUND --out $R/revisit/e3c_repaint_laundered
echo "[JA_DONE]"
