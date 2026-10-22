#!/bin/bash
true
export SPECWM_NO_FLASH=1; export OMP_NUM_THREADS=4; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V1=$S/lingbot-world; V2=$S/lingbot-world-v2; CK=$V1/lingbot-world-base-cam
echo "##### N1: refine the first 4.5 s of the 55 s Fast run (naive fix, no cloud regeneration) #####"
cd $V1; PYTHONPATH=$V1 python -m specwm.hybrid_refine --ckpt $CK --scene 05 --action-dir $S/traj55_slice_f0_n73 --prompt-file $S/p_05.txt --fast-latent $R/resync55/fast55_first19.pt --frames 73 --schedules "0.40,0.20,0.08" --out $R/naive55/refine
echo "##### N2: write those refined frames into KV + anchors, continue 55 s #####"
cd $V2; PYTHONPATH=$V2 python -m specwm.resync_v4 --ckpt $V2/lingbot-world-v2-14b-causal-fast --action-root $S/traj55_action_05 --prompt-file $S/p_05.txt --scene 05 --frames 873 --seeds 1234 \
  --configs none --extra-configs "W36_sink18:kvrw_naive" --kv-providers "naive=$R/naive55/refine/REFINE_0.40_0.20_0.08/video.mp4" --out $R/resync55
echo "[R4N_DONE]"
