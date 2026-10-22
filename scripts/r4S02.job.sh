#!/bin/bash
true
export SPECWM_NO_FLASH=1; export OMP_NUM_THREADS=4; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V1=$S/lingbot-world; V2=$S/lingbot-world-v2; CK=$V1/lingbot-world-base-cam
B16=$R/repro/S02/base16_s1234/video.mp4; TGT=$R/repro/S02/target_s1234/video.mp4; OUT=$R/scene55/S02
v2() { cd $V2; PYTHONPATH=$V2 python -m specwm.resync_v4 --ckpt $V2/lingbot-world-v2-14b-causal-fast --action-root $S/traj55_action_02 --prompt-file $S/p_02.txt --scene 02 --frames 873 --seeds 1234 "$@"; }
echo "##### S02-1: pre-encode base16 latent #####"
v2 --base-video $B16 --configs none --extra-configs "W36_sink18:fast_alone" --encode-only $OUT/base16_lat.pt --out $OUT/tmp_enc
echo "##### S02-2: fast_alone + upper bound + system (one pinned correction) #####"
v2 --base-video $B16 --base-latent $OUT/base16_lat.pt --configs none --extra-configs "W36_sink18:fast_alone+resync_k6+kvrw_system" --kv-providers "system=$TGT" --out $OUT
echo "##### S02-3: refine the first 4.5 s of that fast_alone run (naive provider) #####"
python - << 'PY'
import torch, os
R = os.environ["SCRATCH"] + "/lingbot-world-v2/specwm_out"; d = R + "/scene55/S02"
lat = torch.load(d + "/W36_sink18/fast_alone_s1234/latent.pt", map_location="cpu"); torch.save(lat[:, :19].contiguous(), d + "/fast55_first19.pt"); print("slice", tuple(lat[:, :19].shape))
PY
cd $V1; PYTHONPATH=$V1 python -m specwm.hybrid_refine --ckpt $CK --scene 02 --action-dir $S/traj55_slice02_f0_n73 --prompt-file $S/p_02.txt --fast-latent $OUT/fast55_first19.pt --frames 73 --schedules "0.40,0.20,0.08" --out $OUT/naive_refine
echo "##### S02-4: naive arm #####"
v2 --base-latent $OUT/base16_lat.pt --configs none --extra-configs "W36_sink18:kvrw_naive" --kv-providers "naive=$OUT/naive_refine/REFINE_0.40_0.20_0.08/video.mp4" --out $OUT
echo "[R4S02_DONE]"
