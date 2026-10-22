#!/bin/bash
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out
true
cd $S/lingbot-world-v2 && python -c "import ast; ast.parse(open('specwm/resync_v4.py').read()); print('v4 OK')"
mk() { SC=$1
cat > $S/r4S$SC.slurm << SEOF
#!/bin/bash
true
export SPECWM_NO_FLASH=1; export OMP_NUM_THREADS=4; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=\$SCRATCH; R=\$S/lingbot-world-v2/specwm_out; V1=\$S/lingbot-world; V2=\$S/lingbot-world-v2; CK=\$V1/lingbot-world-base-cam
B16=\$R/repro/S$SC/base16_s1234/video.mp4; TGT=\$R/repro/S$SC/target_s1234/video.mp4; OUT=\$R/scene55/S$SC
v2() { cd \$V2; PYTHONPATH=\$V2 python -m specwm.resync_v4 --ckpt \$V2/lingbot-world-v2-14b-causal-fast --action-root \$S/traj55_action_$SC --prompt-file \$S/p_$SC.txt --scene $SC --frames 873 --seeds 1234 "\$@"; }
echo "##### S$SC-1: pre-encode base16 latent #####"
v2 --base-video \$B16 --configs none --extra-configs "W36_sink18:fast_alone" --encode-only \$OUT/base16_lat.pt --out \$OUT/tmp_enc
echo "##### S$SC-2: fast_alone + upper bound + system (one pinned correction) #####"
v2 --base-video \$B16 --base-latent \$OUT/base16_lat.pt --configs none --extra-configs "W36_sink18:fast_alone+resync_k6+kvrw_system" --kv-providers "system=\$TGT" --out \$OUT
echo "##### S$SC-3: refine the first 4.5 s of that fast_alone run (naive provider) #####"
python - << 'PY'
import torch, os
R = os.environ["SCRATCH"] + "/lingbot-world-v2/specwm_out"; d = R + "/scene55/S$SC"
lat = torch.load(d + "/W36_sink18/fast_alone_s1234/latent.pt", map_location="cpu"); torch.save(lat[:, :19].contiguous(), d + "/fast55_first19.pt"); print("slice", tuple(lat[:, :19].shape))
PY
cd \$V1; PYTHONPATH=\$V1 python -m specwm.hybrid_refine --ckpt \$CK --scene $SC --action-dir \$S/traj55_slice${SC}_f0_n73 --prompt-file \$S/p_$SC.txt --fast-latent \$OUT/fast55_first19.pt --frames 73 --schedules "0.40,0.20,0.08" --out \$OUT/naive_refine
echo "##### S$SC-4: naive arm #####"
v2 --base-latent \$OUT/base16_lat.pt --configs none --extra-configs "W36_sink18:kvrw_naive" --kv-providers "naive=\$OUT/naive_refine/REFINE_0.40_0.20_0.08/video.mp4" --out \$OUT
echo "[R4S${SC}_DONE]"
SEOF
}
# 궤적 슬라이스 (교정 창 조건화용)
python - << 'PY'
import numpy as np, os
S = os.environ["SCRATCH"]
for sc in ["01", "02"]:
    src = S + "/traj55_action_%s" % sc; d = "%s/traj55_slice%s_f0_n73" % (S, sc); os.makedirs(d, exist_ok=True)
    for f in ["poses.npy", "intrinsics.npy", "action.npy", "wasd_action.npy"]:
        a = np.load(os.path.join(src, f)); np.save(os.path.join(d, f), a[:73].astype(np.float32))
    print(d, np.load(d + "/poses.npy").shape)
PY
mk 01; mk 02
J1=$(sbatch --parsable $S/r4S01.slurm | tail -1); J2=$(sbatch --parsable --dependency=afterany:$J1 $S/r4S02.slurm | tail -1); echo "S01=$J1 S02=$J2"
squeue -u $USER -h -o "%.9i %.8j %.9T %.9M %.14R"
