#!/bin/bash
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out
# 1) resync 분석 잡
cat > $S/ra.slurm << 'SEOF'
#!/bin/bash
true
SEOF
sbatch $S/ra.slurm | tail -1
# 2) SolarWM 프레임 추출 (stage1 두 패스 + stage2 clean)
cp $S/solar_out/stage1_s05/generation/ema_nfe50_autoregressive/slot-000000/video.mp4 $S/solar_stage1_nfe50.mp4
cp $S/solar_out/stage1_s05/generation/ema_nfe4_autoregressive/slot-000000/video.mp4 $S/solar_stage1_nfe4.mp4
cp $S/solar_out/stage2_s05/runs/lingbot_s05_stage2/generation/model_self_forcing_nfe4/slot-000000/video.mp4 $S/solar_stage2_nfe4.mp4
true
python - << 'PY'
import os; os.environ["OMP_NUM_THREADS"]="1"
import cv2; cv2.setNumThreads(1); import numpy as np
S="$SCRATCH"
for n in ["stage1_nfe50","stage1_nfe4","stage2_nfe4"]:
    c=cv2.VideoCapture(S+"/solar_%s.mp4"%n); fr=[]
    while True:
        ok,f=c.read()
        if not ok: break
        fr.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
    V=np.stack(fr); np.save(S+"/solar_%s_frames.npy"%n, V[::4]); print(n, V.shape)
PY
# 3) VBench 배치
python3 - << 'PY'
p="$SCRATCH/vbench2.slurm"; s=open(p).read()
head=s.split("--videos")[0]; tail="\necho \"[VBENCH_JOB_DONE] exit=$?\"\n"
R="$SCRATCH/lingbot-world-v2/specwm_out"; S="$SCRATCH"
vids=[("resync_fast_alone",R+"/resync_s05/fast_alone/video.mp4"),("resync_ctrl_k6",R+"/resync_s05/ctrl_selfcommit_k6/video.mp4"),("resync_k3",R+"/resync_s05/resync_k3/video.mp4"),("resync_k6",R+"/resync_s05/resync_k6/video.mp4"),("resync_k9",R+"/resync_s05/resync_k9/video.mp4"),("resync_every3",R+"/resync_s05/resync_every3/video.mp4"),
("base_S05_s4",R+"/revisit/base_official_S05_s4/video.mp4"),("base_S05_s8",R+"/revisit/base_official_S05_s8/video.mp4"),("base_S05_s16",R+"/revisit/base_official_S05_s16/video.mp4"),("base_hands_s4",R+"/hands/base_official_p1_s4/video.mp4"),("base_hands_s16",R+"/hands/base_official_p1_s16/video.mp4"),
("solar_stage1_nfe50",S+"/solar_stage1_nfe50.mp4"),("solar_stage1_nfe4",S+"/solar_stage1_nfe4.mp4"),("solar_stage2_nfe4",S+"/solar_stage2_nfe4.mp4")]
body="--videos \\\n"+" \\\n".join("  %s=%s"%(k,v) for k,v in vids)+"\n"
s2=head.replace("--out $SCRATCH/vbench_out2","--out $SCRATCH/vbench_out3").replace("vbench2.%j.out","vbench3.%j.out")+body+tail
open("$SCRATCH/vbench3.slurm","w").write(s2); print("vbench3.slurm 작성, 영상", len(vids))
PY
sed -i 's/-p gh-dev -t 01:00:00/-p gh -t 01:00:00/' $S/vbench3.slurm
sbatch $S/vbench3.slurm | tail -1
squeue -u $USER -h -o "%.9i %.10j %.9T %.8M"
