#!/bin/bash
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out
true
python $S/mkact55.py
cd $S/lingbot-world-v2 && python -c "import ast; ast.parse(open('specwm/resync_v4.py').read()); print('v4 OK')"
cd $S/lingbot-world && python -c "import ast; ast.parse(open('specwm/bench_base_prefix2.py').read()); print('prefix2 OK')"
HDR='#!/bin/bash
true
export SPECWM_NO_FLASH=1; export OMP_NUM_THREADS=4; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V1=$S/lingbot-world; V2=$S/lingbot-world-v2; CK=$V1/lingbot-world-base-cam
B16=$R/revisit/base_official_S05_s16/video.mp4
v2() { cd $V2; PYTHONPATH=$V2 python -m specwm.resync_v4 --ckpt $V2/lingbot-world-v2-14b-causal-fast "$@"; }
launder() { cd $V1; PYTHONPATH=$V1 python -m specwm.hybrid_refine --ckpt $CK --scene $1 --action-dir $S/rt2_action_$1 --prompt-file $S/p_$1.txt --fast-latent $R/repro/S$1/W18_sink6/fast_alone_s$2/latent.pt --frames 153 --schedules "0.40,0.20,0.08" --out $R/repro/S$1/launder_s$2; }
repaint() { cd $V1; PYTHONPATH=$V1 python -m specwm.bench_base_prefix2 --ckpt $CK --image $V2/examples/$1/image.jpg --action-path $S/rt2_action_$1 --prompt-file $S/p_$1.txt --frames 161 --steps 16 --prefix-latents 9 --mode repaint --seed $2 --prefix-video $R/repro/S$1/launder_s$2/REFINE_0.40_0.20_0.08/video.mp4 --out $R/repro/S$1/target_s$2; }
base16() { cd $V1; PYTHONPATH=$V1 python -m specwm.bench_base_official --ckpt $CK --image $V2/examples/$1/image.jpg --action-path $S/rt2_action_$1 --prompt-file $S/p_$1.txt --frames 161 --steps 16 --seed $2 --out $R/repro/S$1/base16_s$2; }
v1trio() { echo "##### v1 trio S$1 s$2 #####"; launder $1 $2; repaint $1 $2; base16 $1 $2; }
'
# ---- A: stale anchors (55 s) ----
cat > $S/r4A.slurm << SEOF
$HDR
echo "##### A: stale anchors 55 s #####"
v2 --action-root \$S/traj55_action_05 --prompt-file \$S/p_05.txt --scene 05 --frames 873 --seeds 1234 --base-video \$B16 --configs W24_sink18,W18_sink6 --arms fast_alone,resync_k6,stale_d15_k20,stale_d30_k40,stale_d45_k60,ctrl_stale_k20 --stale-inject d15=20,d30=40,d45=60 --out \$R/resync55
echo "[R4A_DONE]"
SEOF
# ---- B: fast_alone all scenes + v1 trios (S05 s4321,s9999 ; S01 s1234) ----
cat > $S/r4B.slurm << SEOF
$HDR
mkdir -p \$R/repro/S05/W18_sink6; [ -d \$R/repro/S05/W18_sink6/fast_alone_s1234 ] || cp -r \$R/resync_v2/W18_sink6/fast_alone_s1234 \$R/repro/S05/W18_sink6/
for SC in 05 01 02; do echo "##### fast_alone S\$SC #####"; v2 --action-root \$S/rt2_action_\$SC --prompt-file \$S/p_\$SC.txt --scene \$SC --frames 161 --seeds 1234,4321,9999 --configs W18_sink6 --arms fast_alone --out \$R/repro/S\$SC; done
v1trio 05 4321; v1trio 05 9999; v1trio 01 1234
echo "[R4B_DONE]"
SEOF
# ---- C: v1 trios S01 s4321,s9999 ; S02 s1234,s4321 + lock-on laundering sweep ----
cat > $S/r4C.slurm << SEOF
$HDR
v1trio 01 4321; v1trio 01 9999; v1trio 02 1234; v1trio 02 4321
echo "##### lock-on laundering sweep (S05 s1234) #####"
cd \$V1; PYTHONPATH=\$V1 python -m specwm.hybrid_refine --ckpt \$CK --scene 05 --action-dir \$S/rt2_action_05 --prompt-file \$S/p_05.txt --fast-latent \$R/resync_v2/W18_sink6/fast_alone_s1234/latent.pt --frames 153 --schedules "0.15,0.07,0.03;0.25,0.12,0.05;0.60,0.30,0.12;0.80,0.40,0.16" --out \$R/lockon/launder
echo "[R4C_DONE]"
SEOF
# ---- D: v1 trio S02 s9999 ; target cost ; kvrw arms (repro + lockon) ----
cat > $S/r4D.slurm << SEOF
$HDR
v1trio 02 9999
echo "##### target cost: RePaint window 73 f (2.25 s prefix + 2.25 s new) #####"
LAU=\$R/hybrid/s05_launder/REFINE_0.40_0.20_0.08/video.mp4
for CFG in "386048 16" "386048 8" "225280 16" "225280 8" "99840 16" "99840 8"; do set -- \$CFG
  cd \$V1; PYTHONPATH=\$V1 python -m specwm.bench_base_prefix2 --ckpt \$CK --image \$V2/examples/05/image.jpg --action-path \$S/rt2_action_05 --prompt-file \$S/p_05.txt --frames 73 --steps \$2 --max-area \$1 --prefix-latents 9 --mode repaint --seed 1234 --prefix-video \$LAU --out \$R/targetcost/area\$1_s\$2
done
echo "##### kvrw arms: repro #####"
for SC in 05 01 02; do v2 --action-root \$S/rt2_action_\$SC --prompt-file \$S/p_\$SC.txt --scene \$SC --frames 161 --seeds 1234,4321,9999 --configs W18_sink6 --arms kvrw_laundered,kvrw_ideal --kv-providers "laundered=\$R/repro/S\$SC/target_s{seed}/video.mp4,ideal=\$R/repro/S\$SC/base16_s{seed}/video.mp4" --out \$R/repro/S\$SC; done
echo "##### kvrw arms: lock-on sigma sweep #####"
KV="l15=\$R/lockon/launder/REFINE_0.15_0.07_0.03/video.mp4,l25=\$R/lockon/launder/REFINE_0.25_0.12_0.05/video.mp4,l40=\$R/hybrid/s05_launder/REFINE_0.40_0.20_0.08/video.mp4,l60=\$R/lockon/launder/REFINE_0.60_0.30_0.12/video.mp4,l80=\$R/lockon/launder/REFINE_0.80_0.40_0.16/video.mp4,raw=\$R/resync_v2/W18_sink6/fast_alone_s1234/video.mp4"
v2 --action-root \$S/rt2_action_05 --prompt-file \$S/p_05.txt --scene 05 --frames 161 --seeds 1234 --configs W18_sink6 --arms kvrw_l15,kvrw_l25,kvrw_l40,kvrw_l60,kvrw_l80,kvrw_raw --kv-providers "\$KV" --out \$R/lockon
echo "[R4D_DONE]"
SEOF
for J in A B C D; do sed -i "s/^#SBATCH -N 1/#SBATCH -N 1/" $S/r4$J.slurm; done
# HDR 의 #SBATCH 줄이 shebang 다음에 오도록 확인
head -3 $S/r4A.slurm; grep -c SBATCH $S/r4A.slurm
JA=$(sbatch --parsable $S/r4A.slurm | tail -1); JB=$(sbatch --parsable --dependency=afterany:$JA $S/r4B.slurm | tail -1); JC=$(sbatch --parsable --dependency=afterany:$JB $S/r4C.slurm | tail -1); JD=$(sbatch --parsable --dependency=afterany:$JC $S/r4D.slurm | tail -1)
echo "A=$JA B=$JB C=$JC D=$JD"; sleep 10; squeue -u $USER -h -o "%.9i %.6j %.9T %.8M %.14R %.20E"
