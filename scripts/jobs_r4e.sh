#!/bin/bash
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out
true
cd $S/lingbot-world-v2 && python -c "import ast; ast.parse(open('specwm/resync_v4.py').read()); print('v4 OK')"
python - << 'PY'
import os; S=os.environ["SCRATCH"]
# 캐시 구조 확인: binding_v2 에서 _caches["target"]["kv"] 가 층별 리스트인지
src=open(S+"/lingbot-world-v2/specwm/binding_v2.py").read(); i=src.find('_caches'); print("binding _caches refs:", src.count('_caches'), "| kv list:", '["kv"]' in src)
PY
HDR='#!/bin/bash
true
export SPECWM_NO_FLASH=1; export OMP_NUM_THREADS=4; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V1=$S/lingbot-world; V2=$S/lingbot-world-v2; CK=$V1/lingbot-world-base-cam
B16=$R/revisit/base_official_S05_s16/video.mp4
v2() { cd $V2; PYTHONPATH=$V2 python -m specwm.resync_v4 --ckpt $V2/lingbot-world-v2-14b-causal-fast "$@"; }
launder() { cd $V1; PYTHONPATH=$V1 python -m specwm.hybrid_refine --ckpt $CK --scene $1 --action-dir $S/rt2_action_$1 --prompt-file $S/p_$1.txt --fast-latent $R/repro/S$1/W18_sink6/fast_alone_s$2/latent.pt --frames 153 --schedules "0.40,0.20,0.08" --out $R/repro/S$1/launder_s$2; }
repaint() { cd $V1; PYTHONPATH=$V1 python -m specwm.bench_base_prefix2 --ckpt $CK --image $V2/examples/$1/image.jpg --action-path $S/rt2_action_$1 --prompt-file $S/p_$1.txt --frames 161 --steps 16 --prefix-latents 9 --mode repaint --seed $2 --prefix-video $R/repro/S$1/launder_s$2/REFINE_0.40_0.20_0.08/video.mp4 --out $R/repro/S$1/target_s$2; }
base16() { [ -f $R/repro/S$1/base16_s$2/video.mp4 ] && return; cd $V1; PYTHONPATH=$V1 python -m specwm.bench_base_official --ckpt $CK --image $V2/examples/$1/image.jpg --action-path $S/rt2_action_$1 --prompt-file $S/p_$1.txt --frames 161 --steps 16 --seed $2 --out $R/repro/S$1/base16_s$2; }
v1trio() { echo "##### v1 trio S$1 s$2 #####"; launder $1 $2; repaint $1 $2; base16 $1 $2; }
'
cat > $S/r4E1.slurm << SEOF
$HDR
echo "##### E1a: fast_alone all scenes #####"
for SC in 05 01 02; do v2 --action-root \$S/rt2_action_\$SC --prompt-file \$S/p_\$SC.txt --scene \$SC --frames 161 --seeds 1234,4321,9999 --configs W18_sink6 --arms fast_alone --out \$R/repro/S\$SC; done
echo "##### E1b: stale anchors 55 s (sink overwrite) #####"
v2 --action-root \$S/traj55_action_05 --prompt-file \$S/p_05.txt --scene 05 --frames 873 --seeds 1234 --base-video \$B16 --configs W24_sink18 --arms stale_d15_k20,stale_d30_k40,stale_d45_k60,ctrl_stale_k20 --stale-inject d15=20,d30=40,d45=60 --out \$R/resync55
echo "[R4E1_DONE]"
SEOF
cat > $S/r4E2.slurm << SEOF
$HDR
v1trio 05 4321; v1trio 05 9999; v1trio 01 1234; v1trio 01 4321; v1trio 01 9999
echo "[R4E2_DONE]"
SEOF
cat > $S/r4E3.slurm << SEOF
$HDR
v1trio 02 1234; v1trio 02 4321; v1trio 02 9999
echo "##### kvrw arms: repro #####"
for SC in 05 01 02; do v2 --action-root \$S/rt2_action_\$SC --prompt-file \$S/p_\$SC.txt --scene \$SC --frames 161 --seeds 1234,4321,9999 --configs W18_sink6 --arms kvrw_laundered,kvrw_ideal --kv-providers "laundered=\$R/repro/S\$SC/target_s{seed}/video.mp4,ideal=\$R/repro/S\$SC/base16_s{seed}/video.mp4" --out \$R/repro/S\$SC; done
echo "##### kvrw arms: lock-on sigma sweep #####"
KV="l15=\$R/lockon/launder/REFINE_0.15_0.07_0.03/video.mp4,l25=\$R/lockon/launder/REFINE_0.25_0.12_0.05/video.mp4,l40=\$R/hybrid/s05_launder/REFINE_0.40_0.20_0.08/video.mp4,l60=\$R/lockon/launder/REFINE_0.60_0.30_0.12/video.mp4,l80=\$R/lockon/launder/REFINE_0.80_0.40_0.16/video.mp4,raw=\$R/resync_v2/W18_sink6/fast_alone_s1234/video.mp4"
v2 --action-root \$S/rt2_action_05 --prompt-file \$S/p_05.txt --scene 05 --frames 161 --seeds 1234 --configs W18_sink6 --arms kvrw_l15,kvrw_l25,kvrw_l40,kvrw_l60,kvrw_l80,kvrw_raw --kv-providers "\$KV" --out \$R/lockon
echo "[R4E3_DONE]"
SEOF
# SBATCH 지시문이 쉘 명령 뒤에 오면 무시됨 -> -J 줄을 위로 올림
for J in E1 E2 E3; do python3 - "$S/r4$J.slurm" << 'PY'
import sys; p=sys.argv[1]; L=open(p).read().split("\n"); j=[l for l in L if l.startswith("#SBATCH -J")]; L=[l for l in L if not l.startswith("#SBATCH -J")]
i=next(k for k,l in enumerate(L) if l.startswith("#SBATCH -o")); L[i+1:i+1]=j; open(p,"w").write("\n".join(L))
PY
done; head -5 $S/r4E1.slurm | cut -c1-80
JE1=$(sbatch --parsable --dependency=afterany:972387 $S/r4E1.slurm | tail -1); JE2=$(sbatch --parsable --dependency=afterany:$JE1 $S/r4E2.slurm | tail -1); echo "E1=$JE1 E2=$JE2 (E3 는 한도 때문에 나중에)"
squeue -u $USER -h -o "%.9i %.6j %.9T %.8M %.14R %.22E"
