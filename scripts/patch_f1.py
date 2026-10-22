import os; S = os.environ["SCRATCH"]; p = S + "/r4F1.slurm"; s = open(p).read()
add = '''echo "##### F1c: target cost with both experts resident on GPU #####"
V1=$S/lingbot-world; CK=$V1/lingbot-world-base-cam; LAU=$R/hybrid/s05_launder/REFINE_0.40_0.20_0.08/video.mp4
for CFG in "386048 16" "386048 8" "225280 16" "225280 8" "99840 16" "99840 8"; do set -- $CFG
  cd $V1; PYTHONPATH=$V1 python -m specwm.bench_base_prefix2 --ckpt $CK --image $V2/examples/05/image.jpg --action-path $S/rt2_action_05 --prompt-file $S/p_05.txt --frames 73 --steps $2 --max-area $1 --prefix-latents 9 --mode repaint --seed 1234 --prefix-video $LAU --resident --out $R/targetcost_res/area$1_s$2
done
echo "[R4F1_DONE]"'''
assert s.count('echo "[R4F1_DONE]"') == 1; s = s.replace('echo "[R4F1_DONE]"', add); s = s.replace("#SBATCH -N 1 -n 1 -p gh-dev -t 01:40:00", "#SBATCH -N 1 -n 1 -p gh-dev -t 02:00:00")
open(p, "w").write(s); print("r4F1.slurm 패치 완료")
