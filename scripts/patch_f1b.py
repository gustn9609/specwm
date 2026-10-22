import os; S = os.environ["SCRATCH"]; p = S + "/r4F1.slurm"; s = open(p).read()
add = '''echo "##### F1d: stale anchors 55 s rerun (sink overwrite) #####"
v2 --action-root $S/traj55_action_05 --prompt-file $S/p_05.txt --scene 05 --frames 873 --seeds 1234 --base-video $B16 --configs W24_sink18 --arms stale_d15_k20,stale_d30_k40,stale_d45_k60,ctrl_stale_k20 --stale-inject d15=20,d30=40,d45=60 --out $R/resync55
echo "##### F1c: target cost with both experts resident on GPU #####"'''
assert s.count('echo "##### F1c: target cost with both experts resident on GPU #####"') == 1; s = s.replace('echo "##### F1c: target cost with both experts resident on GPU #####"', add)
open(p, "w").write(s); print("F1 에 stale 재실행 추가")
