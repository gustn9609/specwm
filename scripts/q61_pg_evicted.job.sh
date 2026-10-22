#!/bin/bash
# GPU7: PoseGram/oracle 블렌딩 — 매칭 후보를 sink 밖(evicted, 잠재프레임 >= 18 이면서 window 밖) 으로 제한: --min-gap 18 은 유지, 후보 하한 --evicted-only
true
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out; CK=$PWD/lingbot-world-v2-14b-causal-fast
run() { SC=$1; ACT=$2; shift 2; echo "########## PG-evicted S$SC $* ($(date)) ##########"; python -m specwm.posegram --ckpt $CK --action-root $ACT --prompt-file $SCRATCH/p_$SC.txt --scene $SC --frames 873 --window 18 --sink 18 --evicted-only --out $R/posegram/S${SC}_W18_evicted "$@" || echo "!!! FAILED $*"; }
for SD in 1234 4321 9999; do run 05 $SCRATCH/traj55_action_05 --mode posegram_blend --alpha 0.3 --write first --seed $SD; run 05 $SCRATCH/traj55_action_05 --mode oracle_blend_all --alpha 0.3 --seed $SD; done
for SD in 1234 4321; do run 05 $SCRATCH/traj55_action_05 --mode posegram_rand_blend --alpha 0.3 --write first --seed $SD; done
python $SCRATCH/scripts/geo55.py $R/geo55.json --scan; touch $SCRATCH/exp/Q61_DONE; echo "[Q61_DONE] $(date)"
