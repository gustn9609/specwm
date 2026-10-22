true
# v2 causal-pretrain S00 timing @360x512 (480x832 는 cond+uncond 전체 KV 캐시로 1 GPU OOM) — Q73C 뒤 GPU 2
for i in $(seq 1 900); do [ -f $SCRATCH/exp/Q73C_DONE ] && break; sleep 60; done
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out/timing_s00/v2_pretrain_s00_360; mkdir -p $R
echo "########## V2 pretrain360 S00 timing ($(date)) ##########"
python generate.py --task i2v-A14B --ckpt_dir $PWD/lingbot-world-v2-14b-causal-pretrain --assets_dir $PWD/lingbot-world-v2-14b-causal-fast --infer_mode causal_pretrain --t5_cpu --offload_model True \
  --size 480*832 --frame_num 161 --image $SCRATCH/lingbot-world/examples/00/image.jpg --action_path $SCRATCH/rt2_action_00 --prompt "$(cat $SCRATCH/p_00.txt)" --base_seed 9999 --convert_model_dtype --save_file $R/S00_s9999_480.mp4 2>&1 | grep -v "^\s*$" | tail -12
echo "########## V2 pretrain360 S00 timing 360x512 ($(date)) ##########"
python - <<PY
# generate.py 는 --size 로 480*832 계열만 받으므로 360x512 는 max_area 로 직접 호출
PY
python -c "
import sys,os; sys.argv=['generate.py','--task','i2v-A14B','--ckpt_dir','$PWD/lingbot-world-v2-14b-causal-pretrain','--assets_dir','$PWD/lingbot-world-v2-14b-causal-fast','--infer_mode','causal_pretrain','--t5_cpu','--offload_model','True','--size','480*832','--frame_num','161','--image','$SCRATCH/lingbot-world/examples/00/image.jpg','--action_path','$SCRATCH/rt2_action_00','--prompt',open('$SCRATCH/p_00.txt').read().strip(),'--base_seed','9999','--convert_model_dtype','--save_file','$R/S00_s9999_360.mp4']
import wan.configs as C; C.MAX_AREA_CONFIGS['480*832']=184320
import runpy; runpy.run_path('generate.py', run_name='__main__')" 2>&1 | grep -v "^\s*$" | tail -12
touch $SCRATCH/exp/Q80_DONE; echo "[Q80_DONE] $(date)"
