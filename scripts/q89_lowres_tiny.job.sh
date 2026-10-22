true
# 더 낮은 해상도 롤아웃: max_area 73728(208x320) / 46080(160x256) / 30720(128x208) / 20480(112x160), Fast 14B + 1.3B, 그 뒤 SR 평가, 그 뒤 H3-World 워커. GPU 4 (q88 arms 종료 후)
for i in $(seq 1 240); do grep -q "Q83_DONE" $SCRATCH/logs/q88_wcoff.3102066.out 2>/dev/null && break; sleep 30; done
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out/lowres_sr; CK=$PWD/lingbot-world-v2-14b-causal-fast; IMG=$PWD/examples/00/image.jpg; P="$(cat $SCRATCH/p_00.txt)"
for AREA in 73728 46080 30720 20480; do echo "########## FAST14B rollout area=$AREA ($(date)) ##########"
  python -m specwm.stevo_gen --ckpt $CK --tasks $SCRATCH/data/stevo/tasks_s00.json --out $R/fast14b_$AREA --arm fast_bf16 --frames 161 --max-area $AREA --seed 9999 --t5-gpu; done
for AREA in 73728 46080 30720 20480; do echo "########## 1.3B rollout area=$AREA ($(date)) ##########"; mkdir -p $R/v2_1p3b_$AREA
  SPECWM_MAX_AREA=$AREA python generate.py --task i2v-1.3B --ckpt_dir $PWD/lingbot-world-v2-1.3b-causal-fast --assets_dir $CK --infer_mode causal_fast --local_attn_size 18 --sink_size 6 --size 480*832 --frame_num 161 --image $IMG --action_path $SCRATCH/rt2_action_00 --prompt "$P" --base_seed 9999 --convert_model_dtype --save_file $R/v2_1p3b_$AREA/S00_s9999.mp4 2>&1 | grep -E "INFO: Generating|INFO: Saving|Error|Traceback"; done
rm -f $SCRATCH/exp/Q86_DONE; touch $SCRATCH/exp/Q86_DONE
cd $SCRATCH; rm -f exp/Q87_DONE; bash scripts/q87_sr_eval.job.sh
SHARD=1/3 bash $SCRATCH/scripts/q84_h3world_shard.job.sh
