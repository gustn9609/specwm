true
# S00 seed 9999 (Great Wall, rt2_action_00, p_00): (a) LingBot 1.3B 480x832 재실행(올바른 이미지) (b) Fast 14B 저해상도 롤아웃 256x464 / 360x512 (c) 1.3B 저해상도 롤아웃 → super-res 는 별도 스크립트(sr_eval.py)
for i in $(seq 1 120); do [ -f $SCRATCH/lingbot-world-v2/specwm_out/timing_s00/base_wcoff3_360x512/video.mp4 ] && break; sleep 30; done
kill -- -2855417 2>/dev/null; sleep 5; for p in $(ps -eo pid,cmd | grep "[b]ench_base_official.*base_wc8_480x832\|[b]ench_base_official.*v2bid_bf16_480x832" | awk '{print $1}'); do kill $p; done; sleep 5
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out/lowres_sr; mkdir -p $R; CK=$PWD/lingbot-world-v2-14b-causal-fast; IMG=$PWD/examples/00/image.jpg; P="$(cat $SCRATCH/p_00.txt)"
for AREA in 399360 184320 122880; do echo "########## FAST14B rollout area=$AREA ($(date)) ##########"
  python -m specwm.stevo_gen --ckpt $CK --tasks $SCRATCH/data/stevo/tasks_s00.json --out $R/fast14b_$AREA --arm fast_bf16 --frames 161 --max-area $AREA --seed 9999 --t5-gpu; done
for AREA in 399360 184320 122880; do echo "########## 1.3B rollout area=$AREA ($(date)) ##########"; mkdir -p $R/v2_1p3b_$AREA
  SPECWM_MAX_AREA=$AREA python generate.py --task i2v-1.3B --ckpt_dir $PWD/lingbot-world-v2-1.3b-causal-fast --assets_dir $CK --infer_mode causal_fast --local_attn_size 18 --sink_size 6 --size 480*832 --frame_num 161 --image $IMG --action_path $SCRATCH/rt2_action_00 --prompt "$P" --base_seed 9999 --convert_model_dtype --save_file $R/v2_1p3b_$AREA/S00_s9999.mp4 2>&1 | grep -E "INFO: Generating|INFO: Saving|Error|Traceback"; done
touch $SCRATCH/exp/Q86_DONE; echo "[Q86_DONE] $(date)"
