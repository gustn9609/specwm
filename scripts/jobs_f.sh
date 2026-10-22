#!/bin/bash
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out
true
cd $S/lingbot-world-v2 && python -c "import ast; ast.parse(open('specwm/resync_v4.py').read()); print('v4 OK')"; grep -n '"hw"' specwm/binding_v2.py | head -2
cat > $S/r4F1.slurm << 'SEOF'
#!/bin/bash
true
export SPECWM_NO_FLASH=1; export OMP_NUM_THREADS=4; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V2=$S/lingbot-world-v2; B16=$R/revisit/base_official_S05_s16/video.mp4
v2() { cd $V2; PYTHONPATH=$V2 python -m specwm.resync_v4 --ckpt $V2/lingbot-world-v2-14b-causal-fast "$@"; }
echo "##### F1a: low-res / 8-step targets as KV providers (S05 s1234) #####"
KV=""; for A in 386048 225280 99840; do for ST in 16 8; do KV="$KV,tc${A}_s${ST}=$R/targetcost/area${A}_s${ST}/video.mp4"; done; done; KV=${KV#,}
v2 --action-root $S/rt2_action_05 --prompt-file $S/p_05.txt --scene 05 --frames 161 --seeds 1234 --configs W18_sink6 --arms kvrw_tc386048_s16,kvrw_tc386048_s8,kvrw_tc225280_s16,kvrw_tc225280_s8,kvrw_tc99840_s16,kvrw_tc99840_s8 --kv-providers "$KV" --out $R/targetcost_kv
echo "##### F1b: sink capacity sweep 55 s #####"
v2 --action-root $S/traj55_action_05 --prompt-file $S/p_05.txt --scene 05 --frames 873 --seeds 1234 --base-video $B16 --configs none --extra-configs "W12_sink6:fast_alone+resync_k2,W18_sink12:fast_alone+resync_k4,W42_sink36:fast_alone+resync_k12" --out $R/resync55
echo "[R4F1_DONE]"
SEOF
echo "r4F1.slurm 작성"; squeue -u $USER -h -o "%.9i %.6j %.9T %.8M %.14R"
