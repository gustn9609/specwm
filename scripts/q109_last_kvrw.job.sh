true
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; O=$R/stevo/reground; V2=$S/lingbot-world-v2; cd $V2; export PYTHONPATH=$V2; t=wine_color_fabric
IMG=$(python -c "import json;T=json.load(open('$S/data/stevo/tasks_camera.json'));print([x for x in T if x['id']=='$t'][0]['image'])"); ACT=$(python -c "import json;T=json.load(open('$S/data/stevo/tasks_camera.json'));print([x for x in T if x['id']=='$t'][0]['action'])")
python -m specwm.resync_v4 --ckpt $V2/lingbot-world-v2-14b-causal-fast --action-root $ACT --prompt-file $O/$t.prompt.txt --scene 00 --image $IMG --frames 253 --seeds 1234 --base-video $R/stevo/lingbot_base/$t/video.mp4 --configs W18_sink6 --arms kvrw_laundered --kv-providers "laundered=$O/$t/target/video.mp4" --out $O/$t 2>&1 | grep -E "kvrw_laundered|실패|Traceback"
echo "[Q109_DONE] $(date)"
