#!/bin/bash
# Q5 (GPU 4, VisionReward 와 공유): 다양한 프롬프트 데모. Fast alone (11개) -> Base 16-step 참조 (~10분/개) -> Fast resync_k6 (Base ctx) -> 갤러리.
true
LIST=${LIST:-$SCRATCH/exp/demo_list.json}; MARK=${MARK:-Q5_DONE}
source $SCRATCH/venv/bin/activate
export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
R=$SCRATCH/lingbot-world-v2/specwm_out; D=$R/demo; mkdir -p $D
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD
echo "########## DEMO fast_alone ($(date)) ##########"
python -m specwm.demo_prompts --ckpt $PWD/lingbot-world-v2-14b-causal-fast --list $LIST --out $D
cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD
echo "########## DEMO Base 16-step refs ($(date)) ##########"
# Base 는 ~60-70GB 필요: 이 GPU 의 여유 메모리가 70GB 이상 될 때까지 대기 (최대 3h), 아니면 Base16/resync 단계 건너뜀
G=${CUDA_VISIBLE_DEVICES:-0}; OK=0
for i in $(seq 1 180); do FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $G); [ "$FREE" -ge 70000 ] && { OK=1; break; }; sleep 60; done
[ $OK -eq 1 ] || { echo "[Q5] GPU $G 여유 메모리 부족 (${FREE}MiB) -> Base16 단계 건너뜀"; python $SCRATCH/scripts/demo_gallery.py; touch $SCRATCH/exp/$MARK; exit 0; }
python - << 'PY'
import json, os, subprocess, sys
S=os.environ["SCRATCH"]; D=S+"/lingbot-world-v2/specwm_out/demo"
for it in json.load(open(os.environ["LIST"])):
    out=os.path.join(D, it["name"], "base16")
    if os.path.exists(os.path.join(out, "revisit.json")): print("[skip]", out, flush=True); continue
    img=it["image"] if it["image"].startswith("/") else S+"/lingbot-world-v2/"+it["image"]; act=it["action"] if it["action"].startswith("/") else S+"/lingbot-world-v2/"+it["action"]
    cmd=[sys.executable,"-m","specwm.bench_base_official","--ckpt",S+"/lingbot-world/lingbot-world-base-cam","--image",img,"--action-path",act,"--prompt-file",it["prompt_file"],"--frames",str(it.get("frames",161)),"--steps","16","--out",out]
    print("##########", it["name"], "base16", flush=True); subprocess.run(cmd)
    if os.path.exists(os.path.join(out,"video.mp4")): 
        import shutil; shutil.copy(it["prompt_file"], os.path.join(out,"prompt.txt"))
PY
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD
echo "########## DEMO resync_k6 with Base16 ctx ($(date)) ##########"
python - << 'PY'
# base16 참조를 demo_prompts 의 --base-root 형식(<root>/<name>/video.mp4)으로 링크
import json, os; S=os.environ["SCRATCH"]; D=S+"/lingbot-world-v2/specwm_out/demo"; B=D+"/_base16_root"; os.makedirs(B, exist_ok=True)
for it in json.load(open(os.environ["LIST"])):
    src=os.path.join(D, it["name"], "base16", "video.mp4"); dst=os.path.join(B, it["name"]); os.makedirs(dst, exist_ok=True)
    if os.path.exists(src) and not os.path.exists(dst+"/video.mp4"): os.symlink(src, dst+"/video.mp4")
PY
python -m specwm.demo_prompts --ckpt $PWD/lingbot-world-v2-14b-causal-fast --list $LIST --out $D --base-root $D/_base16_root --k 6
echo "########## DEMO gallery ($(date)) ##########"
python $SCRATCH/scripts/demo_gallery.py
touch $SCRATCH/exp/$MARK; echo "[$MARK] $(date)"
