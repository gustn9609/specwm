#!/bin/bash
S=$SCRATCH/lingbot-world
echo "=== v1 wan/__init__.py ==="; cat $S/wan/__init__.py
echo; echo "=== v1 image2video.py (Base) 카메라/조건 관련 ==="
grep -nE "only_rays|plucker|dit_cond|c2ws|action_path|wasd|cam" $S/wan/image2video.py | head -40
echo; echo "=== v1 image2video_fast.py 495-545 ==="; sed -n 495,545p $S/wan/image2video_fast.py
echo; echo "=== only_rays_d 정의/설정 위치 ==="
grep -rn "only_rays_d" $S/wan/ $S/generate.py $S/*.py 2>/dev/null | grep -v "^Binary" | head
echo; echo "=== get_plucker_embeddings 시그니처 ==="
grep -n "def get_plucker_embeddings" -A12 $S/wan/utils/cam_utils.py
echo; echo "=== generate.py 카메라 인자 ==="
grep -nE "action_path|allow_act2cam|only_rays|act_mode|cam_mode|add_argument" $S/generate.py | head -30
