#!/bin/bash
set -e
true
source $SCRATCH/venv/bin/activate
cd $SCRATCH/lingbot-world-v2
python3 -c "import ast; ast.parse(open('specwm/multi_scene.py').read())" && echo "multi_scene 서버 문법 OK"
python3 -c "import ast; ast.parse(open('$SCRATCH/lingbot-world/specwm/hybrid_refine.py').read())" && echo "hybrid_refine 서버 문법 OK"
# 장면별 액션 디렉토리 (00 은 기존 rt2_action 재사용)
[ -d $SCRATCH/rt2_action_00 ] || cp -r $SCRATCH/rt2_action $SCRATCH/rt2_action_00
python $SCRATCH/build_action.py examples/01 $SCRATCH/rt2_action_01 $SCRATCH/revisit_rt_poses_01.npy
python $SCRATCH/build_action.py examples/05 $SCRATCH/rt2_action_05 $SCRATCH/revisit_rt_poses_05.npy
# 장면별 프롬프트 (00 은 Base/Fast 가 쓴 기본 프롬프트, 01/05 는 예제 자체 프롬프트)
cp $SCRATCH/pA.txt $SCRATCH/p_00.txt
cp examples/01/prompt.txt $SCRATCH/p_01.txt
cp examples/05/prompt.txt $SCRATCH/p_05.txt
for s in 00 01 05; do printf "  p_%s: %s...\n" $s "$(head -c 70 $SCRATCH/p_$s.txt)"; done
# 하이브리드용: v1 저장소에 hands 장면
mkdir -p $SCRATCH/lingbot-world/examples/hands
cp $SCRATCH/hands_first.jpg $SCRATCH/lingbot-world/examples/hands/image.jpg
ls $SCRATCH/lingbot-world/examples/hands/
