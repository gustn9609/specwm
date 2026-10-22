true
# v2 causal-pretrain StEvo: 1 GPU 메모리 한계로 360x512 + 173 프레임(논문의 HY-WorldPlay 기본 pan "s-3, right-10, left-20, right-10") 변형 프로토콜
export SPECWM_MAX_AREA=184320
VARIANT=pretrain SHARD=0/1 TASKS=$SCRATCH/data/stevo/tasks_camera_173.json OUT=$SCRATCH/lingbot-world-v2/specwm_out/stevo/lingbot_v2_pretrain_360_173 FRAMES=173 bash $SCRATCH/scripts/q75_stevo_v2.job.sh
touch $SCRATCH/exp/Q81_DONE; echo "[Q81_DONE] $(date)"
