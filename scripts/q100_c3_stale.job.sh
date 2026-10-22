true
# C3: stale anchor injection (10 s 프로토콜, S05/S01/S02 seed 1234): Fast 가 k 청크 생성 후 Base 청크 0..5 를 sink 에 주입 (k=6,9,12 → Δ≈4.5/6.75/9 s) + ctrl(자기 latent 주입). 설정 (24,18).
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V2=$S/lingbot-world-v2; cd $V2; export PYTHONPATH=$V2
for SC in 05 01 02; do BV=$R/floor/base/S${SC}_s1234/video.mp4; echo "########## C3 STALE S$SC ($(date)) ##########"
  python -m specwm.resync_v4 --ckpt $V2/lingbot-world-v2-14b-causal-fast --action-root $S/rt2_action_$SC --prompt-file $S/p_$SC.txt --scene $SC --frames 161 --seeds 1234 --base-video $BV --configs W24_sink18 --arms fast_alone,resync_k6,stale_d6_k6,stale_d9_k9,stale_d12_k12,ctrl_stale_k6 --stale-inject "d6=6,d9=9,d12=12" --out $R/c3_stale/S$SC
done; touch $S/exp/Q100_DONE; echo "[Q100_DONE] $(date)"
