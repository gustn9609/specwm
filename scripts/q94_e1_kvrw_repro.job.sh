true
# E1': KV-rewrite 재현 — 3 scenes (05,01,02) x 3 seeds (1234,4321,9999). 각 조합: launder(v1 low expert, sched 0.40,0.20,0.08) -> RePaint 16-step (target) -> Fast kvrw arms (laundered / ideal=Base70) + resync_k6 + fast_alone
# LAUNDER_CK 로 launderer 교체 가능 (D1a: bid_as_v1_cam). TAG 로 출력 폴더 구분.
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; V1=$S/lingbot-world; V2=$S/lingbot-world-v2; CK=${LAUNDER_CK:-$V1/lingbot-world-base-cam}; TAG=${TAG:-e1repro}; EXTRA=${LAUNDER_EXTRA:-}; SCHED=${SCHED:-0.40,0.20,0.08}
O=$R/$TAG; mkdir -p $O
for SC in ${SCENES:-05 01 02}; do for SD in ${SEEDS:-1234 4321 9999}; do
  FL=$R/floor/fast/S$SC/W18_sink6/fast_alone_s$SD/latent.pt; BV=$R/floor/base/S${SC}_s$SD/video.mp4; IMG=$V2/examples/$SC/image.jpg
  [ -f $FL ] && [ -f $BV ] || { echo "[skip] S$SC s$SD inputs missing"; continue; }
  LD=$O/S$SC/launder_s$SD; TD=$O/S$SC/target_s$SD
  if [ ! -f $TD/video.mp4 ]; then
    echo "########## LAUNDER S$SC s$SD ($(date)) ##########"; cd $V1; PYTHONPATH=$V1 python -m specwm.hybrid_refine --ckpt $CK --scene $SC --image $IMG --action-dir $S/rt2_action_$SC --prompt-file $S/p_$SC.txt --fast-latent $FL --frames 153 --schedules "$SCHED" --out $LD $EXTRA
    LV=$(ls -d $LD/REFINE_*/video.mp4 | head -1)
    echo "########## REPAINT S$SC s$SD ($(date)) ##########"; PYTHONPATH=$V1 python -m specwm.bench_base_prefix2 --ckpt $V1/lingbot-world-base-cam --image $IMG --action-path $S/rt2_action_$SC --prompt-file $S/p_$SC.txt --frames 161 --steps 16 --prefix-latents 9 --mode repaint --seed $SD --prefix-video $LV --out $TD
  fi
  echo "########## KVRW S$SC s$SD ($(date)) ##########"; cd $V2; PYTHONPATH=$V2 python -m specwm.resync_v4 --ckpt $V2/lingbot-world-v2-14b-causal-fast --action-root $S/rt2_action_$SC --prompt-file $S/p_$SC.txt --scene $SC --frames 161 --seeds $SD --base-video $BV --configs W18_sink6 --arms fast_alone,resync_k6,kvrw_laundered,kvrw_ideal --kv-providers "laundered=$TD/video.mp4,ideal=$BV" --out $O/S$SC
done; done
touch $S/exp/Q94_${TAG}_DONE; echo "[Q94_DONE $TAG] $(date)"
