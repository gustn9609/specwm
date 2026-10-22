#!/bin/bash
source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
true
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
source $SCRATCH/venv/bin/activate
cd $SCRATCH
until huggingface-cli download robbyant/lingbot-world-base-cam --local-dir $SCRATCH/lingbot-world/lingbot-world-base-cam; do
  echo "[retry v1 dl] $(date)"; sleep 30; done
echo "[v1 weights DONE] $(date)" >> $SCRATCH/specwm_status.txt
until huggingface-cli download robbyant/lingbot-world-v2-14b-causal-fast --local-dir $SCRATCH/lingbot-world-v2/lingbot-world-v2-14b-causal-fast; do
  echo "[retry v2 dl] $(date)"; sleep 30; done
echo "[v2 weights DONE] $(date)" >> $SCRATCH/specwm_status.txt
SM=$(sbatch --parsable $SCRATCH/smoke.sbatch)
echo "[smoke resubmitted: $SM] $(date)" >> $SCRATCH/specwm_status.txt
L1=$(sbatch --parsable --dependency=afterok:$SM $SCRATCH/specwm-lingbot/slurm/l1_stage_a.sbatch)
echo "[L1-A chained after smoke: $L1] $(date)" >> $SCRATCH/specwm_status.txt
