#!/bin/bash
S=$SCRATCH; module load gcc cuda python3 >/dev/null 2>&1; source $S/SolarWM/.venv-solar/bin/activate
python $S/patch_co.py
# Stage1 slurm: Stage2 것을 복사해 config/ckpt/run_id/out 교체
sed -e 's#infer_stage2_sgf_camera_length.yaml#infer_stage1_tf_anyflow_v1_5_81f.yaml#' \
    -e 's#SolarWM-5B-sgf-stage2-81f#SolarWM-5B-tf-stage1-81f#' \
    -e 's#inference.run_id=lingbot_s05_stage2#inference.run_id=lingbot_s05_stage1 --set distributed.world_size=1 --set inference.length=camera#' \
    -e 's#solar_out/stage2_s05#solar_out/stage1_s05#' \
    -e 's#-J solars2#-J solars1#' -e 's#solar_s2.%j.out#solar_s1.%j.out#' -e 's#-t 00:45:00#-t 01:30:00#' \
    -e 's#SOLAR_S2_DONE#SOLAR_S1_DONE#' $S/solar_s2.slurm > $S/solar_s1.slurm
grep -nE "config|checkpoint|run_id|output_dir|-t |-p " $S/solar_s1.slurm
rm -rf $S/solar_out/stage1_s05 $S/solar_out/stage2_s05
sbatch $S/solar_s1.slurm | tail -1
# Stage2 도 커밋 패치 적용본으로 재실행 (깨끗한 출력 확보용)
sbatch $S/solar_s2.slurm | tail -1
squeue -u $USER -h -o "%.9i %.10j %.9T %.8M"
