#!/bin/bash
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out
echo "=== jb2 OOM 위치 ==="; grep -nE "File \".*wan/|File \".*specwm" $S/jb2.970782.out | tail -3 | cut -c1-120
# J-B3: 241 frames (61 latents = 15 s), seeds 1234, 9999 (gh)
sed -e 's/for SD in 1234 4321; do/for SD in 1234 9999; do/' -e 's/--frames 321/--frames 241/' -e 's#base20_S05_s16_seed#base15_S05_s16_seed#' -e 's/-J jbBase20/-J jbBase15/' -e 's#jb.%j.out#jb3.%j.out#' $S/jb.slurm > $S/jb3.slurm
grep -nE "frames|out " $S/jb3.slurm | head -3 | cut -c1-160; sbatch $S/jb3.slurm | tail -1
# xfade node 3 변형 (gh-dev): 별도 out 이름으로 (기존 xfade arm 과 구분)
sed -e 's/--xfade --prefix-videos [^ ]*/--xfade --xfade-node 3/' -e 's#--out $R/resync_v2#--out $R/resync_v2_xf3#' -e 's/-J jc1/-J xf3/' -e 's#jc1.%j.out#xf3.%j.out#' $S/jc1.slurm > $S/xf3.slurm
grep -E "xfade-node|--out" $S/xf3.slurm | cut -c1-120
# resync_v2_xf3 는 새 out 이라 fast_alone 도 다시 돈다(필요: xfade 가 fast_alone latent 를 씀) - 빠름
sbatch $S/xf3.slurm | tail -1
squeue -u $USER -h -o "%.9i %.10j %.9T %.8M %.12R"
