#!/bin/bash
S=$SCRATCH; A=$S/aria
true
taskset -c 0-3 python -u aria_win.py loc2_script3_seq3_rec1 210 161 B2_pour_liquid_pan 2>&1 | grep -E "WIN_DONE|Error|assert"
deactivate; source $S/specwm-venv/bin/activate
# 프롬프트
w() { echo "$2" > $A/win_$1/prompt.txt; }
w A_egg_into_pan "First-person view in a home kitchen, wearing smart glasses. The person holds a raw egg, cracks it on the edge of the frying pan and drops the egg into the hot pan on the stove, then discards the shell. Natural hand motion, realistic lighting, steady egocentric camera."
w B2_pour_liquid_pan "First-person view in a home kitchen. The person holds a small container and pours liquid into a pan on the stove, then sets the container down on the counter. Natural hand motion, realistic lighting, steady egocentric camera."
w C_pour_stir_pan "First-person view in a home kitchen. The person holds a bowl of chopped vegetables, tips it to pour the vegetables into the pan on the stove and stirs them with a utensil. Natural hand motion, realistic lighting, steady egocentric camera."
w D_wash_tomato "First-person view at a kitchen sink. The person turns on the faucet, holds a tomato under the running water, rinses it with the fingers and turns the water off. Natural hand motion, realistic lighting, steady egocentric camera."
w E_spatula_to_plate "First-person view in a home kitchen. The person uses a spatula to scrape scrambled eggs from the frying pan onto a plate held in the other hand. Natural hand motion, realistic lighting, steady egocentric camera."
mk() { # $1 jobname $2.. windows
cat > $S/aria_$1.slurm << SEOF
#!/bin/bash
true
export SPECWM_NO_FLASH=1; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd \$SCRATCH/lingbot-world; export PYTHONPATH=\$PWD
for W in ${@:2}; do for SD in 1234 42; do E=\$SCRATCH/aria/win_\$W; echo "##### \$W 81f seed \$SD #####"
  python -m specwm.bench_base_official --ckpt \$PWD/lingbot-world-base-cam --image \$E/image.jpg --action-path \$E/action --prompt-file \$E/prompt.txt --frames 81 --steps 70 --seed \$SD --out \$SCRATCH/lingbot-world-v2/specwm_out/aria/\${W}_base70_81f_s\$SD
  echo "[ARIA_BASE_DONE] \$W s\$SD"
done; done
SEOF
}
mk H1 C_pour_stir_pan A_egg_into_pan E_spatula_to_plate; mk H2 D_wash_tomato B2_pour_liquid_pan
J1=$(sbatch --parsable $S/aria_H1.slurm | tail -1); J2=$(sbatch --parsable $S/aria_H2.slurm | tail -1); echo "H1=$J1 H2=$J2"
squeue -u $USER -h -o "%.9i %.8j %.9T %.8M %.14R"; squeue -u $USER -h --start -o "%.9i %.8j %.20S" | grep aria
