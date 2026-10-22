true
# B1 verifier comparison on GPU 2: clips = Base/Fast (S05,S01,S00,S02 seed1234) + StEvo (falling_glass, pouring, rocket, balloon: base/fast) + controls (shuffle/teleport/freeze).
# per clip: V-JEPA2 surprise, VGGT pose recovery (+commanded-pose error for floor clips), Base diffusion loss (floor clips + controls), DINO jumps (already).
true
S=$SCRATCH; O=$S/lingbot-world-v2/specwm_out; B=$O/b1; mkdir -p $B
source $S/venv/bin/activate; python $S/scripts/b1_make_controls.py
CLIPS=""
for sc in 05 01 00 02; do CLIPS="$CLIPS base_S$sc=$O/floor/base/S${sc}_s1234/video.mp4 fast_S$sc=$O/floor/fast/S$sc/W18_sink6/fast_alone_s1234/video.mp4 ctrl_S${sc}_shuffle=$B/controls/S${sc}_shuffle.mp4 ctrl_S${sc}_teleport=$B/controls/S${sc}_teleport.mp4 ctrl_S${sc}_freeze=$B/controls/S${sc}_freeze.mp4"; done
for t in falling_glass pouring_water_into_cup rocket_launch inflating_balloon; do CLIPS="$CLIPS stevo_base_$t=$O/stevo/lingbot_base/$t/video.mp4 stevo_fast_$t=$O/stevo/lingbot_fast/$t.mp4"; done
source $S/b1-venv/bin/activate
for kv in $CLIPS; do n=${kv%%=*}; v=${kv#*=}; [ -f $B/${n}_surprise.json ] || python $S/scripts/b1_vjepa_surprise.py --video $v --out $B/${n}_surprise.json 2>&1 | grep -E "VJEPA_DONE|Error|Traceback"; done
for kv in $CLIPS; do n=${kv%%=*}; v=${kv#*=}; P=""; case $n in base_S*|fast_S*|ctrl_S*) sc=$(echo $n | grep -o "S0[0-9]" | head -1); P="--poses $S/rt2_action_${sc#S}/poses.npy";; esac
  [ -f $B/${n}_vggt.json ] || python $S/scripts/b1_vggt_pose.py --video $v --out $B/${n}_vggt.json $P 2>&1 | grep -E "VGGT_DONE|Error|Traceback"; done
source $S/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd $S/lingbot-world; export PYTHONPATH=$PWD
for kv in $CLIPS; do n=${kv%%=*}; v=${kv#*=}; case $n in stevo_*) continue;; esac; sc=$(echo $n | grep -o "S0[0-9]" | head -1); sc=${sc#S}
  [ -f $B/${n}_diffloss.json ] || python -m specwm.base_diffloss --ckpt $PWD/lingbot-world-base-cam --video $v --image $S/lingbot-world-v2/examples/$sc/image.jpg --action-dir $S/rt2_action_$sc --prompt-file $S/p_$sc.txt --frames 153 --out $B/${n}_diffloss.json 2>&1 | grep -E "DIFFLOSS_DONE|Error|Traceback|SystemExit"; done
touch $S/exp/B1_DONE; echo "[B1_DONE] $(date)"
