#!/bin/bash
# bundle definitions: bundle <name> runs sequentially on one GPU pair (GPUS env). Uses run() from the worker.
V0=/home/hk28965/specwm/lingbot-world-v2/specwm_out/spec_v0; PK="/home/hk28965/specwm/venv/bin/python /home/hk28965/specwm/scripts/spec_v0_pick.py"
bundle() { case $1 in
 diag_castle_4321) run sandcastle_wave_impact 4321 always_c0base always --chunk0-from $V0/sandcastle_wave_impact/base70_s4321/video.mp4 --steps 16 ;;
 corrmask_castle_4321) run sandcastle_wave_impact 4321 always_c0base_mask always --chunk0-from $V0/sandcastle_wave_impact/base70_s4321/video.mp4 --steps 16 --corr-mode mask ;;
 corrboth_castle_4321) run sandcastle_wave_impact 4321 always_c0base_both always --chunk0-from $V0/sandcastle_wave_impact/base70_s4321/video.mp4 --steps 16 --corr-mode both ;;
 base16_castle_4321) D=$V0/sandcastle_wave_impact/base16_s4321; [ -f $D/video.mp4 ] || { echo "########## [$GPUS] BASE16 castle s4321 ($(date)) ##########"; IMG=$(python -c "import json;print([t for t in json.load(open('/home/hk28965/specwm/data/stevo/tasks_camera.json')) if t['id']=='sandcastle_wave_impact'][0]['image'])")
   (cd lingbot-world; PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=${GPUS%%,*} python -m specwm.bench_base_official --ckpt $PWD/lingbot-world-base-cam --image $IMG --action-path /home/hk28965/specwm/data/stevo/actions/s-3_right-30_left-30 --prompt-file /home/hk28965/specwm/exp/empty_prompt.txt --max-area 399360 --frames 161 --steps 16 --seed 4321 --out $D 2>&1 | grep -E "^\[BASE|Error|Traceback"); echo "[SPEC_V0_DONE] base16 castle s4321"; } ;;
 diag_castle_9999) run sandcastle_wave_impact 9999 always_c0base always --chunk0-from $V0/sandcastle_wave_impact/base70_s9999/video.mp4 --steps 16 ;;
 d360win3_castle_1234) run sandcastle_wave_impact 1234 d360_win3_g1 window:3:$($PK thr2 sandcastle_wave_impact 1234 g1 0.5 probe2_360) --chk g1 --warmup 2 --steps 16 --max-area 184320 --nodes 0,3 --corr-area 399360 ;;
 padcheck) for P in 0 1; do run sandcastle_wave_impact 4321 _pad${P}_always61 always --frames 61 --steps 16 --end-pad $P; done ;;
 adapt_*) T=${1#adapt_}; T=${T%_*}; S=${1##*_}; [ $T = jar ] && T=sand_filling_up_jar; [ $T = castle ] && T=sandcastle_wave_impact
   run $T $S early_adapt50_a240g1c3 early:q0.5:$($PK thr2 $T $S a240g1c3 0.5) --chk a240g1c3 --warmup 2 --run 2 --steps 16
   run $T $S early_adapt70_a240g1c3 early:q0.7:$($PK thr2 $T $S a240g1c3 0.7) --chk a240g1c3 --warmup 2 --run 2 --steps 16
   run $T $S early_adapt50_kv early:q0.5:$($PK thr2 $T $S kv 0.5 probe2kv) --chk kv --warmup 2 --run 2 --steps 16 ;;
 sched_*) T=${1#sched_}; T=${T%_*}; S=${1##*_}; [ $T = jar ] && T=sand_filling_up_jar; [ $T = castle ] && T=sandcastle_wave_impact
   run $T $S sched_123 sched:1,2,3 --steps 16
   run $T $S sched_123_12 sched:1,2,3,12 --steps 16
   run $T $S sched_123_6_9_12 sched:1,2,3,6,9,12 --steps 16 ;;
 cam_*) T=${1#cam_}; T=${T%_*}; S=${1##*_}; [ $T = jar ] && T=sand_filling_up_jar; [ $T = castle ] && T=sandcastle_wave_impact
   run $T $S sched_123_11 sched:1,2,3,11 --steps 16
   run $T $S sched_123_6_11 sched:1,2,3,6,11 --steps 16 ;;
 sctx_*) T=${1#sctx_}; T=${T%_*}; S=${1##*_}; [ $T = jar ] && T=sand_filling_up_jar; [ $T = castle ] && T=sandcastle_wave_impact
   run $T $S sched_123_6_9_12_ctx18 sched:1,2,3,6,9,12 --steps 16 --ctx 18
   run $T $S sched_123_6_9_12_ctx12 sched:1,2,3,6,9,12 --steps 16 --ctx 12 ;;
 sem_*) T=${1#sem_}; T=${T%_*}; S=${1##*_}; [ $T = jar ] && T=sand_filling_up_jar; [ $T = castle ] && T=sandcastle_wave_impact
   run $T $S sem50_w3 sem:0.5 --warmup 3 --steps 16
   run $T $S sem35_w3 sem:0.35 --warmup 3 --steps 16
   run $T $S sem50_w3_first sem:0.5 --warmup 3 --steps 16 --sem-bank 0 ;;
 semc_*) T=${1#semc_}; T=${T%_*}; S=${1##*_}; [ $T = jar ] && T=sand_filling_up_jar; [ $T = castle ] && T=sandcastle_wave_impact
   run $T $S semc100_w3 sem:1.0 --sem-colour 1 --warmup 3 --steps 16
   run $T $S semc80_w3 sem:0.8 --sem-colour 1 --warmup 3 --steps 16
   run $T $S semc100_w3_ctx18 sem:1.0 --sem-colour 1 --warmup 3 --steps 16 --ctx 18 ;;
 semr_*) T=${1#semr_}; T=${T%_*}; S=${1##*_}; [ $T = jar ] && T=sand_filling_up_jar; [ $T = castle ] && T=sandcastle_wave_impact
   run $T $S semc100_w3_recent_retry1 sem:1.0 --sem-colour 1 --sem-ref recent --warmup 3 --steps 16 --sem-retry 1
   run $T $S semc100_w3_retry1 sem:1.0 --sem-colour 1 --warmup 3 --steps 16 --sem-retry 1
   run $T $S semc100_w3_ctx18_retry1 sem:1.0 --sem-colour 1 --warmup 3 --steps 16 --ctx 18 --sem-retry 1 ;;
 semr7_*) T=${1#semr7_}; T=${T%_*}; S=${1##*_}; [ $T = jar ] && T=sand_filling_up_jar; [ $T = castle ] && T=sandcastle_wave_impact
   run $T $S semc70_w3_recent_retry1 sem:0.7 --sem-colour 1 --sem-ref recent --warmup 3 --steps 16 --sem-retry 1
   run $T $S semc100_w3_recent_retry1 sem:1.0 --sem-colour 1 --sem-ref recent --warmup 3 --steps 16 --sem-retry 1 ;;
 exp1_lora) L="--base-lora 1022 --base-sampler euler --steps 4 --shift 5 --guide 1"; SEM="sem:1.0 --sem-colour 1 --sem-ref recent --warmup 3 --sem-retry 1"
   run sand_filling_up_jar 1234 _smoke_lora1022_always61 always --frames 61 $L || return 1
   for TS in sandcastle_wave_impact:1234 sand_filling_up_jar:1234 sandcastle_wave_impact:9999 sand_filling_up_jar:9999 sandcastle_wave_impact:4321 sand_filling_up_jar:4321; do run ${TS%%:*} ${TS##*:} lora1022_semc100 $SEM $L; done ;;
 exp2_kvrp) SEM="sem:1.0 --sem-colour 1 --sem-ref recent --warmup 3 --sem-retry 1 --steps 16"
   run sand_filling_up_jar 1234 _smoke_kvrph12_always61 always --frames 61 --steps 16 --corr-kv 1 || return 1
   for TS in sandcastle_wave_impact:1234 sand_filling_up_jar:1234 sandcastle_wave_impact:9999 sand_filling_up_jar:9999 sandcastle_wave_impact:4321 sand_filling_up_jar:4321; do run ${TS%%:*} ${TS##*:} kvrph12_semc100 $SEM --corr-kv 1 --kv-rp-win 12; done ;;
 exp4_hand) SEM="sem:1.0 --sem-colour 1 --sem-ref recent --warmup 3 --sem-retry 1 --steps 16"
   run sand_filling_up_jar 1234 _smoke_hand1_always61 always --frames 61 --steps 16 --handoff 1 || return 1
   for V in "hand1:--handoff 1" "hand2:--handoff 2" "hand1_cfg3:--handoff 1 --cfg-steps 3" "hand1_reuse24:--handoff 1 --flow-reuse 2,4"; do
     for TS in sandcastle_wave_impact:4321 sand_filling_up_jar:9999; do run ${TS%%:*} ${TS##*:} ${V%%:*}_semc100 $SEM ${V#*:}; done; done
   for TS in sandcastle_wave_impact:1234 sand_filling_up_jar:1234 sandcastle_wave_impact:9999 sand_filling_up_jar:4321; do run ${TS%%:*} ${TS##*:} hand1_semc100 $SEM --handoff 1; done ;;
 exp5_mixed) SEM="sem:1.0 --sem-colour 1 --sem-ref recent --warmup 3 --sem-retry 1 --steps 16"
   run sand_filling_up_jar 1234 _smoke_mixed46_always61 always --frames 61 --steps 16 --corr-mixed 4:6 || return 1
   for TS in sandcastle_wave_impact:4321 sand_filling_up_jar:9999 sandcastle_wave_impact:1234 sand_filling_up_jar:1234 sandcastle_wave_impact:9999 sand_filling_up_jar:4321; do run ${TS%%:*} ${TS##*:} mixed46_semc100 $SEM --corr-mixed 4:6; done
   for TS in sandcastle_wave_impact:4321 sand_filling_up_jar:9999; do run ${TS%%:*} ${TS##*:} mixed412_semc100 $SEM --corr-mixed 4:12; done ;;
 exp6_mixed2) SEM="sem:1.0 --sem-colour 1 --sem-ref recent --warmup 3 --sem-retry 1 --steps 16"
   for V in "mixed26:2:6" "mixed212:2:12"; do for TS in sand_filling_up_jar:9999 sandcastle_wave_impact:4321; do run ${TS%%:*} ${TS##*:} ${V%%:*}_semc100 $SEM --corr-mixed ${V#*:}; done; done ;;
 exp7_cfgreuse) SEM="sem:1.0 --sem-colour 1 --sem-ref recent --warmup 3 --sem-retry 1 --steps 16"
   for TS in sandcastle_wave_impact:4321 sand_filling_up_jar:9999 sand_filling_up_jar:1234 sandcastle_wave_impact:1234 sandcastle_wave_impact:9999 sand_filling_up_jar:4321; do run ${TS%%:*} ${TS##*:} cfg6reuse_semc100 $SEM --cfg-steps 6 --flow-reuse 9,11,13; done ;;
 exp8_cfg4) SEM="sem:1.0 --sem-colour 1 --sem-ref recent --warmup 3 --sem-retry 1 --steps 16"
   for TS in sand_filling_up_jar:9999 sandcastle_wave_impact:4321; do run ${TS%%:*} ${TS##*:} cfg4reuse_semc100 $SEM --cfg-steps 4 --flow-reuse 7,9,11,13; done ;;
 exp9_act) SEM="sem:1.0 --sem-colour 1 --sem-ref recent --warmup 3 --sem-retry 1 --steps 16 --cfg-steps 6 --flow-reuse 9,11,13"
   for TS in sandcastle_wave_impact:1234 sand_filling_up_jar:9999 sandcastle_wave_impact:4321 sand_filling_up_jar:1234 sandcastle_wave_impact:9999 sand_filling_up_jar:4321; do run ${TS%%:*} ${TS##*:} act90_cr_semc100 $SEM --act-theta 90; done
   for TS in sandcastle_wave_impact:1234 sand_filling_up_jar:9999 sandcastle_wave_impact:4321 sand_filling_up_jar:1234 sandcastle_wave_impact:9999 sand_filling_up_jar:4321; do run ${TS%%:*} ${TS##*:} act72_cr_semc100 $SEM --act-theta 72; done ;;
 exp10_guide) SEM="sem:1.0 --sem-colour 1 --sem-ref recent --warmup 3 --sem-retry 1 --steps 16 --cfg-steps 6 --flow-reuse 9,11,13"
   run sand_filling_up_jar 1234 _smoke_ag2_always61 always --frames 61 --steps 16 --cfg-steps 6 --autoguide 2 || return 1
   run sand_filling_up_jar 1234 _smoke_hg1_never61 never --frames 61 --histguide 1 || return 1
   for V in "ag2:--autoguide 2" "hg1:--histguide 1" "ag3:--autoguide 3" "hg05:--histguide 0.5"; do
     for TS in sandcastle_wave_impact:4321 sand_filling_up_jar:9999; do run ${TS%%:*} ${TS##*:} ${V%%:*}_cr_semc100 $SEM ${V#*:}; done; done ;;
 exp11_lora) SEM="sem:1.0 --sem-colour 1 --sem-ref recent --warmup 3 --sem-retry 1 --steps 16 --cfg-steps 6 --flow-reuse 9,11,13"; ML=/home/hk28965/specwm/lingbot-world-v2/specwm_out/mixlora/lora
   for TS in sandcastle_wave_impact:4321 sand_filling_up_jar:9999 sand_filling_up_jar:1234 sandcastle_wave_impact:1234 sandcastle_wave_impact:9999 sand_filling_up_jar:4321; do run ${TS%%:*} ${TS##*:} mixlora46_cr_semc100 $SEM --corr-mixed 4:6 --mixed-lora $ML; done ;;
 exp10b_ag) SEM="sem:1.0 --sem-colour 1 --sem-ref recent --warmup 3 --sem-retry 1 --steps 16 --cfg-steps 6 --flow-reuse 9,11,13"
   for V in "ag2:--autoguide 2" "ag3:--autoguide 3"; do for TS in sandcastle_wave_impact:4321 sand_filling_up_jar:9999 sand_filling_up_jar:1234 sandcastle_wave_impact:1234; do run ${TS%%:*} ${TS##*:} ${V%%:*}_cr_semc100 $SEM ${V#*:}; done; done ;;
 exp10c_ag) SEM="sem:1.0 --sem-colour 1 --sem-ref recent --warmup 3 --sem-retry 1 --steps 16 --cfg-steps 6 --flow-reuse 9,11,13"
   run sand_filling_up_jar 9999 ag2_cr_semc100 $SEM --autoguide 2
   for TS in sandcastle_wave_impact:4321 sand_filling_up_jar:9999; do run ${TS%%:*} ${TS##*:} ag15_cr_semc100 $SEM --autoguide 1.5; done ;;
 exp12_fd) SEM="sem:1.0 --sem-colour 1 --sem-ref recent --warmup 3 --sem-retry 1 --steps 16 --cfg-steps 6 --flow-reuse 9,11,13"; ML=/home/hk28965/specwm/lingbot-world-v2/specwm_out/mixlora/lora
   for TS in sandcastle_wave_impact:4321 sand_filling_up_jar:9999 sand_filling_up_jar:1234 sandcastle_wave_impact:1234 sandcastle_wave_impact:9999 sand_filling_up_jar:4321; do run ${TS%%:*} ${TS##*:} fd1_cr_semc100 $SEM --corr-from-draft 1; done
   for TS in sandcastle_wave_impact:4321 sand_filling_up_jar:9999; do run ${TS%%:*} ${TS##*:} fd2_cr_semc100 $SEM --corr-from-draft 2; done
   for TS in sandcastle_wave_impact:4321 sand_filling_up_jar:9999; do run ${TS%%:*} ${TS##*:} fd1_mixlora46_cr_semc100 $SEM --corr-from-draft 1 --corr-mixed 4:6 --mixed-lora $ML; done ;;
 exp13_always) for TS in sand_filling_up_jar:9999 sandcastle_wave_impact:4321 sand_filling_up_jar:1234 sandcastle_wave_impact:1234 sand_filling_up_jar:4321 sandcastle_wave_impact:9999; do run ${TS%%:*} ${TS##*:} always16 always --steps 16; done ;;
 exp14_polish) SEM="sem:1.0 --sem-colour 1 --sem-ref recent --warmup 3 --sem-retry 1 --steps 16 --cfg-steps 6 --flow-reuse 9,11,13"
   for TS in sand_filling_up_jar:4321 sand_filling_up_jar:9999 sandcastle_wave_impact:4321 sand_filling_up_jar:1234 sandcastle_wave_impact:1234 sandcastle_wave_impact:9999; do run ${TS%%:*} ${TS##*:} pol77_cr_semc100 $SEM --polish 0.77:24; done
   for TS in sand_filling_up_jar:4321 sandcastle_wave_impact:4321; do run ${TS%%:*} ${TS##*:} pol60_cr_semc100 $SEM --polish 0.6:32; done ;;
 exp15_seam) F16="--steps 16"
   run sand_filling_up_jar 1234 _smoke_seam61 always --frames 61 $F16 --ctx 18 --noise-corr 0.9 --seam 0.6:32 || return 1
   for TS in sand_filling_up_jar:9999 sandcastle_wave_impact:4321; do
     run ${TS%%:*} ${TS##*:} al_ctx18 always $F16 --ctx 18
     run ${TS%%:*} ${TS##*:} al_ctx18_nc always $F16 --ctx 18 --noise-corr 0.9
     run ${TS%%:*} ${TS##*:} al_ctx18_nc_seam always $F16 --ctx 18 --noise-corr 0.9 --seam 0.6:32
     run ${TS%%:*} ${TS##*:} al_nc_seam always $F16 --noise-corr 0.9 --seam 0.6:32
   done ;;
 exp16_bguide) run sand_filling_up_jar 1234 _smoke_bg1_never61 never --frames 61 --bguide 1.0 --bguide-nodes 0,1 || return 1
   for TS in sandcastle_wave_impact:4321 sand_filling_up_jar:9999; do
     run ${TS%%:*} ${TS##*:} bg1_n01 never --bguide 1.0 --bguide-nodes 0,1
     run ${TS%%:*} ${TS##*:} bg05_n0123 never --bguide 0.5 --bguide-nodes 0,1,2,3
     run ${TS%%:*} ${TS##*:} bg1_n0123 never --bguide 1.0 --bguide-nodes 0,1,2,3
     run ${TS%%:*} ${TS##*:} hg02 never --histguide 0.2
     run ${TS%%:*} ${TS##*:} hg04 never --histguide 0.4
   done ;;
 exp17_seamdbg) run sand_filling_up_jar 1234 _smoke_nc61 always --frames 61 --steps 16 --noise-corr 0.9
   run sand_filling_up_jar 1234 _smoke_seamonly61 always --frames 61 --steps 16 --seam 0.6:32
   run sand_filling_up_jar 1234 _smoke_ctx18only61 always --frames 61 --steps 16 --ctx 18 ;;
 data70_*) T=${1#data70_}; D=$V0/$T/base70_s1234; [ -f $D/video.mp4 ] && { echo "[skip] $D"; } || { echo "########## [$GPUS] BASE70 $T ($(date)) ##########"; mkdir -p $D
   IMG=$(python3 -c "import json;print([t for t in json.load(open('/home/hk28965/specwm/data/stevo/tasks_camera.json')) if t['id']=='$T'][0]['image'])"); ACT=$(python3 -c "import json;print([t for t in json.load(open('/home/hk28965/specwm/data/stevo/tasks_camera.json')) if t['id']=='$T'][0]['action'])")
   (cd lingbot-world; PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=${GPUS%%,*} python -m specwm.bench_base_official --ckpt $PWD/lingbot-world-base-cam --image $IMG --action-path $ACT --prompt-file /home/hk28965/specwm/exp/empty_prompt.txt --max-area 399360 --frames 161 --steps 70 --seed 1234 --out $D 2>&1 | grep -E "^\[BASE|Error|Traceback|OutOfMemory"); echo "[SPEC_V0_DONE] base70 $T"; } ;;
 evalfl_*) CK=${1#evalfl_}; FL=/home/hk28965/specwm/lingbot-world-v2/specwm_out/fastlora/$CK/fast_lora.pt
   for TS in sand_filling_up_jar:9999 sandcastle_wave_impact:4321 sand_filling_up_jar:1234 sandcastle_wave_impact:1234 sand_filling_up_jar:4321 sandcastle_wave_impact:9999; do run ${TS%%:*} ${TS##*:} fl_${CK}_never never --fast-lora $FL; done ;;
 fastroll_*) T=${1#fastroll_}; run $T 1234 never never ;;
 onpol_*) T=${1#onpol_}; echo "########## [$GPUS] ONPOLICY $T ($(date)) ##########"; (cd lingbot-world-v2; PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=${GPUS%%,*} python -m specwm.onpolicy_gen --out /home/hk28965/specwm/lingbot-world-v2/specwm_out/onpolicy --tasks $T 2>&1 | grep -E "onpolicy|Error|Traceback|OutOfMemory"); echo "[SPEC_V0_DONE] onpolicy $T" ;;
 fullkv) for TS in sand_filling_up_jar:9999 sandcastle_wave_impact:4321 sand_filling_up_jar:1234 sandcastle_wave_impact:1234 sand_filling_up_jar:4321 sandcastle_wave_impact:9999; do run ${TS%%:*} ${TS##*:} never_win41 never --fast-win 41; done ;;
 evalhl_*) CK=${1#evalhl_}; HLP=/home/hk28965/specwm/lingbot-world-v2/specwm_out/hintlora/$CK/hint_lora.pt
   for TS in sand_filling_up_jar:9999 sandcastle_wave_impact:4321 sand_filling_up_jar:1234 sandcastle_wave_impact:1234 sand_filling_up_jar:4321 sandcastle_wave_impact:9999; do run ${TS%%:*} ${TS##*:} hl_${CK}_never never --hint-lora $HLP; done ;;
 plan_*) X=${1#plan_}; T=${X%%:*}; R=${X#*:}; SD=${R%%:*}; A=${R##*:}; D=$V0/$T/base16_a${A}_s$SD; [ -f $D/video.mp4 ] && { echo "[skip] $D"; } || { echo "########## [$GPUS] BASE16 low-res plan $T s$SD area $A ($(date)) ##########"; mkdir -p $D
   IMG=$(python3 -c "import json;print([t for t in json.load(open('/home/hk28965/specwm/data/stevo/tasks_camera.json')) if t['id']=='$T'][0]['image'])"); ACT=$(python3 -c "import json;print([t for t in json.load(open('/home/hk28965/specwm/data/stevo/tasks_camera.json')) if t['id']=='$T'][0]['action'])")
   (cd lingbot-world; PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=${GPUS%%,*} python -m specwm.bench_base_official --ckpt $PWD/lingbot-world-base-cam --image $IMG --action-path $ACT --prompt-file /home/hk28965/specwm/exp/empty_prompt.txt --max-area $A --frames 161 --steps 16 --seed $SD --out $D 2>&1 | grep -E "^\[BASE|Error|Traceback|OutOfMemory"); echo "[SPEC_V0_DONE] plan $T s$SD a$A"; } ;;
 evalplan_*) CK=${1#evalplan_}; HLP=/home/hk28965/specwm/lingbot-world-v2/specwm_out/hintlora/$CK/hint_lora.pt
   for TS in sand_filling_up_jar:9999 sandcastle_wave_impact:4321 sand_filling_up_jar:1234 sandcastle_wave_impact:1234 sand_filling_up_jar:4321 sandcastle_wave_impact:9999; do T=${TS%%:*}; SD=${TS##*:}; PV=$V0/$T/base16_a184320_s$SD/video.mp4; [ -f $PV ] && run $T $SD pl_${CK}_never never --hint-lora $HLP --plan-video $PV; done ;;
 evalplg_*) X=${1#evalplg_}; CK=${X%%:*}; TAG=${X##*:}; HLP=/home/hk28965/specwm/lingbot-world-v2/specwm_out/hintlora/$CK/hint_lora.pt
   for TS in sand_filling_up_jar:1234 sandcastle_wave_impact:1234 sand_filling_up_jar:4321 sandcastle_wave_impact:4321 sand_filling_up_jar:9999 sandcastle_wave_impact:9999; do T=${TS%%:*}; SD=${TS##*:}; PV=$V0/$T/plan_${TAG}_s$SD/video.mp4; [ -f $PV ] && run $T $SD pl_${CK}_${TAG}_never never --hint-lora $HLP --plan-video $PV; done ;;
 evalfix_*) X=${1#evalfix_}; CK=${X%%:*}; R=${X#*:}; ST=${R%%:*}; TAG=${R##*:}; HLP=/home/hk28965/specwm/lingbot-world-v2/specwm_out/hintlora/$CK/hint_lora.pt
   for TS in sand_filling_up_jar:1234 sandcastle_wave_impact:1234 sand_filling_up_jar:4321 sandcastle_wave_impact:4321 sand_filling_up_jar:9999 sandcastle_wave_impact:9999; do T=${TS%%:*}; SD=${TS##*:}
     if [ $TAG = b16 ]; then PV=$V0/$T/base16_a184320_s$SD/video.mp4; else PV=$V0/$T/plan_${TAG}_s$SD/video.mp4; fi
     [ -f $PV ] && run $T $SD plf_${CK}_${TAG}_st${ST}_never never --hint-lora $HLP --plan-video $PV --start $ST --no-base 1; done ;;
 evallo_*) X=${1#evallo_}; CK=${X%%:*}; R=${X#*:}; ST=${R%%:*}; TAG=${R##*:}; HLP=/home/hk28965/specwm/lingbot-world-v2/specwm_out/hintlora/$CK/hint_lora.pt
   for TS in sand_filling_up_jar:1234 sandcastle_wave_impact:4321 sand_filling_up_jar:4321 sandcastle_wave_impact:1234 sand_filling_up_jar:9999 sandcastle_wave_impact:9999; do T=${TS%%:*}; SD=${TS##*:}
     if [ $TAG = b16 ]; then PV=$V0/$T/base16_a184320_s$SD/video.mp4; else PV=$V0/$T/plan_${TAG}_s$SD/video.mp4; fi
     [ -f $PV ] && run $T $SD pll_${CK}_${TAG}_st${ST}_never never --hint-lora $HLP --plan-video $PV --start $ST --no-base 1 --plan-hint-mode lo; done ;;
 evalhlp_*) X=${1#evalhlp_}; CK=${X%%:*}; R=${X#*:}; ST=${R%%:*}; TAG=${R##*:}; HLP=/home/hk28965/specwm/lingbot-world-v2/specwm_out/hintlora/$CK/hint_lora.pt
   for TS in sand_filling_up_jar:1234 sandcastle_wave_impact:4321 sand_filling_up_jar:4321 sandcastle_wave_impact:1234 sand_filling_up_jar:9999 sandcastle_wave_impact:9999; do T=${TS%%:*}; SD=${TS##*:}
     if [ $TAG = b16 ]; then PV=$V0/$T/base16_a184320_s$SD/video.mp4; else PV=$V0/$T/plan_${TAG}_s$SD/video.mp4; fi
     [ -f $PV ] && run $T $SD plh_${CK}_${TAG}_st${ST}_never never --hint-lora $HLP --plan-video $PV --start $ST --no-base 1 --hint-low plain; done ;;
 evaltex_*) X=${1#evaltex_}; CK=${X%%:*}; R=${X#*:}; TX=${R%%:*}; TAG=${R##*:}; HLP=/home/hk28965/specwm/lingbot-world-v2/specwm_out/hintlora/$CK/hint_lora.pt; TXP=/home/hk28965/specwm/lingbot-world-v2/specwm_out/hintlora/$TX/tex_lora.pt
   for TS in sand_filling_up_jar:1234 sandcastle_wave_impact:4321 sand_filling_up_jar:4321 sandcastle_wave_impact:1234 sand_filling_up_jar:9999 sandcastle_wave_impact:9999; do T=${TS%%:*}; SD=${TS##*:}
     if [ $TAG = b16 ]; then PV=$V0/$T/base16_a184320_s$SD/video.mp4; else PV=$V0/$T/plan_${TAG}_s$SD/video.mp4; fi
     [ -f $PV ] && run $T $SD plt_${CK}_${TX}_${TAG}_st0_never never --hint-lora $HLP --tex-lora $TXP --plan-video $PV --start 0 --no-base 1 --hint-low plain; done ;;
 evalkv_*) X=${1#evalkv_}; CK=${X%%:*}; TAG=${X##*:}; KVP=/home/hk28965/specwm/lingbot-world-v2/specwm_out/hintlora/$CK/kv_lora.pt
   for TS in sand_filling_up_jar:1234 sandcastle_wave_impact:4321 sand_filling_up_jar:4321 sandcastle_wave_impact:1234 sand_filling_up_jar:9999 sandcastle_wave_impact:9999; do T=${TS%%:*}; SD=${TS##*:}; PL=$V0/$T/plan_${TAG}_s$SD/plan_lat.pt
     [ -f $PL ] && run $T $SD plk_${CK}_${TAG}_never never --kv-mem $KVP --plan-lat $PL --start 0 --resident; done ;;
 evalkvl_*) X=${1#evalkvl_}; CK=${X%%:*}; TAG=${X##*:}; KVP=/home/hk28965/specwm/lingbot-world-v2/specwm_out/hintlora/$CK/kv_lora.pt
   for TS in sand_filling_up_jar:1234 sandcastle_wave_impact:4321 sand_filling_up_jar:4321 sandcastle_wave_impact:1234 sand_filling_up_jar:9999 sandcastle_wave_impact:9999; do T=${TS%%:*}; SD=${TS##*:}; PL=$V0/$T/plan_${TAG}_s$SD/plan_lat.pt
     [ -f $PL ] && run $T $SD plkl_${CK}_${TAG}_never never --kv-mem $KVP --plan-lat $PL --start 0 --resident --kv-low plain; done ;;
 evalkvy_*) X=${1#evalkvy_}; CK=${X%%:*}; TAG=${X##*:}; KVP=/home/hk28965/specwm/lingbot-world-v2/specwm_out/hintlora/$CK/kv_lora.pt
   for TS in sand_filling_up_jar:1234 sandcastle_wave_impact:4321 sand_filling_up_jar:4321 sandcastle_wave_impact:1234 sand_filling_up_jar:9999 sandcastle_wave_impact:9999; do T=${TS%%:*}; SD=${TS##*:}; PL=$V0/$T/plan_${TAG}_s$SD/plan_lat.pt; PV=$V0/$T/plan_${TAG}_s$SD/video.mp4
     [ -f $PL ] && run $T $SD plky_${CK}_${TAG}_never never --hint-lora $KVP --plan-video $PV --hint-low plain --kv-mem $KVP --plan-lat $PL --kv-low plain --start 0 --resident; done ;;
 evaltexs_*) X=${1#evaltexs_}; CK=${X%%:*}; R=${X#*:}; TX=${R%%:*}; R2=${R#*:}; TAG=${R2%%:*}; SC=${R2##*:}; HLP=/home/hk28965/specwm/lingbot-world-v2/specwm_out/hintlora/$CK/hint_lora.pt; TXP=/home/hk28965/specwm/lingbot-world-v2/specwm_out/hintlora/$TX/tex_lora.pt
   for TS in sand_filling_up_jar:1234 sandcastle_wave_impact:4321 sand_filling_up_jar:4321 sandcastle_wave_impact:1234 sand_filling_up_jar:9999 sandcastle_wave_impact:9999; do T=${TS%%:*}; SD=${TS##*:}; PV=$V0/$T/plan_${TAG}_s$SD/video.mp4
     [ -f $PV ] && run $T $SD plts_${CK}_${TX}_${TAG}_x${SC}_st0_never never --hint-lora $HLP --tex-lora $TXP --tex-scale $SC --plan-video $PV --start 0 --no-base 1 --hint-low plain; done ;;
 renderpl_*) X=${1#renderpl_}; T=${X%%:*}; R=${X#*:}; SD=${R%%:*}; TAG=${R##*:}; HLP=/home/hk28965/specwm/lingbot-world-v2/specwm_out/hintlora/plan6_ck400/hint_lora.pt; PV=$V0/$T/plan_${TAG}_s$SD/video.mp4
   [ -f $PV ] && run $T $SD plh_plan6_ck400_${TAG}_st0_never never --hint-lora $HLP --plan-video $PV --start 0 --no-base 1 --hint-low plain || echo "[missing plan] $PV" ;;
 fastonly_*) X=${1#fastonly_}; T=${X%%:*}; SD=${X##*:}; run $T $SD never never --no-base 1 ;;
 data70p_*) T=${1#data70p_}; D=$V0/$T/base70_s1234; [ -f $D/video.mp4 ] && { echo "[skip] $D"; } || { echo "########## [$GPUS] BASE70 (prompted) $T ($(date)) ##########"; mkdir -p $D
   IMG=$(python3 -c "import json;print([t for t in json.load(open('/home/hk28965/specwm/data/stevo/tasks_camera.json')) if t['id']=='$T'][0]['image'])"); ACT=$(python3 -c "import json;print([t for t in json.load(open('/home/hk28965/specwm/data/stevo/tasks_camera.json')) if t['id']=='$T'][0]['action'])")
   (cd lingbot-world; PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=${GPUS%%,*} python -m specwm.bench_base_official --ckpt $PWD/lingbot-world-base-cam --image $IMG --action-path $ACT --prompt-file /home/hk28965/specwm/exp/prompt_$T.txt --max-area 399360 --frames 161 --steps 70 --seed 1234 --out $D 2>&1 | grep -E "^\[BASE|Error|Traceback|OutOfMemory"); echo "[SPEC_V0_DONE] base70 $T"; } ;;
 specstep_*) X=${1#specstep_}; T=${X%%:*}; R=${X#*:}; SD=${R%%:*}; NS=${R##*:}; run $T $SD specstep_n${NS//,/}c9_never never --specstep 1 --spec-nodes $NS --resident --ctx 9 ;;
 base70s_*) X=${1#base70s_}; T=${X%%:*}; SD=${X##*:}; D=$V0/$T/base70_s$SD; [ -f $D/video.mp4 ] && { echo "[skip] $D"; } || { echo "########## [$GPUS] BASE70 $T s$SD ($(date)) ##########"; mkdir -p $D
   IMG=$(python3 -c "import json;print([t for t in json.load(open('/home/hk28965/specwm/data/stevo/tasks_camera.json')) if t['id']=='$T'][0]['image'])"); ACT=$(python3 -c "import json;print([t for t in json.load(open('/home/hk28965/specwm/data/stevo/tasks_camera.json')) if t['id']=='$T'][0]['action'])")
   (cd lingbot-world; PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=${GPUS%%,*} python -m specwm.bench_base_official --ckpt $PWD/lingbot-world-base-cam --image $IMG --action-path $ACT --prompt-file /home/hk28965/specwm/exp/empty_prompt.txt --max-area 399360 --frames 161 --steps 70 --seed $SD --out $D 2>&1 | grep -E "^\[BASE|Error|Traceback|OutOfMemory"); echo "[SPEC_V0_DONE] base70 $T s$SD"; } ;;
 nullarm_*) X=${1#nullarm_}; T=${X%%:*}; SD=${X##*:}; run $T $SD tgtonly_n012c9_never never --bguide 1 --bguide-nodes 0,1,2 --ctx 9 --resident ;;
 specstepcfg_*) X=${1#specstepcfg_}; T=${X%%:*}; R=${X#*:}; SD=${R%%:*}; G=${R##*:}; run $T $SD specstep_n012c9g${G//./p}_never never --specstep 1 --spec-nodes 0,1,2 --resident --ctx 9 --bguide-cfg $G ;;
 nullarmcfg_*) X=${1#nullarmcfg_}; T=${X%%:*}; R=${X#*:}; SD=${R%%:*}; G=${R##*:}; run $T $SD tgtonly_n012c9g${G//./p}_never never --bguide 1 --bguide-nodes 0,1,2 --ctx 9 --resident --bguide-cfg $G ;;
 plang_*) X=${1#plang_}; T=${X%%:*}; R=${X#*:}; SD=${R%%:*}; TAG=${R##*:}; D=$V0/$T/plan_${TAG}_s$SD; case $TAG in cr) EX="--cfg-steps 6 --reuse 9,11,13";; c8) EX="--cfg-steps 8 --reuse 9,11,13";; c24) EX="--steps 24 --cfg-steps 8 --reuse 14,17,20";; g15) EX="--steps 4 --cfg-steps 2 --guide 1.5 --shift 5 --lora 1022 --lora-scale 1.0";; g2s1) EX="--steps 4 --cfg-steps 1 --guide 2 --shift 5 --lora 1022 --lora-scale 1.0";; l4) EX="--steps 4 --cfg-steps 0 --shift 5 --lora 1022";; sk4) EX="--steps 4 --cfg-steps 0 --shift 5 --lora seko";; l6) EX="--steps 6 --cfg-steps 0 --shift 5 --lora 1022";; pa_*) EX="--steps 4 --cfg-steps 0 --shift 5 --lora 1022 --planlora /home/hk28965/specwm/lingbot-world-v2/specwm_out/hintlora/${TAG#pa_}/plan_lora.pt";; crw) EX="--steps 16 --cfg-steps 6 --reuse 9,11,13 --window 81";; l4w) EX="--steps 4 --cfg-steps 0 --shift 5 --lora 1022 --window 81";; l4c) EX="--steps 6 --cfg-steps 2 --guide 3 --shift 5 --lora 1022 --lora-scale 0.8";; l8c) EX="--steps 8 --cfg-steps 3 --guide 3 --shift 5 --lora 1022 --lora-scale 0.7";; *) EX="";; esac
   [ -f $D/video.mp4 ] && { echo "[skip] $D"; } || { echo "########## [$GPUS] PLAN $T s$SD $TAG ($(date)) ##########"; mkdir -p $D; (cd lingbot-world-v2; PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=${GPUS%%,*} python -m specwm.plan_gen --task $T --seed $SD --out $D --area 184320 $(case $TAG in c24|l4|sk4|l6|pa_*|crw|l4w|l4c|l8c) ;; *) echo --steps 16;; esac) $EX 2>&1 | grep -E "plan\]|Error|Traceback|OutOfMemory"); echo "[SPEC_V0_DONE] plan $T s$SD $TAG"; } ;;
 pqcal_*) X=${1#pqcal_}; T=${X%%:*}; R=${X#*:}; SD=${R%%:*}; ARMV=${R#*:}   # pq verifier calibration: score an existing clip chunk by chunk (teacher-forced)
   run $T $SD pqcal_$ARMV pq --pq-teacher $V0/$T/${ARMV}_s$SD/video.mp4 ;;
 pqgen_*) X=${1#pqgen_}; T=${X%%:*}; R=${X#*:}; SD=${R%%:*}; R=${R#*:}; TAU=${R%%:*}; R=${R#*:}; NR=${R%%:*}; FB=${R#*:}   # draft -> p/q verify -> Fast redraw (smooth)
   run $T $SD pq_t${TAU//[.-]/}_r${NR}_$FB pq --pq-tau $TAU --pq-retries $NR --pq-fallback $FB ;;
 base70v2_*) X=${1#base70v2_}; T=${X%%:*}; SD=${X##*:}; D=$V0/$T/base70v2_s$SD; [ -f $D/video.mp4 ] && { echo "[skip] $D"; } || { echo "########## [$GPUS] BASE70v2 $T s$SD ($(date)) ##########"; mkdir -p $D   # LingBot v2 14B bid, 70-step reference (v1 camera pipeline)
   IMG=$(python3 -c "import json;print([t for t in json.load(open('/home/hk28965/specwm/data/stevo/tasks_camera.json')) if t['id']=='$T'][0]['image'])"); ACT=$(python3 -c "import json;print([t for t in json.load(open('/home/hk28965/specwm/data/stevo/tasks_camera.json')) if t['id']=='$T'][0]['action'])")
   (cd lingbot-world; PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=${GPUS%%,*} python -m specwm.bench_base_official --ckpt /home/hk28965/specwm/lingbot-world-v2/bid_as_v1_cam --image $IMG --action-path $ACT --prompt-file /home/hk28965/specwm/exp/empty_prompt.txt --max-area 399360 --frames 161 --steps 70 --seed $SD --out $D 2>&1 | grep -E "^\[BASE|Error|Traceback|OutOfMemory"); } ;;
 plangv2_*) X=${1#plangv2_}; T=${X%%:*}; R=${X#*:}; SD=${R%%:*}; TAG=${R##*:}; D=$V0/$T/planv2_${TAG}_s$SD; case $TAG in cr) EX="--cfg-steps 6 --reuse 9,11,13";; l4c) EX="--steps 6 --cfg-steps 2 --guide 3 --shift 5 --lora 1022 --lora-scale 0.8";; esac   # low-res whole-clip plan by LingBot v2 14B bid
   [ -f $D/video.mp4 ] && { echo "[skip] $D"; } || { echo "########## [$GPUS] PLANv2 $T s$SD $TAG ($(date)) ##########"; mkdir -p $D; (cd lingbot-world-v2; SPECWM_BASE_CKPT=/home/hk28965/specwm/lingbot-world-v2/bid_as_v1_cam PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=${GPUS%%,*} python -m specwm.plan_gen --task $T --seed $SD --out $D --area 184320 $(case $TAG in l4c) ;; *) echo --steps 16;; esac) $EX 2>&1 | grep -E "plan\]|Error|Traceback|OutOfMemory"); } ;;
 pqplanb_*) X=${1#pqplanb_}; T=${X%%:*}; R=${X#*:}; SD=${R%%:*}; R=${R#*:}; TAU=${R%%:*}; NR=${R##*:}; HLP=/home/hk28965/specwm/lingbot-world-v2/specwm_out/hintlora/plan6_ck400/hint_lora.pt; PV=$V0/$T/planv2_l4c_s$SD/video.mp4   # ALL-v2: v2-bid plan + Fast render + p/q verify (v2-bid target) + Fast redraw
   [ -f $PV ] && run $T $SD pqplanv2_t${TAU//[.-]/}_r$NR pq --pq-tau $TAU --pq-retries $NR --pq-fallback best --hint-lora $HLP --plan-video $PV --start 0 --hint-low plain --base-ckpt /home/hk28965/specwm/lingbot-world-v2/bid_as_v1_cam || echo "[missing plan] $PV" ;;
 planrc_*) X=${1#planrc_}; T=${X%%:*}; R=${X#*:}; SD=${R%%:*}; K=${R##*:}; HLP=/home/hk28965/specwm/lingbot-world-v2/specwm_out/hintlora/plan6_ck400/hint_lora.pt   # plan-level V/R: candidate plan K (v2 bid, l4c, noise seed SD+1000K; K=0 = the standard planv2_l4c) + its Fast render, one GPU
   if [ "$K" = 0 ]; then PD=$V0/$T/planv2_l4c_s$SD; RN=plhv2_plan6_ck400_l4c_st0_never; else PD=$V0/$T/planv2c${K}_l4c_s$SD; RN=plhv2c${K}_plan6_ck400_l4c_st0_never; fi
   [ -f $PD/video.mp4 ] && { echo "[skip plan] $PD"; } || { echo "########## [$GPUS] PLANv2 cand $K $T s$SD ($(date)) ##########"; mkdir -p $PD; (cd lingbot-world-v2; SPECWM_BASE_CKPT=/home/hk28965/specwm/lingbot-world-v2/bid_as_v1_cam PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=${GPUS%%,*} python -m specwm.plan_gen --task $T --seed $((SD+1000*K)) --out $PD --area 184320 --steps 6 --cfg-steps 2 --guide 3 --shift 5 --lora 1022 --lora-scale 0.8 2>&1 | grep -E "plan\]|Error|Traceback|OutOfMemory"); }
   [ -f $PD/video.mp4 ] && run $T $SD $RN never --hint-lora $HLP --plan-video $PD/video.mp4 --start 0 --no-base 1 --hint-low plain || echo "[missing plan] $PD" ;;
 plancr_*) X=${1#plancr_}; T=${X%%:*}; SD=${X##*:}; HLP=/home/hk28965/specwm/lingbot-world-v2/specwm_out/hintlora/plan6_ck400/hint_lora.pt; PD=$V0/$T/planv2_cr_s$SD   # v2-bid SAFE planner (16 steps, CFG 6 + flow reuse) + Fast render, one GPU
   [ -f $PD/video.mp4 ] && { echo "[skip plan] $PD"; } || { echo "########## [$GPUS] PLANv2 cr $T s$SD ($(date)) ##########"; mkdir -p $PD; (cd lingbot-world-v2; SPECWM_BASE_CKPT=/home/hk28965/specwm/lingbot-world-v2/bid_as_v1_cam PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=${GPUS%%,*} python -m specwm.plan_gen --task $T --seed $SD --out $PD --area 184320 --steps 16 --cfg-steps 6 --reuse 9,11,13 2>&1 | grep -E "plan\]|Error|Traceback|OutOfMemory"); }
   [ -f $PD/video.mp4 ] && run $T $SD plhv2_plan6_ck400_cr_st0_never never --hint-lora $HLP --plan-video $PD/video.mp4 --start 0 --no-base 1 --hint-low plain || echo "[missing plan] $PD" ;;
 planvar_*) X=${1#planvar_}; T=${X%%:*}; R=${X#*:}; SD=${R%%:*}; TAG=${R##*:}; HLP=/home/hk28965/specwm/lingbot-world-v2/specwm_out/hintlora/plan6_ck400/hint_lora.pt; PD=$V0/$T/planv2_${TAG}_s$SD   # v2-bid planner variants + Fast render, one GPU
   case $TAG in l4) EX="--steps 4 --cfg-steps 0 --shift 5 --lora 1022 --lora-scale 1.0";; l4c8) EX="--steps 8 --cfg-steps 3 --guide 3 --shift 5 --lora 1022 --lora-scale 0.8";; c24) EX="--steps 24 --cfg-steps 8 --reuse 14,17,20";; g15) EX="--steps 4 --cfg-steps 2 --guide 1.5 --shift 5 --lora 1022 --lora-scale 1.0";; g2s1) EX="--steps 4 --cfg-steps 1 --guide 2 --shift 5 --lora 1022 --lora-scale 1.0";; *) echo "[bad tag] $TAG"; return 1;; esac
   [ -f $PD/video.mp4 ] && { echo "[skip plan] $PD"; } || { echo "########## [$GPUS] PLANv2 $TAG $T s$SD ($(date)) ##########"; mkdir -p $PD; (cd lingbot-world-v2; SPECWM_BASE_CKPT=/home/hk28965/specwm/lingbot-world-v2/bid_as_v1_cam PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=${GPUS%%,*} python -m specwm.plan_gen --task $T --seed $SD --out $PD --area 184320 $EX 2>&1 | grep -E "plan\]|Error|Traceback|OutOfMemory"); }
   [ -f $PD/video.mp4 ] && run $T $SD plhv2_plan6_ck400_${TAG}_st0_never never --hint-lora $HLP --plan-video $PD/video.mp4 --start 0 --no-base 1 --hint-low plain || echo "[missing plan] $PD" ;;
 rfast_*) X=${1#rfast_}; T=${X%%:*}; R=${X#*:}; SD=${R%%:*}; VAR=${R##*:}; HLP=/home/hk28965/specwm/lingbot-world-v2/specwm_out/hintlora/plan6_ck400/hint_lora.pt; PV=$V0/$T/planv2_l4_s$SD/video.mp4   # faster render of the l4 plan: fewer Fast nodes or lower render res
   case $VAR in n023) EX="--nodes 0,2,3";; n03) EX="--nodes 0,3";; a576) EX="--max-area 184320";; a448) EX="--max-area 114688";; esac
   [ -f $PV ] && run $T $SD plhv2_plan6_ck400_l4_${VAR}_never never --hint-lora $HLP --plan-video $PV --start 0 --no-base 1 --hint-low plain $EX || echo "[missing plan] $PV" ;;
 planlo_*) X=${1#planlo_}; T=${X%%:*}; R=${X#*:}; SD=${R%%:*}; AR=${R##*:}; HLP=/home/hk28965/specwm/lingbot-world-v2/specwm_out/hintlora/plan6_ck400/hint_lora.pt; PD=$V0/$T/planv2_l4a${AR}_s$SD   # l4 plan at a lower area + standard Fast render
   [ -f $PD/video.mp4 ] && { echo "[skip plan] $PD"; } || { echo "########## [$GPUS] PLANv2 l4 area $AR $T s$SD ($(date)) ##########"; mkdir -p $PD; (cd lingbot-world-v2; SPECWM_BASE_CKPT=/home/hk28965/specwm/lingbot-world-v2/bid_as_v1_cam PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=${GPUS%%,*} python -m specwm.plan_gen --task $T --seed $SD --out $PD --area $AR --steps 4 --cfg-steps 0 --shift 5 --lora 1022 --lora-scale 1.0 2>&1 | grep -E "plan\]|Error|Traceback|OutOfMemory"); }
   [ -f $PD/video.mp4 ] && run $T $SD plhv2_plan6_ck400_l4a${AR}_st0_never never --hint-lora $HLP --plan-video $PD/video.mp4 --start 0 --no-base 1 --hint-low plain || echo "[missing plan] $PD" ;;
 fast1p3b_*) X=${1#fast1p3b_}; T=${X%%:*}; SD=${X##*:}   # LingBot v2 1.3B causal-fast alone (action-conditioned AR world model), same actions/prompt/seed as the 14B 'never' runs
   run $T $SD never1p3b never --no-base 1 --fast-ckpt /home/hk28965/specwm/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast ;;
 shadow_*) X=${1#shadow_}; T=${X%%:*}; SD=${X##*:}; D=/home/hk28965/specwm/lingbot-world-v2/specwm_out/shadow/${T}_s$SD   # latent-verifier shadow data: 1.3B draft + 14B target chunks from the same context/noise, one GPU
   [ -f $D/shadow.pt ] && { echo "[skip] $D"; } || { echo "########## [$GPUS] SHADOW $T s$SD ($(date)) ##########"; mkdir -p $D; (cd lingbot-world-v2; PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=${GPUS%%,*} python -m specwm.spec_ar_shadow --task $T --seed $SD --out $D 2>&1 | grep -E "SHADOW_DONE|Error|Traceback|OutOfMemory|line [0-9]+"); } ;;
 loop_*) X=${1#loop_}; TASKS=${X%%:*}; R=${X#*:}; SEEDS=${R%%:*}; VERS=${R#*:}; ROOT=/home/hk28965/specwm/lingbot-world-v2/specwm_out/loop   # speculative AR loop sweep in ONE process: loop_<tasks csv|@file>:<seeds csv>:<verifiers csv, ':' written as '~'>
   echo "########## [$GPUS] LOOP tasks=$TASKS seeds=$SEEDS verifiers=$VERS ($(date)) ##########"; (cd lingbot-world-v2; PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=${GPUS%%,*} python -m specwm.spec_ar_loop --tasks "$TASKS" --seeds "$SEEDS" --verifiers "$VERS" --out $ROOT 2>&1 | grep -E "LOOP_DONE|LOOP_FAIL|Error|Traceback|OutOfMemory|line [0-9]+") ;;
 critv_*) X=${1#critv_}; T=${X%%:*}; R=${X#*:}; SD=${R%%:*}; TAU=${R##*:}; HLP=/home/hk28965/specwm/lingbot-world-v2/specwm_out/hintlora/plan6_ck400/hint_lora.pt; PV=$V0/$T/planv2_l4_s$SD/video.mp4; CP=/home/hk28965/specwm/lingbot-world-v2/specwm_out/drift/noprev/critic.pt   # plan(l4) + Fast render + DRIFT-CRITIC chunk verify + Fast redraw (no Base)
   [ -f $PV ] && run $T $SD plhv2_l4_critic_t${TAU//[.-]/}_r3 pq --pq-verifier critic:$CP --pq-tau $TAU --pq-retries 3 --pq-fallback best --hint-lora $HLP --plan-video $PV --start 0 --no-base 1 --hint-low plain || echo "[missing plan] $PV" ;;
 v2render_*) X=${1#v2render_}; T=${X%%:*}; SD=${X##*:}; HLP=/home/hk28965/specwm/lingbot-world-v2/specwm_out/hintlora/plan6_ck400/hint_lora.pt; PV=$V0/$T/planv2_l4c_s$SD/video.mp4   # v2-fast-CFG with the v2-bid plan, no verification
   [ -f $PV ] && run $T $SD plhv2_plan6_ck400_l4c_st0_never never --hint-lora $HLP --plan-video $PV --start 0 --no-base 1 --hint-low plain || echo "[missing plan] $PV" ;;
 pqplan_*) X=${1#pqplan_}; T=${X%%:*}; R=${X#*:}; SD=${R%%:*}; R=${R#*:}; TAU=${R%%:*}; NR=${R##*:}; HLP=/home/hk28965/specwm/lingbot-world-v2/specwm_out/hintlora/plan6_ck400/hint_lora.pt; PV=$V0/$T/plan_l4c_s$SD/video.mp4   # v2-fast-CFG render as draft + p/q verify + plan-anchored Fast redraw
   [ -f $PV ] && run $T $SD pqplan_t${TAU//[.-]/}_r$NR pq --pq-tau $TAU --pq-retries $NR --pq-fallback best --hint-lora $HLP --plan-video $PV --start 0 --hint-low plain || echo "[missing plan] $PV" ;;
 cmask_*) X=${1#cmask_}; T=${X%%:*}; R=${X#*:}; SD=${R%%:*}; CM=${R##*:}   # best sem pipeline, Base correction sees the prefix CLEAN through its y/mask channel (mask) or y/mask + RePaint (both)
   run $T $SD cfg6reuse_semc100_cm$CM sem:1.0 --sem-colour 1 --sem-ref recent --warmup 3 --sem-retry 1 --steps 16 --cfg-steps 6 --flow-reuse 9,11,13 --corr-mode $CM ;;
 semb_*) X=${1#semb_}; T=${X%%:*}; SD=${X##*:}   # best draft->verify->Base-correction pipeline (replace), corrector = LingBot v2 14B bid (Fast's own teacher)
   run $T $SD cfg6reuse_semc100_v2bid sem:1.0 --sem-colour 1 --sem-ref recent --warmup 3 --sem-retry 1 --steps 16 --cfg-steps 6 --flow-reuse 9,11,13 --base-ckpt /home/hk28965/specwm/lingbot-world-v2/bid_as_v1_cam ;;
 pqcalb_*) X=${1#pqcalb_}; T=${X%%:*}; R=${X#*:}; SD=${R%%:*}; ARMV=${R#*:}   # same as pqcal_ but the target is LingBot v2 14B bid (Fast's own teacher)
   run $T $SD pqcalb_$ARMV pq --pq-teacher $V0/$T/${ARMV}_s$SD/video.mp4 --base-ckpt /home/hk28965/specwm/lingbot-world-v2/bid_as_v1_cam ;;
 pqgenb_*) X=${1#pqgenb_}; T=${X%%:*}; R=${X#*:}; SD=${R%%:*}; R=${R#*:}; TAU=${R%%:*}; R=${R#*:}; NR=${R%%:*}; FB=${R#*:}   # pq generation with the v2 bid target
   run $T $SD pqb_t${TAU//[.-]/}_r${NR}_$FB pq --pq-tau $TAU --pq-retries $NR --pq-fallback $FB --base-ckpt /home/hk28965/specwm/lingbot-world-v2/bid_as_v1_cam ;;
 blend_*) X=${1#blend_}; T=${X%%:*}; R=${X#*:}; SD=${R%%:*}; BL=${R#*:}   # Guandao 9/28 frame-jump fix on the best sem pipeline (baseline = cfg6reuse_semc100): BL = soft:A | low:K
   run $T $SD cfg6reuse_semc100_bl${BL//[:.]/} sem:1.0 --sem-colour 1 --sem-ref recent --warmup 3 --sem-retry 1 --steps 16 --cfg-steps 6 --flow-reuse 9,11,13 --corr-blend $BL ;;
 bandprobe_*) X=${1#bandprobe_}; T=${X%%:*}; R=${X#*:}; SD=${R%%:*}; G=${R##*:}   # direction 2: frequency split of the draft-target gap inside the exact sampler (logging only)
   run $T $SD bandprobe_c9g${G//./p}_never never --specstep 1 --spec-nodes 0,1,2 --resident --ctx 9 --bguide-cfg $G --band-probe 1 ;;
 sdvg_*) X=${1#sdvg_}; T=${X%%:*}; R=${X#*:}; SD=${R%%:*}; TAU=${R##*:}   # SDVG baseline (Hu & Zhang 2026, arXiv 2604.17397, branch dev): Fast draft, ImageReward worst-frame router with fixed tau, first generated chunk forced to the target, rejected chunks regenerated by Base (16-step RePaint)
   run $T $SD sdvg_t${TAU//./p}_ir ir:$TAU --start 1 --warmup 1 --ir-prompt /home/hk28965/specwm/exp/irprompt_$T.txt ;;
 best_*) T=${1#best_}; T=${T%_*}; S=${1##*_}; [ $T = jar ] && T=sand_filling_up_jar; [ $T = castle ] && T=sandcastle_wave_impact
   [ -f $V0/$T/base70_s$S/video.mp4 ] || echo "[note] no Base rollout for $T s$S"
   run $T $S semc100_w3_recent_retry1 sem:1.0 --sem-colour 1 --sem-ref recent --warmup 3 --steps 16 --sem-retry 1 ;;
 viz_*) T=${1#viz_}; T=${T%_*}; S=${1##*_}; [ $T = jar ] && T=sand_filling_up_jar; [ $T = castle ] && T=sandcastle_wave_impact
   run $T $S viz_best sem:1.0 --sem-colour 1 --sem-ref recent --warmup 3 --steps 16 --sem-retry 1 --save-drafts 1 ;;
 d360n_*) T=${1#d360n_}; T=${T%_*}; S=${1##*_}; [ $T = jar ] && T=sand_filling_up_jar; [ $T = castle ] && T=sandcastle_wave_impact
   run $T $S d360_never never --max-area 184320 --nodes 0,3 ;;
 smoke_kvmc) T=sand_filling_up_jar; S=1; F="--frames 61"
   run $T $S _smoke_kv_probe2 probe2:1 --chk-list full,kv,kvg1 $F && run $T $S _smoke_kv_early early:0.5 --chk kv --warmup 1 --run 1 --steps 4 $F && \
   run $T $S _smoke_mc_probe mc --beta 0 --mc-chk kv --warmup 1 --steps 4 $F && run $T $S _smoke_mc_rej mc --beta 1000 --mc-chk kv --warmup 1 --steps 4 $F
   for n in _smoke_kv_probe2 _smoke_kv_early _smoke_mc_probe _smoke_mc_rej; do [ -f $V0/$T/${n}_s1/video.mp4 ] || { echo "[SMOKE_FAIL] $n"; touch /home/hk28965/specwm/exp/KVMC_SMOKE_FAIL; return; }; done
   echo "[SMOKE_OK kvmc] $(date)"; touch /home/hk28965/specwm/exp/KVMC_SMOKE_OK ;;
 kvmc_*) T=${1#kvmc_}; T=${T%_*}; S=${1##*_}; [ $T = jar ] && T=sand_filling_up_jar; [ $T = castle ] && T=sandcastle_wave_impact
   while [ ! -f /home/hk28965/specwm/exp/KVMC_SMOKE_OK ] && [ ! -f /home/hk28965/specwm/exp/KVMC_SMOKE_FAIL ]; do sleep 60; done
   [ -f /home/hk28965/specwm/exp/KVMC_SMOKE_OK ] || { echo "[skip $1: smoke failed]"; return; }
   run $T $S probe2kv probe2:1 --chk-list full,kv,kvg1
   run $T $S early_kv early:$($PK thr2 $T $S kv 0.5 probe2kv) --chk kv --warmup 2 --run 2 --steps 16
   run $T $S mcprobe_kv mc --beta 0 --mc-chk kv --warmup 2 --steps 16
   run $T $S mc_kv mc --beta $($PK beta $T $S mcprobe_kv) --mc-chk kv --warmup 2 --steps 16
   run $T $S mcprobe_full mc --beta 0 --mc-chk full --mc-k 1 --warmup 2 --steps 16
   run $T $S mc_full mc --beta $($PK beta $T $S mcprobe_full) --mc-chk full --mc-k 1 --warmup 2 --steps 16 ;;
 dis_*) T=${1#dis_}; T=${T%_*}; S=${1##*_}; [ $T = jar ] && T=sand_filling_up_jar; [ $T = castle ] && T=sandcastle_wave_impact
   run $T $S probe2n3 probe2:3 --chk-list full,g1
   run $T $S win3_full_n3 window:3:$($PK thr2 $T $S full 0.5 probe2n3) --chk full --check-at 3 --warmup 2 --steps 16
   run $T $S win3_full_q90 window:3:$($PK thr2 $T $S full 0.9) --chk full --warmup 2 --steps 16 ;;
 esac; }
