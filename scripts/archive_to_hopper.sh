#!/bin/bash
# [10/15] copy closed-experiment outputs to hopper (no deletion here; delete only after the verify pass prints OK)
# usage: scripts/archive_to_hopper.sh copy|verify [A|B|X|all]   (needs key auth: ssh-copy-id hopper.ece.utexas.edu)
H=${ARCHIVE_HOST:-lamarr.ece.utexas.edu}; DST=specwm_archive; S=/home/hk28965/specwm; O=$S/lingbot-world-v2/specwm_out
A="specwm_out/bench43 specwm_out/actspec2 specwm_out/fused specwm_out/early_cmp specwm_out/cmp10 specwm_out/stevo38 specwm_out/stevo specwm_out/fm_data specwm_out/edit_data specwm_out/edit_data_val specwm_out/gap_dump_nob43 specwm_out/res_data"
B="specwm_out/spec_v0 specwm_out/edit14 specwm_out/selfd14 specwm_out/distill47 specwm_out/distill_data47 specwm_out/distill4 specwm_out/actspec specwm_out/paths2 specwm_out/sdvg_paths specwm_out/lever2_b43 specwm_out/loop specwm_out/loop1005 specwm_out/loop1005lat specwm_out/loop_lr576 specwm_out/loop_lr576_sr specwm_out/loop_lr448_sr specwm_out/commit_pipe specwm_out/commit_pipe_lr576_sr specwm_out/latency specwm_out/pipe_ff3 specwm_out/fast_gen"
X="exp/drift_lat exp/drift_feat paper"
cd $S; G=${2:-all}; [ "$G" = A ] && { B=""; X=""; }; [ "$G" = B ] && { A=""; X=""; }; [ "$G" = X ] && { A=""; B=""; }
case "$1" in
  copy)
    ssh $H "mkdir -p $DST/lingbot-world-v2/specwm_out $DST/exp"
    for d in $A $B; do echo "== $d $(date +%H:%M)"; rsync -a --partial --info=progress2 lingbot-world-v2/$d $H:$DST/lingbot-world-v2/specwm_out/ || echo "[RSYNC_FAIL] $d"; done
    for d in $X; do echo "== $d $(date +%H:%M)"; rsync -a --partial --info=progress2 $d $H:$DST/$(dirname $d)/ || echo "[RSYNC_FAIL] $d"; done
    echo ARCHIVE_COPY_DONE ;;
  verify)   # size+mtime comparison in dry-run: any line printed = difference
    for d in $A $B; do n=$(rsync -ani lingbot-world-v2/$d $H:$DST/lingbot-world-v2/specwm_out/ | grep -c "^[<>]"); echo "$d: $n differing files"; done
    for d in $X; do n=$(rsync -ani $d $H:$DST/$(dirname $d)/ | grep -c "^[<>]"); echo "$d: $n differing files"; done
    echo ARCHIVE_VERIFY_DONE ;;
  *) echo "usage: $0 copy|verify" ;;
esac
