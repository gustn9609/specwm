#!/bin/bash
# [10/16 frozen baselines; 10/17 --batch] Matrix-Game 2.0 in ITS setting (official stack: flash-attn 2 + torch.compile, 640x352, 3 steps, 12 fps native, 10 s).
# usage: baselines/mg2/run.sh IMAGE PROGRAM_ID SEED OUTDIR [extra run_mg2.py flags]   -> OUTDIR/video.mp4 (120 frames @ 12 fps) + OUTDIR/timing.json
#        (OUTDIR/warmup/ holds the discarded warm-up clip; pass --warm 0 to skip it)
#        baselines/mg2/run.sh --batch JOBS.json [extra run_mg2.py flags]              -> ONE model load for every job; JOBS.json = [{image, program, seed, out}, ...]
#        (gen/make_queue.py writes them); a job whose out/video.mp4 + timing.json exist is skipped (--redo re-runs), a failing job writes out/ERROR.txt and
#        the batch goes on; exit 1 if any job failed, 0 otherwise (also when nothing was pending). The 2 warm-up clips go to <first pending out>/warmup*/.
# GPU: the caller sets CUDA_VISIBLE_DEVICES and holds the GPU lock, e.g.
#        flock /tmp/gpu4.lock env CUDA_VISIBLE_DEVICES=4 /home/hk28965/specwm/baselines/mg2/run.sh IMG P2 1234 OUT
#        flock /tmp/gpu4.lock env CUDA_VISIBLE_DEVICES=4 /home/hk28965/specwm/baselines/mg2/run.sh --batch JOBS.json
S=/home/hk28965/specwm; V=$S/tools/mg2/venv; C=$S/tools/cuda128_tk
# triton / inductor JIT need python headers + a CUDA toolkit (same exports as tools/abot/run.sh); caches kept next to the venv so recompiles are rare
export CPATH=$S/tools/pydev/usr/include/python3.12:$S/tools/pydev/usr/include/x86_64-linux-gnu/python3.12:$S/tools/pydev/usr/include:$C/include:$CPATH
export LD_LIBRARY_PATH=$C/lib64:$LD_LIBRARY_PATH LIBRARY_PATH=$C/lib64:$C/lib64/stubs:$LIBRARY_PATH CUDA_HOME=$C PATH=$C/bin:$PATH OMP_NUM_THREADS=8
export TORCHINDUCTOR_CACHE_DIR=$S/tools/mg2/inductor_cache TRITON_CACHE_DIR=$S/tools/mg2/triton_cache HF_HUB_OFFLINE=1
if [ "$1" = "--batch" ]; then
  J=$2; shift 2; [ -n "$J" ] && [ -f "$J" ] || { echo "usage: $0 --batch JOBS.json [flags]   (no such file: $J)"; exit 2; }
  exec $V/bin/python $S/baselines/mg2/run_mg2.py --jobs "$J" --warm 2 "$@"
fi
[ $# -ge 4 ] || { echo "usage: $0 IMAGE PROGRAM_ID SEED OUTDIR [flags]   |   $0 --batch JOBS.json [flags]"; exit 1; }
mkdir -p "$4"
exec $V/bin/python $S/baselines/mg2/run_mg2.py --image "$1" --program "$2" --seed "$3" --out "$4" "${@:5}"
