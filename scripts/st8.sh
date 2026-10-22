#!/bin/bash
S=$SCRATCH; R=$S/lingbot-world-v2/specwm_out; date "+지금 %H:%M"
squeue -u $USER -h -o "%.9i %.8j %.9T %.9M %.14R"; echo; sacct -S now-8hours -X -o JobID,JobName%9,Elapsed,State -n | grep -vE "PENDING" | tail -12
echo "== D: lock-on arms =="; ls $R/lockon/W18_sink6/ 2>/dev/null | tr "\n" " "; echo; grep -cE "^\[W.*\] \{" ~/slurm-972387.out; grep -E "실패" ~/slurm-972387.out | cut -c1-100 | head -3
echo "== E1 =="; f=$(ls -t $S/r4E1.*.out 2>/dev/null | head -1); [ -n "$f" ] && grep -E "^\[W|실패|R4E1_DONE|Error" $f | cut -c1-110 | tail -14
echo "== E2 =="; f=$(ls -t $S/r4E2.*.out 2>/dev/null | head -1); [ -n "$f" ] && grep -E "#####|BASE_PREFIX_DONE|R4E2_DONE|Error|Traceback" $f | cut -c1-100 | tail -6
echo "== Aria H1/H2 =="; for f in $(ls -t $S/aria_H*.out 2>/dev/null); do grep -E "#####|ARIA_BASE_DONE|gen_s|Error" $f | tail -3; done
echo "== repro 상태 =="; for sc in 05 01 02; do echo "S$sc fast=[$(ls $R/repro/S$sc/W18_sink6 2>/dev/null | grep -c fast)] target=[$(for d in $R/repro/S$sc/target_s*; do [ -f $d/video.mp4 ] && echo x; done | wc -l)] base16=[$(for d in $R/repro/S$sc/base16_s*; do [ -f $d/video.mp4 ] && echo x; done | wc -l)]"; done
ls $R/resync55/W24_sink18/ 2>/dev/null | tr "\n" " "; echo
echo "== SU (allocation) =="; /usr/local/etc/taccinfo 2>/dev/null || taccinfo 2>/dev/null || echo "taccinfo 없음"
echo "== 내 잡 사용량 (sacct, node-hours) =="; sacct -u $USER -S 2026-08-25 -X -o JobID,Partition,Elapsed,State,AllocNodes -n -P | awk -F'|' '{split($3,t,":"); h=t[1]; if(split($3,d,"-")>1){split(d[2],t,":"); h=d[1]*24+t[1]} sec=h*3600+t[2]*60+t[3]; tot+=sec*$5; if($2=="gh-dev")dev+=sec*$5; else gh+=sec*$5; n++} END{printf "jobs %d | total %.1f node-h (gh-dev %.1f, gh %.1f)\n", n, tot/3600, dev/3600, gh/3600}'
sacct -u $USER -S 2026-08-25 -X -o JobID,Partition,Elapsed,State -n -P | awk -F'|' '$4=="COMPLETED"||$4=="FAILED"||$4=="TIMEOUT"||$4=="CANCELLED"{split($3,t,":"); h=t[1]; if(split($3,d,"-")>1){split(d[2],t,":"); h=d[1]*24+t[1]} sec=h*3600+t[2]*60+t[3]; c[$4]+=sec} END{for(k in c) printf "  %s: %.1f h\n", k, c[k]/3600}'
