#!/bin/bash
# GPU 4: 데모(Q5, Fast 70GB + Base16 67GB 라 VR 과 동시 불가) 먼저, 끝나면 VisionReward 큐(Q4)
bash $SCRATCH/scripts/q5_demo.job.sh
bash $SCRATCH/scripts/q4_vr.job.sh
