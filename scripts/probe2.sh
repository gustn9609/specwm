#!/bin/bash
echo "=== [Base] 잠재 shape 과 VAE encode 존재 ==="
grep -nE 'shape|def encode|vae\.encode|latent_channels|z_dim' $SCRATCH/lingbot-world/specwm/binding_v1.py | head -8
grep -n 'def encode' $SCRATCH/lingbot-world/wan/modules/vae*.py 2>/dev/null | head -3
echo
echo "=== [Base/Fast] VAE 가중치 동일한가 ==="
md5sum $SCRATCH/lingbot-world/lingbot-world-base-cam/Wan2.*_VAE.pth $SCRATCH/lingbot-world-v2/lingbot-world-v2-14b-causal-fast/Wan2.*_VAE.pth 2>/dev/null
ls $SCRATCH/lingbot-world/lingbot-world-base-cam/ | grep -i vae
ls $SCRATCH/lingbot-world-v2/lingbot-world-v2-14b-causal-fast/ | grep -i vae
echo
echo "=== [SolarWM] archive.py 가 tar 에서 읽는 멤버 ==="
grep -nE 'member|\.name|endswith|getmember|extractfile' $SCRATCH/SolarWM/src/solarwm/data/archive.py | head -20
echo
echo "=== [SolarWM] 샘플 dict 키 ==="
grep -rhoE 'sample\[.[a-z_]+.\]' $SCRATCH/SolarWM/src/solarwm/ 2>/dev/null | sort -u | head -20
