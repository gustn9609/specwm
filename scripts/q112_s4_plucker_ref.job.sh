#!/bin/bash
# S4: Base 70-step 저해상도에서 Plücker 만 480x832 grid 로 계산(토큰맵 보간) — C1(320x192 0.48 / 320x208 0.71 / 240x136 0.85) 와 비교
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out; CK=$PWD/lingbot-world-base-cam
A="--image $SCRATCH/lingbot-world-v2/examples/00/image.jpg --action-path $SCRATCH/rt2_action_00 --prompt-file $SCRATCH/p_00.txt --frames 161 --seed 9999"
export SPECWM_PLUCKER_REF=480,832
for AREA in 61440 73728 32640; do [ -f $R/lowres_sr/base70_${AREA}_pref/video.mp4 ] && continue; echo "########## S4 BASE plucker-ref area=$AREA ($(date)) ##########"; python -m specwm.bench_base_official --ckpt $CK $A --max-area $AREA --out $R/lowres_sr/base70_${AREA}_pref; done
unset SPECWM_PLUCKER_REF
echo "########## S4 DINO ($(date)) ##########"
python - <<'PY'
import os, torch, numpy as np, json
S=os.environ["SCRATCH"]; R=S+"/lingbot-world-v2/specwm_out"; torch.hub.set_dir(S+"/.torch_hub")
m=torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",trust_repo=True).eval().cuda()
mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).cuda(); std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).cuda()
@torch.no_grad()
def feats(V,bs=4):
    out=[]
    for i in range(0,len(V),bs):
        b=torch.from_numpy(V[i:i+bs]).permute(0,3,1,2).float().cuda()/255.; b=torch.nn.functional.interpolate(b,size=(518,518),mode="bicubic",align_corners=False)
        out.append(torch.nn.functional.normalize(m((b-mean)/std).flatten(1).float(),dim=-1))
    return torch.cat(out)
ref=feats(np.load(R+"/floor/base/S00_s9999/frames_all.npy")); res={}
for a in ["base70_61440","base70_61440_pref","base70_73728","base70_73728_pref","base70_32640","base70_32640_pref"]:
    p=R+"/lowres_sr/%s/frames_all.npy"%a
    if not os.path.exists(p): continue
    f=feats(np.load(p)); n=min(len(f),len(ref)); d=[1-float(x@y) for x,y in zip(f[:n],ref[:n])]
    res[a]=dict(dino=round(float(np.mean(d)),3), late=round(float(np.mean(d[n//2:])),3)); print(a,res[a],flush=True)
json.dump(res,open(R+"/lowres_sr/s4_plucker_ref.json","w"),indent=1)
PY
touch $SCRATCH/exp/Q112_DONE; echo "[Q112_DONE] $(date)"
