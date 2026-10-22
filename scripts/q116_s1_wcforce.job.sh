#!/bin/bash
# S1b: Fast 에서 WorldCache 강제 skip 상한 (thr=100): 360p 4-step ret0.25/0.5, 2-step ret0.5 → S00 seed9999 영상 + fps; DINO vs Base 480p / vs same-config no-WC
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=$SCRATCH/lingbot-world-v2/specwm_out/a3_fast; CK=$PWD/lingbot-world-v2-14b-causal-fast
g() { [ -f $R/$1/S00_s9999.mp4 ] && return; echo "########## S1b $1 ($(date)) ##########"; python -m specwm.stevo_gen --ckpt $CK --tasks $SCRATCH/data/stevo/tasks_s00.json --out $R/$1 --frames 161 --seed 9999 --t5-gpu "${@:2}"; }
g s4_bf16_360 --arm fast_bf16 --max-area 184320
g s4_bf16_360_wcforce_ret25 --arm fast_bf16 --max-area 184320 --wc 3 --wc-thresh 100 --wc-ret 0.25
g s4_bf16_360_wcforce_ret50 --arm fast_bf16 --max-area 184320 --wc 3 --wc-thresh 100 --wc-ret 0.5
g s2_bf16_360_wcforce --arm fast_bf16 --max-area 184320 --nodes 0,3 --wc 3 --wc-thresh 100 --wc-ret 0.5
python - <<'PY'
import os, torch, numpy as np, json, imageio
S=os.environ["SCRATCH"]; R=S+"/lingbot-world-v2/specwm_out"; torch.hub.set_dir(S+"/.torch_hub")
m=torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",trust_repo=True).eval().cuda()
mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).cuda(); std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).cuda()
def rd(p): return np.stack([np.asarray(f) for f in imageio.get_reader(p)])[::4]
@torch.no_grad()
def feats(V,bs=4):
    out=[]
    for i in range(0,len(V),bs):
        b=torch.from_numpy(np.ascontiguousarray(V[i:i+bs])).permute(0,3,1,2).float().cuda()/255.; b=torch.nn.functional.interpolate(b,size=(518,518),mode="bicubic",align_corners=False)
        out.append(torch.nn.functional.normalize(m((b-mean)/std).flatten(1).float(),dim=-1))
    return torch.cat(out)
def dist(a,b): n=min(len(a),len(b)); return float((1-(a[:n]*b[:n]).sum(-1)).mean())
ref=feats(np.load(R+"/floor/base/S00_s9999/frames_all.npy")); F={}
for a in ["s4_bf16_360","s4_bf16_360_wcforce_ret25","s4_bf16_360_wcforce_ret50","s2_bf16_360","s2_bf16_360_wcforce"]:
    p=R+"/a3_fast/%s/S00_s9999.mp4"%a
    if os.path.exists(p): F[a]=feats(rd(p))
res={}
for a,f in F.items():
    import glob; t=json.load(open(sorted(glob.glob(R+"/a3_fast/%s/timing.json*"%a))[0])); t=list(t.values())[0]
    res[a]=dict(dino_to_base=round(dist(f,ref),3), dino_to_nowc=round(dist(f,F["s4_bf16_360" if a.startswith("s4") else "s2_bf16_360"]),3), gen_s=t.get("gen_s"), fps_gen=round(t["frames"]/t["gen_s"],2), wc=t.get("wc") and {k:t["wc"][k] for k in ("skip_rate","full","skip")})
    print(a,res[a],flush=True)
json.dump(res,open(R+"/a3_fast/s1b_wcforce.json","w"),indent=1)
PY
touch $SCRATCH/exp/Q116_DONE; echo "[Q116_DONE] $(date)"
