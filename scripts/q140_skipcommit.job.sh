#!/bin/bash
# Commit lever: skip the clean t=0 KV pass (KV = last denoise step's) — speed + quality (DINO to Base, S00 seed 9999) at 360p 2-step / 480p 4-step; plus a 3-scene DINO check vs standard.
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd $SCRATCH/lingbot-world-v2; export PYTHONPATH=$PWD; R=specwm_out/a3_fast; CK=$PWD/lingbot-world-v2-14b-causal-fast
g() { [ -f $R/$1/S00_s9999.mp4 ] && return; echo "########## SKIPCOMMIT $1 ($(date)) ##########"; python -m specwm.stevo_gen --ckpt $CK --tasks $SCRATCH/data/stevo/tasks_s00.json --out $R/$1 --frames 161 --seed 9999 --t5-gpu "${@:2}" 2>&1 | grep -E "^\[|Error|Traceback"; }
SPECWM_SKIP_COMMIT=1 g s2_bf16_360_skipcommit --arm fast_bf16 --max-area 184320 --nodes 0,3
SPECWM_SKIP_COMMIT=1 g s4_bf16_480_skipcommit --arm fast_bf16 --max-area 399360
g s4_bf16_480 --arm fast_bf16 --max-area 399360
# floor scenes S05/S01/S02 seed 1234 480p 4-step, skip-commit vs the existing fast_alone (resync_v4 dirs)
for SC in 05 01 02; do
  SPECWM_SKIP_COMMIT=1 python -m specwm.resync_v4 --ckpt $CK --action-root $SCRATCH/rt2_action_$SC --prompt-file $SCRATCH/p_$SC.txt --scene $SC --frames 161 --seeds 1234 --base-video specwm_out/floor/base/S${SC}_s1234/video.mp4 --configs W18_sink6 --arms fast_alone --out specwm_out/skipcommit/S$SC 2>&1 | grep -E "^\[|Error|Traceback"
done
python - <<'PY'
import os, json, glob, numpy as np, torch, imageio
S=os.environ["SCRATCH"]; R=S+"/lingbot-world-v2/specwm_out"; torch.hub.set_dir(S+"/.torch_hub")
m=torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",trust_repo=True).eval().cuda(); mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).cuda(); std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).cuda()
def feats(V):
    o=[]
    with torch.no_grad():
        for i in range(0,len(V),4):
            b=torch.from_numpy(np.ascontiguousarray(V[i:i+4])).permute(0,3,1,2).float().cuda()/255.; b=torch.nn.functional.interpolate(b,size=(518,518),mode="bicubic",align_corners=False); o.append(torch.nn.functional.normalize(m((b-mean)/std).flatten(1).float(),dim=-1))
    return torch.cat(o)
def dist(a,b): n=min(len(a),len(b)); return float((1-(a[:n]*b[:n]).sum(-1)).mean())
def rd(p): return np.stack([np.asarray(f) for f in imageio.get_reader(p)])[::4]
out={}; refB=feats(np.load(R+"/floor/base/S00_s9999/frames_all.npy"))
for a in ["s4_bf16_480","s4_bf16_480_skipcommit","s2_bf16_360","s2_bf16_360_skipcommit"]:
    p=R+"/a3_fast/%s/S00_s9999.mp4"%a
    if not os.path.exists(p): continue
    F=feats(rd(p)); t=json.load(open(sorted(glob.glob(R+"/a3_fast/%s/timing.json*"%a))[0])); t=list(t.values())[0]
    out[a]=dict(dino_to_base=round(dist(F,refB),3), gen_s=t["gen_s"], fps_gen=round(t["frames"]/t["gen_s"],2), total_s=t["total_s"]); print(a,out[a],flush=True)
for sc in ["05","01","02"]:
    FB=feats(np.load(R+"/floor/base/S%s_s1234/frames_all.npy"%sc)); std_=feats(np.load(R+"/floor/fast/S%s/W18_sink6/fast_alone_s1234/frames_all.npy"%sc)); p=R+"/skipcommit/S%s/W18_sink6/fast_alone_s1234/frames_all.npy"%sc
    if os.path.exists(p): sk=feats(np.load(p)); out["S%s"%sc]=dict(std_to_base=round(dist(std_,FB),3), skip_to_base=round(dist(sk,FB),3), skip_vs_std=round(dist(sk,std_),3)); print(sc,out["S%s"%sc],flush=True)
json.dump(out,open(R+"/skipcommit/skipcommit.json","w"),indent=1)
PY
touch $SCRATCH/exp/Q140_DONE; echo "[Q140_DONE] $(date)"
