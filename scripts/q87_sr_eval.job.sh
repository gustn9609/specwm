true
for i in $(seq 1 120); do [ -f $SCRATCH/exp/Q86_DONE ] && break; sleep 20; done
cd $SCRATCH; $SCRATCH/sr-venv/bin/python scripts/sr_eval.py
source $SCRATCH/venv/bin/activate; export SPECWM_NO_FLASH=1; python - <<'PY'
import numpy as np, torch, os, json, glob
S="/home/hk28965/specwm"; R=S+"/lingbot-world-v2/specwm_out/lowres_sr"; torch.hub.set_dir(S+"/.torch_hub"); dev="cuda"
m=torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",trust_repo=True).eval().to(dev)
mean=torch.tensor([0.485,0.456,0.406]).view(3,1,1).to(dev); std=torch.tensor([0.229,0.224,0.225]).view(3,1,1).to(dev)
def feats_np(V):
    V=V[::4] if len(V)>100 else V
    X=torch.from_numpy(np.ascontiguousarray(V)).permute(0,3,1,2).float().to(dev)/255
    X=torch.nn.functional.interpolate(X,size=(518,518),mode="bilinear"); X=(X-mean)/std; o=[]
    with torch.no_grad():
        for i in range(0,len(X),8): o.append(torch.nn.functional.normalize(m(X[i:i+8]).flatten(1),dim=-1).cpu())
    return torch.cat(o)
import imageio
def load_mp4(p): return np.stack([np.asarray(f) for f in imageio.get_reader(p)])
refs={"base480": feats_np(np.load(S+"/lingbot-world-v2/specwm_out/floor/base/S00_s9999/frames_all.npy")), "fast14b_480": feats_np(load_mp4(R+"/fast14b_399360/S00_s9999.mp4"))}
def d(A,B): n=min(len(A),len(B)); x=(1-(A[:n]*B[:n]).sum(-1)).numpy(); return round(float(x.mean()),3), round(float(x[18:].mean()),3)
res=json.load(open(R+"/sr_results.json")); rows=[]
for name in sorted(glob.glob(R+"/*_*/S00_s9999.mp4")):
    k=name.split("/")[-2]; F=feats_np(load_mp4(name)); rows.append((k,"as generated (resized by DINO)")+d(F,refs["base480"])+d(F,refs["fast14b_480"]))
for k,v in res.items():
    F=feats_np(np.load(R+"/sr/"+k+".npy")); rows.append((v["src"],v["method"]+" (SR %.1fs)"%v["sr_s"])+d(F,refs["base480"])+d(F,refs["fast14b_480"]))
tim={}
for p in glob.glob(R+"/fast14b_*/timing.json.0"): tim.update({os.path.basename(os.path.dirname(p)): json.load(open(p))["S00_s9999"]})
out=["# Low-res rollout + super-resolution (S00 Great Wall, seed 9999, 161 f) — DINO 1-cos vs Base bf16 480x832 (seed floor 0.17) and vs Fast14B 480x832","","| rollout | upsampling | vs Base (all/late) | vs Fast14B-480 (all/late) | rollout gen+dec (s) |","|---|---|---|---|---|"]
for r in rows:
    t=tim.get(r[0]); ts="%.1f"%(t["gen_s"]+t["dec_s"]) if t else "-"
    out.append("| %s | %s | %.3f / %.3f | %.3f / %.3f | %s |"%(r[0],r[1],r[2],r[3],r[4],r[5],ts))
open(R+"/lowres_sr.md","w").write("\n".join(out)); print("\n".join(out))
PY
touch $SCRATCH/exp/Q87_DONE; echo "[Q87_DONE] $(date)"
