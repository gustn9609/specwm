#!/usr/bin/env python3
"""P1: anchor-frame restart on StEvo tasks. Fast prefix = reground fast_alone (frames 0..71 shown), anchor = frame 72 of {corrected target | Fast itself (control) | Base70 (upper bound)},
Fast restarted from the anchor image with the remaining camera path for 181 f -> composite 253 f video (72 Fast + 181 restart). Outputs per task: composite mp4 per arm + json (DINO chunks 6-12 to Base70).
usage: c2_anchor_stevo.py --shard i/n"""
import argparse, json, os, sys, time, numpy as np, torch, imageio
from PIL import Image
S="/home/hk28965/specwm"; R=S+"/lingbot-world-v2/specwm_out"; RG=R+"/stevo/reground"; O=R+"/stevo/anchor"; os.makedirs(O, exist_ok=True)
ap=argparse.ArgumentParser(); ap.add_argument("--shard", default="0/1"); ap.add_argument("--anchor-frame", type=int, default=72); ap.add_argument("--frames", type=int, default=181); ap.add_argument("--arms", default="anchor_target480,anchor_fast,anchor_base70")
a=ap.parse_args(); si,sn=[int(x) for x in a.shard.split("/")]; F0=a.anchor_frame
tasks=[t for i,t in enumerate(json.load(open(S+"/data/stevo/tasks_camera.json"))) if i%sn==si]
sys.path.insert(0, S+"/lingbot-world-v2"); os.chdir(S+"/lingbot-world-v2")
from specwm.binding_v2 import V2Binding
b=V2Binding(S+"/lingbot-world-v2/lingbot-world-v2-14b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0,250,500,750))
torch.hub.set_dir(S+"/.torch_hub"); dino=torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",trust_repo=True).eval().cuda()
mean=torch.tensor([0.485,0.456,0.406]).view(3,1,1).cuda(); std=torch.tensor([0.229,0.224,0.225]).view(3,1,1).cuda()
def feats(V):
    X=torch.from_numpy(np.ascontiguousarray(V)).permute(0,3,1,2).float().cuda()/255; X=torch.nn.functional.interpolate(X,size=(518,518),mode="bilinear"); X=(X-mean)/std; o=[]
    with torch.no_grad():
        for i in range(0,len(X),8): o.append(torch.nn.functional.normalize(dino(X[i:i+8]).flatten(1),dim=-1).cpu())
    return torch.cat(o)
def d(A,B): n=min(len(A),len(B)); return float((1-(A[:n]*B[:n]).sum(-1)).mean())
def rd(p): return np.stack([np.asarray(f) for f in imageio.get_reader(p)])
for t in tasks:
    tid=t["id"]; td=RG+"/"+tid; fv=td+"/W18_sink6/fast_alone_s1234/video.mp4"; tv=td+"/target/video.mp4"; bv=R+"/stevo/lingbot_base/%s/video.mp4"%tid
    if not (os.path.exists(fv) and os.path.exists(tv) and os.path.exists(bv)): print("[skip]",tid,flush=True); continue
    od=O+"/"+tid; os.makedirs(od,exist_ok=True); rp=od+"/anchor.json"; rep=json.load(open(rp)) if os.path.exists(rp) else {}
    VF,VT,VB=rd(fv),rd(tv),rd(bv); n_rest=min(a.frames, len(VB)-F0); n_rest=((n_rest-1)//12)*12+1   # 12 px frames per chunk (+1 image frame)
    anchors={"anchor_target480":VT[F0],"anchor_fast":VF[F0],"anchor_base70":VB[F0]}
    act=t["action"]; ad=od+"/action_from_%d"%F0; os.makedirs(ad,exist_ok=True); P=np.load(act+"/poses.npy"); np.save(ad+"/poses.npy",P[F0:F0+n_rest]); K=np.load(act+"/intrinsics.npy"); np.save(ad+"/intrinsics.npy",K[F0:F0+n_rest] if len(K)>=F0+n_rest else np.repeat(K[:1],n_rest,0))
    prompt=t.get("camera_WM",""); FB_ref=feats(VB[F0:F0+n_rest][::4]); FF_ref=feats(VF[F0:F0+n_rest][::4]); rep["fast_alone_to_base"]=d(FF_ref,FB_ref)
    for name in a.arms.split(","):
        if name in rep and "to_base" in rep[name] and os.path.exists(od+"/%s.mp4"%name): continue
        ip=od+"/%s.png"%name; Image.fromarray(anchors[name]).save(ip)
        try:
            cond=b.prepare(ip,prompt,ad,frame_num=n_rest); N=cond["num_chunks"]; sig=b.node_sigmas(); b.reset_caches(); torch.cuda.empty_cache(); g=torch.Generator().manual_seed(1234); t0=time.time(); outs=[]
            for ci in range(N):
                z=b.chunk_noise(ci,g); x0=None
                for i in range(len(sig)):
                    x0=b.target_x0(z,i,ci)
                    if i<len(sig)-1: s=float(sig[i+1]); z=(1-s)*x0+s*b.randn_like(x0,g)
                b.commit(x0,ci); outs.append(x0.detach().float().cpu())
            gen_s=time.time()-t0; lat=torch.cat(outs,dim=1)
            with torch.no_grad(): im=b.pipe.vae.decode([lat.to(b.device)])[0].detach().float().cpu()
            V=((im.clamp(-1,1)+1)/2).permute(1,2,3,0).numpy(); V=(V*255).astype("uint8")
            imageio.mimwrite(od+"/%s.mp4"%name, list(VF[:F0])+list(V), fps=16, quality=8)   # composite: Fast prefix + restart
            FG=feats(V[::4]); rep[name]=dict(to_base=d(FG,FB_ref), to_fast_alone=d(FG,FF_ref), gen_s=round(gen_s,1), frames=int(len(V)))
            print("[%s] %s to_base %.3f (fast %.3f) gen %.0fs"%(name,tid,rep[name]["to_base"],rep["fast_alone_to_base"],gen_s),flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); rep[name]=dict(error=str(e)[:200]); torch.cuda.empty_cache()
        json.dump(rep,open(rp,"w"),indent=1)
print("[ANCHOR_STEVO_DONE shard %s]"%a.shard)
