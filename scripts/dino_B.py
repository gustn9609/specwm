import os
import torch, numpy as np, json, os
torch.set_num_threads(4); torch.hub.set_dir(os.environ["SCRATCH"]+"/.torch_hub")
m = torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",trust_repo=True).eval()
mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
R=os.environ["SCRATCH"]+"/lingbot-world-v2/specwm_out"; I=R+"/ideasAB"
@torch.no_grad()
def f(img):
    b=torch.from_numpy(img[None]).permute(0,3,1,2).float()/255.
    b=torch.nn.functional.interpolate(b,size=(518,518),mode="bicubic",align_corners=False)
    return torch.nn.functional.normalize(m(((b-mean)/std)).flatten(1).float(),dim=-1)[0]
ref=np.load(R+"/revisit/base_official_S05_s16/frames_all.npy")     # seed 1234 base16, 4fps
seeds=[1234,4321,9999]; S={sd:np.load("%s/B_stream_s%d/frames_all.npy"%(I,sd)) for sd in seeds}   # 16fps 전체
V={sd:{r["chunk"]:r["dist_rel"] for r in json.load(open("%s/B_score_s%d/verify.json"%(I,sd)))["rows"]} for sd in seeds}
out=[]
for k in [2,5,8,11]:
    fr=12*k-3; ri=fr//4                                             # 청크 k 마지막 부근 프레임, ref 는 4fps
    fref=f(ref[min(ri,len(ref)-1)])
    truth={sd:1-float(f(S[sd][min(fr,len(S[sd])-1)])@fref) for sd in seeds}
    ver={sd:V[sd][k] for sd in seeds}
    pick=min(ver,key=ver.get); best=min(truth,key=truth.get)
    out.append(dict(chunk=k,frame=fr,verifier=ver,truth=truth,pick=pick,best=best,hit=pick==best,
                    picked_truth=truth[pick],best_truth=truth[best],mean_truth=float(np.mean(list(truth.values())))))
    print("k=%2d  verifier %s  truth %s  pick=%d best=%d %s"%(k,{s:round(v,3) for s,v in ver.items()},{s:round(v,3) for s,v in truth.items()},pick,best,"HIT" if pick==best else "miss"),flush=True)
hits=sum(o["hit"] for o in out); print("\n적중 %d/4   선택 평균 %.4f   무작위(세 후보 평균) %.4f   최선 %.4f"%(hits,np.mean([o["picked_truth"] for o in out]),np.mean([o["mean_truth"] for o in out]),np.mean([o["best_truth"] for o in out])))
# 창 전체(12개 후보) 상관
vv=[o["verifier"][s] for o in out for s in seeds]; tt=[o["truth"][s] for o in out for s in seeds]
print("12개 후보 전체 Pearson r = %.3f"%np.corrcoef(vv,tt)[0,1])
json.dump(out,open(os.environ["SCRATCH"]+"/B_result.json","w"),indent=1); print("DONE",flush=True)
