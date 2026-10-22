import os
import torch, os, json, glob, numpy as np
from PIL import Image
torch.hub.set_dir(os.environ["SCRATCH"]+"/.torch_hub")
R=os.environ["SCRATCH"]+"/lingbot-world-v2/specwm_out/revisit"
runs=["fast_s4","fast_yaw"]

def load_model():
    try:
        m=torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",trust_repo=True); return m.eval(),"dinov2_vitb14",518
    except Exception as e:
        print("dinov2 실패:",type(e).__name__,str(e)[:80])
    try:
        import torchvision; from torchvision.models import resnet50, ResNet50_Weights
        m=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2); m.fc=torch.nn.Identity(); return m.eval(),"resnet50",224
    except Exception as e:
        print("resnet50 실패:",type(e).__name__,str(e)[:80]); raise

m,name,S=load_model(); print("feature extractor:",name)
mean=torch.tensor([0.485,0.456,0.406]).view(3,1,1); std=torch.tensor([0.229,0.224,0.225]).view(3,1,1)
def feat(p):
    im=Image.open(p).convert("RGB").resize((S,S))
    x=(torch.from_numpy(np.asarray(im)).permute(2,0,1).float()/255-mean)/std
    with torch.no_grad(): f=m(x[None])
    return torch.nn.functional.normalize(f.flatten(1),dim=-1)[0]
cos=lambda a,b: float((a*b).sum())

# 통제군: 서로 다른 run의 첫 프레임끼리 (같은 입력 이미지 → 거의 1이어야 함, 지표 sanity)
F={r:{t:feat(f"{R}/{r}/{t}.jpg") for t in ("first","mid","last")} for r in runs}
print(f"\n{'run':10}{'first↔last':>12}{'first↔mid':>11}{'mid↔last':>10}   판정(last가 mid보다 first에 가까운가)")
out={}
for r in runs:
    fl=cos(F[r]["first"],F[r]["last"]); fm=cos(F[r]["first"],F[r]["mid"]); ml=cos(F[r]["mid"],F[r]["last"])
    out[r]=dict(first_last=fl,first_mid=fm,mid_last=ml,revisit_gain=fl-fm)
    print(f"{r:10}{fl:12.4f}{fm:11.4f}{ml:10.4f}   {'YES' if fl>fm else 'no'} (Δ={fl-fm:+.4f})")
# 지표 검정력: 다른 run의 first끼리 (같은 조건 이미지) vs base first ↔ fast mid (다른 장면 시점)
print(f"\n검정력 확인: base.first↔fast4.first (같은 출발 이미지) = {cos(F['base_s40']['first'],F['fast_s4']['first']):.4f}")
print(f"           base.first↔fast4.mid   (다른 시점)      = {cos(F['base_s40']['first'],F['fast_s4']['mid']):.4f}")
json.dump(dict(extractor=name,runs=out),open(f"{R}/dino_revisit.json","w"),indent=1)
