import os
import torch, numpy as np, json, os
from PIL import Image
torch.set_num_threads(4); torch.hub.set_dir(os.environ["SCRATCH"]+"/.torch_hub")
m = torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",trust_repo=True).eval()
mean=torch.tensor([0.485,0.456,0.406]).view(3,1,1); std=torch.tensor([0.229,0.224,0.225]).view(3,1,1)
@torch.no_grad()
def fimg(p):
    x=(torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((518,518)))).permute(2,0,1).float()/255-mean)/std
    return torch.nn.functional.normalize(m(x[None]).flatten(1),dim=-1)[0]
R=os.environ["SCRATCH"]+"/lingbot-world-v2/specwm_out/revisit"
IN=os.environ["SCRATCH"]+"/lingbot-world-v2/examples/00/image.jpg"
fin=fimg(IN)
print("== 검증: 생성된 첫 프레임이 입력 이미지와 같은가 (0이면 동일) ==",flush=True)
for a in ["base_s40","fast_s4","fast_s16","fast_s40"]:
    p=os.path.join(R,a,"first.jpg")
    if os.path.exists(p): print("  %-10s first vs input image: %.4f"%(a,1-float(fimg(p)@fin)),flush=True)
print("== 되돌아보기: 첫 프레임 vs 마지막 프레임 (같은 시점이어야 함) ==",flush=True)
for a in ["base_s40","fast_s4","fast_s16","fast_s40"]:
    d=os.path.join(R,a)
    if os.path.exists(os.path.join(d,"last.jpg")):
        print("  %-10s first vs last: %.4f"%(a,1-float(fimg(os.path.join(d,"first.jpg"))@fimg(os.path.join(d,"last.jpg")))),flush=True)
print("DONE",flush=True)
