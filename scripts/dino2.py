import os
import torch, numpy as np
from PIL import Image
torch.hub.set_dir(os.environ["SCRATCH"]+"/.torch_hub")
m=torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",trust_repo=True).eval()
R=os.environ["SCRATCH"]+"/lingbot-world-v2/specwm_out/revisit"
mean=torch.tensor([0.485,0.456,0.406]).view(3,1,1); std=torch.tensor([0.229,0.224,0.225]).view(3,1,1)
def feat(p):
    x=(torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((518,518)))).permute(2,0,1).float()/255-mean)/std
    with torch.no_grad(): return torch.nn.functional.normalize(m(x[None]).flatten(1),dim=-1)[0]
cos=lambda a,b: float((a*b).sum())
for r in ["base_own70","base_s40","fast_s4","fast_yaw"]:
    F={t:feat(f"{R}/{r}/{t}.jpg") for t in ("first","mid","last")}
    fm=cos(F['first'],F['mid']); fl=cos(F['first'],F['last']); ml=cos(F['mid'],F['last'])
    print(f"{r:12} first-mid {fm:.3f}  first-last {fl:.3f}  mid-last {ml:.3f}")
