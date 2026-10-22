import os, torch
torch.hub.set_dir(os.environ["SCRATCH"]+"/.torch_hub")
import clip
from clip import clip as _c
root = os.environ["SCRATCH"]+"/vbench_cache/clip_model"
for name in ["ViT-B/32", "ViT-L/14"]:
    p = _c._download(_c._MODELS[name], root); print("downloaded", name, "->", os.path.basename(p), os.path.getsize(p)//1_000_000, "MB", flush=True)
try:
    d = torch.hub.load("facebookresearch/dino:main", "dino_vitb16", trust_repo=True); print("dino_vitb16 OK", flush=True); del d
except Exception as e: print("dino hub 실패:", str(e)[:150], flush=True)
print("[PRE_DONE]", flush=True)
