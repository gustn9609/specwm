import os, torch
os.environ.setdefault("TORCH_HOME", os.environ["SCRATCH"]+"/.torch_hub")
torch.hub.set_dir(os.environ["SCRATCH"]+"/.torch_hub")
import clip
for name in ["ViT-B/32", "ViT-L/14"]:
    m, _ = clip.load(name, device="cpu", download_root=os.environ["SCRATCH"]+"/vbench_cache/clip_model"); print("clip", name, "OK", flush=True); del m
# subject_consistency: DINO ViT-B/16 via torch.hub
try:
    d = torch.hub.load("facebookresearch/dino:main", "dino_vitb16", trust_repo=True); print("dino_vitb16 OK", flush=True); del d
except Exception as e: print("dino hub 실패:", str(e)[:150], flush=True)
# imaging_quality: pyiqa musiq — 캐시 경로 확인
import glob; print("cache:", [os.path.basename(p) for p in glob.glob(os.environ["SCRATCH"]+"/vbench_cache/*")], flush=True)
print("[PRE_DONE]", flush=True)
