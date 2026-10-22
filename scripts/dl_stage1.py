import os
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="junchaoh-cs/SolarWM-Wan2.2-5B",
    allow_patterns=["SolarWM-5B-tf-stage1-81f/*"],
    local_dir=os.environ["SCRATCH"]+"/SolarWM-models", max_workers=2)
print("[done]", p, flush=True)
