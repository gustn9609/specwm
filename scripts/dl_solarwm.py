import os
"""SolarWM 5B: 실험에 필요한 3개 체크포인트만 (base + 교사 + 학생) = 약 114 GB"""
from huggingface_hub import snapshot_download
import sys
DIRS = ["SolarWM-5B-base", "SolarWM-5B-bid-stage0p5-81f", "SolarWM-5B-sgf-stage2-81f"]
p = snapshot_download(
    repo_id="junchaoh-cs/SolarWM-Wan2.2-5B",
    allow_patterns=[d + "/*" for d in DIRS] + ["*.md", "*.json"],
    local_dir=os.environ["SCRATCH"]+"/SolarWM-models",
    max_workers=2,
)
print("[done]", p, flush=True)
