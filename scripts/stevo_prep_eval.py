#!/usr/bin/env python3
"""StEvo outputs 폴더에 output_map.json (task_id -> mp4) 생성. usage: stevo_prep_eval.py <out_dir> [<out_dir> ...]"""
import json, os, sys, glob
for d in sys.argv[1:]:
    for j in glob.glob(os.path.join(d, "*.json")):
        if os.path.basename(j) != os.path.basename(os.path.normpath(d)) + ".json": os.rename(j, j + ".bak")        # eval 은 폴더에 .json 이 정확히 하나여야 함
    m = {os.path.basename(p)[:-4]: os.path.basename(p) for p in sorted(glob.glob(os.path.join(d, "*.mp4"))) if not p.endswith(".tmp.mp4")}
    json.dump(m, open(os.path.join(d, os.path.basename(os.path.normpath(d)) + ".json"), "w"), indent=1); print(d, len(m), "videos")
