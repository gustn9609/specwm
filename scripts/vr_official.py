#!/usr/bin/env python3
"""공식 VisionReward-Video 프로토콜: 클립 전체를 1fps(최대 24프레임)로 한 번 채점 (창 분할 없음). vr_score.py 의 모델/질문/가중치 재사용.
사용: python vr_official.py <out.json> tag:video.mp4:prompt.txt ..."""
import os, sys, json, time
S = os.environ.get("SCRATCH", "/home/hk28965/specwm"); sys.path.insert(0, S + "/scripts")
import vr_score as VR
out_path = sys.argv[1]; res = json.load(open(out_path)) if os.path.exists(out_path) else {}
for arg in sys.argv[2:]:
    tag, vp, pp = arg.split(":")
    if tag in res: continue
    t0 = time.time(); score, per = VR.score_with_dims(vp, open(pp).read().strip())
    res[tag] = dict(score=score, per_dimension={k: v["signed"] for k, v in per.items()}, elapsed=round(time.time() - t0, 1))
    json.dump(res, open(out_path, "w"), indent=1); print("[vr-official] %-45s %.4f (%.0fs)" % (tag, score, time.time() - t0), flush=True)
print("[VR_OFFICIAL_DONE]")
