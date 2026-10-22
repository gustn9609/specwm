#!/usr/bin/env python3
"""VisionReward-Video 스코어링 데몬 (action item #1).
specwm_out 아래에서 새 video.mp4 를 계속 찾아 6초 창 단위로 채점하고 specwm_out/vr/vr_scores.json 에 누적.
프롬프트는 경로의 씬 토큰(S05, /05/, hands) 으로 data/p_*.txt 에 매핑.
종료: $SCRATCH/exp/ALL_GEN_DONE 가 있고 대기 중인 영상이 없을 때, 또는 --max-hours.
vr_score.py (vr-venv, 모델은 import 시 로드) 의 채점 함수를 재사용.
"""
import os, sys, json, time, glob, argparse, re, traceback
S = os.environ.get("SCRATCH", "/home/hk28965/specwm"); R = S + "/lingbot-world-v2/specwm_out"
sys.path.insert(0, S + "/scripts")
ap = argparse.ArgumentParser(); ap.add_argument("--roots", default="floor,handoff_grid,ctx_grid,revisit,repro,resync_s05,edge_sim,compare_lowres,demo")
ap.add_argument("--poll", type=int, default=180); ap.add_argument("--max-hours", type=float, default=48); ap.add_argument("--out", default=R + "/vr/vr_scores.json")
a = ap.parse_args(); os.makedirs(os.path.dirname(a.out), exist_ok=True)
import numpy as np
import vr_score as VR                                        # 모델 로드 (한 번)
VR.WINDOWS_DIR = R + "/vr/windows"
PROMPTS = {"00": S + "/data/p_00.txt", "01": S + "/data/p_01.txt", "02": S + "/data/p_02.txt", "05": S + "/data/p_05.txt", "hands": S + "/data/p_hands1.txt"}


def scene_of(p):
    mm = re.search(r"[/_]S(\d\d)[/_]", p) or re.search(r"/(\d\d)/", p)
    if mm and mm.group(1) in PROMPTS: return mm.group(1)
    return "hands" if "hands" in p else None


def load():
    return json.load(open(a.out)) if os.path.exists(a.out) else {"model_path": VR.MODEL_PATH, "window_sec": VR.WINDOW_SEC, "videos": {}}


def save(res):
    tmp = a.out + ".tmp"; json.dump(res, open(tmp, "w"), indent=1); os.replace(tmp, a.out)


def pending(res):
    vids = []
    for root in a.roots.split(","):
        vids += glob.glob(R + "/%s/**/video.mp4" % root, recursive=True)
    todo = []
    for v in sorted(vids):
        tag = os.path.relpath(os.path.dirname(v), R)
        if tag in res["videos"] and res["videos"][tag].get("status") == "done": continue
        if time.time() - os.path.getmtime(v) < 60: continue                 # 아직 쓰는 중일 수 있음
        if scene_of(v) is None and not os.path.exists(os.path.join(os.path.dirname(v), "prompt.txt")): continue
        todo.append((tag, v))
    return todo


def score_one(tag, v, res):
    sc = scene_of(v); pp = os.path.join(os.path.dirname(v), "prompt.txt")
    prompt = open(pp).read().strip() if os.path.exists(pp) else open(PROMPTS[sc]).read().strip()
    e = {"tag": tag, "video_path": v, "scene": sc, "windows": [], "status": "running"}; res["videos"][tag] = e; save(res)
    try:
        dur, nfr, fps = VR.get_duration_sec(v); e.update(duration_sec=dur, nframes=nfr, fps=fps)
        starts = list(np.arange(0.0, dur - 1e-6, VR.WINDOW_SEC)); t0 = time.time()
        for wi, st in enumerate(starts):
            d = min(VR.WINDOW_SEC, dur - st); clip = os.path.join(VR.WINDOWS_DIR, tag.replace("/", "__"), "win_%03d.mp4" % wi)
            VR.cut_window(v, st, d, clip); sc_, per = VR.score_with_dims(clip, prompt)
            e["windows"].append(dict(window_index=wi, start_sec=float(st), duration_sec=float(d), score=sc_, per_dimension={k: vv["signed"] for k, vv in per.items()}))
        scores = [w["score"] for w in e["windows"]]
        e.update(mean_score=float(np.mean(scores)), min_score=float(np.min(scores)), last_window_score=scores[-1], num_windows=len(scores), status="done", elapsed_sec=round(time.time() - t0, 1))
        print("[vr] %-60s mean %.4f min %.4f (%d win, %.0fs)" % (tag, e["mean_score"], e["min_score"], len(scores), e["elapsed_sec"]), flush=True)
    except Exception as ex:
        e["status"] = "error"; e["error"] = "%s: %s" % (type(ex).__name__, ex); print("[vr] ERROR", tag, ex, flush=True); traceback.print_exc()
    save(res)


t_start = time.time(); res = load(); idle = 0
while True:
    todo = pending(res)
    if todo:
        idle = 0
        for tag, v in todo: score_one(tag, v, res)
    else:
        idle += 1
        if os.path.exists(S + "/exp/ALL_GEN_DONE") or all(os.path.exists(S + "/exp/Q%d_DONE" % q) for q in (1, 2, 3, 5)) or (time.time() - t_start) > a.max_hours * 3600:
            print("[vr] 대기 영상 없음 + 생성 종료 -> 종료", flush=True); break
        print("[vr] 대기 중 (%d 채점 완료)" % len(res["videos"]), flush=True); time.sleep(a.poll)
print("[VR_DAEMON_DONE]", flush=True)
