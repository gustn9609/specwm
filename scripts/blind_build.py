"""Blind pairwise study: every candidate clip vs the 14B 4-step teacher clip of the same prompt. Transcodes clips to opaque names (sha1 of the
source path), writes paper/1012/blind/mapping.json (opaque id -> arm, prompt, source; NOT published) and trials.json (published: trial id,
prompt label, start image id, the two opaque clip ids). usage: python scripts/blind_build.py"""
import os, json, hashlib, subprocess, glob
S = "/home/hk28965/specwm"; O = S + "/lingbot-world-v2/specwm_out"; B = S + "/paper/1012/blind"
T = ["easy_lake", "easy_greatwall", "easy_stonehenge_sunset", "easy_urban_rain", "easy_dragon_night", "pouring_water_into_cup", "domino_falling", "hourglass_accumulation", "newtons_cradle", "billiard_break"]
ARMS = {"teacher_14b_4step": O + "/cmp10/%s/uniform4_s1234/video.mp4", "14b_2step_832": O + "/cmp10/%s/uniform2_s1234/video.mp4", "chunkwise": O + "/cmp10/%s/diag_learned_dense_0.24_chunk0.7_s1234/video.mp4",
        "576_esrgan": O + "/cmp10_lr_sr/%s/lr576_u2_s1234/video.mp4", "576_refiner": O + "/cmp10/%s/ressplit_pix_s1234/video.mp4", "edge_1p3b_832": O + "/cmp10/%s/draft_1p3b_s1234/video.mp4",
        "matrixgame2": O + "/mg2/cmp10/%s/video.mp4", "waypoint": O + "/waypoint/cmp10/%s/output.mp4"}
extra = json.load(open(B + "/extra_arms.json")) if os.path.exists(B + "/extra_arms.json") else {}; ARMS.update(extra)
TASKS = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}
mapping = json.load(open(B + "/mapping.json")) if os.path.exists(B + "/mapping.json") else {}; trials = []; imgs = {}
def opaque(src): return hashlib.sha1(("specwm-blind|" + src).encode()).hexdigest()[:12]
for t in T:
    im = TASKS[t]["image"]; iid = opaque(im); imgs[t] = iid; dst = B + "/v/%s.jpg" % iid
    if not os.path.exists(dst): subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", im, "-vf", "scale=640:-2", "-q:v", "4", dst])
    clip = {}
    for arm, pat in ARMS.items():
        src = pat % t
        if not os.path.exists(src): continue
        cid = opaque(src); clip[arm] = cid; mapping[cid] = dict(arm=arm, prompt=t, src=src); dst = B + "/v/%s.mp4" % cid
        if not os.path.exists(dst): subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", "scale=-2:360,fps=16", "-frames:v", "153", "-c:v", "libx264", "-preset", "slow", "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", dst])
    if "teacher_14b_4step" not in clip: continue
    for arm, cid in clip.items():
        if arm == "teacher_14b_4step": continue
        trials.append(dict(id=opaque(t + "|" + arm)[:10], prompt=t.replace("easy_", "").replace("_", " "), text=(TASKS[t].get("camera_WM") or "")[:220], image=imgs[t], a=clip["teacher_14b_4step"], b=cid))
json.dump(mapping, open(B + "/mapping.json", "w"), indent=1); json.dump(trials, open(B + "/trials.json", "w"))
print(len(trials), "trials;", len(glob.glob(B + "/v/*.mp4")), "clips; MB", round(sum(os.path.getsize(p) for p in glob.glob(B + "/v/*")) / 1e6, 1))
