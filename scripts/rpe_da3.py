"""[10/16] Relative pose error of a generated clip against the commanded camera trajectory, following arXiv 2610.22347 App. B.1: camera poses
recovered with Depth Anything 3 (DA3NESTED-GIANT-LARGE-1.1, local src from tools/mg35), Sim(3) Umeyama alignment to the commanded trajectory
independently per segment (their 5 camera-control segments -> here the clip split into --segments equal parts), then relative transforms between
consecutive (sampled) frames: Rot = mean angular error (deg), Trans = sum of translation errors / sum of commanded displacements (dimensionless).
The commanded trajectory is the task's poses.npy (c2w, one per frame, as the binding uses it: c2ws[:frame_num]); frames of a clip with a different
count (ABot 12 fps) are mapped onto the pose index range linearly. Results per video (absolute path) with a label, like vlm_clip_score.
usage: CUDA_VISIBLE_DEVICES=g scripts/envrun.sh python /home/hk28965/specwm/scripts/rpe_da3.py OUT.json LABEL=GLOB ... [--segments 5 --stride 2 --res 504]
[10/17 frozen baselines, additive, default off] --program-from-path: the commanded trajectory is baselines/programs/<P>.npz (c2w, 16 fps, 161 frames),
<P> parsed from the clip's run-dir name (<set>__<scene>__P<id>...); frame i of the clip is mapped to the pose at time i / fps_native (fps_native from the
clip's timing.json, else calib.json[<model of the arm dir>], else the container fps), and frames are sampled on a --sample-fps time grid (8 fps = the
--stride 2 default at our 16 fps) so every model is scored with the same step length (Waypoint 60 fps, MG2/ABot 12 fps, Astra 20 fps, ours 16 fps).
usage: ... rpe_da3.py OUT.json LABEL=/home/hk28965/specwm/baselines/runs/<arm>/*/video.mp4 --program-from-path [--sample-fps 8]"""
import sys, os, glob, json, argparse, re, numpy as np, torch, cv2
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("specs", nargs="+"); ap.add_argument("--segments", type=int, default=5); ap.add_argument("--stride", type=int, default=2)
ap.add_argument("--res", type=int, default=504); ap.add_argument("--redo", action="store_true"); ap.add_argument("--model", default="/home/hk28965/specwm/hf/DA3NESTED-GIANT-LARGE-1.1")
ap.add_argument("--ref-task", default="", help="sanity: score every video against THIS task's commanded trajectory (a wrong trajectory must give larger errors)")
ap.add_argument("--program-from-path", action="store_true", help="[baselines] commanded c2w = baselines/programs/<P>.npz, P from the run-dir name; frames mapped to poses by time at the model's native fps; sampling on the --sample-fps grid")
ap.add_argument("--sample-fps", type=float, default=8.0, help="with --program-from-path: frames sampled on this time grid instead of --stride (8 = stride 2 at 16 fps)")
ap.add_argument("--programs-dir", default="/home/hk28965/specwm/baselines/programs"); ap.add_argument("--calib", default="/home/hk28965/specwm/baselines/calib.json")
ap.add_argument("--seg-seconds", type=float, default=0.0, help="[10/19 EXP 1, additive, default 0 = off] with --program-from-path: use max(--segments, round(clip seconds / S)) Sim(3) segments, i.e. ~S-second pieces for LONG clips (P6 60 s, S 2 -> 30 segments of 16 frames at --sample-fps 8, as a 10-s clip's 5 segments; DA3 then never sees more frames per call than on a 10-s clip). Every 10-s clip (10.0-10.06 s) keeps exactly --segments (round(10.x / 2) = 5)")
a = ap.parse_args()
S = "/home/hk28965/specwm"; sys.path.insert(0, S + "/tools/mg35/Matrix-Game-3.5/third_party/depth-anything-3/src")
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.pose_align import align_poses_umeyama
TASKS = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}
def task_of(path):
    parts = path.split("/")
    for p in reversed(parts):
        if p in TASKS: return p
    for p in reversed(parts):
        m = re.match(r"(.+?)_s\d+$", p)
        if m and m[1] in TASKS: return m[1]
    raise KeyError(path)
def frames_of(path):
    cap = cv2.VideoCapture(path); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return fr
def inv(T): R = T[..., :3, :3]; t = T[..., :3, 3:]; Ri = np.swapaxes(R, -1, -2); out = np.tile(np.eye(4), T.shape[:-2] + (1, 1)); out[..., :3, :3] = Ri; out[..., :3, 3:] = -Ri @ t; return out
def rel(T): return T[1:] @ inv(T[:-1])   # w2c: T_{i+1} T_i^{-1}
def rot_deg(R): return np.degrees(np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))
OUT = json.load(open(a.out)) if os.path.exists(a.out) else {}
model = None
PROGRAM_FPS = 16.0   # baselines/programs/*.npz c2w are at 16 fps (161 frames = 10 s)
CALIB_MODEL = {"waypoint": "waypoint", "wp": "waypoint", "mg2": "mg2", "abot": "abot", "astra": "astra", "ours": "ours", "sf1p3b": "ours", "sf": "ours"}
def program_ref(path):
    """[--program-from-path] (run-dir name, P id, c2w [161,4,4], fps_native, fps source) for a clip at .../<arm>/<set>__<scene>__P<id>.../video.mp4"""
    d = os.path.dirname(os.path.abspath(path)); name = os.path.basename(d); m = re.search(r"__(P\d+p?)(?=__|_|$)", name)
    if not m: raise KeyError("no __P<id> in the run-dir name: " + d)
    c2w = np.load(os.path.join(a.programs_dir, m[1] + ".npz"))["c2w"].astype(np.float64); fps, src = None, ""
    tj = os.path.join(d, "timing.json")
    if os.path.exists(tj):
        try: fps = float(json.load(open(tj)).get("fps_native") or 0) or None; src = "timing.json"
        except Exception: fps = None
    if not fps:
        arm = os.path.basename(os.path.dirname(d)).split("_")[0].lower()
        try: fps = float(json.load(open(a.calib))[CALIB_MODEL[arm]]["fps"]); src = "calib.json[%s]" % CALIB_MODEL[arm]
        except Exception: fps = None
    if not fps: cap = cv2.VideoCapture(path); fps = float(cap.get(cv2.CAP_PROP_FPS)) or PROGRAM_FPS; cap.release(); src = "container"
    return name, m[1], c2w, fps, src
def score(path):
    global model
    if model is None: model = DepthAnything3.from_pretrained(a.model).to("cuda").eval()
    fr = frames_of(path); N = len(fr); extra = {}
    if a.program_from_path:
        task, pid, c2w, fps, src = program_ref(path); step = max(fps / a.sample_fps, 1.0)
        idx_pose = np.clip(np.round(np.arange(N) / fps * PROGRAM_FPS), 0, len(c2w) - 1).astype(int)   # frame i at t = i / fps_native -> program pose
        extra = dict(program=pid, fps_native=fps, fps_source=src, sample_fps=a.sample_fps, stride=round(step, 3), pose_index_max=int(idx_pose[-1]))
    else:
        task = a.ref_task or task_of(path); c2w = np.load(os.path.join(TASKS[task]["action"], "poses.npy")).astype(np.float64); step = a.stride
        n_pose = min(len(c2w), 161); idx_pose = np.round(np.linspace(0, n_pose - 1, len(fr))).astype(int)   # frame -> commanded pose index
    ref_w2c_all = inv(c2w[idx_pose]); rot, terr, tref, segs = [], 0.0, 0.0, []
    n_seg = a.segments
    if a.program_from_path and a.seg_seconds > 0: n_seg = max(a.segments, int(round(N / fps / a.seg_seconds)))   # [10/19] long clips only change
    if n_seg != a.segments: extra["n_segments"] = n_seg
    bounds = np.linspace(0, N, n_seg + 1).astype(int)
    for s in range(n_seg):
        if a.program_from_path: ii = sorted({int(round(x)) for x in np.arange(bounds[s], bounds[s + 1], step) if int(round(x)) < bounds[s + 1]})   # time grid at --sample-fps
        else: ii = list(range(bounds[s], bounds[s + 1], a.stride))
        if len(ii) < 4: continue
        pred = model.inference([fr[i] for i in ii], process_res=a.res)
        est = pred.extrinsics.astype(np.float64); ref = ref_w2c_all[ii]
        try: _, _, _, est_al = align_poses_umeyama(ref, est, return_aligned=True)
        except Exception as e: print("[rpe] align failed", path, s, e); continue
        rr, re_ = rel(ref), rel(est_al); dR = re_[:, :3, :3] @ np.swapaxes(rr[:, :3, :3], -1, -2); rot += list(rot_deg(dR))
        te = np.linalg.norm(re_[:, :3, 3] - rr[:, :3, 3], axis=-1).sum(); tr = np.linalg.norm(rr[:, :3, 3], axis=-1).sum(); terr += te; tref += tr
        segs.append(dict(rot=float(np.mean(rot_deg(dR))), trans=float(te / max(tr, 1e-9))))
    return dict(task=task, frames=N, rot=float(np.mean(rot)), trans=float(terr / max(tref, 1e-9)), segments=segs, **extra)
for spec in a.specs:
    label, pat = spec.split("=", 1); vids = sorted(glob.glob(pat)); assert vids, pat
    todo = [v for v in vids if a.redo or v not in OUT or OUT[v].get("label") != label]
    print("[rpe] %s: %d videos, %d to do" % (label, len(vids), len(todo)), flush=True)
    for v in todo:
        r = score(v); r["label"] = label; OUT[v] = r; json.dump(OUT, open(a.out, "w"), indent=1)
        print("[rpe] %s %s rot %.3f trans %.3f" % (label, os.path.relpath(v, S), r["rot"], r["trans"]), flush=True)
for label in sorted({r["label"] for r in OUT.values()}):
    rs = [r for r in OUT.values() if r["label"] == label]
    print("%-22s n=%3d Rot %.3f deg  Trans %.3f" % (label, len(rs), np.mean([r["rot"] for r in rs]), np.mean([r["trans"] for r in rs])))
