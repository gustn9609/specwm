"""[baselines/sf1p3b] Build the task entry for one (image, control program) pair the way our arms see it (PROTOCOL §2/§3, baselines/make_tasks.py):
OUTDIR/task/{image.<ext>, poses.npy (N x 4 x 4 c2w, OpenCV camera: x right, y down, z forward, frame 0 = identity), intrinsics.npy (N x [fx, fy, cx, cy]
at the 832x480 reference frame = [416, 416, 416, 240], the convention of the existing bench tasks that binding_v2 rescales), caption.txt, task.json}.
PROGRAM = P1..P5 (uses the frozen baselines/programs/<P>.npz when it exists, else generates the PROTOCOL §2 table: 160 steps at 16 fps, yaw in deg/s
with positive yaw = atan2(Rr[0,2], Rr[2,2]) > 0 for Rr = R_i^T R_{i+1}, positive pitch = camera up, translation in the camera frame, z forward)
or a directory holding poses.npy [+ intrinsics.npy] (e.g. lingbot-world-v2/examples/03 = a recorded camera path).
usage: python make_task.py IMAGE PROGRAM OUTDIR [--prompt ""] [--fps 16] [--steps 160]   -> prints the task id on the last line"""
import os, sys, json, shutil, argparse, numpy as np
B = "/home/hk28965/specwm/baselines"
ap = argparse.ArgumentParser(); ap.add_argument("image"); ap.add_argument("program"); ap.add_argument("outdir"); ap.add_argument("--prompt", default="")
ap.add_argument("--fps", type=float, default=16.0); ap.add_argument("--steps", type=int, default=160); a = ap.parse_args()
TABLE = {   # id: 4 segments of 2.5 s: (forward speed u/s, yaw deg/s, pitch deg/s)
    "P1": [(1, 0, 0)] * 4, "P2": [(0, 18, 0)] * 4, "P3": [(1, 12, 0)] * 4, "P4": [(1, 18, 0), (1, -18, 0), (1, -18, 0), (1, 18, 0)], "P5": [(1, 0, 0), (0, 0, 0), (-1, 0, 0), (1, 0, 0)]}
def Ry(t): c, s = np.cos(t), np.sin(t); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def Rx(t): c, s = np.cos(t), np.sin(t); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def program_to_c2w(pid, fps, steps):
    seg = TABLE[pid]; dt = 1.0 / fps; R = np.eye(3); t = np.zeros(3); out = [np.eye(4)]
    for i in range(steps):
        v, yaw, pitch = seg[min(len(seg) - 1, int(i * len(seg) / steps))]
        t = t + R @ np.array([0.0, 0.0, v * dt]); R = R @ Ry(np.radians(yaw * dt)) @ Rx(np.radians(pitch * dt))
        M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t; out.append(M)
    return np.stack(out).astype(np.float32)
d = os.path.join(a.outdir, "task"); os.makedirs(d, exist_ok=True)
if os.path.isdir(a.program):   # a recorded path
    c2w = np.load(os.path.join(a.program, "poses.npy")).astype(np.float32); src = a.program; pid = os.path.basename(os.path.normpath(a.program))
    K = np.load(os.path.join(a.program, "intrinsics.npy")).astype(np.float32) if os.path.exists(os.path.join(a.program, "intrinsics.npy")) else None
else:
    pid = a.program; npz = "%s/programs/%s.npz" % (B, pid)
    if os.path.exists(npz): c2w = np.load(npz)["c2w"].astype(np.float32); src = npz
    else: c2w = program_to_c2w(pid, a.fps, a.steps); src = "PROTOCOL table (fallback: %s missing)" % npz
    K = None
if K is None: K = np.tile(np.array([416.0, 416.0, 416.0, 240.0], np.float32), (len(c2w), 1))
ext = os.path.splitext(a.image)[1].lower() or ".png"; img = os.path.join(d, "image" + ext)
if os.path.abspath(a.image) != os.path.abspath(img): shutil.copy(a.image, img)
np.save(d + "/poses.npy", c2w); np.save(d + "/intrinsics.npy", K); open(d + "/caption.txt", "w").write(a.prompt)
scene = os.path.splitext(os.path.basename(a.image))[0]; tid = "sf1p3b__%s__%s" % (scene, pid)
T = dict(id=tid, base=tid, image=img, video_WM="", camera_WM=a.prompt, pose="program", action=d, program=pid, scene=scene, set="sf1p3b", poses_from=src, n_poses=int(len(c2w)))
json.dump([T], open(d + "/task.json", "w"), indent=1); print(json.dumps({k: T[k] for k in ("poses_from", "n_poses")})); print(tid)
