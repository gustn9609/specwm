"""Convert one of our bench tasks' camera trajectories (poses.npy c2w, 16 fps) into ABot-World per-block keyboard actions: one 8-dim multi-hot
[W,A,S,D,I,J,K,L] per block of 12 frames (W/S forward/back, A/D strafe, I/K camera up/down, J/L camera left/right). Per block we sum the
frame-to-frame yaw / pitch (deg) and the translation in the camera frame; a key is set when the block's motion exceeds a threshold
(translation: 0.25 x median per-frame speed x 12; rotation: 2 deg). Lossy by construction (8 discrete keys vs a continuous camera path).
usage: python tools/abot/poses_to_keys.py TASK_ID OUT.json [--frames 161]   → prints the action list and a summary"""
import sys, json, numpy as np
T = {t["id"]: t for t in json.load(open("/home/hk28965/specwm/data/stevo/tasks_camera.json"))}; tid, out = sys.argv[1], sys.argv[2]; N = int(sys.argv[4]) if len(sys.argv) > 4 else 161
P = np.load(T[tid]["action"] + "/poses.npy")[:N]; R = P[:, :3, :3]; t = P[:, :3, 3]; mv = []
for i in range(N - 1):
    Rr = R[i].T @ R[i + 1]; yaw = np.degrees(np.arctan2(Rr[0, 2], Rr[2, 2])); pitch = np.degrees(np.arcsin(np.clip(-Rr[1, 2], -1, 1))); tr = R[i].T @ (t[i + 1] - t[i]); mv.append((yaw, pitch, tr))
spd = np.array([np.linalg.norm(m[2]) for m in mv]); thr = 0.25 * np.median(spd[spd > 0]) * 12 if (spd > 0).any() else 1e9
acts = []
for c in range(0, N - 1, 12):
    seg = mv[c:c + 12]; yaw = sum(m[0] for m in seg); pitch = sum(m[1] for m in seg); tr = sum(m[2] for m in seg); k = dict.fromkeys("WASDIJKL", 0)
    if np.linalg.norm(tr) > thr:
        if abs(tr[2]) >= abs(tr[0]): k["W" if tr[2] > 0 else "S"] = 1
        else: k["D" if tr[0] > 0 else "A"] = 1
    if abs(yaw) > 2: k["L" if yaw > 0 else "J"] = 1
    if abs(pitch) > 2: k["I" if pitch > 0 else "K"] = 1
    acts.append([k[x] for x in "WASDIJKL"])
json.dump(acts, open(out, "w")); c = np.array(acts).sum(0)
print(tid, len(acts), "blocks; key counts W/A/S/D/I/J/K/L =", c.tolist(), "; total yaw %.0f deg pitch %.0f deg" % (sum(m[0] for m in mv), sum(m[1] for m in mv)))
