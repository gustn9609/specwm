"""[10/16 frozen baselines] Control program (PROTOCOL Section 2) -> commanded c2w (baselines/convert.py, the one shared definition) -> Astra camera
input (per-LATENT-frame relative 3x4 pose + mask bit, the `sekai` modality of tools/astra/Astra/scripts/infer_demo.py).

Commanded trajectory: convert.program_to_c2w(pid, fps=20, n_frames) — OpenCV camera (x right, y down, z forward), frame 0 = identity, per-frame
yaw = atan2(Rr[0,2], Rr[2,2]) with Rr = R_i^T R_{i+1}; convert.py fixes POSITIVE YAW = PAN RIGHT (the formula every converter uses; the 10/9
'easy_lake' check). So P2 (+18 deg/s) is a 180-degree pan to the RIGHT for every model, Astra included.

Astra camera input (infer_demo.py: generate_sekai_camera_embeddings_sliding / prepare_framepack_sliding_window_with_camera_moe; training:
train_single.py::create_sekai_pose_embeddings):
  * one 13-vector per latent frame j (= 4 video frames at 20 fps): flattened 3x4 matrix M_j, then a mask bit (1 = history latent). The inference
    code overwrites the mask from the history length, so it is written as 0 here.
  * M_j = compute_relative_pose(E[4j], E[4j+4]) = E[4j+4] @ inv(E[4j]) on the dataset's 4x4 extrinsics -> the motion from latent j to latent j+1
    (video frame 4j -> 4j+4); entry 0 belongs to the condition latent (the start image).
  * semantics fixed by Astra's own presets (cam_type 1..7) and its icon overlay (overlay_controls): forward = M[2,3] < 0 (preset speed
    0.03 / latent frame); rotate LEFT = M[0,2] = +sin(yaw) (preset 0.03 rad / latent frame); pitch down = M[1,2] < 0.  Both consistent readings of
    the dataset extrinsics (OpenGL-style c2w, or OpenCV w2c) give the same M for pure pans / pure translations, and differ only in second-order
    terms for combined motion, so M is taken as the OpenCV camera-frame relative c2w conjugated by the OpenCV<->OpenGL flip S = diag(1,-1,-1):
        T_rel = inv(c2w[4j]) @ c2w[4j+4],   M_j = S @ T_rel[:3,:] @ S     (yaw sign flips: protocol right pan -> M[0,2] < 0 = Astra 'right';
                                                                         forward +z -> M[2,3] < 0 = Astra 'forward')
    Strafe (x) and pitch signs are not exercised by P1..P5 and are unverified. The pan direction on screen is checked on the P2 smoke run (ENV.md).
  * scale: Astra's pose units are those of its training data (SpatialVID / MegaSaM, roughly metric). The program's translation unit is mapped to
    Astra's OWN preset forward speed: 1 u/s == 0.03 per latent frame == 0.15 units/s at 20 fps (u_scale = 0.03 * fps / 4). Rotation is passed
    exactly (18 deg/s = 3.6 deg = 0.0628 rad per latent frame; Astra's presets use 0.03 rad = 8.6 deg/s).

Closest Astra presets (README table), recorded for reference only — run.sh always uses the exact program via the real-pose path:
  P1 forward -> cam_type 1;  P2 pan right -> cam_type 3 (Rotate Right, 8.6 deg/s);  P3 -> cam_type 5 (Forward + Rotate Right);
  P4 (R,L,L,R) -> cam_type 6 'S-shape' (left half then right half, ONE reversal, opposite order);  P5 (stop / back) -> none (cam_type 1 forward).

usage (CPU):  python program_to_astra.py P2 [--fps 20 --frames 201 --out P2.npz]
  -> npz: c2w (frames,4,4) commanded (reference for RPE), intrinsics (3,3) fx=fy=0.9W, astra (L,13) float32, fps, frames, u_scale, n_latents
"""
import argparse, importlib.util, json, os, numpy as np

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # baselines/
_spec = importlib.util.spec_from_file_location("convert", os.path.join(B, "convert.py")); convert = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(convert)
PROGRAMS = convert.PROGRAMS

ASTRA_FPS = 20                    # imageio writer fps in infer_demo.py
ASTRA_T = 4                       # video frames per latent frame (Wan2.1 VAE)
ASTRA_FWD_PER_LATENT = 0.03       # Astra preset forward speed per latent frame (cam_type 1/4/5/6)
S_FLIP = np.diag([1.0, -1.0, -1.0])
ASTRA_PRESET = {"P1": 1, "P2": 3, "P3": 5, "P4": 6, "P5": 1, "P2p": 3, "P4p": 6}     # 10/17: P2p/P4p added (16 deg/s variants, PROTOCOL §2)
ASTRA_PRESET_NOTE = {"P1": "cam_type 1 Move Forward (0.03/latent frame)", "P2": "cam_type 3 Rotate Right In Place (8.6 deg/s vs 18)",
                     "P3": "cam_type 5 Forward + Rotate Right (8.6 deg/s vs 12)", "P4": "cam_type 6 S-shape: left then right, one reversal (P4: R,L,L,R)",
                     "P5": "no stop/back preset; cam_type 1 forward only",
                     "P2p": "cam_type 3 Rotate Right In Place (8.6 deg/s vs 16)", "P4p": "cam_type 6 S-shape: left then right, one reversal (P4p: R,L,L,R at 16 deg/s)"}


def intrinsics(width=832, height=480):
    return np.array([[0.9 * width, 0.0, width / 2.0], [0.0, 0.9 * width, height / 2.0], [0.0, 0.0, 1.0]])


def encode_astra(c2w, n_latents=None, u_scale=1.0):
    """(L,13) float32 Astra embedding from an OpenCV c2w sequence at 20 fps; entry j = motion latent j -> j+1 (frames 4j -> 4j+4)."""
    N = len(c2w)
    if n_latents is None:
        n_latents = (N - 1) // ASTRA_T + 1
    rows = []
    for j in range(n_latents):
        a, b = ASTRA_T * j, ASTRA_T * j + ASTRA_T
        if b < N:
            T_rel = np.linalg.inv(c2w[a]) @ c2w[b]
            M = S_FLIP @ np.concatenate([T_rel[:3, :3], T_rel[:3, 3:4] * u_scale], axis=1) @ np.diag([1.0, -1.0, -1.0, 1.0])
        else:
            M = np.eye(3, 4)                                   # beyond the program: identity, as the official code pads
        rows.append(np.concatenate([M.reshape(-1), [0.0]]))
    return np.asarray(rows, dtype=np.float32)


def decode_astra(emb):
    """inverse check: per-latent (yaw_deg in the protocol's sign, forward translation) from an Astra embedding."""
    out = []
    for row in emb:
        M = row[:12].reshape(3, 4); Rr = S_FLIP @ M[:, :3] @ S_FLIP; tr = S_FLIP @ M[:, 3]
        out.append((np.degrees(np.arctan2(Rr[0, 2], Rr[2, 2])), tr[2]))
    return np.asarray(out)


def _program_to_c2w_local(pid, fps, n_frames):
    """Local copy of convert.program_to_c2w (2026-10-16 22:11 version) so this file keeps working if convert.py changes; checked against it."""
    seg = PROGRAMS[pid]["seg"]; c2w = np.tile(np.eye(4), (n_frames, 1, 1))
    for i in range(n_frames - 1):
        k = min(int(np.floor((i / fps) / convert.SEG_S + 1e-9)), len(seg) - 1); fwd, yaw, pitch = seg[k]
        Rr = convert.Ry(yaw / fps) @ convert.Rx(pitch / fps); tr = np.array([0.0, 0.0, fwd / fps])
        c2w[i + 1, :3, :3] = c2w[i, :3, :3] @ Rr; c2w[i + 1, :3, 3] = c2w[i, :3, 3] + c2w[i, :3, :3] @ tr
    return c2w


def commanded_c2w(pid, fps, n_frames):
    """convert.program_to_c2w (the shared definition) in float64; falls back to the local copy with a warning if its interface changed."""
    local = _program_to_c2w_local(pid, fps, n_frames)
    try:
        r = convert.program_to_c2w(pid, fps, n_frames)
        c2w = np.asarray(r[0] if isinstance(r, tuple) else r, dtype=np.float64)
        if c2w.shape != local.shape or not np.allclose(c2w, local, atol=1e-4):
            print("WARNING program_to_astra: convert.program_to_c2w differs from the local copy; using convert.py's", flush=True)
        return c2w
    except Exception as e:
        print("WARNING program_to_astra: convert.program_to_c2w unusable (%s); using the local copy" % e, flush=True); return local


def build(pid, fps=ASTRA_FPS, n_frames=201, width=832, height=480, u_scale=None):
    """u_scale = Astra units per program unit per second; default (None) = Astra's preset forward speed (0.03/latent = 0.15 at 20 fps), the
    frozen value (calib_astra.json, verified 10/17); an explicit value is for re-calibration runs only (astra_gen.py --u-scale)."""
    if u_scale is None:
        u_scale = ASTRA_FWD_PER_LATENT * fps / ASTRA_T                  # Astra units per program unit (1 u/s == preset forward speed)
    n_lat = (n_frames - 1 + ASTRA_T - 1) // ASTRA_T                     # latent frames to generate (50 for 201 frames)
    c2w_ext = commanded_c2w(pid, fps, ASTRA_T * n_lat + ASTRA_T + 1)    # a few frames past the clip so the last entry is defined
    emb = encode_astra(c2w_ext, n_latents=n_lat + 1, u_scale=u_scale)  # entry 0 = condition latent, 1..n_lat = generated latents
    return dict(c2w=c2w_ext[:n_frames], intrinsics=intrinsics(width, height), astra=emb, fps=fps, frames=n_frames, u_scale=u_scale, n_latents=n_lat)


def preset_embedding(cam_type, n_latents):
    """Astra's own synthetic preset (cam_type 1..7), via the official function, for the 'their example' run."""
    import sys, torch
    A = "/home/hk28965/specwm/tools/astra/Astra"
    for p in (A, A + "/scripts"):
        if p not in sys.path: sys.path.insert(0, p)
    import infer_demo
    return infer_demo.generate_sekai_camera_embeddings_sliding(None, 0, 1, n_latents, 0, use_real_poses=False, cam_type=int(cam_type)).float().numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("program", choices=sorted(PROGRAMS)); ap.add_argument("--fps", type=int, default=ASTRA_FPS)
    ap.add_argument("--frames", type=int, default=201); ap.add_argument("--width", type=int, default=832); ap.add_argument("--height", type=int, default=480); ap.add_argument("--out", default=None)
    a = ap.parse_args(); d = build(a.program, a.fps, a.frames, a.width, a.height); dec = decode_astra(d["astra"]); mv = convert.frame_motion(d["c2w"])
    print(json.dumps(dict(program=a.program, name=PROGRAMS[a.program]["name"], frames=a.frames, fps=a.fps, latents=int(d["n_latents"]), total_yaw_deg=float(sum(m[0] for m in mv)),
                          path_u=float(sum(np.linalg.norm(m[2]) for m in mv)), astra_rows=int(len(d["astra"])), u_scale=float(d["u_scale"]),
                          astra_entry1=dict(M02=float(d["astra"][1, 2]), tz=float(d["astra"][1, 11]), yaw_deg=float(dec[1, 0])),
                          roundtrip_yaw_sum_deg=float(dec[:-1, 0].sum()), preset=ASTRA_PRESET.get(a.program, "none"),
                          preset_note=ASTRA_PRESET_NOTE.get(a.program, "no preset recorded for %s (reference only; run.sh uses the exact program)" % a.program))))
    if a.out:
        np.savez(a.out, program=np.array(a.program), **{k: np.asarray(v) for k, v in d.items()}); print("wrote", a.out)
