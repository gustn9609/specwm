"""[10/16 frozen baselines] Control program (PROTOCOL.md Section 2) -> camera path -> ABot-World per-block keys. Self-contained copy of the
`program_to_c2w` / `encode_abot` functions that also live in baselines/convert.py (kept here so baselines/abot/run.sh never depends on the shared file).

program_to_c2w(pid, fps, n_frames) -> c2w (n_frames, 4, 4) float64, OpenCV camera (x right, y down, z forward), frame 0 = identity, piecewise-constant
  2.5 s segments (the last segment holds beyond 10 s). Sign convention = the existing converters (tools/*/poses_to_*.py): with Rr = R_i^T R_{i+1},
  yaw = atan2(Rr[0,2], Rr[2,2]) > 0 for the program's positive yaw, pitch = asin(-Rr[1,2]) > 0 for positive pitch (camera up); translation is
  R_i^T (t_{i+1} - t_i) = [0, 0, fwd/fps] (camera z forward, 1 u/s).
encode_abot(c2w, first_block_frames=9, frames_per_block=12, n_blocks=None) -> list of per-block 8-key multi-hot [W,A,S,D,I,J,K,L]
  (ABot's native input: one held-key set per block of 3 latent frames = 12 decoded frames; the first block decodes to 9 frames because its first
  latent frame IS the start image). Rules copied from tools/abot/poses_to_keys.py: per block sum the frame-to-frame yaw / pitch (deg) and the
  translation in the camera frame; W/S (fwd/back) or A/D (strafe, the larger axis wins) when |translation| > 0.25 x median per-frame speed x
  (motions in the block); L when yaw > +2 deg, J when yaw < -2 deg; I when pitch > +2 deg, K when pitch < -2 deg. The L/J assignment follows
  poses_to_keys.py (yaw > 0 -> L); the physical pan direction of ABot's L key is recorded in baselines/calib.json ("abot") from its own output.
usage: python baselines/abot/encode_abot.py P2 [--blocks 11] [--fps 12] [--out keys.json]   -> prints the key list (JSON) and a summary to stderr
"""
import sys, json, argparse, numpy as np

SEG_S = 2.5
PROGRAMS = {  # per 2.5 s segment: (yaw deg/s, pitch deg/s, forward u/s)   (PROTOCOL.md Section 2 table)
    "P1": [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)],
    "P2": [(18, 0, 0), (18, 0, 0), (18, 0, 0), (18, 0, 0)],
    "P3": [(12, 0, 1), (12, 0, 1), (12, 0, 1), (12, 0, 1)],
    "P4": [(18, 0, 1), (-18, 0, 1), (-18, 0, 1), (18, 0, 1)],
    "P5": [(0, 0, 1), (0, 0, 0), (0, 0, -1), (0, 0, 1)],
    "P2p": [(16, 0, 0), (16, 0, 0), (16, 0, 0), (16, 0, 0)],                 # 10/17: P2 / P4 at 16 deg/s (MG2's cap saturates 18 deg/s), same rows as convert.PROGRAMS
    "P4p": [(16, 0, 1), (-16, 0, 1), (-16, 0, 1), (16, 0, 1)],
}
KEYS = "WASDIJKL"


def program_to_c2w(pid, fps, n_frames):
    segs = PROGRAMS[pid]; T = np.eye(4); out = [T.copy()]
    for i in range(n_frames - 1):
        yaw, pitch, fwd = segs[min(int((i / fps) // SEG_S), len(segs) - 1)]   # motion i -> i+1 belongs to the segment of frame i's time
        dy, dp = np.radians(yaw / fps), np.radians(pitch / fps)
        Ry = np.array([[np.cos(dy), 0, np.sin(dy)], [0, 1, 0], [-np.sin(dy), 0, np.cos(dy)]])
        Rx = np.array([[1, 0, 0], [0, np.cos(dp), -np.sin(dp)], [0, np.sin(dp), np.cos(dp)]])
        step = np.eye(4); step[:3, :3] = Ry @ Rx; step[:3, 3] = [0, 0, fwd / fps]
        T = T @ step; out.append(T.copy())
    return np.array(out)


def relative_motions(c2w):
    R = c2w[:, :3, :3]; t = c2w[:, :3, 3]; mv = []
    for i in range(len(c2w) - 1):
        Rr = R[i].T @ R[i + 1]; yaw = np.degrees(np.arctan2(Rr[0, 2], Rr[2, 2])); pitch = np.degrees(np.arcsin(np.clip(-Rr[1, 2], -1, 1)))
        mv.append((yaw, pitch, R[i].T @ (t[i + 1] - t[i])))
    return mv


def abot_block_frames(n_frames, first_block_frames=9, frames_per_block=12, n_blocks=None):
    """[start, end) decoded-frame ranges of ABot's blocks: block 0 = 9 frames (start image + 2 latent frames x 4), then 12 per block."""
    if n_blocks is None: n_blocks = 1 + int(np.ceil(max(n_frames - first_block_frames, 0) / frames_per_block))
    b = [(0, first_block_frames)]
    for k in range(1, n_blocks): b.append((first_block_frames + frames_per_block * (k - 1), first_block_frames + frames_per_block * k))
    return b


def encode_abot(c2w, first_block_frames=9, frames_per_block=12, n_blocks=None, rot_thr_deg=2.0, trans_frac=0.25):
    mv = relative_motions(np.asarray(c2w, dtype=np.float64)); N = len(mv)
    spd = np.array([np.linalg.norm(m[2]) for m in mv]); med = np.median(spd[spd > 0]) if (spd > 0).any() else None
    acts = []
    for s, e in abot_block_frames(len(c2w), first_block_frames, frames_per_block, n_blocks):
        seg = mv[min(s, N):min(e, N)]
        if not seg: acts.append(acts[-1] if acts else [1, 0, 0, 0, 0, 0, 0, 0]); continue   # beyond the pose array: hold the last keys
        yaw = sum(m[0] for m in seg); pitch = sum(m[1] for m in seg); tr = sum(m[2] for m in seg); k = dict.fromkeys(KEYS, 0)
        if med is not None and np.linalg.norm(tr) > trans_frac * med * len(seg):
            if abs(tr[2]) >= abs(tr[0]): k["W" if tr[2] > 0 else "S"] = 1
            else: k["D" if tr[0] > 0 else "A"] = 1
        if abs(yaw) > rot_thr_deg: k["L" if yaw > 0 else "J"] = 1
        if abs(pitch) > rot_thr_deg: k["I" if pitch > 0 else "K"] = 1
        acts.append([k[x] for x in KEYS])
    return acts


def frames_for_blocks(n_blocks, first_block_frames=9, frames_per_block=12): return first_block_frames + frames_per_block * (n_blocks - 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("pid"); ap.add_argument("--blocks", type=int, default=11); ap.add_argument("--fps", type=int, default=12); ap.add_argument("--out", default="")
    a = ap.parse_args(); n = frames_for_blocks(a.blocks); c2w = program_to_c2w(a.pid, a.fps, n); acts = encode_abot(c2w, n_blocks=a.blocks)
    mv = relative_motions(c2w); print("%s: %d frames at %d fps -> %d blocks; total yaw %.1f deg pitch %.1f deg; keys per block: %s" % (
        a.pid, n, a.fps, len(acts), sum(m[0] for m in mv), sum(m[1] for m in mv), " ".join("".join(k for k, v in zip(KEYS, b) if v) or "-" for b in acts)), file=sys.stderr)
    s = json.dumps(acts)
    if a.out: open(a.out, "w").write(s)
    print(s)
