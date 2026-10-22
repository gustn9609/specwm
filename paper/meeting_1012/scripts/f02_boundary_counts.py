"""f02_boundary_counts: raw counts behind facts.md section 5 (boundary jumps) and the mid-chunk control, per arm.

For every arm of the f02 jump table this recomputes, from the videos, with the functions of scripts/boundary_jump.py
(imported, not copied):
  - k / N chunk boundaries with r > 1.5 (b = frame 8|9, then every 12 frames)
  - k / N mid-chunk control positions (b + 4, b + 8) with r > 1.5
and writes paper/meeting_1012/data/f02_boundary_counts.json, which f02_partial_commit.py reads.
Re-run:  CUDA_VISIBLE_DEVICES="" /home/hk28965/specwm/venv/bin/python paper/meeting_1012/scripts/f02_boundary_counts.py
"""
import contextlib
import glob
import importlib.util
import io
import json
import os
import sys

import numpy as np

ROOT = "/home/hk28965/specwm"
OUTDIR = os.path.join(ROOT, "lingbot-world-v2/specwm_out")
DST = os.path.join(ROOT, "paper/meeting_1012/data/f02_boundary_counts.json")
SEEDS3 = ("1234", "4321", "9999")

# facts.md section-5 row key -> list of video globs (relative to OUTDIR)
ARMS = {
    "teacher 832 (s1234)": ["cmp10/*/uniform4_s1234/video.mp4"],
    "704 separate": ["fused/*/dg704_s%s/cloud/spec.mp4" % s for s in SEEDS3],
    "704 fused": ["fused/*/fc704_s%s/cloud/spec.mp4" % s for s in SEEDS3],
    "704 fused + partial 1": ["fused/*/pc1_704_s%s/cloud/spec.mp4" % s for s in SEEDS3],
    "704 fused + partial 2": ["fused/*/pc2_704_s%s/cloud/spec.mp4" % s for s in SEEDS3],
    "832 fused (s1234)": ["fused/*/f832_n0_500_s1234/video.mp4"],
    "832 fused + partial 1 (s1234)": ["fused/*/pc1_832_s1234/cloud/spec.mp4"],
    "672 fused": ["fused/*/fc672_s%s/cloud/spec.mp4" % s for s in SEEDS3],
    "576 fused": ["fused/*/fc576_s%s/cloud/spec.mp4" % s for s in SEEDS3],
}


def load_boundary_jump():
    """Import scripts/boundary_jump.py with an empty argv (its table loop then does nothing) and return the module."""
    spec = importlib.util.spec_from_file_location("boundary_jump", os.path.join(ROOT, "scripts/boundary_jump.py"))
    mod = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod


def main():
    bj = load_boundary_jump()
    bj.cv2.setNumThreads(2)
    res = {}
    for key, globs in ARMS.items():
        files = sorted(f for g in globs for f in glob.glob(os.path.join(OUTDIR, g)))
        R, C, used = [], [], 0
        for f in files:
            d = bj.diffs(f)
            if len(d) < 40:
                continue
            used += 1
            R += bj.ratios(d, 0)
            C += bj.ratios(d, 4) + bj.ratios(d, 8)
        R, C = np.array(R), np.array(C)
        res[key] = dict(
            clips=used, globs=globs,
            boundary_k=int((R > 1.5).sum()), boundary_n=int(R.size), boundary_ratio_mean=round(float(R.mean()), 4),
            control_k=int((C > 1.5).sum()), control_n=int(C.size), control_ratio_mean=round(float(C.mean()), 4),
        )
        r = res[key]
        print("%-32s clips %2d  boundary %3d/%3d = %4.1f %%  control %3d/%3d = %4.1f %%" % (
            key, used, r["boundary_k"], r["boundary_n"], 100 * r["boundary_k"] / r["boundary_n"],
            r["control_k"], r["control_n"], 100 * r["control_k"] / r["control_n"]), flush=True)
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    with open(DST, "w") as fh:
        json.dump(dict(source="scripts/boundary_jump.py (diffs, ratios); threshold r > 1.5; videos under "
                              "lingbot-world-v2/specwm_out", arms=res), fh, indent=1)
    print("wrote", DST)


if __name__ == "__main__":
    main()
