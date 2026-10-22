#!/usr/bin/env python
"""baselines/convert.py — PROTOCOL.md §2: control programs P1..P5 (+ P2p, P4p) -> camera path (c2w per frame) -> every model's native input.

    program_to_c2w(pid, fps=16, n_frames=161, speed=1.0, W=832, H=480) -> (c2w [N,4,4] float32, intrinsics [N,4] = fx,fy,cx,cy with fx = fy = 0.5 W)
    encode_waypoint(c2w, K_yaw=68.0)                -> list of {mouse:[K*yaw, K*pitch], buttons:[W|S|A|D]} per 4-frame call (tools/waypoint/poses_to_actions.py rules)
    encode_mg2(c2w, K=0.0704, clip=0.1)             -> {keyboard:[[fwd,back,left,right]]*N, mouse:[[K*pitch, K*yaw]]*N}      (tools/mg2/poses_to_cond.py rules)
    encode_abot(c2w, frames_per_block=12, rot_thr=2.0, first_block_frames=9, n_blocks=11)
                                                    -> 8-key multi-hot [W,A,S,D,I,J,K,L] per block; block 0 = 9 decoded frames (start image + 2 latent
                                                       frames x 4), then 12 per block = baselines/abot/encode_abot.py (the runner); first_block_frames=None
                                                       reproduces tools/abot/poses_to_keys.py (12-frame blocks from frame 0)
    encode_astra(c2w, u_scale=0.15, n_latents=None) -> [L,13] float32: Astra 'sekai' rows (flattened 3x4 M_j + mask), M_j = S [inv(c2w[4j]) c2w[4j+4]] S
                                                       with S = diag(1,-1,-1) = baselines/astra/program_to_astra.py (the runner's encoder)

Pose conventions (identical to the three existing converters):  Rr = R_i^T R_{i+1};  yaw = atan2(Rr[0,2], Rr[2,2]);  pitch = asin(-Rr[1,2]);
tr = R_i^T (t_{i+1} - t_i) with z forward (OpenCV c2w: x right, y down, z forward).  Positive yaw rotates the forward axis toward camera +x, i.e. the
camera pans to its RIGHT (content flows left): ABot key L (its camera-right key), Waypoint +mouse-x, MG2 +mouse-yaw (camera_r = [0, +0.1]), Astra
M[0,2] < 0 ('Rotate Right').  All four baseline P2 smoke clips (Waypoint, MG2, ABot, Astra) pan right.  PROTOCOL.md §2 labels it RIGHT (10/17).

Calibration constants: `CALIB` below holds the ACTIVE constants (decisions) + their provenance; `calib.json` additionally keeps the raw measurements
written by the calibration scripts (baselines/waypoint/calib_yaw.py, baselines/mg2/calib_merge.py, ...).  `build` LOADS calib.json and MERGES it with
CALIB (see merge_calib: file keys are never dropped; measured keys keep the file's value; constants and documentation come from CALIB; every
disagreement is listed under calib.json['_merge']).  It never replaces the file blindly; the previous file is kept as calib.json.prev.

usage (non-interactive, CPU only):
  python baselines/convert.py validate                   # (a) round trip on every program at every model's native fps, equality with the runners'
                                                         #     own encoders (abot/encode_abot.py, astra/program_to_astra.py); (b) bit-exact reproduction
                                                         #     of the old tools/*/poses_to_*.py scripts on existing tasks
  python baselines/convert.py build                      # validate, then programs/<P>.npz, calib.json (merged), scenes/neutral/ (missing files only),
                                                         #     programs/validation.json, and the per-program summary
  python baselines/convert.py summary                    # print + re-check the contents of programs/<P>.npz
  python baselines/convert.py encode PID MODEL OUT       # MODEL in waypoint|mg2|abot|astra|ours; OUT = .json (or a dir for ours/astra)
  python baselines/convert.py export-ours PID OUT_DIR [--fps 16 --frames 161 --W 832 --H 480]   # poses.npy + intrinsics.npy for our loader
  python baselines/convert.py build-long [P6]            # [10/19 EXP 1] validate_long, then programs/P6.npz + programs/P6_validation.json ONLY (the frozen
                                                         #     programs/<P>.npz and calib.json are never rewritten: md5 checked before/after)
  python baselines/convert.py validate-long [P6]         # [10/19] checks of the 60-s program at every model's long layout (long_layout)
[10/19 EXP 1] P6 = 60-s long-rollout program (12 segments of 5 s, PROGRAMS['P6'] seg_s / clip_s); FROZEN_PROGRAMS = the 7 frozen 10-s programs that
validate / build / summary / abot_summary iterate; LONG_PROGRAMS = ['P6'] encoded by encode_long at long_layout(pid) (ours 969 poses -> 961 frames,
Waypoint 900 calls, MG2 729 -> 720 frames, ABot 61 blocks = 729 -> 720 frames; no Astra).
Run through /home/hk28965/specwm/scripts/envrun.sh python (plain python3 has no numpy).
"""
import argparse, copy, importlib.util, json, os, subprocess, sys, tempfile, time
import numpy as np

BASE = "/home/hk28965/specwm/baselines"
SPECWM = "/home/hk28965/specwm"
CALIB_PATH = os.path.join(BASE, "calib.json")
SEG_S = 2.5          # every program is piecewise-constant over 4 segments of 2.5 s
CLIP_S = 10.0
FX_OVER_W = 0.5      # PROTOCOL §2: fx = fy = 416 at 832x480 (90 deg horizontal FOV)

# ----------------------------------------------------------------------------------------------------------------------------------------
# programs: per 2.5 s segment (forward u/s, yaw deg/s, pitch deg/s).  Positive yaw = pan RIGHT (see the module docstring).
PROGRAMS = {
    "P1":  dict(name="forward",      seg=[(1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0)],        tests="look-ahead consistency, scale drift"),
    "P2":  dict(name="turn",         seg=[(0, 18, 0), (0, 18, 0), (0, 18, 0), (0, 18, 0)],    tests="180 deg pan (right): revisit / parallax, look-back"),
    "P3":  dict(name="forward+turn", seg=[(1, 12, 0), (1, 12, 0), (1, 12, 0), (1, 12, 0)],    tests="combined motion"),
    "P4":  dict(name="S-curve",      seg=[(1, 18, 0), (1, -18, 0), (1, -18, 0), (1, 18, 0)],  tests="input flips (2 reversals), state persistence"),
    "P5":  dict(name="stop-and-go",  seg=[(1, 0, 0), (0, 0, 0), (-1, 0, 0), (1, 0, 0)],       tests="sudden input change -> reaction frames; look-back"),
    # 10/17: MG2's mouse cap (0.1 = 17.04 deg/s at K 0.0704) saturates P2/P4 at 18 deg/s -> the same programs at 16 deg/s, unsaturated for every model
    "P2p": dict(name="turn-16",      seg=[(0, 16, 0), (0, 16, 0), (0, 16, 0), (0, 16, 0)],    tests="P2 at MG2's largest unsaturated rate: 160 deg pan (right), no translation"),
    "P4p": dict(name="S-curve-16",   seg=[(1, 16, 0), (1, -16, 0), (1, -16, 0), (1, 16, 0)],  tests="P4 with +-16 deg/s: 2 input flips without MG2 mouse saturation"),
    # [10/19 EXP 1, additive] 60-s long-rollout program: 12 segments of 5 s (seg_s / clip_s override SEG_S / CLIP_S for this program only; the 7 frozen
    # programs above keep 4 x 2.5 s and their programs/<P>.npz stay byte-identical: they are never rebuilt by `build-long`). Turns <= 16 deg/s (MG2's
    # cap 0.1 at K 0.0704 = 17.04 deg/s: unsaturated). Heading after each segment: 0, 80, 160, 160, 160, 160, 80, 0, -60, 0, 0, 0 deg.
    #   1 fwd | 2 fwd + right 16 | 3 right 16 in place | 4 fwd | 5 STOP | 6 BACK-UP (undoes 4: the t=30 s view should re-show the t=15 s view) |
    #   7 left 16 in place | 8 fwd + left 16 (heading back at 0 = the START heading at t=40 s: look-back toward the first frames' scene) |
    #   9 fwd + left 12 | 10 right 12 in place (input flip, heading 0 again) | 11 STOP facing the start heading | 12 fwd
    "P6":  dict(name="long-60s", seg_s=5.0, clip_s=60.0,
                seg=[(1, 0, 0), (1, 16, 0), (0, 16, 0), (1, 0, 0), (0, 0, 0), (-1, 0, 0), (0, -16, 0), (1, -16, 0), (1, -12, 0), (0, 12, 0), (0, 0, 0), (1, 0, 0)],
                tests="60-s long rollout: forward, right/left turns <= 16 deg/s (MG2 unsaturated), turn in place, stop (static stability), back-up that "
                      "retraces a forward segment (t=30 s revisits the t=15 s pose), return to the start heading at t=40 s (look-back consistency), "
                      "an S-flip at 12 deg/s, a stop facing the start heading; drift over time (per-10-s judge)"),
}
FROZEN_PROGRAMS = ["P1", "P2", "P3", "P4", "P5", "P2p", "P4p"]   # [10/19] the 7 frozen 10-s programs (validate / build / summary / abot_summary iterate these only)
LONG_PROGRAMS = ["P6"]                                           # [10/19] long programs: programs/<P>.npz written by `build-long` only


def seg_s_of(pid): return float(PROGRAMS[pid].get("seg_s", SEG_S))
def clip_s_of(pid): return float(PROGRAMS[pid].get("clip_s", CLIP_S))

# ----------------------------------------------------------------------------------------------------------------------------------------
# ACTIVE calibration constants (decisions) with their provenance.  Raw measurements are repeated here as defaults so that a lost calib.json can be
# rebuilt; when calib.json has them, the file's copy wins (FILE_OWNED).  Everything else here overrides the file (constants + documentation).
CALIB = {
    "version": "2026-10-17b (PROTOCOL v1; measured values merged 10/16-17; Waypoint K_yaw 68 FINAL fit, Astra translation VERIFIED, 7 programs incl. P2p/P4p)",
    "yaw_pitch_convention": {
        "yaw_deg": "atan2(Rr[0,2], Rr[2,2]) with Rr = R_i^T R_{i+1}, c2w OpenCV (x right, y down, z forward)",
        "pitch_deg": "asin(-Rr[1,2]); positive = camera up (ABot I, MG2 +mouse[0], Waypoint +mouse[1])",
        "translation": "tr = R_i^T (t_{i+1} - t_i); tr[2] > 0 forward (W), tr[0] > 0 strafe right (D)",
        "positive_yaw_maps_to": "ABot key L, Waypoint +mouse[0], MG2 +mouse[1], Astra M[0,2] < 0 (its 'Rotate Right' form)",
        "geometric_note": "positive yaw rotates the forward axis toward camera +x = the camera pans RIGHT (content flows left) in OpenCV c2w; PROTOCOL.md §2 "
                          "labels it RIGHT since 10/17 (the 10/16 text said 'left'). Verified on the baselines' own P2 outputs: Waypoint (DA3, "
                          "waypoint/smoke/calib_da3.json 'camera pans RIGHT'), MG2 (DA3, mg2/smoke/calib_P2_universal_0000.json sign_ok), ABot (L key = "
                          "web_client/config.py 右, smoke frames), Astra (astra/calib_astra.json direction_check: content moves LEFT = camera pans RIGHT). "
                          "Code: StEvo generator 'right-30' = +48 deg/s by this formula, Matrix-Game-2 utils/conditions.py camera_r = [0, +0.1], "
                          "tools/waypoint/run_waypoint.py '--mouse-x (right turn)', Astra cam_type 3 'Rotate Right' = Ry(-0.03) relative = our +yaw.",
        "source": "tools/waypoint/poses_to_actions.py, tools/mg2/poses_to_cond.py, tools/abot/poses_to_keys.py (read 2026-10-16)",
    },
    "ours": {
        "fps": 16, "n_frames_10s": 161, "n_frames_written": 161, "n_frames_generated": 165, "n_latents_generated": 42, "n_chunks": 14,
        "chunk_frames": 12, "latent_frames_per_chunk": 3,
        "frames_note": "lingbot-world-v2/specwm/binding_v2.py prepare() rounds the latent count DOWN to a multiple of the chunk size (3): 161 poses -> 41 -> 39 "
                       "latents = 153 frames = 9.5 s (the 10/16 sf1p3b smoke timing.json reports frames 153). A 10 s clip therefore needs 165 poses and "
                       "--frames 165 -> 42 latents = 14 chunks = 165 frames; the clip is trimmed to its first 161 frames (10 s) for the metrics. "
                       "programs/<P>.npz carries c2w (161, the RPE reference) and c2w_165 (the same program continued for 4 frames at the last segment's rate).",
        "kv_window": "18 latent frames (baselines/run_ours.sh --window 18); 12 when the output area is above 600000 px, i.e. at ABot's 1280x704 "
                     "(window 18 does not fit in 96 GB there, facts §42); Waypoint 1024x512, MG2 640x352 and Astra 832x480 run with 18",
        "intrinsics": "per frame [fx, fy, cx, cy] = [0.5 W, 0.5 W, W/2, H/2] = [416, 416, 416, 240] at the 832x480 reference frame (PROTOCOL §2; 90 deg "
                      "horizontal FOV; = lingbot-world-v2/examples/03 fx 415.5); our loader (wan/image2video.py) assumes intrinsics.npy is for 832x480 and "
                      "rescales to the generation size; baselines/make_tasks.py writes the same [416, 416, 416, 240]",
        "intrinsics_note": "the 10/16 build of programs/<P>.npz wrote fx = 0.9 W (748.8) by mistake (fixed 10/17); data/stevo/actions/* use fx = 502.9 (0.60 W)",
        "translation": "normalised per video: relative translations / max |t| (wan/utils/cam_utils.py compute_relative_poses normalize_trans=True), "
                       "so only the shape of the path matters for ours",
        "source": "PROTOCOL.md §2 + rule 8; lingbot-world-v2/wan/image2video.py L608-709; wan/utils/cam_utils.py L56-75; specwm/binding_v2.py L105-135; "
                  "baselines/run_ours.sh (window rule)",
    },
    "waypoint": {
        "K_yaw_units_per_deg_per_call": 68.0, "K_yaw": 68.0,
        "unit": "mouse-x units per degree of yaw per 4-frame call (15 calls/s); mouse = [K_yaw*dyaw_deg, K_yaw*dpitch_deg]; positive mouse x = camera pans right",
        "fps": 60, "frames_per_call": 4, "calls_10s": 150,
        "buttons": {"W": 87, "S": 83, "A": 65, "D": 68},
        "button_rule": "a button is set when |sum of the call's 4 per-frame camera-frame translations| > 4 * 0.25 * median(per-frame speed > 0); "
                       "W if tr_z > |tr_x|, S if -tr_z > |tr_x|, else D (tr_x > 0) / A",
        "mouse": "[K_yaw * yaw_deg_per_call, K_yaw * pitch_deg_per_call]; raw, unclipped (K 68: P2: 81.6 units/call, P2p: 72.6, P3: 54.4)",
        "boundary_note": "at 60 fps a 2.5 s segment is 37.5 calls, so calls 37 and 112 straddle the P4/P4p/P5 switches (their mouse/translation sums "
                         "to the average of both segments); the effective switch time for Waypoint is 2.533 s / 7.533 s",
        "source": "K_yaw 68 (10/17 FINAL): proportional least-squares fit total_yaw = c * mouse_units over eight 10-s P2 clips of highland_castle_loch "
                  "generated through baselines/waypoint/run.sh with 64-95 mouse units/call (K 53.6/64.7/66/69/79, seeds 1234/1235; DA3 pose recovery "
                  "baselines/waypoint/calib_yaw.py -> waypoint/smoke/calib_da3.json, fit waypoint/smoke/kyaw_fit.json): c = 2.214 deg/unit -> K = 150/c = "
                  "67.8 -> 68 (affine 67.8, power-law 67.9). Direct clips bracket it: K 66 -> 161/168 deg, K 69 -> 203/176, K 79 -> 220/207 (the 10/16 "
                  "one-clip proposal 79 over-rotates). The realised turn at a fixed input varies ~6 % rms between seeds and processes, so K is a fit, not "
                  "a one-clip value; expected P2 pan at K 68 = 181 +- 11 deg. Response ~linear in 64-95 units/call (exponent 0.97), ramps over the first "
                  "~2 s. fps: tools/waypoint/Waypoint-1.5-1B/config.yaml inference_fps 60, temporal_compression 4 (15 calls/s)",
        "k_yaw_previous": "118 (10/9 optical-flow calibration, paper/meeting_1012/research_mg2_waypoint_eval.md L166) — INVALID: those runs re-appended "
                          "the start image every call (baselines/waypoint/ENV.md)",
        "measured_on": "baselines/waypoint/smoke (highland_castle_loch, P2, seed 1234) + constant-mouse sweep; DA3 pose recovery (calib_yaw.py)",
        "date": "2026-10-16T22:40",
        "measurements": {   # compact copy of waypoint/smoke/calib_da3.json (the file's copy wins when present)
            "P2_k118": {"mouse_x_per_call": 141.6, "calls": 150.0, "yaw_total_deg": 269.68, "yaw_deg_per_call": 1.7979, "K_yaw_units_per_deg": 78.8,
                        "phase_shift_px_per_frame": -1.606, "da3_fx_px_at_full_width": 597.5, "flow_yaw_deg_per_call": 0.6161, "K_yaw_from_flow": 229.8},
            "m0": {"mouse_x_per_call": 0.0, "calls": 30.0, "yaw_total_deg": -0.02, "yaw_deg_per_call": -0.0007, "K_yaw_units_per_deg": -0.0,
                   "phase_shift_px_per_frame": 0.0, "da3_fx_px_at_full_width": 948.6, "flow_yaw_deg_per_call": -0.0001, "K_yaw_from_flow": -0.0},
            "m60": {"mouse_x_per_call": 60.0, "calls": 30.0, "yaw_total_deg": 24.12, "yaw_deg_per_call": 0.8039, "K_yaw_units_per_deg": 74.6,
                    "phase_shift_px_per_frame": -1.711, "da3_fx_px_at_full_width": 829.9, "flow_yaw_deg_per_call": 0.4726, "K_yaw_from_flow": 127.0},
            "m120": {"mouse_x_per_call": 120.0, "calls": 30.0, "yaw_total_deg": 34.02, "yaw_deg_per_call": 1.1339, "K_yaw_units_per_deg": 105.8,
                     "phase_shift_px_per_frame": -1.583, "da3_fx_px_at_full_width": 753.4, "flow_yaw_deg_per_call": 0.4816, "K_yaw_from_flow": 249.2},
            "m240": {"mouse_x_per_call": 240.0, "calls": 30.0, "yaw_total_deg": 71.03, "yaw_deg_per_call": 2.3678, "K_yaw_units_per_deg": 101.4,
                     "phase_shift_px_per_frame": -1.726, "da3_fx_px_at_full_width": 634.8, "flow_yaw_deg_per_call": 0.6231, "K_yaw_from_flow": 385.2},
            "m480": {"mouse_x_per_call": 480.0, "calls": 30.0, "yaw_total_deg": 139.0, "yaw_deg_per_call": 4.6333, "K_yaw_units_per_deg": 103.6,
                     "phase_shift_px_per_frame": -2.003, "da3_fx_px_at_full_width": 604.5, "flow_yaw_deg_per_call": 0.7594, "K_yaw_from_flow": 632.1},
        },
        "verification": "VERIFIED 10/17 on GPU 4 (baselines/waypoint/smoke/{wp__highland_castle_loch__P2_k79, batch_k, batch_k69, batch_k66}; fit in "
                        "smoke/kyaw_fit.json): P2 realised pan vs K: 53.6 -> 151, 64.7 -> 165, 66 -> 161/168, 69 -> 203/176, 79 -> 220/207 deg (commanded 180); "
                        "K = 68 is the fit through all of them (expected 181 deg, residual rms 11 deg = +-6 %); the file's copy (FILE_OWNED) wins when present",
    },
    "mg2": {
        "K_per_deg_per_frame": 0.0704, "K_flow_1009": 0.067, "K_per_deg_per_frame_da3": 0.0704, "clip": 0.1,
        "K_decision": "10/17: the ACTIVE K is the DA3 value 0.0704 measured on MG2's own P2 output (PROTOCOL rule 3: calibrate on the baseline's outputs); "
                      "the 10/9 optical-flow value 0.067 is kept as K_flow_1009 (0.1 mouse ~ 1.5 deg/frame, paper/meeting_1012/research_mg2_waypoint_eval.md "
                      "L243). The two differ by 5 %; at 0.0704 the 18 deg/s programs (P2, P4) exceed the 0.1 cap on every moving frame (1.5 deg/frame -> 0.1056), "
                      "so P2p/P4p (16 deg/s -> 0.0939, unsaturated) are run for all models (PROTOCOL §2)",
        "fps": 12, "n_frames_generated": 129, "n_frames_written": 120, "n_frames_10s": 120,
        "n_frames_note": "MG2 needs frames % 4 == 1 and latents % 3 == 0: 121 frames = 31 latents is not generable; 129 frames (33 latents, 10.67 s) are "
                         "generated and the first 120 (10.0 s at 12 fps) written (baselines/mg2/run_mg2.py --frames 129 --keep-frames 120)",
        "largest_unsaturated_yaw_deg_per_s": 17.04,
        "keyboard": "[fwd, back, left, right] one-hot per frame; set when |tr| > 0.25 * median(per-frame speed > 0); frame 0 = all zero",
        "mouse": "[K * pitch_deg_per_frame, K * yaw_deg_per_frame] clipped to +-clip; frame 0 = [0, 0]",
        "saturation_note": "K 0.0704: 18 deg/s = 1.5 deg/frame -> 0.1056 -> clipped to 0.1 on 100 % of the moving frames of P2/P4 (effective 17.04 deg/s, "
                           "reported in timing.json control.saturated); 16 deg/s (P2p/P4p) -> 0.0939, unsaturated; 12 deg/s (P3) -> 0.0704",
        "measurement_da3": {   # copy of baselines/mg2/smoke/calib_P2_universal_0000.json as merged by mg2/calib_merge.py (the file's copy wins when present)
            "date": "2026-10-16 23:50 UTC", "video": "/home/hk28965/specwm/baselines/mg2/smoke/universal_0000__P2/video.mp4", "program": "P2",
            "commanded_mouse_yaw": 0.1, "commanded_deg_per_s": 17.999998092651367, "realized_deg_per_frame": 1.4201, "realized_deg_per_s": 17.04,
            "realized_total_deg": 169.5, "per_segment_deg_per_frame": [1.4299, 1.4149, 1.3991, 1.5647], "sign_ok": True, "flow_median_px_per_frame": -9.633,
            "da3_model": "/home/hk28965/specwm/hf/DA3NESTED-GIANT-LARGE-1.1",
            "method": "DA3 (DA3NESTED-GIANT-LARGE-1.1) poses of the MG2 clip -> per-frame yaw via atan2(Rr[0,2],Rr[2,2]); K = |commanded mouse yaw| / "
                      "|realized deg per frame|; largest unsaturated rate = cap / K * fps",
            "program_saturated_at_K_0067": True,
        },
        "source": "K: DA3 on MG2's own P2 output (baselines/mg2/calib_da3.py -> smoke/calib_P2_universal_0000.json, merged by calib_merge.py); clip: the "
                  "official bench uses CAM_VALUE = 0.1 (Matrix-Game-2/utils/conditions.py, camera_r = [0, +0.1]); fps: official Matrix-Game-2/utils/visualize.py "
                  "writes 12 fps (the 10/9-10/12 runs wrote 16 fps); frames: baselines/mg2/run_mg2.py; rules: tools/mg2/poses_to_cond.py",
    },
    "abot": {
        "fps": 12, "first_block_frames": 9, "frames_per_block": 12, "blocks_10s": 11, "n_frames_generated": 129, "n_frames_written": 120,
        "block_frames": "block 0 = decoded frames [0, 9) (the start image + 2 latent frames x 4), block b >= 1 = [9 + 12 (b-1), 9 + 12 b); 11 blocks = 129 "
                        "frames (10.75 s), the first 120 (10.0 s) written (baselines/abot/bench_abot.py --blocks 11 --keep-frames 120)",
        "keys": "WASDIJKL", "key_meaning": "W/S forward/back, A/D strafe left/right, I/K camera up/down, J/L camera left/right (web_client/config.py)",
        "rot_thr_deg_per_block": 2.0,
        "trans_rule": "a move key when |block translation| > 0.25 * median(per-frame speed > 0) * (motions in the block); W/S if |tr_z| >= |tr_x| else D/A",
        "boundary_note": "blocks take the NET motion of their frames (tools/abot/poses_to_keys.py rule), so a block straddling a 2.5 s program switch keeps "
                         "the majority key: block 2 = frames [21, 33) holds 9 frames of segment 0 + 3 of segment 1, block 7 = [81, 93) 9 of segment 2 + 3 of "
                         "segment 3. The ABot input therefore switches at the block boundaries 33 / 57 / 93 = 2.75 / 4.75 / 7.75 s (frame index / 12 fps, "
                         "frame 0 = the start image at t = 0; counted from the first generated frame these are 2.67 / 4.67 / 7.67 s — abot/ENV.md's "
                         "'6.67 s' for the P5 S->W switch is a typo for 7.67). keys_per_program / effective_switches below are computed from the encoding",
        "source": "baselines/abot/encode_abot.py + bench_abot.py (the runner, 10/16); fps: tools/abot/web_client/config.py VIDEO_FPS = 12; block = 3 latent "
                  "frames = 12 decoded frames, first block 9 (abot/ENV.md)",
    },
    "astra": {
        "fps": 20, "n_frames_10s": 201, "n_frames_written": 201, "latent_stride": 4, "latents_for_201_frames": 50, "embedding_rows": 57, "embedding_rows_npz": 51,
        "latents_generated": 56, "latents_per_block": 8, "blocks": 7, "n_frames_generated": 225, "steps": 50, "resolution": [832, 480],
        "fps_source": "tools/astra/Astra/scripts/infer_demo.py: imageio.get_writer(output_path, fps=20); latent stride 4 (Wan2.1 VAE) -> 5 latent frames per second",
        "latents_note": "run.sh generates 7 FULL blocks of 8 latents (56 = 225 frames) and cuts the video to 201 frames (10.05 s); a partial 2-latent final "
                        "block (50 = 6x8+2) collapsed into noise in the 10/16 P2 smoke (astra/ENV.md). The program embedding handed to the official "
                        "sliding-window code has 57 rows = 1 + 56 (entry 0 = condition latent, 1..56 = the program continued over all 225 generated frames: "
                        "astra_gen.py calls program_to_astra.build(pid, 20, 225)); with a 51-row (201-frame) embedding the official code zero-pads on the CPU "
                        "and crashes at block 7 (torch.cat device mismatch, infer_demo.py L934), and a zero 3x4 is not 'no motion' (10/17). programs/<P>.npz "
                        "astra_emb keeps the 51-row 10 s version (program_to_astra.build(pid, 20, 201)) = rows 0..50 of the runner's 57 (checked by validate)",
        "pose_format": "one 13-vector per latent j: flattened 3x4 M_j + mask (0; the inference code overwrites it); M_j = E[4j+4] @ inv(E[4j]) on the "
                       "training extrinsics = motion latent j -> j+1 (train_single.py create_sekai_pose_embeddings; infer_demo.py real-pose path)",
        "mapping_from_protocol_c2w": "T_rel = inv(c2w[4j]) @ c2w[4j+4] (OpenCV camera-frame relative c2w); M_j = S T_rel[:3,:] S' with S = diag(1,-1,-1), "
                                     "S' = diag(1,-1,-1,1): protocol +yaw (pan right) -> M[0,2] < 0 = Astra 'Rotate Right' (cam_type 3 = Ry(-0.03)); forward +z -> "
                                     "M[2,3] < 0 = Astra 'Move Forward' (cam_type 1: M[2,3] = -0.03). Strafe / pitch signs not exercised by the programs (unverified)",
        "rotation": "exact: yaw_deg_per_latent = yaw_deg_per_s * 4 / 20 (P2: 3.6 deg = 0.0628 rad per latent; P2p: 3.2 deg; Astra's presets: 0.03 rad = 1.72 deg "
                    "per latent = 8.6 deg/s)",
        "translation_units_per_u": 0.15,
        "translation_rule": "1 u/s == Astra's own preset forward speed 0.03 per latent step (cam_type 1/4/5/6 forward_speed) = 0.03 * 20 / 4 = 0.15 units/s, "
                            "i.e. 0.0075 per video frame at 20 fps",
        "translation_status": "VERIFIED 10/17 on Astra's OWN outputs with DA3 metric poses (astra/calib_da3.py -> astra/calib_astra.json, merged into "
                              "calib.json['astra'] as measurement_da3 / translation_status_da3 / u_scale_decision_da3): 1 Astra unit ~ 7.3 m; its preset forward "
                              "speed 0.03 per latent (= u_scale 0.15 for 1 u/s) realises 1.1 m/s walking speed on P1. Under combined motion (P3) Astra drops "
                              "almost all translation whatever the scale (0.00 / 0.11 / 0.17 m/s at 1x / 2x / 3x the preset), so its realised P3 forward/turn "
                              "ratio cannot be matched to ours by the unit; the INPUT ratio at 0.15 (0.092 m/deg) matches ours' realised 0.080 m/deg within "
                              "15 %, so u_scale 0.15 (Astra's own operating point, PROTOCOL rule 3) is KEPT and frozen",
        "translation_note": "(superseded 10/17; was: 0.12 per second at 16 fps) 1 u/s = 0.03 per latent step = 0.15 units/s at Astra's native 20 fps; "
                            "a turning + moving latent step is the chord of the 4-frame arc (P4: 0.029987 instead of 0.03)",
        "rotation_check": "VERIFIED 10/17 (DA3 poses + median DIS optical flow on Astra's own clips, rotation_check_da3): the radians-per-latent mapping is "
                          "exact while the generated view is still faithful (P2 first 2.5 s: 18.2 deg/s for 18 commanded; P3 first 5 s: 11.3-13.0 deg/s for 12) "
                          "and the realised pan rises to 1.5-1.9x once the content is hallucinated (P2 whole clip 272.6 deg instead of 180; P3 178.7 instead of "
                          "120) — treated as drift behaviour (scored by RPE-Rot), not re-calibrated: rotation_scale 1.0. Astra's own demo turns are 0.03 rad per "
                          "latent = 1.72 deg = 8.6 deg/s; P2 (3.6 deg per latent) is 2.1x the demo rate, P2p (3.2 deg) 1.9x",
        "direction_check": "VERIFIED 10/16 (frames) + 10/17 (DA3 + flow, direction_check_da3): P2 smoke (garden_1, seed 1234, baselines/astra/smoke/"
                           "garden_1_P2_lat50) — scene content moves LEFT on screen from the first generated frames (flow -15 px/frame) = camera pans RIGHT, as "
                           "the protocol's +yaw demands; Astra's own cam_type 4 ('left') example moves content RIGHT; P1 flow divergence +0.63 px/frame = forward",
        "presets_closest": {"P1": 1, "P2": 3, "P2p": 3, "P3": 5, "P4": "6 (left then right: opposite order, one reversal)", "P4p": "6", "P5": "none"},
        "source": "baselines/astra/program_to_astra.py + run.sh + astra_gen.py (the runner; 57-row embedding); baselines/astra/calib_astra.json + calib_da3.py "
                  "(10/17 DA3 calibration of translation / rotation / direction); tools/astra/Astra/scripts/infer_demo.py L295-655 (presets), L863-905 (overlay "
                  "semantics), L934 (zero-pad crash), train_single.py L366-385; commit 3fb82e1 (2026-10-16)",
    },
}

# keys that calibration scripts write into calib.json: the file's value always wins over the copy above
FILE_OWNED = {"measurements", "measurement_da3", "verification", "date", "measured_on", "source_previous", "k_yaw_previous",
              "K_per_deg_per_frame_da3", "K_per_deg_per_frame_10_09", "largest_unsaturated_yaw_deg_per_s", "translation_units_per_u_da3"}
MERGE_POLICY = ("calib.json = merge(file, convert.CALIB): every key of the file survives (keys unknown to convert.py are kept as they are); keys in "
                "FILE_OWNED (raw measurements: %s) keep the file's value; all other keys (the ACTIVE constants = decisions, and documentation) take "
                "convert.CALIB's value, so re-calibrating a constant means recording the measurement in the file AND changing CALIB; every disagreement "
                "is listed in _merge.decisions and the previous file is kept as calib.json.prev" % ", ".join(sorted(FILE_OWNED)))


# ----------------------------------------------------------------------------------------------------------------------------------------
def Ry(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def Rx(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def n_frames_for(fps, seconds=CLIP_S):
    return int(round(seconds * fps)) + 1


def intrinsics_for(W, H, n):
    """fx = fy = 0.5 W, cx = W/2, cy = H/2 (PROTOCOL §2: [416, 416, 416, 240] at 832x480) repeated per frame -> [n, 4] float32."""
    return np.tile(np.array([FX_OVER_W * W, FX_OVER_W * W, W / 2.0, H / 2.0], dtype=np.float32), (n, 1))


def program_to_c2w(pid, fps=16, n_frames=161, speed=1.0, W=832, H=480):
    """Integrate the piecewise-constant program into OpenCV c2w poses (frame 0 = identity).  Step i (t = i/fps) uses the segment
    floor(t / 2.5); beyond 10 s the last segment holds.  Returns (c2w [n_frames,4,4] float32, intrinsics [n_frames,4] float32 for a W x H image).
    [10/19] a program may set its own segment length (PROGRAMS[pid]['seg_s'], P6: 5 s); the 7 frozen programs use SEG_S = 2.5 exactly as before."""
    seg = PROGRAMS[pid]["seg"]; seg_len = seg_s_of(pid)
    c2w = np.tile(np.eye(4, dtype=np.float64), (n_frames, 1, 1))
    for i in range(n_frames - 1):
        k = min(int(np.floor((i / fps) / seg_len + 1e-9)), len(seg) - 1)
        fwd, yaw, pitch = seg[k]
        Rr = Ry(yaw / fps) @ Rx(pitch / fps)
        tr = np.array([0.0, 0.0, fwd * speed / fps])
        R, t = c2w[i, :3, :3], c2w[i, :3, 3]
        c2w[i + 1, :3, :3] = R @ Rr
        c2w[i + 1, :3, 3] = t + R @ tr
    return c2w.astype(np.float32), intrinsics_for(W, H, n_frames)


def program_per_second(pid, fps=16):
    """(yaw_deg, fwd_u) per whole second of the program, integrated the way program_to_c2w steps it (a second straddling a 2.5 s switch mixes both segments)."""
    seg = PROGRAMS[pid]["seg"]; yaw, fwd = [], []
    for s in range(int(CLIP_S)):
        ks = [min(int(np.floor((i / fps) / SEG_S + 1e-9)), len(seg) - 1) for i in range(fps * s, fps * s + fps)]
        yaw.append(round(sum(seg[k][1] for k in ks) / fps, 6)); fwd.append(round(sum(seg[k][0] for k in ks) / fps, 6))
    return yaw, fwd


def frame_motion(c2w):
    """Per-frame relative motion exactly as the three converters compute it: list of (yaw_deg, pitch_deg, tr[3]) for i -> i+1."""
    P = np.asarray(c2w)
    R = P[:, :3, :3]
    t = P[:, :3, 3]
    mv = []
    for i in range(len(P) - 1):
        Rr = R[i].T @ R[i + 1]
        yaw = np.degrees(np.arctan2(Rr[0, 2], Rr[2, 2]))
        pitch = np.degrees(np.arcsin(np.clip(-Rr[1, 2], -1, 1)))
        tr = R[i].T @ (t[i + 1] - t[i])
        mv.append((yaw, pitch, tr))
    return mv


def _speed_thr(mv, factor=0.25):
    spd = np.array([np.linalg.norm(m[2]) for m in mv])
    return factor * np.median(spd[spd > 0]) if (spd > 0).any() else 1e9


# ----------------------------------------------------------------------------------------------------------------------------------------
W_, S_, A_, D_ = 87, 83, 65, 68
KEYS8 = "WASDIJKL"


def encode_waypoint(c2w, K_yaw=68.0, frames_per_call=4):
    """tools/waypoint/poses_to_actions.py on a pose array: one {mouse, buttons} per call of `frames_per_call` frames."""
    mv = frame_motion(c2w)
    thr = _speed_thr(mv)
    acts = []
    for c in range(0, len(mv), frames_per_call):
        seg = mv[c:c + frames_per_call]
        yaw = sum(m[0] for m in seg); pitch = sum(m[1] for m in seg); tr = sum(m[2] for m in seg); b = []
        if np.linalg.norm(tr) > frames_per_call * thr:
            if tr[2] > abs(tr[0]): b.append(W_)
            elif -tr[2] > abs(tr[0]): b.append(S_)
            elif tr[0] > 0: b.append(D_)
            else: b.append(A_)
        acts.append(dict(mouse=[float(K_yaw * yaw), float(K_yaw * pitch)], buttons=b))
    return acts


def encode_mg2(c2w, K=0.0704, clip=0.1):
    """tools/mg2/poses_to_cond.py on a pose array (+ optional clipping to +-clip; clip=None reproduces the old script bit for bit)."""
    mv = frame_motion(c2w)
    thr = _speed_thr(mv)
    kb = [[0, 0, 0, 0]]; ms = [[0.0, 0.0]]
    for yaw, pitch, tr in mv:
        k = [0, 0, 0, 0]
        if np.linalg.norm(tr) > thr:
            if tr[2] > abs(tr[0]): k[0] = 1
            elif -tr[2] > abs(tr[0]): k[1] = 1
            elif tr[0] > 0: k[3] = 1
            else: k[2] = 1
        kb.append(k)
        m = [float(K * pitch), float(K * yaw)]
        if clip is not None:
            m = [float(np.clip(v, -clip, clip)) for v in m]
        ms.append(m)
    return dict(keyboard=kb, mouse=ms)


def abot_block_frames(n_frames, first_block_frames=9, frames_per_block=12, n_blocks=None):
    """[start, end) decoded-frame ranges of ABot's blocks (= baselines/abot/encode_abot.py abot_block_frames)."""
    if n_blocks is None:
        n_blocks = 1 + int(np.ceil(max(n_frames - first_block_frames, 0) / frames_per_block))
    b = [(0, first_block_frames)]
    for k in range(1, n_blocks):
        b.append((first_block_frames + frames_per_block * (k - 1), first_block_frames + frames_per_block * k))
    return b


def encode_abot(c2w, frames_per_block=12, rot_thr=2.0, first_block_frames=9, n_blocks=11):
    """ABot per-block 8-key multi-hot [W,A,S,D,I,J,K,L].  Default = the runner's layout (baselines/abot/encode_abot.py: block 0 = `first_block_frames`
    decoded frames, then `frames_per_block`; blocks beyond the pose array hold the last keys).  first_block_frames=None = tools/abot/poses_to_keys.py
    (blocks of `frames_per_block` motions from frame 0, as many as the array has)."""
    mv = frame_motion(c2w)
    thr = _speed_thr(mv)
    if first_block_frames is None:
        blocks = [(c, c + frames_per_block) for c in range(0, len(mv), frames_per_block)]
    else:
        blocks = abot_block_frames(len(c2w), first_block_frames, frames_per_block, n_blocks)
    acts = []
    for s, e in blocks:
        seg = mv[min(s, len(mv)):min(e, len(mv))]
        if not seg:
            acts.append(acts[-1] if acts else [1, 0, 0, 0, 0, 0, 0, 0]); continue
        n_thr = frames_per_block if first_block_frames is None else len(seg)
        yaw = sum(m[0] for m in seg); pitch = sum(m[1] for m in seg); tr = sum(m[2] for m in seg); k = dict.fromkeys(KEYS8, 0)
        if thr < 1e8 and np.linalg.norm(tr) > thr * n_thr:
            if abs(tr[2]) >= abs(tr[0]): k["W" if tr[2] > 0 else "S"] = 1
            else: k["D" if tr[0] > 0 else "A"] = 1
        if abs(yaw) > rot_thr: k["L" if yaw > 0 else "J"] = 1
        if abs(pitch) > rot_thr: k["I" if pitch > 0 else "K"] = 1
        acts.append([k[x] for x in KEYS8])
    return acts


def abot_keys_str(acts):
    return " ".join("".join(k for k, v in zip(KEYS8, b) if v) or "-" for b in acts)


S_FLIP = np.diag([1.0, -1.0, -1.0])
ASTRA_FWD_PER_LATENT = 0.03   # Astra preset forward speed per latent frame (cam_type 1/4/5/6)


def encode_astra(c2w, u_scale=0.15, n_latents=None, latent_stride=4):
    """(L,13) float32 Astra 'sekai' embedding from an OpenCV c2w sequence; entry j = motion latent j -> j+1 (frames 4j -> 4j+4), M_j = S T_rel[:3,:] S'
    with the translation scaled by u_scale; rows beyond the array = identity (as the official code pads).  Bit-identical to
    baselines/astra/program_to_astra.py encode_astra (the runner)."""
    P = np.asarray(c2w, dtype=np.float64)
    N = len(P)
    if n_latents is None:
        n_latents = (N - 1) // latent_stride + 1
    rows = []
    for j in range(n_latents):
        a, b = latent_stride * j, latent_stride * j + latent_stride
        if b < N:
            T_rel = np.linalg.inv(P[a]) @ P[b]
            M = S_FLIP @ np.concatenate([T_rel[:3, :3], T_rel[:3, 3:4] * u_scale], axis=1) @ np.diag([1.0, -1.0, -1.0, 1.0])
        else:
            M = np.eye(3, 4)
        rows.append(np.concatenate([M.reshape(-1), [0.0]]))
    return np.asarray(rows, dtype=np.float32)


def decode_astra(emb):
    """inverse check: per-latent (yaw_deg in the protocol's sign, forward translation in Astra units) from an Astra embedding."""
    out = []
    for row in emb:
        M = np.asarray(row[:12], dtype=np.float64).reshape(3, 4); Rr = S_FLIP @ M[:, :3] @ S_FLIP; tr = S_FLIP @ M[:, 3]
        out.append((np.degrees(np.arctan2(Rr[0, 2], Rr[2, 2])), tr[2]))
    return np.asarray(out)


def export_ours(pid, out_dir, fps=16, n_frames=161, W=832, H=480, speed=1.0):
    """poses.npy + intrinsics.npy in the layout our loader reads (action_path)."""
    c2w, K = program_to_c2w(pid, fps, n_frames, speed, W, H)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "poses.npy"), c2w)
    np.save(os.path.join(out_dir, "intrinsics.npy"), K)
    return c2w, K


# ----------------------------------------------------------------------------------------------------------------------------------------
def long_layout(pid):
    """[10/19 EXP 1] per-model frame layout of a LONG program (clip_s seconds), the same rules as the 10-s layout (CALIB) applied to clip_s:
    ours      16 fps: n_written = 16 clip_s + 1 (961 for 60 s, the RPE reference c2w); latents = ceil((n_written - 1) / 4 + 1) rounded UP to a multiple of the
              chunk (3) -> 243 = 81 chunks -> n_generated = 4 (243 - 1) + 1 = 969 poses / frames (binding_v2.prepare rounds the latent count DOWN to a multiple
              of 3: 965 poses would give 241 -> 240 latents = 957 frames < 961, so 969 are needed); trimmed to 961 by ours_to_timing.py
    waypoint  60 fps, 4 frames / call: 15 clip_s calls (900), c2w at 60 fps with 60 clip_s + 1 poses, 60 clip_s frames written (3600)
    mg2       12 fps: n_written = 12 clip_s (720); n_generated = smallest N >= n_written + 1 with N % 4 == 1 and latents (N - 1) / 4 + 1 % 3 == 0 (729 = 183 latents)
    abot      12 fps: block 0 = 9 frames, then 12: blocks = 1 + ceil((n_written - 9) / 12) (61) -> n_generated = 9 + 12 (blocks - 1) (729), n_written 720
    astra     not defined for long programs (not part of EXP 1)"""
    cs = clip_s_of(pid); C = CALIB
    nw = int(round(cs * C["ours"]["fps"])) + 1; lat = (nw - 1) // 4 + 1; lat += (-lat) % C["ours"]["latent_frames_per_chunk"]
    ours = dict(fps=C["ours"]["fps"], n_frames_written=nw, n_latents_generated=lat, n_chunks=lat // C["ours"]["latent_frames_per_chunk"], n_frames_generated=4 * (lat - 1) + 1)
    wf, fpc = C["waypoint"]["fps"], C["waypoint"]["frames_per_call"]
    wp = dict(fps=wf, frames_per_call=fpc, n_calls=int(round(cs * wf / fpc)), n_poses=int(round(cs * wf)) + 1, n_frames_written=int(round(cs * wf)))
    mf = C["mg2"]["fps"]; mw = int(round(cs * mf)); mn = mw + 1
    while not (mn % 4 == 1 and ((mn - 1) // 4 + 1) % 3 == 0): mn += 1
    mg = dict(fps=mf, n_frames_written=mw, n_frames_generated=mn, n_latents=(mn - 1) // 4 + 1)
    af, fb, fpb = C["abot"]["fps"], C["abot"]["first_block_frames"], C["abot"]["frames_per_block"]; aw = int(round(cs * af)); nb = 1 + int(np.ceil(max(aw - fb, 0) / fpb))
    ab = dict(fps=af, first_block_frames=fb, frames_per_block=fpb, n_blocks=nb, n_frames_generated=fb + fpb * (nb - 1), n_frames_written=aw)
    return dict(clip_s=cs, seg_s=seg_s_of(pid), n_segments=len(PROGRAMS[pid]["seg"]), ours=ours, waypoint=wp, mg2=mg, abot=ab)


def encode_long(pid, model, speed=1.0, W=832, H=480):
    """[10/19 EXP 1] encode_program for a LONG program (PROGRAMS[pid] with clip_s): the same encoders and constants as the frozen 10-s programs, only the
    lengths come from long_layout(pid).  Returns (meta, native) like encode_program."""
    L = long_layout(pid); C = CALIB
    if model == "waypoint":
        w = L["waypoint"]; K = C["waypoint"]["K_yaw_units_per_deg_per_call"]
        c2w, _ = program_to_c2w(pid, w["fps"], w["n_poses"], speed, W, H); acts = encode_waypoint(c2w, K, w["frames_per_call"])
        return dict(model="waypoint", fps=w["fps"], frames_per_call=w["frames_per_call"], n_calls=len(acts), n_frames=len(c2w) - 1, K_yaw=K, clip_s=L["clip_s"],
                    call_time_s="c * frames_per_call / fps", actions=acts), acts
    if model == "mg2":
        m = L["mg2"]; K, cap = C["mg2"]["K_per_deg_per_frame"], C["mg2"]["clip"]
        c2w, _ = program_to_c2w(pid, m["fps"], m["n_frames_generated"], speed, W, H)      # = baselines/mg2/run_mg2.py encode_program(pid, 12, 729, K, 0.1)
        cond = encode_mg2(c2w, K, cap); raw = np.abs(np.array(encode_mg2(c2w, K, None)["mouse"], np.float64))
        moving = int((raw.max(1) > 1e-9).sum()); sat = int((raw.max(1) > cap + 1e-9).sum())
        return dict(model="mg2", fps=m["fps"], n_frames_generated=len(cond["keyboard"]), n_frames_written=m["n_frames_written"], K=K, clip=cap, clip_s=L["clip_s"],
                    saturated_frames=sat, moving_frames=moving, saturated_frac_of_moving=(sat / moving if moving else 0.0), max_abs_mouse_unclipped=float(raw.max()),
                    run_flags="--frames %d --keep-frames %d" % (m["n_frames_generated"], m["n_frames_written"]), **cond), cond
    if model == "abot":
        b = L["abot"]; n = b["n_frames_generated"]
        c2w, _ = program_to_c2w(pid, b["fps"], n, speed, W, H)
        acts = encode_abot(c2w, b["frames_per_block"], C["abot"]["rot_thr_deg_per_block"], b["first_block_frames"], b["n_blocks"])
        bf = abot_block_frames(n, b["first_block_frames"], b["frames_per_block"], b["n_blocks"]); sw = [i for i in range(1, len(acts)) if acts[i] != acts[i - 1]]
        return dict(model="abot", fps=b["fps"], first_block_frames=b["first_block_frames"], frames_per_block=b["frames_per_block"], n_blocks=len(acts), n_frames_generated=n,
                    n_frames_written=b["n_frames_written"], clip_s=L["clip_s"], block_frames=[list(x) for x in bf], keys=KEYS8, keys_str=abot_keys_str(acts), switch_blocks=sw,
                    switch_frames=[bf[i][0] for i in sw], switch_times_s=[round(bf[i][0] / b["fps"], 4) for i in sw],
                    run_flags="--blocks %d --keep-frames %d (actions passed per job: bench_abot.py's own encode_abot.py has no long programs)" % (b["n_blocks"], b["n_frames_written"]),
                    actions=acts), acts
    if model == "ours":
        o = L["ours"]
        c2w, K = program_to_c2w(pid, o["fps"], o["n_frames_written"], speed, W, H); c2wg, _ = program_to_c2w(pid, o["fps"], o["n_frames_generated"], speed, W, H)
        return dict(model="ours", fps=o["fps"], n_frames_written=len(c2w), n_frames_generated=len(c2wg), n_latents_generated=o["n_latents_generated"], n_chunks=o["n_chunks"],
                    intrinsics=K[0].tolist(), kv_window=C["ours"]["kv_window"], clip_s=L["clip_s"],
                    frames_note="long program: %d poses (c2w_%d) = %d latents = %d chunks of 3 are generated and trimmed to the first %d frames (%.0f s) by "
                                "baselines/ours_to_timing.py" % (o["n_frames_generated"], o["n_frames_generated"], o["n_latents_generated"], o["n_chunks"], o["n_frames_written"], L["clip_s"])), (c2w, K, c2wg)
    raise ValueError("long program %s: model %s not defined (astra is not part of EXP 1)" % (pid, model))


def encode_program(pid, model, speed=1.0, W=832, H=480):
    """Program -> native encoding at the model's native fps / clip length (CALIB), wrapped with meta.  Returns (meta_dict, native)."""
    if pid in LONG_PROGRAMS: return encode_long(pid, model, speed, W, H)   # [10/19] long programs: own lengths; the 10-s path below is unchanged
    if model == "waypoint":
        C = CALIB["waypoint"]; fps, fpc, K = C["fps"], C["frames_per_call"], C["K_yaw_units_per_deg_per_call"]
        c2w, _ = program_to_c2w(pid, fps, n_frames_for(fps), speed, W, H)
        acts = encode_waypoint(c2w, K, fpc)
        return dict(model="waypoint", fps=fps, frames_per_call=fpc, n_calls=len(acts), n_frames=len(c2w) - 1, K_yaw=K,
                    call_time_s="c * frames_per_call / fps", actions=acts), acts
    if model == "mg2":
        C = CALIB["mg2"]; fps, N, K, cap = C["fps"], C["n_frames_generated"], C["K_per_deg_per_frame"], C["clip"]
        c2w, _ = program_to_c2w(pid, fps, N, speed, W, H)                 # = baselines/mg2/run_mg2.py encode_program(pid, 12, 129, K, 0.1)
        cond = encode_mg2(c2w, K, cap)
        raw = np.abs(np.array(encode_mg2(c2w, K, None)["mouse"], np.float64))
        moving = int((raw.max(1) > 1e-9).sum())
        sat = int((raw.max(1) > cap + 1e-9).sum())
        return dict(model="mg2", fps=fps, n_frames_generated=len(cond["keyboard"]), n_frames_written=C["n_frames_written"], K=K, clip=cap,
                    saturated_frames=sat, moving_frames=moving, saturated_frac_of_moving=(sat / moving if moving else 0.0),
                    max_abs_mouse_unclipped=float(raw.max()), **cond), cond
    if model == "abot":
        C = CALIB["abot"]; fps, fb, fpb, nb = C["fps"], C["first_block_frames"], C["frames_per_block"], C["blocks_10s"]
        n = fb + fpb * (nb - 1)                                            # 129 = bench_abot.py frames_for_blocks(11)
        c2w, _ = program_to_c2w(pid, fps, n, speed, W, H)
        acts = encode_abot(c2w, fpb, C["rot_thr_deg_per_block"], fb, nb)
        bf = abot_block_frames(n, fb, fpb, nb)
        sw = [i for i in range(1, len(acts)) if acts[i] != acts[i - 1]]
        return dict(model="abot", fps=fps, first_block_frames=fb, frames_per_block=fpb, n_blocks=len(acts), n_frames_generated=n,
                    n_frames_written=C["n_frames_written"], block_frames=[list(b) for b in bf], keys=KEYS8, keys_str=abot_keys_str(acts),
                    switch_blocks=sw, switch_frames=[bf[i][0] for i in sw], switch_times_s=[round(bf[i][0] / fps, 4) for i in sw], actions=acts), acts
    if model == "astra":
        C = CALIB["astra"]; fps, T, nf = C["fps"], C["latent_stride"], C["n_frames_10s"]
        u = ASTRA_FWD_PER_LATENT * fps / T                                  # 0.15: computed as program_to_astra.build does (bit-equality)
        n_lat = (nf - 1 + T - 1) // T                                       # 50 latents cover 201 frames
        c2w_ext, _ = program_to_c2w(pid, fps, T * n_lat + T + 1, speed, W, H)
        emb = encode_astra(c2w_ext, u, n_lat + 1, T)
        dec = decode_astra(emb)
        meta = dict(model="astra", fps=fps, n_frames_written=nf, latents_for_clip=n_lat, latents_generated=C["latents_generated"], latents_per_block=C["latents_per_block"],
                    embedding_rows=int(len(emb)), embedding_rows_runner=C["embedding_rows"], u_scale=u, latent_stride=T,
                    format="rows = program_to_astra.build(pid, 20, 201)['astra'] = the first 51 rows of the runner's 57-row input (astra_gen.py builds "
                           "build(pid, 20, 225) for all 225 generated frames); npz astra_emb [51,13], astra_c2w [201,4,4] commanded c2w at 20 fps",
                    per_latent_yaw_deg=[round(float(x), 4) for x in dec[:, 0]], per_latent_tz_units=[round(float(x), 4) for x in dec[:, 1]])
        return meta, dict(c2w=c2w_ext[:nf], emb=emb)
    if model == "ours":
        C = CALIB["ours"]; fps = C["fps"]
        c2w, K = program_to_c2w(pid, fps, C["n_frames_written"], speed, W, H)
        c2w165, _ = program_to_c2w(pid, fps, C["n_frames_generated"], speed, W, H)
        return dict(model="ours", fps=fps, n_frames_written=len(c2w), n_frames_generated=len(c2w165), n_latents_generated=C["n_latents_generated"],
                    n_chunks=C["n_chunks"], intrinsics=K[0].tolist(), kv_window=C["kv_window"], frames_note=C["frames_note"]), (c2w, K, c2w165)
    raise ValueError(model)


# ----------------------------------------------------------------------------------------------------------------------------------------
def _short(v):
    s = json.dumps(v, default=str)
    return s if len(s) <= 70 else s[:67] + "..."


COMPUTED = {"abot": ("keys_per_program", "effective_switches")}   # recomputed by build_calib from the encodings (abot_summary)


def merge_calib(file_d, code_d=CALIB):
    """file ∪ code (see MERGE_POLICY).  Returns (merged, decisions)."""
    out = copy.deepcopy(code_d); dec = []
    for sec, fv in (file_d or {}).items():
        if sec.startswith("_"):
            continue
        if sec not in out:
            out[sec] = copy.deepcopy(fv); dec.append("kept file section '%s' (unknown to convert.CALIB)" % sec); continue
        if not isinstance(fv, dict) or not isinstance(out[sec], dict):
            if fv != out[sec]:
                dec.append("%s: code wins: %s  [file had %s]" % (sec, _short(out[sec]), _short(fv)))
            continue
        for k, v in fv.items():
            if k in COMPUTED.get(sec, ()):
                continue
            if k not in out[sec]:
                out[sec][k] = copy.deepcopy(v); dec.append("kept file key %s.%s (unknown to convert.CALIB)" % (sec, k))
            elif v != out[sec][k]:
                if k in FILE_OWNED:
                    dflt = out[sec][k]; out[sec][k] = copy.deepcopy(v); dec.append("%s.%s: file (measured) wins: %s  [code default %s]" % (sec, k, _short(v), _short(dflt)))
                else:
                    dec.append("%s.%s: code wins: %s  [file had %s]" % (sec, k, _short(out[sec][k]), _short(v)))
    return out, dec


def calib_warnings(c):
    """active constants vs the latest measurements (printed by build; recorded in _merge.warnings)."""
    w = []
    wp = c["waypoint"]
    if wp["K_yaw"] != wp["K_yaw_units_per_deg_per_call"]:
        w.append("waypoint: K_yaw %s != K_yaw_units_per_deg_per_call %s" % (wp["K_yaw"], wp["K_yaw_units_per_deg_per_call"]))
    # 10/17: the reference is the fit over eight 10-s P2 clips (waypoint/smoke/kyaw_fit.json), not the single K-118 clip
    fit_path = os.path.join(BASE, "waypoint", "smoke", "kyaw_fit.json")
    try:
        k_ref, ref = float(json.load(open(fit_path))["K_proportional"]), "the 10/17 P2 fit (waypoint/smoke/kyaw_fit.json)"
    except Exception:
        m = (wp.get("measurements") or {}).get("P2_k118"); k_ref, ref = (float(m["K_yaw_units_per_deg"]) if m else None), "the P2 DA3 measurement"
    if k_ref and abs(wp["K_yaw"] - k_ref) > 0.05 * k_ref:
        w.append("waypoint: active K_yaw %s is > 5 %% from %s %s" % (wp["K_yaw"], ref, k_ref))
    mg = c["mg2"]
    if "K_per_deg_per_frame_da3" in mg and abs(mg["K_per_deg_per_frame"] - mg["K_per_deg_per_frame_da3"]) > 1e-4:
        w.append("mg2: active K %s != DA3 measurement %s" % (mg["K_per_deg_per_frame"], mg["K_per_deg_per_frame_da3"]))
    md = mg.get("measurement_da3") or {}
    if md and "largest_unsaturated_yaw_deg_per_s" in mg and abs(mg["largest_unsaturated_yaw_deg_per_s"] - md.get("realized_deg_per_s", mg["largest_unsaturated_yaw_deg_per_s"])) > 0.05:
        w.append("mg2: largest_unsaturated_yaw_deg_per_s %s != measurement realized_deg_per_s %s" % (mg["largest_unsaturated_yaw_deg_per_s"], md.get("realized_deg_per_s")))
    return w


def abot_summary():
    """keys per program + effective switch times, computed from the runner layout (recorded in calib.json['abot'])."""
    keys, sw = {}, {}
    for pid in FROZEN_PROGRAMS:   # [10/19] the 7 frozen programs only (long programs: build-long)
        m, _ = encode_program(pid, "abot")
        keys[pid] = m["keys_str"]
        sw[pid] = dict(blocks=m["switch_blocks"], frames=m["switch_frames"], times_s=m["switch_times_s"])
    return dict(keys_per_program=keys, effective_switches=sw)


def build_calib(path=CALIB_PATH):
    """LOAD calib.json, MERGE with CALIB (never replace), add the computed ABot schedule, write; previous file -> calib.json.prev."""
    old = json.load(open(path)) if os.path.exists(path) else {}
    merged, dec = merge_calib(old, CALIB)
    merged["abot"].update(abot_summary())
    warn = calib_warnings(merged)
    merged["_merge"] = dict(policy=MERGE_POLICY, last_build=time.strftime("%Y-%m-%d %H:%M:%S %Z"), previous_file="calib.json.prev",
                            decisions=dec, warnings=warn)
    if old:
        json.dump(old, open(path + ".prev", "w"), indent=1)
    json.dump(merged, open(path, "w"), indent=1)
    for d in dec:
        print("  calib merge:", d)
    for w in warn:
        print("  calib WARNING:", w)
    return merged, dec, warn


def build_programs(calib, out_dir=os.path.join(BASE, "programs")):
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for pid, prog in ((p, PROGRAMS[p]) for p in FROZEN_PROGRAMS):   # [10/19] never rebuilds a long program here
        om, (c2w, K, c2w165) = encode_program(pid, "ours")
        enc = {m: encode_program(pid, m) for m in ("waypoint", "mg2", "abot", "astra")}
        astra_meta, astra_nat = enc["astra"]
        meta = dict(id=pid, name=prog["name"], segments_s=SEG_S, segments=[dict(fwd_u_per_s=s[0], yaw_deg_per_s=s[1], pitch_deg_per_s=s[2]) for s in prog["seg"]],
                    tests=prog["tests"], sign="positive yaw = pan RIGHT (yaw = atan2(Rr[0,2], Rr[2,2]), Rr = R_i^T R_{i+1}, OpenCV c2w)",
                    c2w="OpenCV c2w, 16 fps, 161 frames = 10 s, frame 0 = identity, translation 1 u/s = 1/16 per frame; the RPE reference for every model",
                    c2w_165="the same program continued to 165 frames (42 latents = 14 chunks of 3 for our loader, which rounds latents down to a multiple of 3; "
                            "the clip is trimmed back to 161 frames)",
                    intrinsics="fx,fy,cx,cy per frame for 832x480: [416, 416, 416, 240] (fx = fy = 0.5 W, PROTOCOL §2)",
                    keys=dict(ours="c2w [161,4,4], c2w_165 [165,4,4], intrinsics [161,4]; JSON meta in 'ours'",
                              waypoint="JSON (60 fps, 4 frames/call, 150 calls; K_yaw %s)" % CALIB["waypoint"]["K_yaw_units_per_deg_per_call"],
                              mg2="JSON (12 fps, 129 frames generated, first 120 written; K %s, cap %s)" % (CALIB["mg2"]["K_per_deg_per_frame"], CALIB["mg2"]["clip"]),
                              abot="JSON (12 fps; 11 blocks = 9 + 10 x 12 = 129 frames generated, first 120 written)",
                              astra="JSON meta; astra_emb [51,13] = program_to_astra.build(pid, 20, 201)['astra'] = the first 51 rows of the runner's 57-row cam "
                                    "input (astra_gen.py: build(pid, 20, 225), 1 + 56 generated latents); astra_c2w [201,4,4] commanded c2w at 20 fps (RPE "
                                    "reference for Astra clips)"),
                    built=time.strftime("%Y-%m-%d %H:%M:%S %Z"), calib=calib)
        path = os.path.join(out_dir, pid + ".npz")
        np.savez(path, c2w=c2w, c2w_165=c2w165, intrinsics=K, astra_c2w=astra_nat["c2w"], astra_emb=astra_nat["emb"], meta=json.dumps(meta),
                 ours=json.dumps(om), waypoint=json.dumps(enc["waypoint"][0]), mg2=json.dumps(enc["mg2"][0]), abot=json.dumps(enc["abot"][0]),
                 astra=json.dumps(astra_meta))
        written.append(path)
    return written


def summarize_programs(out_dir=os.path.join(BASE, "programs"), check=True, verbose=True):
    """Print (and re-check against a fresh encoding) the per-model contents of programs/<P>.npz.  Returns {pid: summary}; raises on a mismatch."""
    out = {}
    for pid in FROZEN_PROGRAMS:   # [10/19] the 7 frozen programs only
        z = np.load(os.path.join(out_dir, pid + ".npz"))
        J = {k: json.loads(str(z[k])) for k in ("meta", "ours", "waypoint", "mg2", "abot", "astra")}
        mv = frame_motion(z["c2w"]); yaw = np.array([m[0] for m in mv]); tz = np.array([m[2][2] for m in mv])
        wp = J["waypoint"]; wm = np.array([a["mouse"] for a in wp["actions"]]); wb = [a["buttons"] for a in wp["actions"]]
        mg = J["mg2"]; kb = np.array(mg["keyboard"]); ms = np.array(mg["mouse"])
        ab = J["abot"]; asx = J["astra"]
        s = dict(
            c2w=dict(frames=int(len(z["c2w"])), frames_165=int(len(z["c2w_165"])), intrinsics=z["intrinsics"][0].tolist(), total_yaw_deg=round(float(yaw.sum()), 3),
                     total_fwd_u=round(float(tz.sum()), 3), yaw_per_s=np.round(yaw.reshape(10, 16).sum(1), 2).tolist(), fwd_per_s=np.round(tz.reshape(10, 16).sum(1), 2).tolist()),
            ours=dict(fps=J["ours"]["fps"], generated=J["ours"]["n_frames_generated"], latents=J["ours"]["n_latents_generated"], written=J["ours"]["n_frames_written"]),
            waypoint=dict(calls=wp["n_calls"], K_yaw=wp["K_yaw"], mouse_x_per_call=sorted(set(np.round(wm[:, 0], 2).tolist())), yaw_sum_deg=round(float(wm[:, 0].sum() / wp["K_yaw"]), 2),
                          W=sum(b == [W_] for b in wb), S=sum(b == [S_] for b in wb), none=sum(b == [] for b in wb)),
            mg2=dict(frames_generated=mg["n_frames_generated"], written=mg["n_frames_written"], K=mg["K"], keys_fwd_back_left_right=kb.sum(0).tolist(),
                     mouse_yaw_values=sorted(set(np.round(ms[:, 1], 4).tolist())), saturated_frames="%d/%d moving (%.0f %%)" % (mg["saturated_frames"], mg["moving_frames"], 100 * mg["saturated_frac_of_moving"]),
                     max_abs_unclipped=round(mg["max_abs_mouse_unclipped"], 4), yaw_sum_10s_deg_clipped=round(float(ms[1:121, 1].sum() / mg["K"]), 2)),
            abot=dict(blocks=ab["n_blocks"], generated=ab["n_frames_generated"], written=ab["n_frames_written"], keys=ab["keys_str"], switch_frames=ab["switch_frames"], switch_times_s=ab["switch_times_s"]),
            astra=dict(fps=asx["fps"], frames_written=asx["n_frames_written"], latents_generated=asx["latents_generated"], emb_rows=int(len(z["astra_emb"])), c2w_frames=int(len(z["astra_c2w"])),
                       u_scale=asx["u_scale"], yaw_per_latent_deg=sorted(set(np.round(decode_astra(z["astra_emb"])[:, 0], 3).tolist())),
                       tz_per_latent=sorted(set(np.round(decode_astra(z["astra_emb"])[:, 1], 4).tolist()))),
        )
        if check:
            om, (c2w, K, c2w165) = encode_program(pid, "ours")
            assert np.array_equal(z["c2w"], c2w) and np.array_equal(z["c2w_165"], c2w165) and np.array_equal(z["intrinsics"], K), pid + " ours arrays differ"
            assert np.array_equal(z["c2w_165"][:161], z["c2w"]), pid + " c2w_165[:161] != c2w"
            for m in ("waypoint", "mg2", "abot"):
                assert J[m] == json.loads(json.dumps(encode_program(pid, m)[0])), pid + " " + m + " JSON differs from a fresh encoding"
            am, an = encode_program(pid, "astra")
            assert np.array_equal(z["astra_emb"], an["emb"]) and np.array_equal(z["astra_c2w"], an["c2w"]) and J["astra"] == json.loads(json.dumps(am)), pid + " astra differs"
            assert J["meta"]["calib"]["abot"]["keys_per_program"][pid] == ab["keys_str"], pid + " calib abot keys != npz abot keys"
            s["consistent_with_fresh_encoding"] = True
        out[pid] = s
        if verbose:
            print("%s %-11s seg %s" % (pid, PROGRAMS[pid]["name"], PROGRAMS[pid]["seg"]))
            for m in ("c2w", "ours", "waypoint", "mg2", "abot", "astra"):
                print("   %-8s %s" % (m, json.dumps(s[m])))
    return out


NEUTRAL = "balloon_block_cut,billiard_break,bowling_inertia,domino_falling,easy_dragon_night,easy_greatwall,easy_lake,easy_urban_rain," \
          "inflating_balloon,newtons_cradle,pouring_water_into_cup,traffic_light_green".split(",")


def build_scenes(out_dir=os.path.join(BASE, "scenes", "neutral")):
    """scenes/neutral/<id>.{png,txt} + index.json from data/stevo/tasks_camera.json; existing files are left untouched (frozen)."""
    from PIL import Image
    T = {t["id"]: t for t in json.load(open(os.path.join(SPECWM, "data/stevo/tasks_camera.json")))}
    os.makedirs(out_dir, exist_ok=True)
    index = []
    for tid in NEUTRAL:
        t = T[tid]; png = os.path.join(out_dir, tid + ".png"); txt = os.path.join(out_dir, tid + ".txt")
        if not os.path.exists(png):
            im = Image.open(t["image"])
            if im.mode != "RGB":
                im = im.convert("RGB")
            im.save(png)
        if not os.path.exists(txt):
            open(txt, "w").write(t["camera_WM"])
        size = list(Image.open(png).size)
        index.append(dict(id=tid, source=t["image"], size=size, caption_chars=len(t["camera_WM"]), stevo_pose=t.get("pose"), category=t.get("category")))
    if not os.path.exists(os.path.join(out_dir, "index.json")):
        json.dump(index, open(os.path.join(out_dir, "index.json"), "w"), indent=1)
    return index


# ----------------------------------------------------------------------------------------------------------------------------------------
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


EXPECTED_ABOT = {  # 11 blocks (abot/ENV.md): WL... etc.
    "P1": "W W W W W W W W W W W", "P2": "L L L L L L L L L L L", "P2p": "L L L L L L L L L L L", "P3": "WL WL WL WL WL WL WL WL WL WL WL",
    "P4": "WL WL WL WJ WJ WJ WJ WJ WL WL WL", "P4p": "WL WL WL WJ WJ WJ WJ WJ WL WL WL", "P5": "W W W - - S S S W W W",
}


def validate(verbose=True):
    """(a) round trip on every program at every model's native fps + equality with the runners' own encoders; (b) reproduce the old scripts on
    existing tasks (bit-exact JSON, with the old scripts' own constants 118 / 0.067 / 12-frame blocks)."""
    rep = dict(a={}, b={}, ok=True)

    def chk(cond, msg):
        rep["ok"] &= bool(cond)
        if verbose or not cond:
            print(("  PASS " if cond else "  FAIL ") + msg)
        return bool(cond)

    Kw = CALIB["waypoint"]["K_yaw_units_per_deg_per_call"]; Km = CALIB["mg2"]["K_per_deg_per_frame"]; cap = CALIB["mg2"]["clip"]
    try:
        P2A = _load_module("program_to_astra", os.path.join(BASE, "astra", "program_to_astra.py"))
    except Exception as e:
        P2A = None; chk(False, "import baselines/astra/program_to_astra.py: %s" % e)
    try:
        AB = _load_module("encode_abot", os.path.join(BASE, "abot", "encode_abot.py"))
    except Exception as e:
        AB = None; chk(False, "import baselines/abot/encode_abot.py: %s" % e)

    # ---- (a) round trip ------------------------------------------------------------------------------------------------------------
    for pid in FROZEN_PROGRAMS:   # [10/19] the 7 frozen programs (P6: validate_long)
        r = {}; seg = PROGRAMS[pid]["seg"]; rate = seg[0][1]
        wp, wa = encode_program(pid, "waypoint"); mg, mc = encode_program(pid, "mg2"); ab, aa = encode_program(pid, "abot"); asd, asn = encode_program(pid, "astra")
        om, (c2w16, K16, c2w165) = encode_program(pid, "ours")
        mv16 = frame_motion(c2w16)
        yaw16 = np.array([m[0] for m in mv16]); tz16 = np.array([m[2][2] for m in mv16])
        r["c2w16"] = dict(n=len(c2w16), total_yaw_deg=float(yaw16.sum()), total_fwd_u=float(tz16.sum()), yaw_per_s=np.round(yaw16.reshape(10, 16).sum(1), 3).tolist(),
                          fwd_per_s=np.round(tz16.reshape(10, 16).sum(1), 3).tolist(), K832=K16[0].tolist())
        wb = [a["buttons"] for a in wa]; wm = np.array([a["mouse"] for a in wa]); kb = np.array(mc["keyboard"]); ms = np.array(mc["mouse"]); A = np.array(aa)
        raw = np.array(encode_mg2(program_to_c2w(pid, 12, CALIB["mg2"]["n_frames_generated"])[0], Km, None)["mouse"])
        r["waypoint"] = dict(n_calls=len(wa), W=sum(b == [W_] for b in wb), S=sum(b == [S_] for b in wb), none=sum(b == [] for b in wb),
                             yaw_sum_deg=float(wm[:, 0].sum() / Kw), pitch_sum_deg=float(wm[:, 1].sum() / Kw), pos=int((wm[:, 0] > 1e-6).sum()), neg=int((wm[:, 0] < -1e-6).sum()),
                             mouse_x_abs_max=float(np.abs(wm[:, 0]).max()))
        r["mg2"] = dict(n_frames=len(kb), fwd=int(kb[:, 0].sum()), back=int(kb[:, 1].sum()), left=int(kb[:, 2].sum()), right=int(kb[:, 3].sum()),
                        yaw_sum_10s_deg_clipped=float(ms[1:121, 1].sum() / Km), yaw_sum_10s_deg_unclipped=float(raw[1:121, 1].sum() / Km),
                        pos=int((ms[:, 1] > 1e-9).sum()), neg=int((ms[:, 1] < -1e-9).sum()), saturated_frames=mg["saturated_frames"], moving_frames=mg["moving_frames"],
                        saturated_frac_of_moving=mg["saturated_frac_of_moving"], max_abs_unclipped=mg["max_abs_mouse_unclipped"])
        r["abot"] = dict(n_blocks=len(A), counts=dict(zip(KEYS8, A.sum(0).tolist())), keys=ab["keys_str"], switch_frames=ab["switch_frames"], switch_times_s=ab["switch_times_s"])
        emb = asn["emb"]; dec = decode_astra(emb)
        r["astra"] = dict(rows=int(len(emb)), yaw_per_latent_deg=np.round(dec[:, 0], 4).tolist(), tz_per_latent=np.round(dec[:, 1], 4).tolist(), u_scale=asd["u_scale"])
        r["ours"] = dict(n_frames_written=om["n_frames_written"], n_frames_generated=om["n_frames_generated"], n_latents=om["n_latents_generated"])
        print(pid, PROGRAMS[pid]["name"], "| c2w16 yaw/s", r["c2w16"]["yaw_per_s"], "fwd/s", r["c2w16"]["fwd_per_s"], "K", r["c2w16"]["K832"])
        print("   waypoint", r["waypoint"], "\n   mg2", r["mg2"], "\n   abot", r["abot"]["counts"], r["abot"]["keys"], "switch", r["abot"]["switch_times_s"],
              "\n   astra rows", r["astra"]["rows"], "yaw/latent", sorted(set(r["astra"]["yaw_per_latent_deg"])), "tz/latent", sorted(set(r["astra"]["tz_per_latent"])))

        # -- every program: c2w per second = the program; intrinsics; ours 165 continuation; runner-encoder equality
        exp_yaw, exp_fwd = program_per_second(pid)   # second 2 / 7 straddle the 2.5 / 7.5 s switches (half of each segment)
        chk(np.allclose(r["c2w16"]["yaw_per_s"], exp_yaw, atol=2e-3) and np.allclose(r["c2w16"]["fwd_per_s"], exp_fwd, atol=2e-5),
            "%s c2w: yaw/s %s fwd/s %s = the program integrated per second" % (pid, exp_yaw, exp_fwd))
        chk(K16[0].tolist() == [416.0, 416.0, 416.0, 240.0] and K16.shape == (161, 4), "%s intrinsics [416, 416, 416, 240] x 161 (fx = fy = 0.5 W)" % pid)
        mv165 = frame_motion(c2w165)
        chk(len(c2w165) == 165 and np.array_equal(c2w165[:161], c2w16) and all(abs(m[0] - seg[3][1] / 16.0) < 1e-3 and abs(m[2][2] - seg[3][0] / 16.0) < 1e-6 for m in mv165[160:]),
            "%s c2w_165: first 161 poses = c2w, 4 extra frames continue the last segment (yaw %s deg/s, fwd %s)" % (pid, seg[3][1], seg[3][0]))
        chk(len(wa) == 150 and len(kb) == 129 and len(aa) == 11 and len(emb) == 51 and len(asn["c2w"]) == 201,
            "%s native lengths: waypoint 150 calls, mg2 129 frames, abot 11 blocks, astra 51 rows / 201 frames" % pid)
        chk(ab["keys_str"] == EXPECTED_ABOT[pid], "%s abot keys (11 blocks): %s  [expected %s]" % (pid, ab["keys_str"], EXPECTED_ABOT[pid]))
        if AB is not None:
            n = AB.frames_for_blocks(11)
            theirs = AB.encode_abot(program_to_c2w(pid, 12, n)[0], n_blocks=11)
            chk(theirs == aa and n == 129, "%s abot: identical keys to baselines/abot/encode_abot.py encode_abot(c2w, first_block_frames=9, n_blocks=11) on %d frames" % (pid, n))
            if pid in AB.PROGRAMS:
                chk(np.allclose(AB.program_to_c2w(pid, 12, n), program_to_c2w(pid, 12, n)[0], atol=1e-5), "%s abot: encode_abot.program_to_c2w == convert.program_to_c2w (12 fps)" % pid)
        if P2A is not None:
            d = P2A.build(pid, 20, 201)
            chk(np.array_equal(d["astra"], emb) and np.array_equal(np.asarray(d["c2w"], np.float32), asn["c2w"]) and abs(d["u_scale"] - asd["u_scale"]) < 1e-12 and d["n_latents"] == 50,
                "%s astra: bit-identical rows (%d x 13) and c2w (201) to baselines/astra/program_to_astra.py build(pid, 20, 201), u_scale %.4f" % (pid, len(emb), d["u_scale"]))
            # 10/17: the runner (astra_gen.py) hands the official sliding window the program over ALL 225 generated frames = 1 + 56 rows (calib astra.embedding_rows)
            d225 = P2A.build(pid, 20, CALIB["astra"]["n_frames_generated"]); rows = CALIB["astra"]["embedding_rows"]
            chk(len(d225["astra"]) == rows == 1 + CALIB["astra"]["latents_generated"] and d225["n_latents"] == CALIB["astra"]["latents_generated"]
                and np.array_equal(d225["astra"][:len(emb)], emb) and abs(d225["u_scale"] - asd["u_scale"]) < 1e-12,
                "%s astra: the runner's %d-row embedding (build(pid, 20, 225), 56 latents) starts with the npz's %d rows, same u_scale" % (pid, len(d225["astra"]), len(emb)))
        # -- program-specific
        if pid == "P1":
            chk(all(b == [W_] for b in wb) and np.abs(wm).max() == 0, "P1 waypoint: W every call (%d/%d), mouse all zero" % (r["waypoint"]["W"], len(wa)))
            chk(all(k == [1, 0, 0, 0] for k in mc["keyboard"][1:]) and np.abs(ms).max() == 0, "P1 mg2: fwd every frame (%d/%d), mouse all zero" % (r["mg2"]["fwd"], len(kb) - 1))
            chk(np.allclose(emb[:, [0, 1, 2, 4, 5, 6, 8, 9, 10]], np.eye(3).reshape(-1), atol=1e-6) and np.allclose(emb[:, 11], -0.03, atol=1e-6) and np.abs(emb[:, [3, 7]]).max() < 1e-6,
                "P1 astra: relative rotation identity, M[2,3] = -0.03 per latent (= Astra demo forward, 1 u/s * 0.2 s * 0.15)")
        if pid in ("P2", "P2p"):
            tot = rate * 10
            chk(all(b == [] for b in wb) and (wm[:, 0] > 0).all() and np.abs(wm[:, 1]).max() == 0 and abs(wm[:, 0].sum() / Kw - tot) < 1e-3 and abs(wm[0, 0] - Kw * rate * 4 / 60) < 1e-3,
                "%s waypoint: no buttons, every call +mouse-x = %.1f units (K %.0f x %.3f deg), total %.3f deg (= %d), pitch 0" % (pid, wm[0, 0], Kw, rate * 4 / 60, wm[:, 0].sum() / Kw, tot))
            expect_sat = rate / 12.0 * Km > cap + 1e-9
            chk(all(k == [0, 0, 0, 0] for k in mc["keyboard"]) and (ms[1:, 1] > 0).all() and np.abs(ms[:, 0]).max() == 0 and abs(raw[1:121, 1].sum() / Km - tot) < 1e-3
                and (mg["saturated_frames"] == (mg["moving_frames"] if expect_sat else 0)),
                "%s mg2: no keys, +mouse-yaw every frame (%.4f unclipped vs cap %.2f), unclipped 10 s total %.3f deg (= %d); saturated %d/%d moving frames (%s)" % (
                    pid, mg["max_abs_mouse_unclipped"], cap, raw[1:121, 1].sum() / Km, tot, mg["saturated_frames"], mg["moving_frames"],
                    "expected: all, clipped total %.2f deg" % r["mg2"]["yaw_sum_10s_deg_clipped"] if expect_sat else "expected: none"))
            if pid == "P2p":
                chk(mg["saturated_frames"] == 0 and mg["max_abs_mouse_unclipped"] < cap, "P2p mg2: unsaturated on 100 %% of frames (max |mouse| %.4f < %.2f)" % (mg["max_abs_mouse_unclipped"], cap))
            chk(abs(r["c2w16"]["total_yaw_deg"] - tot) < 1e-3 and abs(r["c2w16"]["total_fwd_u"]) < 1e-6, "%s c2w: total yaw %.3f deg (= %d), translation 0" % (pid, r["c2w16"]["total_yaw_deg"], tot))
            th = rate * 4 / 20.0
            rot = emb[:, [0, 1, 2, 4, 5, 6, 8, 9, 10]]
            chk(np.allclose(rot, (S_FLIP @ Ry(th) @ S_FLIP).reshape(-1), atol=1e-5) and np.abs(emb[:, [3, 7, 11]]).max() < 1e-6 and (emb[:, 2] < 0).all(),
                "%s astra: every per-latent rotation = S Ry(%.2f deg) S (M[0,2] = %.4f < 0: Astra 'Rotate Right' = our +yaw), translation 0" % (pid, th, emb[0, 2]))
        if pid == "P3":
            chk(all(b == [W_] for b in wb) and (wm[:, 0] > 0).all(), "P3 waypoint: W + positive mouse every call")
            chk((kb[1:, 0] == 1).all() and (ms[1:, 1] > 0).all() and mg["saturated_frames"] == 0, "P3 mg2: fwd + positive mouse-yaw every frame, unsaturated (%.4f)" % mg["max_abs_mouse_unclipped"])
        if pid in ("P4", "P4p"):
            fps, fpc = 60, 4
            tcall = np.arange(len(wa)) * fpc / fps
            s = np.sign(wm[:, 0]); strad = np.where((tcall < 2.5) & (tcall + fpc / fps > 2.5) | (tcall < 7.5) & (tcall + fpc / fps > 7.5))[0]
            chk((s[tcall + fpc / fps <= 2.5 + 1e-9] > 0).all() and (s[(tcall >= 2.5 - 1e-9) & (tcall + fpc / fps <= 7.5 + 1e-9)] < 0).all() and (s[tcall >= 7.5 - 1e-9] > 0).all()
                and all(b == [W_] for b in wb) and list(strad) == [37, 112] and np.abs(wm[strad, 0]).max() < 0.05,
                "%s waypoint: +mouse for calls ending <= 2.5 s, -mouse for 2.5-7.5 s, +mouse after 7.5 s; W every call (straddling calls %s: 2 + 2 frames of opposite "
                "segments, mouse %s ~ 0 vs %.1f elsewhere)" % (pid, [int(i) for i in strad], [round(float(wm[i, 0]), 4) for i in strad], abs(wm[0, 0])))
            tf = np.arange(1, len(ms)) / 12.0   # mg2 frame i's motion covers (i-1)/fps .. i/fps
            y = ms[1:, 1]
            chk((y[tf <= 2.5 + 1e-9] > 0).all() and (y[(tf > 2.5 + 1e-9) & (tf <= 7.5 + 1e-9)] < 0).all() and (y[tf > 7.5 + 1e-9] > 0).all() and (kb[1:, 0] == 1).all()
                and (mg["saturated_frames"] == (mg["moving_frames"] if rate / 12.0 * Km > cap + 1e-9 else 0)),
                "%s mg2: mouse-yaw sign flips exactly at 2.5 s and 7.5 s (frames 31 and 91), fwd every frame; saturated %d/%d" % (pid, mg["saturated_frames"], mg["moving_frames"]))
            chk(ab["switch_frames"] == [33, 93] and ab["switch_times_s"] == [2.75, 7.75],
                "%s abot: L->J at block 3 (frame 33 = 2.75 s), J->L at block 8 (frame 93 = 7.75 s); blocks 2 / 7 straddle the flips and keep the majority key" % pid)
            dy = dec[:, 0]; tl = np.arange(len(dy)) * 4 / 20.0   # row j covers 4j/20 .. (4j+4)/20 s
            chk((dy[tl + 0.2 <= 2.5 + 1e-9] > 0).all() and (dy[(tl >= 2.5 - 1e-9) & (tl + 0.2 <= 7.5 + 1e-9)] < 0).all() and (dy[(tl >= 7.5 - 1e-9) & (tl < 10)] > 0).all()
                and abs(dy[12]) < 1e-3 and abs(dy[37]) < 1e-3 and (dec[:, 1] > 0).all() and np.allclose(dec[:, 1], 0.03, atol=5e-5),
                "%s astra: per-latent yaw +/-/+ at 2.5 / 7.5 s (latents 12 and 37 straddle: net 0), forward ~0.03 units every latent (chord of the 4-frame arc: %.5f)" % (pid, dec[0, 1]))
        if pid == "P5":
            chk([a["buttons"] for a in wa[:37]] == [[W_]] * 37 and wa[37]["buttons"] == [W_] and [a["buttons"] for a in wa[38:75]] == [[]] * 37 and wa[75]["buttons"] == [S_]
                and [a["buttons"] for a in wa[76:112]] == [[S_]] * 36 and wa[112]["buttons"] == [] and [a["buttons"] for a in wa[113:]] == [[W_]] * 37 and np.abs(wm).max() == 0,
                "P5 waypoint: W (calls 0-37) / none (38-74) / S (75-111) / none at straddling call 112 / W (113-149); mouse 0")
            chk([k[0] for k in kb[1:31]] == [1] * 30 and all(k == [0, 0, 0, 0] for k in mc["keyboard"][31:61]) and [k[1] for k in kb[61:91]] == [1] * 30 and [k[0] for k in kb[91:]] == [1] * 38,
                "P5 mg2: fwd 1-30, none 31-60, back 61-90, fwd 91-128 (129 generated frames; 120 written)")
            chk(ab["switch_frames"] == [33, 57, 93] and ab["switch_times_s"] == [2.75, 4.75, 7.75],
                "P5 abot: W,W,W,-,-,S,S,S,W,W,W: switches at frames 33 / 57 / 93 = 2.75 / 4.75 / 7.75 s (block 2 = 9 fwd + 3 still -> W; block 7 = 9 back + 3 fwd -> S)")
            chk(abs(r["c2w16"]["total_fwd_u"] - 2.5) < 1e-6 and np.allclose(c2w16[-1, :3, 3], [0, 0, 2.5], atol=1e-5), "P5 c2w: ends 2.5 u ahead of the start (2.5 - 0 + -2.5 + 2.5)")
            tzl = dec[:, 1]   # protocol sign (forward > 0); latent j covers 0.2 j .. 0.2 j + 0.2 s, latents 12 / 37 straddle 2.5 / 7.5 s
            chk(np.allclose(tzl[:12], 0.03, atol=1e-6) and abs(tzl[12] - 0.015) < 1e-6 and np.abs(tzl[13:25]).max() < 1e-6 and np.allclose(tzl[25:37], -0.03, atol=1e-6)
                and abs(tzl[37]) < 1e-6 and np.allclose(tzl[38:], 0.03, atol=1e-6),
                "P5 astra: per-latent translation +0.03 (fwd, M[2,3] < 0) / 0 / -0.03 (back) / +0.03; straddling latent 12 = +0.015 (0.1 s fwd), latent 37 = 0 (0.1 s back + 0.1 s fwd)")
        rep["a"][pid] = r

    # ---- (b) reproduce the old scripts on existing task trajectories (their own constants: K 118, 0.067, 12-frame blocks from frame 0) -----------
    T = {t["id"]: t for t in json.load(open(os.path.join(SPECWM, "data/stevo/tasks_camera.json")))}
    with tempfile.TemporaryDirectory(prefix="convert_validate_") as tmp:
        for tid in ("easy_lake", "easy_greatwall", "balloon_block_cut", "traffic_light_green"):
            P = np.load(T[tid]["action"] + "/poses.npy")
            res = {}
            for name, script, N, mine in (
                ("waypoint", "tools/waypoint/poses_to_actions.py", 153, lambda P: encode_waypoint(P[:153], 118.0, 4)),
                ("mg2", "tools/mg2/poses_to_cond.py", 153, lambda P: encode_mg2(P[:153], 0.067, None)),
                ("abot", "tools/abot/poses_to_keys.py", 161, lambda P: encode_abot(P[:161], 12, 2.0, first_block_frames=None)),
            ):
                out = os.path.join(tmp, "%s_%s.json" % (name, tid))
                cmd = [sys.executable, os.path.join(SPECWM, script), tid, out] + (["0.067"] if name == "mg2" else [])
                log = subprocess.run(cmd, cwd=SPECWM, capture_output=True, text=True).stdout.strip().splitlines()
                old = json.load(open(out)); new = json.loads(json.dumps(mine(P)))
                same = old == new
                if name == "waypoint":
                    summ = "calls %d, W/S/A/D %s, yaw sum %.3f" % (len(new), [sum(k in a["buttons"] for a in new) for k in (W_, S_, A_, D_)], sum(a["mouse"][0] for a in new) / 118.0)
                elif name == "mg2":
                    summ = "frames %d, fwd/back/left/right %s, yaw sum %.3f, max|mouse| %.4f (%d entries > 0.1)" % (len(new["keyboard"]), np.array(new["keyboard"]).sum(0).tolist(),
                                                                                                                    sum(m[1] for m in new["mouse"]) / 0.067, np.abs(new["mouse"]).max(), int((np.abs(new["mouse"]) > 0.1).sum()))
                else:
                    summ = "blocks %d, counts W/A/S/D/I/J/K/L %s" % (len(new), np.array(new).sum(0).tolist())
                res[name] = dict(identical_json=same, summary=summ, old_script_stdout=log[-1] if log else "")
                chk(same, "%s %s: identical JSON to %s  [%s]" % (tid, name, script, summ))
            rep["b"][tid] = res
    print("VALIDATION", "OK" if rep["ok"] else "FAILED")
    return rep


def validate_long(pid="P6", verbose=True):
    """[10/19 EXP 1] checks of a LONG program: c2w per segment = the program, heading returns to 0, the frozen-rule encoders at the long layout, equality
    with the runners' own encoders (waypoint/encode_program.py encode_via_convert, abot/encode_abot.py encode_abot on the same c2w), MG2 unsaturated."""
    rep = dict(pid=pid, ok=True, checks=[])

    def chk(cond, msg):
        rep["ok"] &= bool(cond); rep["checks"].append(("PASS " if cond else "FAIL ") + msg)
        if verbose or not cond: print(("  PASS " if cond else "  FAIL ") + msg)
        return bool(cond)
    P = PROGRAMS[pid]; seg = P["seg"]; ss = seg_s_of(pid); L = long_layout(pid); rep["layout"] = L
    om, (c2w, K, c2wg) = encode_program(pid, "ours"); mv = frame_motion(c2w); yaw = np.array([m[0] for m in mv]); tz = np.array([m[2][2] for m in mv])
    fps = L["ours"]["fps"]; per = int(round(ss * fps))
    yseg = yaw.reshape(len(seg), per).sum(1); fseg = tz.reshape(len(seg), per).sum(1)
    chk(len(c2w) == L["ours"]["n_frames_written"] == 961 and len(c2wg) == L["ours"]["n_frames_generated"] == 969 and L["ours"]["n_latents_generated"] == 243 and L["ours"]["n_chunks"] == 81,
        "%s ours: c2w %d (RPE reference, 60 s at 16 fps), generated %d poses = %d latents = %d chunks" % (pid, len(c2w), len(c2wg), L["ours"]["n_latents_generated"], L["ours"]["n_chunks"]))
    chk(np.allclose(yseg, [s[1] * ss for s in seg], atol=2e-3) and np.allclose(fseg, [s[0] * ss for s in seg], atol=1e-5),
        "%s c2w per %.0f-s segment: yaw %s deg, fwd %s u = the program" % (pid, ss, np.round(yseg, 2).tolist(), np.round(fseg, 3).tolist()))
    head = np.cumsum(yseg); rep["heading_after_segment_deg"] = np.round(head, 2).tolist()
    chk(abs(head[7]) < 1e-2 and abs(head[-1]) < 1e-2 and max(abs(s[1]) for s in seg) <= 16, "%s heading after each segment %s deg: back at the start heading at t=40 s and at the end; |yaw rate| <= 16 deg/s" % (pid, np.round(head, 1).tolist()))
    pos = c2w[:, :3, 3]; i15, i30 = int(15 * fps), int(30 * fps)
    chk(np.allclose(c2w[i15], c2w[i30], atol=1e-4), "%s back-up retraces segment 4: pose at t=30 s == pose at t=15 s (position %s)" % (pid, np.round(pos[i30], 3).tolist()))
    rep["position_end"] = np.round(pos[-1], 3).tolist(); rep["max_dist_from_start"] = round(float(np.linalg.norm(pos, axis=1).max()), 3)
    chk(np.array_equal(c2wg[:len(c2w)], c2w) and K[0].tolist() == [416.0, 416.0, 416.0, 240.0] and K.shape == (961, 4), "%s c2w_969[:961] == c2w; intrinsics [416, 416, 416, 240] x 961" % pid)
    wm, wa = encode_program(pid, "waypoint"); A = np.array([a_["mouse"] for a_ in wa]); Kw = CALIB["waypoint"]["K_yaw_units_per_deg_per_call"]; cps = int(round(ss * 15))
    wb = [a_["buttons"] for a_ in wa]
    exp_b = [[W_] if s[0] > 0 else ([S_] if s[0] < 0 else []) for s in seg]
    chk(len(wa) == L["waypoint"]["n_calls"] == 900 and all(wb[c] == exp_b[c // cps] for c in range(len(wa))) and np.allclose(A[:, 0].reshape(len(seg), cps).sum(1) / Kw, [s[1] * ss for s in seg], atol=1e-3)
        and np.abs(A[:, 1]).max() == 0, "%s waypoint: %d calls (75 per 5-s segment, no straddling call), buttons W / S / none per segment as the program, mouse-x per segment / K = segment yaw, max |mouse x| %.1f units/call" % (pid, len(wa), np.abs(A[:, 0]).max()))
    try:
        EP = _load_module("wp_encode_program", os.path.join(BASE, "waypoint", "encode_program.py")); r_ = EP.encode_via_convert(pid, Kw, L["clip_s"])
        chk(r_ is not None and json.loads(json.dumps(r_[1])) == json.loads(json.dumps(wa)), "%s waypoint: identical actions to the runner's encoder (waypoint/encode_program.py encode_via_convert, 60 s)" % pid)
    except Exception as e: chk(False, "%s waypoint runner encoder: %r" % (pid, e))
    mm, mc = encode_program(pid, "mg2"); kb = np.array(mc["keyboard"]); ms = np.array(mc["mouse"])
    chk(len(kb) == L["mg2"]["n_frames_generated"] == 729 and mm["saturated_frames"] == 0 and mm["max_abs_mouse_unclipped"] < CALIB["mg2"]["clip"],
        "%s mg2: %d frames generated (183 latents), %d written; unsaturated (max |mouse| %.4f < %.2f)" % (pid, len(kb), L["mg2"]["n_frames_written"], mm["max_abs_mouse_unclipped"], CALIB["mg2"]["clip"]))
    try:
        cal = json.load(open(CALIB_PATH)); chk(abs(float(cal["mg2"]["K_per_deg_per_frame"]) - CALIB["mg2"]["K_per_deg_per_frame"]) < 1e-12, "%s mg2: calib.json K %s == convert K (run_mg2.py reads calib.json)" % (pid, cal["mg2"]["K_per_deg_per_frame"]))
    except Exception as e: chk(False, "calib.json mg2 K: %r" % e)
    am, aa = encode_program(pid, "abot")
    try:
        AB = _load_module("encode_abot", os.path.join(BASE, "abot", "encode_abot.py")); n = am["n_frames_generated"]
        theirs = AB.encode_abot(program_to_c2w(pid, 12, n)[0], n_blocks=am["n_blocks"])
        chk(theirs == aa and am["n_blocks"] == 61 and n == 729, "%s abot: %d blocks / %d frames; identical keys to abot/encode_abot.py encode_abot(c2w, n_blocks=61) on the same c2w; keys %s; switches at %s s" % (
            pid, am["n_blocks"], n, am["keys_str"], am["switch_times_s"]))
    except Exception as e: chk(False, "%s abot runner encoder: %r" % (pid, e))
    rep["summary"] = dict(waypoint=dict(calls=len(wa), mouse_x_values=sorted(set(np.round(A[:, 0], 2).tolist()))), mg2=dict(frames=len(kb), keys_fwd_back_left_right=kb.sum(0).tolist(),
                          mouse_yaw_values=sorted(set(np.round(ms[:, 1], 4).tolist()))), abot=dict(blocks=am["n_blocks"], keys=am["keys_str"], switch_times_s=am["switch_times_s"]))
    print("VALIDATION %s %s" % (pid, "OK" if rep["ok"] else "FAILED"))
    return rep


def build_long(pid="P6", out_dir=os.path.join(BASE, "programs")):
    """[10/19 EXP 1] write programs/<pid>.npz (+ programs/<pid>_validation.json) for a LONG program ONLY; the frozen programs/<P>.npz and calib.json are
    not touched (checked: md5 of every other programs/* file before == after)."""
    import hashlib
    assert pid in LONG_PROGRAMS, pid
    def md5s(): return {f: hashlib.md5(open(os.path.join(out_dir, f), "rb").read()).hexdigest() for f in sorted(os.listdir(out_dir)) if not f.startswith(pid + ".") and not f.startswith(pid + "_")}
    before = md5s(); cal_before = hashlib.md5(open(CALIB_PATH, "rb").read()).hexdigest()
    rep = validate_long(pid, verbose=True)
    if not rep["ok"]: print("validation failed; not building"); return None
    om, (c2w, K, c2wg) = encode_program(pid, "ours"); enc = {m: encode_program(pid, m)[0] for m in ("waypoint", "mg2", "abot")}; L = long_layout(pid); P = PROGRAMS[pid]
    meta = dict(id=pid, name=P["name"], segments_s=seg_s_of(pid), clip_s=clip_s_of(pid), segments=[dict(fwd_u_per_s=s[0], yaw_deg_per_s=s[1], pitch_deg_per_s=s[2]) for s in P["seg"]],
                heading_after_segment_deg=rep["heading_after_segment_deg"], tests=P["tests"], sign="positive yaw = pan RIGHT (yaw = atan2(Rr[0,2], Rr[2,2]), Rr = R_i^T R_{i+1}, OpenCV c2w)",
                c2w="OpenCV c2w, 16 fps, %d frames = %.0f s, frame 0 = identity, translation 1 u/s = 1/16 per frame; the RPE reference for every model" % (len(c2w), clip_s_of(pid)),
                c2w_969="the same program continued to %d frames (%d latents = %d chunks of 3 for our loader; trimmed back to %d frames)" % (len(c2wg), L["ours"]["n_latents_generated"], L["ours"]["n_chunks"], len(c2w)),
                intrinsics="fx,fy,cx,cy per frame for 832x480: [416, 416, 416, 240] (fx = fy = 0.5 W, PROTOCOL §2)", layout=L,
                keys=dict(ours="c2w [961,4,4], c2w_969 [969,4,4], intrinsics [961,4]; JSON meta in 'ours'", waypoint="JSON (60 fps, 4 frames/call, 900 calls; K_yaw %s)" % CALIB["waypoint"]["K_yaw_units_per_deg_per_call"],
                          mg2="JSON (12 fps, 729 frames generated, first 720 written; K %s, cap %s)" % (CALIB["mg2"]["K_per_deg_per_frame"], CALIB["mg2"]["clip"]),
                          abot="JSON (12 fps; 61 blocks = 9 + 60 x 12 = 729 frames generated, first 720 written; actions passed per job)", astra="not built (EXP 1 has no Astra arm)"),
                built=time.strftime("%Y-%m-%d %H:%M:%S %Z"), calib_version=CALIB["version"], validation=rep["checks"])
    path = os.path.join(out_dir, pid + ".npz")
    np.savez(path, c2w=c2w, c2w_969=c2wg, intrinsics=K, meta=json.dumps(meta), ours=json.dumps(om), waypoint=json.dumps(enc["waypoint"]), mg2=json.dumps(enc["mg2"]), abot=json.dumps(enc["abot"]))
    json.dump(rep, open(os.path.join(out_dir, pid + "_validation.json"), "w"), indent=1, default=_jsonable)
    after = md5s(); cal_after = hashlib.md5(open(CALIB_PATH, "rb").read()).hexdigest()
    same = before == after and cal_before == cal_after
    print("[build-long] wrote %s (+ %s_validation.json); other programs/* files and calib.json byte-identical: %s" % (path, pid, same))
    z = np.load(path); assert np.array_equal(z["c2w"], c2w) and np.array_equal(z["c2w_969"], c2wg) and json.loads(str(z["mg2"]))["saturated_frames"] == 0
    return path if same else None


def _jsonable(o):
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)


# ----------------------------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build"); sub.add_parser("validate"); sub.add_parser("summary")
    bl = sub.add_parser("build-long"); bl.add_argument("pid", nargs="?", default="P6", choices=LONG_PROGRAMS)   # [10/19] programs/P6.npz only
    vl = sub.add_parser("validate-long"); vl.add_argument("pid", nargs="?", default="P6", choices=LONG_PROGRAMS)
    e = sub.add_parser("encode"); e.add_argument("pid", choices=sorted(PROGRAMS)); e.add_argument("model", choices=["waypoint", "mg2", "abot", "astra", "ours"]); e.add_argument("out")
    e.add_argument("--speed", type=float, default=1.0); e.add_argument("--W", type=int, default=832); e.add_argument("--H", type=int, default=480)
    x = sub.add_parser("export-ours"); x.add_argument("pid", choices=sorted(PROGRAMS)); x.add_argument("out_dir"); x.add_argument("--fps", type=int, default=16); x.add_argument("--frames", type=int, default=161)
    x.add_argument("--W", type=int, default=832); x.add_argument("--H", type=int, default=480); x.add_argument("--speed", type=float, default=1.0)
    a = ap.parse_args()
    if a.cmd == "build":
        rep = validate(verbose=True)
        if not rep["ok"]:
            print("validation failed; not building"); sys.exit(1)
        calib, dec, warn = build_calib(); print("calib:", CALIB_PATH, "(%d merge decisions, %d warnings)" % (len(dec), len(warn)))
        print("programs:", build_programs(calib))
        idx = build_scenes(); print("scenes/neutral:", [(i["id"], i["size"]) for i in idx])
        print("---- programs/<P>.npz contents (re-checked against a fresh encoding) ----")
        rep["npz_summary"] = summarize_programs(check=True, verbose=True)
        rep["calib_merge"] = dict(decisions=dec, warnings=warn)
        rep["built"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
        json.dump(rep, open(os.path.join(BASE, "programs", "validation.json"), "w"), indent=1, default=_jsonable)
        print("validation.json:", os.path.join(BASE, "programs", "validation.json"))
    elif a.cmd == "validate":
        sys.exit(0 if validate(verbose=True)["ok"] else 1)
    elif a.cmd == "build-long":
        sys.exit(0 if build_long(a.pid) else 1)
    elif a.cmd == "validate-long":
        sys.exit(0 if validate_long(a.pid)["ok"] else 1)
    elif a.cmd == "summary":
        summarize_programs(check=True, verbose=True)
    elif a.cmd == "encode":
        meta, native = encode_program(a.pid, a.model, a.speed, a.W, a.H)
        if a.model == "ours":
            c2w, K, c2w165 = native; os.makedirs(a.out, exist_ok=True)
            np.save(os.path.join(a.out, "poses.npy"), c2w); np.save(os.path.join(a.out, "intrinsics.npy"), K); np.save(os.path.join(a.out, "poses_165.npy"), c2w165)
            json.dump(meta, open(os.path.join(a.out, "ours.json"), "w"), indent=1); print("wrote", a.out, c2w.shape, "(+ poses_165.npy", c2w165.shape, ")", K[0].tolist())
        elif a.model == "astra":
            os.makedirs(a.out, exist_ok=True); np.save(os.path.join(a.out, "astra_emb.npy"), native["emb"]); np.save(os.path.join(a.out, "astra_c2w.npy"), native["c2w"])
            json.dump(meta, open(os.path.join(a.out, "astra_cam.json"), "w"), indent=1); print("wrote", a.out, "(astra_emb.npy [%d,13], astra_c2w.npy [201,4,4], astra_cam.json)" % len(native["emb"]))
        else:
            json.dump(native, open(a.out, "w")); print("wrote", a.out, "(%s native format; fps %s)" % (a.model, meta["fps"]))
    elif a.cmd == "export-ours":
        c2w, K = export_ours(a.pid, a.out_dir, a.fps, a.frames, a.W, a.H, a.speed); print("wrote", a.out_dir, c2w.shape, K[0].tolist())


if __name__ == "__main__":
    main()
