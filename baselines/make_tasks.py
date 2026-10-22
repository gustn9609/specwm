"""[10/16 frozen baselines; 10/17 exact-resolution update] Build OUR task entries for a scene set x the control programs:
baselines/tasks/<SET>/img/<scene>.png (the start image PRE-RESIZED to the baseline's resolution, exactly as that baseline's own loader does it),
baselines/tasks/<SET>/<task>/{poses.npy, intrinsics.npy, caption.txt} and baselines/tasks/<SET>.json (two_machine schema: id, image, camera_WM, action;
load with SPECWM_TASKS=<SET>.json).  SET = wp | mg2 | abot | astra | neutral[_<tag>].

Why pre-resize (PROTOCOL rule 2/4): our loader (binding_v2.prepare = wan/image2video.py) keeps the aspect of the image it is given and buckets it
with lat = round(sqrt(area*aspect) // 8 // 2 * 2), so a source-size image gives 960x528 / 560x384 / 1264x704 / 768x512 instead of the baselines'
1024x512 / 640x352 / 1280x704 / 832x480.  The image is therefore written at the target size and the area passed to our arms is W*H + 1
(the floor formula turns an exact 832x480 at area 399360 into 832x464; +1 gives exactly WxH; stored per task as "area").
Resize rules (= the baseline's own loader):
  wp    1024x512  stretch: PIL resize((1024,512), BILINEAR)                (Waypoint model card load_image(...).resize((1024,512)), official Space)
  mg2    640x352  crop:    MG2 _resizecrop (centre crop to the target aspect) then PIL resize((640,352), BILINEAR) (torchvision v2.Resize on PIL)
  abot  1280x704  fit:     baselines/scenes/abot/fit1280x704/<scene>.png (= ABot's ImageOps.fit LANCZOS centre crop; computed here when missing)
  astra  832x480  stretch: PIL resize((832,480), BILINEAR)                 (Astra's loader stretches, baselines/astra/ENV.md)
  neutral: our 12 a832 scenes, run once per target resolution: --tag wp|mg2|abot|832 selects the rule+size of that baseline (neutral_832 = 832x480 stretch).
Poses = the program's c2w for --frames (default 165 = 42 latent frames with chunk 3 -> 165 generated frames, trimmed to 161 = 10.0 s by
baselines/ours_to_timing.py; our loader clips frame_num to len(poses), so 161 poses would silently give 153 frames); the first 161 poses are asserted
equal to the frozen programs/<P>.npz.  Intrinsics follow the existing bench tasks' convention: per-frame [416, 416, 416, 240] at the 832x480
reference frame, which binding_v2 rescales to the output size.
Scenes (10/17): the default scene list is baselines/scenes/<set>/FROZEN.json (the frozen 12-scene matrix per set, PROTOCOL §3; astra = its 3), in that
order; --scenes a,b overrides it, --scenes all takes every image in the set. Programs default to the 7 frozen ones (P1,P2,P2p,P3,P4,P4p,P5).
Frozen task sets (10/17): wp, mg2 (text-free: their .txt are empty), abot (its own captions: 2 of the 12 frozen scenes), astra (its prompts),
neutral --tag 832 (captioned), neutral --tag 832 --no-caption --suffix _nocap, neutral --tag wp|mg2|abot --no-caption (the text-free baselines'
neutral arms, PROTOCOL rule 5).  --suffix is appended to every task id and to the json name (tasks/neutral_832_nocap.json; ids ..._P1_nocap).
usage: /home/hk28965/specwm/scripts/envrun.sh python baselines/make_tasks.py SET [--tag T] [--res WxH] [--rule stretch|crop|fit] [--programs P1,...]
       [--frames 165] [--no-caption] [--suffix _nocap] [--scenes a,b|all]   e.g.  make_tasks.py neutral --tag 832 ;  make_tasks.py neutral --tag abot --no-caption ;  make_tasks.py wp
[10/19 EXP 1] the 60-s program P6 (baselines/convert.py LONG_PROGRAMS): --programs P6 --frames 969 (243 latents = 81 chunks; trimmed to 961 = 60 s) --suffix _long
       --root <SET>_long --scenes <4 frozen scenes>, e.g. make_tasks.py neutral --tag 832 --programs P6 --frames 969 --suffix _long --root neutral_832_long --scenes a,b,c,d
       -> tasks/neutral_832_long.json, ids neutral_832__<scene>__P6_long (suffix _long: report.py groups them as set neutral_832_long, apart from the 10-s clips)"""
import os, sys, json, glob, argparse, numpy as np
from PIL import Image, ImageOps
B = "/home/hk28965/specwm/baselines"; sys.path.insert(0, B); import convert as CV
REF_K = np.array([416.0, 416.0, 416.0, 240.0], np.float32)
DEF = dict(wp=("1024x512", "stretch"), mg2=("640x352", "crop"), abot=("1280x704", "fit"), astra=("832x480", "stretch"), neutral=("832x480", "stretch"))
TAG = {"wp": DEF["wp"], "mg2": DEF["mg2"], "abot": DEF["abot"], "832": ("832x480", "stretch")}
PROGRAMS7 = "P1,P2,P2p,P3,P4,P4p,P5"    # 10/17: the 7 frozen programs (PROTOCOL §2: P2p/P4p = 16 deg/s, unsaturated for MG2)
ap = argparse.ArgumentParser(); ap.add_argument("set"); ap.add_argument("--programs", default=PROGRAMS7); ap.add_argument("--no-caption", action="store_true", help="write empty captions (text-free baselines' scene sets)")
ap.add_argument("--suffix", default="", help="task-id suffix, e.g. _nocap"); ap.add_argument("--tag", default="", help="neutral only: wp | mg2 | abot | 832 -> set name neutral_<tag> with that baseline's size + rule")
ap.add_argument("--res", default="", help="override WxH"); ap.add_argument("--rule", default="", choices=["", "stretch", "crop", "fit"]); ap.add_argument("--frames", type=int, default=165)
ap.add_argument("--scenes", default="", help="comma list of scene names; default = scenes/<set>/FROZEN.json (the frozen 12-scene matrix, PROTOCOL §3) when it exists, else all; 'all' = every scene in the set")
ap.add_argument("--fps", type=int, default=16)
ap.add_argument("--root", default="", help="[10/19 EXP 1, additive] write tasks/<ROOT>/... and tasks/<ROOT>.json instead of tasks/<SET>/ and tasks/<SET><suffix>.json (task ids keep the <SET>__ prefix): the P6 sets live in their own dirs (e.g. --root neutral_832_long) so the frozen tasks/<SET>/img/*.png are never rewritten while jobs read them")
a = ap.parse_args()
assert a.set in DEF, "SET must be one of %s (use --tag for neutral variants)" % list(DEF)
res, rule = TAG[a.tag] if a.tag else DEF[a.set]
if a.tag: assert a.set == "neutral" and a.tag in TAG, "--tag is for the neutral set: %s" % list(TAG)
if a.res: res = a.res
if a.rule: rule = a.rule
W, H = (int(v) for v in res.lower().split("x")); AREA = W * H + 1
SET = a.set + ("_" + a.tag if a.tag else "")
def resizecrop(image, th, tw):   # verbatim copy of baselines/mg2/run_mg2.py resizecrop (= Matrix-Game-2 _resizecrop): centre crop to th:tw
    w, h = image.size
    if h / w > th / tw: nw, nh = w, int(w * th / tw)
    else: nh, nw = h, int(h * tw / th)
    return image.crop(((w - nw) / 2, (h - nh) / 2, (w + nw) / 2, (h + nh) / 2))
def prep(img_path, name):
    im = Image.open(img_path).convert("RGB")
    if rule == "stretch": out = im.resize((W, H), Image.BILINEAR)
    elif rule == "crop": out = resizecrop(im, H, W).resize((W, H), Image.BILINEAR)
    else:
        fit = "%s/scenes/%s/fit%dx%d/%s.png" % (B, a.set, W, H, name)
        out = Image.open(fit).convert("RGB") if os.path.exists(fit) else ImageOps.fit(im, (W, H), Image.LANCZOS, centering=(0.5, 0.5))
    assert out.size == (W, H), (out.size, (W, H)); return out
scenes = sorted(glob.glob("%s/scenes/%s/*.png" % (B, a.set)) + glob.glob("%s/scenes/%s/*.jpg" % (B, a.set))); assert scenes, "no scenes in %s/scenes/%s" % (B, a.set)
FROZEN = "%s/scenes/%s/FROZEN.json" % (B, a.set); scene_source = "all scenes in scenes/%s" % a.set
if a.scenes == "all": pass
elif a.scenes: keep = a.scenes.split(","); scene_source = "--scenes"
elif os.path.exists(FROZEN): keep = json.load(open(FROZEN)); scene_source = "scenes/%s/FROZEN.json" % a.set
else: keep = None
if a.scenes != "all" and keep is not None:
    names = {os.path.splitext(os.path.basename(s))[0]: s for s in scenes}; missing = [k for k in keep if k not in names]
    assert not missing, "%s names without an image in scenes/%s: %s" % (scene_source, a.set, missing)
    scenes = [names[k] for k in keep]; assert scenes, "no scene matches %s" % scene_source   # FROZEN.json / --scenes order is kept (deterministic)
root = "%s/tasks/%s" % (B, a.root or SET); os.makedirs(root + "/img", exist_ok=True); tasks = []
C2W = {}
for P in a.programs.split(","):
    c2w, _ = CV.program_to_c2w(P, fps=a.fps, n_frames=a.frames); z = np.load("%s/programs/%s.npz" % (B, P)); ref = z["c2w"]
    if P in getattr(CV, "LONG_PROGRAMS", []): assert a.frames >= len(ref), "long program %s: --frames %d < its %d-frame c2w (P6: use --frames %d)" % (P, a.frames, len(ref), CV.long_layout(P)["ours"]["n_frames_generated"])   # [10/19]
    assert np.allclose(c2w[:len(ref)], ref, atol=1e-5), "program %s: regenerated c2w differs from the frozen programs/%s.npz" % (P, P); C2W[P] = c2w.astype(np.float32)
for img in scenes:
    name = os.path.splitext(os.path.basename(img))[0]; cap_f = os.path.splitext(img)[0] + ".txt"
    cap = "" if a.no_caption or not os.path.exists(cap_f) else open(cap_f).read().strip()
    dst = "%s/img/%s.png" % (root, name); prep(img, name).save(dst)
    chk = Image.open(dst); assert chk.size == (W, H) and chk.mode == "RGB", (dst, chk.size, chk.mode)   # verify the written file
    for P in a.programs.split(","):
        c2w = C2W[P]; N = len(c2w); tid = "%s__%s__%s%s" % (SET, name, P, a.suffix); d = "%s/%s" % (root, tid); os.makedirs(d, exist_ok=True)
        np.save(d + "/poses.npy", c2w); np.save(d + "/intrinsics.npy", np.tile(REF_K, (N, 1))); open(d + "/caption.txt", "w").write(cap)
        tasks.append(dict(id=tid, base=tid, image=dst, video_WM="", camera_WM=cap, pose="program", action=d, program=P, scene=name, set=SET, res="%dx%d" % (W, H), area=AREA, rule=rule, frames=N, src_image=img,
                          suffix=a.suffix, caption=("none" if a.no_caption else "scene .txt"), scenes_from=scene_source))
out = "%s/tasks/%s.json" % (B, a.root) if a.root else "%s/tasks/%s%s.json" % (B, SET, a.suffix); json.dump(tasks, open(out, "w"), indent=1)
print(out, len(tasks), "tasks;", len(scenes), "scenes (%s) x" % scene_source, a.programs, "| image %dx%d rule %s area %d poses %d | captions %s" % (W, H, rule, AREA, a.frames, "NONE (--no-caption)" if a.no_caption else "from scene .txt (%d non-empty)" % sum(1 for t in tasks if t["camera_WM"])))
