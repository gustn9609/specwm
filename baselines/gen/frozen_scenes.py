#!/usr/bin/env python3
"""[10/17 frozen baselines] The frozen scene subset of every scene set (orchestrator decision 10/17: 12 scenes per own-scene set, GPU budget).
frozen(set) -> sorted list of scene names.  Authority = baselines/scenes/<set>/FROZEN.json (a JSON list of scene names, written by the task-B agent);
when that file is absent the COMMON fallback rule is applied to the scenes present in baselines/scenes/<set>/ (*.png | *.jpg):
  wp       the first 12 sorted Biome seeds (every scene that is not a waypoint-1-small starter_*; SOURCES.md)
  mg2      the first 8 sorted universal_* + the first 4 sorted mc_* (Minecraft)
  abot     the 7 web_client presets (every scene that is not a case* poster) + the first 5 sorted case* posters
  astra    all (3)
  neutral  all (the 12 a832 scenes)
Every name is checked to exist as an image in the set's directory.  image_path(set, name) -> the scene image (.png preferred, else .jpg).
usage: python3 frozen_scenes.py SET [--json]      (plain python3, no numpy; importable: sys.path.insert(0, '<baselines>/gen'); import frozen_scenes)"""
import os, sys, json, glob
B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # baselines/
SCENES = os.path.join(B, "scenes")
N_FROZEN = 12


def scene_names(set_):
    d = os.path.join(SCENES, set_)
    names = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.jpg"))}
    if not names:
        raise SystemExit("frozen_scenes: no scene images under %s" % d)
    return sorted(names)


def image_path(set_, name):
    for ext in (".png", ".jpg"):
        p = os.path.join(SCENES, set_, name + ext)
        if os.path.exists(p):
            return p
    raise SystemExit("frozen_scenes: scene %s/%s has no .png/.jpg" % (set_, name))


def fallback(set_, names):
    if set_ == "wp":
        return sorted(n for n in names if not n.startswith("starter_"))[:N_FROZEN]
    if set_ == "mg2":
        return sorted(n for n in names if n.startswith("universal_"))[:8] + sorted(n for n in names if n.startswith("mc_"))[:4]
    if set_ == "abot":
        return sorted(n for n in names if not n.startswith("case"))[:7] + sorted(n for n in names if n.startswith("case"))[:5]
    if set_ == "astra":
        return sorted(names)
    if set_ == "neutral":
        return sorted(names)[:N_FROZEN]
    raise SystemExit("frozen_scenes: unknown set %r (wp | mg2 | abot | astra | neutral)" % set_)


def frozen(set_):
    """sorted scene names of the frozen subset; FROZEN.json when present, else the fallback rule (source recorded in frozen.source)"""
    names = scene_names(set_); f = os.path.join(SCENES, set_, "FROZEN.json")
    if os.path.exists(f):
        sel = json.load(open(f))
        if isinstance(sel, dict):                      # tolerate {"scenes": [...]} / {"frozen": [...]}
            sel = sel.get("scenes") or sel.get("frozen") or sel.get("names")
        assert isinstance(sel, list) and sel, "%s: expected a non-empty JSON list of scene names" % f
        missing = [n for n in sel if n not in names]
        assert not missing, "%s names scenes without an image: %s" % (f, missing)
        frozen.source = f
        return sorted(dict.fromkeys(sel))
    frozen.source = "fallback rule (no %s)" % f
    return sorted(fallback(set_, names))


frozen.source = None

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("set"); ap.add_argument("--json", action="store_true"); a = ap.parse_args()
    sel = frozen(a.set)
    if a.json:
        print(json.dumps(sel))
    else:
        print("%s: %d scenes (%s)" % (a.set, len(sel), frozen.source))
        for n in sel: print("  " + n)
