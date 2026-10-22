"""Merge baselines/astra/calib_astra.json into baselines/calib.json["astra"] without touching other models' entries (calib.json is shared and edited
by the other baseline folders' merge scripts; it is re-read right before writing). The previous 'astra' entry (convert.py's 16-fps draft) is kept
under "previous_convert_py_entry" the first time. NOTE: `convert.py build` regenerates calib.json from its own CALIB dict and would drop this merge —
re-run this script after any build.
usage: python3 /home/hk28965/specwm/baselines/astra/calib_merge.py [--dry]"""
import json, os, sys, time
B = "/home/hk28965/specwm/baselines"; src = json.load(open(B + "/astra/calib_astra.json")); dry = "--dry" in sys.argv
C = json.load(open(B + "/calib.json")); old = C.get("astra", {}) if isinstance(C.get("astra"), dict) else {}
new = dict(src); new["merged_from"] = "baselines/astra/calib_astra.json"; new["merged_at"] = time.strftime("%Y-%m-%d %H:%M %Z")
prev = old.get("previous_convert_py_entry") or ({k: v for k, v in old.items() if k not in ("merged_from", "merged_at", "previous_convert_py_entry")} if old and "merged_from" not in old else None)
if prev: new["previous_convert_py_entry"] = prev
C["astra"] = new
if dry: print(json.dumps(new, indent=1)); sys.exit(0)
json.dump(C, open(B + "/calib.json", "w"), indent=1, ensure_ascii=False)
print("calib.json astra updated: fps %s, translation_units_per_u %s, keys %d (other entries untouched: %s)" % (new.get("fps"), new.get("translation_units_per_u"), len(new), [k for k in C if k != "astra"]))
