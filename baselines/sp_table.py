#!/usr/bin/env python3
"""[10/21] Scene-persistence table from metrics_sp/scene_persist.json (scripts/scene_persist.py output): one row per (set, arm) run dir.
Columns: n clips, zs_scene = mean fraction of seconds classified "scene" (CLIP zero-shot vs sky/abstract/ui/blank), zs_last = same over the last 10-s piece,
zs_seg = per-10-s-piece scene fraction (mean over clips), cuts = mean n_cuts_dino (consecutive-second DINO cos < 0.6), dino_auc/clip_auc = mean cosine to the first 2 s
(NOT a drift metric under a moving camera; shown for reference), blank = mean fraction of blank/single-colour seconds.
usage: sp_table.py [--metrics metrics_sp/scene_persist.json] [--sets mg2_long,neutral_832_long,...] [--md OUT.md]"""
import json, argparse, collections, os, re
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--metrics", default="metrics_sp/scene_persist.json"); ap.add_argument("--sets", default=""); ap.add_argument("--md", default=""); ap.add_argument("--min-n", type=int, default=1)
a = ap.parse_args(); M = json.load(open(a.metrics)); G = collections.defaultdict(list)
for v, r in M.items():
    if "error" in r: continue
    d = os.path.basename(os.path.dirname(os.path.dirname(v)))      # runs/<dir>/<clip>/video.mp4
    m = re.match(r"(ours|wp|mg2|abot|astra|sf)_(.+?)_((?:neutral_832|neutral_abot|mg2|wp|astra|abot)(?:_long)?)$", d)
    if not m: continue
    src, arm, st = m.groups(); G[(st, src + ":" + arm if src != "ours" else arm)].append(r)
sets = a.sets.split(",") if a.sets else sorted({s for s, _ in G})
def mean(xs): return float(np.mean(xs)) if xs else float("nan")
lines = []
for st in sets:
    rows = []
    for (s, arm), rs in sorted(G.items()):
        if s != st or len(rs) < a.min_n: continue
        seg = np.array([r["zs_scene_seg"][:6] + [np.nan] * (6 - len(r["zs_scene_seg"][:6])) for r in rs], dtype=float)
        rows.append(dict(arm=arm, n=len(rs), zs=mean([r["zs_frac"]["scene"] for r in rs]), last=mean([r["zs_scene_last"] for r in rs]), seg=np.nanmean(seg, 0),
                         cuts=mean([r["n_cuts_dino"] for r in rs]), dino=mean([r["dino_auc"] for r in rs]), clip=mean([r["clip_auc"] for r in rs]), blank=mean([r["zs_frac"]["blank"] + r["zs_frac"]["abstract"] for r in rs])))
    if not rows: continue
    def mark(key, hi=True):
        vals = sorted({round(r[key], 3) for r in rows}, reverse=hi); best = vals[0]; sec = vals[1] if len(vals) > 1 else None
        for r in rows: r[key + "_s"] = ("**%.2f**" if round(r[key], 3) == best else "<u>%.2f</u>" if round(r[key], 3) == sec else "%.2f") % r[key]
    mark("zs"); mark("last"); mark("cuts", False); mark("blank", False); mark("dino"); mark("clip")
    lines += ["", "### %s (scene persistence; n clips per arm)" % st, "", "| arm | n | zs_scene ↑ | zs_last ↑ | zs per 10 s (6 pieces) | cuts ↓ | blank+abstract ↓ | dino_auc (ref) | clip_auc (ref) |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s |" % (r["arm"], r["n"], r["zs_s"], r["last_s"], " ".join("%.2f" % x for x in r["seg"]), r["cuts_s"], r["blank_s"], r["dino_s"], r["clip_s"]))
out = "\n".join(lines); print(out)
if a.md: open(a.md, "w").write("# Scene persistence (scripts/scene_persist.py; %s)\n\nzs_scene = fraction of 1-fps frames whose CLIP zero-shot class is 'scene' (vs sky / abstract / UI / blank); zs_last = last 10-s piece; cuts = DINOv2 consecutive-second cosine < 0.6; dino/clip_auc = mean cosine to the first 2 s (reference only: P6 pans 160 deg and translates 21 units, so leaving the start view is expected). Best **bold**, second <u>underlined</u>.\n" % a.metrics + out + "\n")
