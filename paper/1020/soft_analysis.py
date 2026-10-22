"""[10/20] Soft re-anchor sweep, CPU analysis: same-model pair (device 14B 2-step <- cloud 14B 4-step T), k = 6, arms alpha 0 (alone) / soft@6_a0.25 / a0.5 /
a0.75 / once@6 (alpha 1) / prog@6 (0.25 -> 0.5 -> 1.0 over chunks 6-8); every = ceiling, self@3 = floor for reference.
Metrics (definitions copied from specwm/inject_analysis.py): follow ratio = mean over chunks 7..12 of d_G / (d_T + d_G), same-seed latent distances
normalised by the mean pairwise T-seed distance at that chunk; envelope distance = |arm seed-mean path - T seed-mean path| / T 90th-pct seed spread in the
scene's PCA (fit on T + alone frames), mean over chunks 7..12.  Boundary jump (data/soft_boundary.json from soft_decode.py): device-view L1 at the chunk
boundary 6|7 / mean(alone, T) at the same boundary (inject_followups.py / inject_followups_report.py convention).  Judge (data/soft_vlm.json): quality 1-10,
drift 0-100 of the device-view mp4s.  Writes data/soft_reanchor.json and prints the markdown tables (English, bold best, underline 2nd among the alpha arms).
usage: CUDA_VISIBLE_DEVICES="" envrun python /home/hk28965/specwm/paper/1020/soft_analysis.py"""
import os, re, json, glob, numpy as np, torch
S = "/home/hk28965/specwm"; R = S + "/lingbot-world-v2/specwm_out/inject/14b_2step"; TD = S + "/lingbot-world-v2/specwm_out/paths2"; D = S + "/paper/1020/data"; torch.set_num_threads(8)
SCENES = ["sandcastle_wave_impact", "easy_stonehenge_sunset", "sand_filling_up_jar"]; SHORT = {"sandcastle_wave_impact": "sandcastle", "easy_stonehenge_sunset": "stonehenge", "sand_filling_up_jar": "jar"}
SEEDS = [1234, 4321, 9999, 11, 22, 33, 44, 55, 66, 77]; K = 6
ARMS = [("soft@6_a0.25", "alpha 0.25"), ("soft@6_a0.5", "alpha 0.5"), ("soft@6_a0.75", "alpha 0.75"), ("once@6", "alpha 1 (once@6)"), ("prog@6", "progressive 0.25/0.5/1")]
REF = [("every", "every (ceiling)"), ("self@3", "self@3 (floor)")]
def L(p): return torch.load(p).float() if os.path.exists(p) else None
def chunks(x): c, f, h, w = x.shape; return x.view(c, f // 3, 3, h, w).permute(1, 0, 2, 3, 4).reshape(f // 3, -1)
def frames(x): return x.permute(1, 0, 2, 3).reshape(x.shape[1], -1)
def se(v): v = np.asarray(v, dtype=np.float64); return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
def mark(vals, hi, fmt="%.2f"):
    """bold best / underline 2nd among the non-None vals (direction hi)"""
    vv = sorted([v for v in vals if v is not None], reverse=hi); out = []
    for v in vals:
        if v is None: out.append("-"); continue
        s = fmt % v
        if vv and v == vv[0]: s = "**%s**" % s
        elif len(vv) > 1 and v == vv[1]: s = "<u>%s</u>" % s
        out.append(s)
    return out
OUT = {}
for sc in SCENES:
    seeds = [s for s in SEEDS if os.path.exists("%s/%s/alone_s%d/lat.pt" % (R, sc, s))]
    if not seeds: continue
    T = {s: L("%s/%s/tau2_s0/K0_uniform4P_fx_s%d/lat.pt" % (TD, sc, s)) for s in seeds}; A = {"alone": {s: L("%s/%s/alone_s%d/lat.pt" % (R, sc, s)) for s in seeds}}
    for a, _ in ARMS + REF: A[a] = {s: x for s in seeds for x in [L("%s/%s/%s_s%d/lat.pt" % (R, sc, a, s))] if x is not None}
    N = T[seeds[0]].shape[1] // 3; CT = torch.stack([chunks(T[s]) for s in seeds]); mT = CT.mean(0)
    pw = torch.stack([(CT[i] - CT[j]).norm(dim=1) for i in range(len(seeds)) for j in range(i + 1, len(seeds))]).mean(0)
    X = torch.cat([frames(T[s]) for s in seeds] + [frames(A["alone"][s]) for s in seeds], 0); mu = X.mean(0); Xc = X - mu; U, Sv, V = torch.pca_lowrank(Xc, q=8, center=False); V2 = V[:, :2]
    proj = lambda x: ((frames(x) - mu) @ V2).view(N, 3, 2).mean(1)
    PT = torch.stack([proj(T[s]) for s in seeds]); mTp = PT.mean(0); sT = torch.quantile((PT - mTp).norm(dim=2), 0.9, dim=0); PA = torch.stack([proj(A["alone"][s]) for s in seeds]); mAp = PA.mean(0)
    cs = list(range(K + 1, N)); rt = dict(seeds=seeds, N=N, arms={}); OUT[sc] = rt
    rt["arms"]["alone"] = dict(n=len(seeds), env_T_after=float(((mAp - mTp).norm(dim=1) / sT)[cs].mean()), follow=0.0, follow_se=0.0, dT_after=float((torch.stack([(chunks(A["alone"][s]) - chunks(T[s])).norm(dim=1) for s in seeds]) / pw)[:, cs].mean()))
    for a, _ in ARMS + REF:
        d = A[a]; ss = [s for s in seeds if s in d]
        if not ss: continue
        dT = torch.stack([(chunks(d[s]) - chunks(T[s])).norm(dim=1) for s in ss]) / pw; dG = torch.stack([(chunks(d[s]) - chunks(A["alone"][s])).norm(dim=1) for s in ss]) / pw
        Pm = torch.stack([proj(d[s]) for s in ss]).mean(0); eT = (Pm - mTp).norm(dim=1) / sT; eG = (Pm - mAp).norm(dim=1) / sT
        fr = (dG[:, cs] / (dT[:, cs] + dG[:, cs]).clamp_min(1e-6)).mean(1); fr4 = (dG[:, cs[:4]] / (dT[:, cs[:4]] + dG[:, cs[:4]]).clamp_min(1e-6)).mean(1)
        rt["arms"][a] = dict(n=len(ss), follow=float(fr.mean()), follow_se=se(fr.numpy()), follow4=float(fr4.mean()), env_T_after=float(eT[cs].mean()), env_G_after=float(eG[cs].mean()), env_T=eT.tolist(),
                             dT_after=float(dT[:, cs].mean()), dG_after=float(dG[:, cs].mean()), dT_mean=dT.mean(0).tolist(), dG_mean=dG.mean(0).tolist(), dT_first=dT[:, cs[:3]].mean(0).tolist())
    print("[%s] n=%d: " % (sc, len(seeds)) + " | ".join("%s follow %.2f env %.2f dT %.2f" % (a, v["follow"], v["env_T_after"], v["dT_after"]) for a, v in rt["arms"].items()))
# ---------------- boundary json
BJ = D + "/soft_boundary.json"; B = json.load(open(BJ)) if os.path.exists(BJ) else {}
bres = {}
for sc in SCENES + ["pooled"]:
    keys = [k for k in B if (sc == "pooled" or k.startswith(sc + "/")) and "alone" in B[k] and "T" in B[k]]
    if not keys: continue
    row = {}
    ordinary = np.array([(B[k]["alone"]["l1_b"][K] + B[k]["T"]["l1_b"][K]) / 2 for k in keys]); row["ordinary"] = (float(ordinary.mean()), se(ordinary), float(np.mean([B[k]["alone"]["l1_b"][K] for k in keys])), float(np.mean([B[k]["T"]["l1_b"][K] for k in keys])))
    for a, _ in ARMS:
        kk = [k for k in keys if "dev:" + a in B[k]]
        if not kk: continue
        sp = np.array([B[k]["dev:" + a]["l1_b"][K] for k in kk]); od = np.array([(B[k]["alone"]["l1_b"][K] + B[k]["T"]["l1_b"][K]) / 2 for k in kk]); ratio = sp / od
        own = np.array([B[k][a]["l1_b"][K] for k in kk if a in B[k]])
        ex = {}
        if a == "prog@6":   # the later re-blend boundaries 7|8 and 8|9 of the device view vs the ordinary ones
            for c in (7, 8):
                spc = np.array([B[k]["dev:" + a]["l1_b"][c] for k in kk]); odc = np.array([(B[k]["alone"]["l1_b"][c] + B[k]["T"]["l1_b"][c]) / 2 for k in kk]); ex[c] = (float(spc.mean()), se(spc), float((spc / odc).mean()))
        row[a] = dict(n=len(kk), splice=float(sp.mean()), splice_se=se(sp), ratio=float(ratio.mean()), ratio_med=float(np.median(ratio)), gt15=int((ratio > 1.5).sum()), own=float(own.mean()) if len(own) else None, extra=ex)
    bres[sc] = row
# ---------------- judge json
VJ = D + "/soft_vlm.json"; J = json.load(open(VJ)) if os.path.exists(VJ) else {}; rows = {}
for p, r in J.items():
    m = re.match(r".*/(alone|soft@6_a0\.25|soft@6_a0\.5|soft@6_a0\.75|once@6|prog@6|T)_(.+)_s(\d+)\.mp4$", p)
    if m: rows[(m.group(1), m.group(2), int(m.group(3)))] = r
jres = {}
JARMS = ["T", "alone"] + [a for a, _ in ARMS]
for key in ("quality", "drift"):
    for sc in SCENES + ["pooled"]:
        for a in JARMS:
            v = [r[key] for (aa, s, sd), r in rows.items() if aa == a and (sc == "pooled" or s == sc) and not np.isnan(r[key])]
            if v: jres.setdefault(key, {}).setdefault(sc, {})[a] = (float(np.mean(v)), se(v), len(v))
            if a != "alone":
                dd = [rows[(a, s, sd)][key] - rows[("alone", s, sd)][key] for (aa, s, sd) in rows if aa == a and (sc == "pooled" or s == sc) and ("alone", s, sd) in rows]
                dd = [x for x in dd if not np.isnan(x)]
                if dd: g = sum(1 for x in dd if (x > 0 if key == "quality" else x < 0)); b = sum(1 for x in dd if (x < 0 if key == "quality" else x > 0)); jres[key][sc]["d:" + a] = (float(np.mean(dd)), se(dd), len(dd), g, b, len(dd) - g - b)
json.dump(dict(latent=OUT, boundary=bres, judge=jres), open(D + "/soft_reanchor.json", "w"), indent=1)
# ---------------- markdown
MD = []; names = [a for a, _ in ARMS]; labels = [l for _, l in ARMS]
MD.append("\n**Table A - following after the re-anchor (chunks 7-12), 14B 2-step device <- 14B 4-step cloud, k = 6. follow ratio ↑ = mean of d_G/(d_T+d_G), same-seed latent distances (1 = on the cloud's same-seed path, 0 = on the device's own path); envelope distance ↓ = |arm seed-mean path - cloud seed-mean path| / cloud seed spread (PCA; < 1 = inside the cloud's seed spread); d_T ↓ = same-seed latent distance to the cloud clip (1 = two independent cloud seeds). Mean over seeds (follow: ± s.e.). Bold best / underline 2nd among the five re-anchor arms; alone = alpha 0, every / self@3 = ceiling / floor for reference.**\n")
MD.append("| scene | n | metric | alpha 0 (alone) | " + " | ".join(labels) + " | every | self@3 |"); MD.append("|---|---|---|---|" + "---|" * (len(labels) + 2))
for sc in SCENES:
    if sc not in OUT: continue
    r = OUT[sc]["arms"]; n = OUT[sc]["arms"]["alone"]["n"]
    for met, hi, lab in (("follow", True, "follow ↑"), ("env_T_after", False, "envelope ↓"), ("dT_after", False, "d_T ↓")):
        vals = [r[a][met] if a in r else None for a in names]; cells = mark(vals, hi)
        if met == "follow": cells = [c + (" ± %.2f" % r[a]["follow_se"] if a in r else "") for c, a in zip(cells, names)]
        MD.append("| %s | %d | %s | %s | %s | %s | %s |" % (SHORT[sc], n, lab, "%.2f" % r["alone"][met], " | ".join(cells), "%.2f" % r["every"][met] if "every" in r else "-", "%.2f" % r["self@3"][met] if "self@3" in r else "-"))
if OUT:
    MD.append("| mean of 3 scenes | | follow ↑ | 0.00 | " + " | ".join(mark([np.mean([OUT[s]["arms"][a]["follow"] for s in OUT if a in OUT[s]["arms"]]) if any(a in OUT[s]["arms"] for s in OUT) else None for a in names], True)) + " | %.2f | %.2f |" % (np.mean([OUT[s]["arms"]["every"]["follow"] for s in OUT if "every" in OUT[s]["arms"]]), np.mean([OUT[s]["arms"]["self@3"]["follow"] for s in OUT if "self@3" in OUT[s]["arms"]])))
    MD.append("| mean of 3 scenes | | envelope ↓ | %.2f | " % np.mean([OUT[s]["arms"]["alone"]["env_T_after"] for s in OUT]) + " | ".join(mark([np.mean([OUT[s]["arms"][a]["env_T_after"] for s in OUT if a in OUT[s]["arms"]]) if any(a in OUT[s]["arms"] for s in OUT) else None for a in names], False)) + " | %.2f | %.2f |" % (np.mean([OUT[s]["arms"]["every"]["env_T_after"] for s in OUT if "every" in OUT[s]["arms"]]), np.mean([OUT[s]["arms"]["self@3"]["env_T_after"] for s in OUT if "self@3" in OUT[s]["arms"]])))
MD.append("\n**Table A2 - same-seed distance to the cloud clip d_T ↓ in the first three chunks after the re-anchor (7 / 8 / 9; 1 = independent cloud seeds), mean over seeds:**\n")
MD.append("| scene | " + " | ".join(labels) + " |"); MD.append("|---|" + "---|" * len(labels))
for sc in SCENES:
    if sc in OUT: MD.append("| %s | %s |" % (SHORT[sc], " | ".join(" / ".join("%.2f" % q for q in OUT[sc]["arms"][a]["dT_first"]) if a in OUT[sc]["arms"] else "-" for a in names)))
if bres:
    MD.append("\n**Table B - re-anchor boundary jump ↓ at chunk boundary 6|7 (device view = the device's own chunks 0-6 followed by the arm's chunks 7-12, decoded jointly; frame-to-frame pixel L1 on 8-bit, pixel frames 80|81). ordinary = mean of the device-alone and the cloud clip at the same boundary; ratio = splice / ordinary (1 = an ordinary chunk boundary; the 1.3B hard re-anchor was 2.55x). n = seeds. Bold best / underline 2nd.**\n")
    MD.append("| scene | n | ordinary L1 (alone / cloud) | " + " | ".join("%s: L1, ratio" % l for l in labels) + " | seeds ratio > 1.5 |"); MD.append("|---|---|---|" + "---|" * (len(labels) + 1))
    for sc in SCENES + ["pooled"]:
        if sc not in bres: continue
        row = bres[sc]; vals = [row[a]["ratio"] if a in row else None for a in names]; cells = mark(vals, False)
        MD.append("| %s | %d | %.1f (%.1f / %.1f) | %s | %s |" % (SHORT.get(sc, sc), max(row[a]["n"] for a in names if a in row), row["ordinary"][0], row["ordinary"][2], row["ordinary"][3],
                  " | ".join("%.1f ± %.1f, %s" % (row[a]["splice"], row[a]["splice_se"], c) if a in row else "-" for a, c in zip(names, cells)), " / ".join("%d" % row[a]["gt15"] if a in row else "-" for a in names)))
    MD.append("\nProgressive arm, later re-blend boundaries of the device view (L1, ratio vs ordinary): " + "; ".join("%s: " % SHORT.get(sc, sc) + ", ".join("7|8 %.1f (%.2fx)" % (v[7][0], v[7][2]) + " 8|9 %.1f (%.2fx)" % (v[8][0], v[8][2]) for v in [bres[sc]["prog@6"]["extra"]] if v) for sc in SCENES + ["pooled"] if sc in bres and "prog@6" in bres[sc] and bres[sc]["prog@6"]["extra"]))
    MD.append("Own boundary of the un-spliced clips (blended context as stored, then the continuation; L1 at 6|7, pooled): " + ", ".join("%s %.1f" % (l, bres["pooled"][a]["own"]) for a, l in ARMS if a in bres.get("pooled", {}) and bres["pooled"][a]["own"] is not None))
if jres:
    MD.append("\n**Table C - judge (Qwen3-VL-32B, 8 frames per clip) on the DEVICE-VIEW clips (own chunks 0-6 + re-anchored continuation): quality 1-10 ↑, drift 0-100 ↓; mean ± s.e. over seeds; T = the cloud clip itself. Bold best / underline 2nd among the re-anchor arms.**\n")
    for key, arrow, hi in (("quality", "↑", True), ("drift", "↓", False)):
        MD.append("| %s %s | T (cloud) | alpha 0 (alone) | " % (key, arrow) + " | ".join(labels) + " |"); MD.append("|---|---|---|" + "---|" * len(labels))
        for sc in SCENES + ["pooled"]:
            rr = jres.get(key, {}).get(sc, {})
            if not rr: continue
            vals = [rr[a][0] if a in rr else None for a in names]; cells = mark(vals, hi)
            MD.append("| %s | %s | %s | %s |" % (SHORT.get(sc, sc), "%.2f ± %.2f" % rr["T"][:2] if "T" in rr else "-", "%.2f ± %.2f" % rr["alone"][:2] if "alone" in rr else "-", " | ".join("%s ± %.2f (n=%d)" % (c, rr[a][1], rr[a][2]) if a in rr else "-" for a, c in zip(names, cells))))
        MD.append("")
    MD.append("Paired vs alpha 0 (alone), pooled over the 3 scenes, mean ± s.e. (better / worse / tie in the metric's own direction): " + "; ".join("%s: quality %+.2f ± %.2f (%d/%d/%d), drift %+.2f ± %.2f (%d/%d/%d)" % ((l,) + jres["quality"]["pooled"]["d:" + a][:2] + jres["quality"]["pooled"]["d:" + a][3:] + jres["drift"]["pooled"]["d:" + a][:2] + jres["drift"]["pooled"]["d:" + a][3:]) for a, l in ARMS if "d:" + a in jres.get("quality", {}).get("pooled", {})))
open(D + "/soft_reanchor_tables.md", "w").write("\n".join(MD) + "\n"); print("\n".join(MD))
