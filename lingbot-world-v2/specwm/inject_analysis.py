"""[10/20 §52] Analysis of specwm/inject_probe.py runs: divergence chunk k*, per-chunk distances, follow ratios, f20-style figure + tables.
Spaces: (i) latent space: a chunk = its 3 latent frames flattened; distances normalised by the mean pairwise distance between T seeds at that chunk
(1.0 = as far apart as two independent 14B seeds); (ii) f20 PCA space per scene: PCA (q = 8, first 2 PCs) fitted on the frames of T + G-alone
(all seeds), chunk position = mean of its 3 frame projections; T spread s_T(c) = 90th percentile over seeds of |T_s(c) - mean_T(c)|.
k* = first chunk c with |mean_G(c) - mean_T(c)| > s_T(c) (reported in both spaces).
follow ratio (arm, c > k) = d_G / (d_T + d_G) with same-seed latent distances; 1 = follows T, 0 = back on its own path.
usage: envrun python -m specwm.inject_analysis [--tags 1p3b_4step,14b_2step]"""
import argparse, os, json, glob, numpy as np, torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = "/home/hk28965/specwm"; R = S + "/lingbot-world-v2/specwm_out/inject"; TD = S + "/lingbot-world-v2/specwm_out/paths2"; PM = S + "/paper/meeting_1012"
ap = argparse.ArgumentParser(); ap.add_argument("--tags", default="1p3b_4step,14b_2step"); ap.add_argument("--fig", default=PM + "/figs/f21_inject.png"); ap.add_argument("--data", default=PM + "/data/f21_inject.json"); a = ap.parse_args()
TASKS = ["sand_filling_up_jar", "sandcastle_wave_impact", "easy_lake", "easy_stonehenge_sunset", "pouring_water_into_cup"]; SEEDS = [1234, 4321, 9999, 11, 22, 33, 44, 55, 66, 77]; torch.set_num_threads(8)
def L(p): return torch.load(p).float() if os.path.exists(p) else None
def chunks(x): c, f, h, w = x.shape; return x.view(c, f // 3, 3, h, w).permute(1, 0, 2, 3, 4).reshape(f // 3, -1)
def frames(x): return x.permute(1, 0, 2, 3).reshape(x.shape[1], -1)
def se(v): v = np.asarray(v, dtype=np.float64); return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
OUT = {}; FIGROWS = {}
for tag in a.tags.split(","):
    root = "%s/%s" % (R, tag)
    if not os.path.isdir(root): continue
    arms = sorted({os.path.basename(d).rsplit("_s", 1)[0] for d in glob.glob(root + "/*/*_s*") if os.path.exists(d + "/lat.pt") and os.path.basename(d).split("@")[0].rsplit("_s", 1)[0] in ("alone", "every", "self", "once")}, key=lambda s: (s.split("@")[0], int(s.split("@")[1]) if "@" in s else -1))
    res = {}; OUT[tag] = res
    for task in TASKS:
        seeds = [s for s in SEEDS if os.path.exists("%s/%s/alone_s%d/lat.pt" % (root, task, s))]
        if not seeds: continue
        T = {s: L("%s/%s/tau2_s0/K0_uniform4P_fx_s%d/lat.pt" % (TD, task, s)) for s in seeds}; A = {arm: {s: L("%s/%s/%s_s%d/lat.pt" % (root, task, arm, s)) for s in seeds} for arm in arms}
        A = {arm: {s: x for s, x in d.items() if x is not None} for arm, d in A.items()}; N = T[seeds[0]].shape[1] // 3
        CT = torch.stack([chunks(T[s]) for s in seeds]); mT = CT.mean(0); pw = torch.stack([(CT[i] - CT[j]).norm(dim=1) for i in range(len(seeds)) for j in range(i + 1, len(seeds))]).mean(0) if len(seeds) > 1 else CT[0].norm(dim=1)   # [N]
        # f20 PCA on T + alone frames
        X = torch.cat([frames(T[s]) for s in seeds] + [frames(A["alone"][s]) for s in seeds], 0); mu = X.mean(0); Xc = X - mu; U, Sv, V = torch.pca_lowrank(Xc, q=8, center=False); V2 = V[:, :2]; ev = (Sv[:2] ** 2 / (Xc ** 2).sum()).tolist()
        proj = (lambda mu_, V2_, N_: (lambda x: ((frames(x) - mu_) @ V2_).view(N_, 3, 2).mean(1)))(mu, V2, N)   # [N, 2]; bound per scene (used again in the figure loop)
        PT = torch.stack([proj(T[s]) for s in seeds]); mTp = PT.mean(0); sT = torch.quantile((PT - mTp).norm(dim=2), 0.9, dim=0) if len(seeds) > 1 else torch.ones(N)
        PA = torch.stack([proj(A["alone"][s]) for s in seeds]); mAp = PA.mean(0)
        CA = torch.stack([chunks(A["alone"][s]) for s in seeds]); mA = CA.mean(0); sTl = torch.quantile((CT - mT).norm(dim=2), 0.9, dim=0) if len(seeds) > 1 else torch.ones(N)
        r_p = ((mAp - mTp).norm(dim=1) / sT).tolist(); r_l = ((mA - mT).norm(dim=1) / sTl).tolist()
        ks_p = next((c for c in range(N) if r_p[c] > 1), None); ks_l = next((c for c in range(N) if r_l[c] > 1), None)
        rt = dict(seeds=seeds, N=N, explained=ev, kstar_pca=ks_p, kstar_lat=ks_l, ratio_pca=r_p, ratio_lat=r_l, T_pairwise=pw.tolist(), arms={}); res[task] = rt; FIGROWS[(tag, task)] = dict(proj=proj, mTp=mTp, PT=PT, sT=sT, ev=ev)
        for arm in arms:
            d = A[arm]; ss = [s for s in seeds if s in d]
            if not ss: continue
            k = int(arm.split("@")[1]) if "@" in arm else (0 if arm == "every" else -1)   # every: chunk 0 is the empty-context chunk of both arms -> measured from chunk 1
            dT = torch.stack([(chunks(d[s]) - chunks(T[s])).norm(dim=1) for s in ss]) / pw; dG = torch.stack([(chunks(d[s]) - chunks(A["alone"][s])).norm(dim=1) for s in ss]) / pw   # [n, N]
            Pm = torch.stack([proj(d[s]) for s in ss]).mean(0); eT = ((Pm - mTp).norm(dim=1) / sT); eG = ((Pm - mAp).norm(dim=1) / sT)
            cs = list(range(k + 1, N)); fr = (dG[:, cs] / (dT[:, cs] + dG[:, cs]).clamp_min(1e-6)).mean(1) if cs else torch.zeros(len(ss))   # per seed
            fr4 = (dG[:, cs[:4]] / (dT[:, cs[:4]] + dG[:, cs[:4]]).clamp_min(1e-6)).mean(1) if cs else torch.zeros(len(ss))
            rt["arms"][arm] = dict(k=k, n=len(ss), dT_mean=dT.mean(0).tolist(), dT_se=[se(dT[:, c].numpy()) for c in range(N)], dG_mean=dG.mean(0).tolist(), dG_se=[se(dG[:, c].numpy()) for c in range(N)],
                                   env_T=eT.tolist(), env_G=eG.tolist(), env_T_after=float(eT[cs].mean()) if cs else None, env_G_after=float(eG[cs].mean()) if cs else None, env_T_alone_same=float(((mAp - mTp).norm(dim=1) / sT)[cs].mean()) if cs else None,
                                   follow=float(fr.mean()), follow_se=se(fr.numpy()), follow4=float(fr4.mean()), follow4_se=se(fr4.numpy()), path_mean=Pm.tolist(),
                                   dT_after=float(dT[:, cs].mean()) if cs else None, dG_after=float(dG[:, cs].mean()) if cs else None, dT_first4=dT[:, cs[:4]].mean(0).tolist() if cs else [], dG_first4=dG[:, cs[:4]].mean(0).tolist() if cs else [])
        print("[%s] %s: k*_pca %s k*_lat %s | ratio_pca %s" % (tag, task, ks_p, ks_l, " ".join("%.1f" % v for v in r_p)))
        for arm, v in rt["arms"].items(): print("   %-8s n=%2d follow %.2f±%.2f (first4 %.2f) env_T %s env_G %s dT_after %s dG_after %s" % (arm, v["n"], v["follow"], v["follow_se"], v["follow4"], "%.2f" % v["env_T_after"] if v["env_T_after"] is not None else "-", "%.2f" % v["env_G_after"] if v["env_G_after"] is not None else "-", "%.2f" % v["dT_after"] if v["dT_after"] is not None else "-", "%.2f" % v["dG_after"] if v["dG_after"] is not None else "-"))
os.makedirs(os.path.dirname(a.data), exist_ok=True); json.dump(OUT, open(a.data, "w"), indent=1)
# ---------------- figure: per tag a row of 5 path panels + a row of d_T / d_G curves ----------------
tags = [t for t in a.tags.split(",") if t in OUT]; COL = {"alone": "tab:orange", "every": "tab:green", "once@1": "tab:red", "once@3": "tab:purple", "once@6": "tab:blue", "once@9": "tab:brown", "self@3": "0.5"}
fig, axes = plt.subplots(3 * len(tags), 5, figsize=(26, 5.2 * 3 * len(tags)), squeeze=False)
for ri, tag in enumerate(tags):
    root = "%s/%s" % (R, tag); GCOL = "tab:orange" if tag.startswith("1p3b") else "tab:cyan"
    for ci, task in enumerate(TASKS):
        ax = axes[3 * ri, ci]; ax2 = axes[3 * ri + 1, ci]; ax3 = axes[3 * ri + 2, ci]
        if task not in OUT[tag]: ax.set_title(task + " (no runs)"); continue
        F = FIGROWS[(tag, task)]; rt = OUT[tag][task]; seeds = rt["seeds"]; N = rt["N"]
        for s in range(F["PT"].shape[0]): p = F["PT"][s].numpy(); ax.plot(p[:, 0], p[:, 1], color="k", lw=0.5, alpha=0.25)
        m = F["mTp"].numpy(); ax.plot(m[:, 0], m[:, 1], color="k", lw=2.6, label="T: 14B 4-step (n=%d)" % len(seeds)); ax.scatter(m[:, 0], m[:, 1], color="k", s=10, zorder=3); ax.scatter(m[-1, 0], m[-1, 1], color="k", marker="x", s=70, lw=2, zorder=4)
        for arm, v in rt["arms"].items():
            col = GCOL if arm == "alone" else COL.get(arm, "0.4"); pm = np.array(v["path_mean"]); k = v["k"]; ls = "--" if arm == "every" else "-"; lw = 2.6 if arm in ("alone", "every") else 1.8
            if arm.startswith("self"): continue
            if arm == "alone":
                for s in seeds:
                    if os.path.exists("%s/%s/alone_s%d/lat.pt" % (root, task, s)): p = F["proj"](torch.load("%s/%s/alone_s%d/lat.pt" % (root, task, s)).float()).numpy(); ax.plot(p[:, 0], p[:, 1], color=col, lw=0.5, alpha=0.25)
            lab = {"alone": "G alone: %s" % tag.replace("_", " "), "every": "inj_every (teacher-forced context)"}.get(arm, "inj_once@%d" % k) + " (n=%d)" % v["n"]
            if k >= 0: ax.plot(pm[k:, 0], pm[k:, 1], color=col, lw=lw, ls=ls, label=lab); ax.scatter(pm[k, 0], pm[k, 1], color=col, marker="D", s=45, zorder=5, edgecolors="k")
            else: ax.plot(pm[:, 0], pm[:, 1], color=col, lw=lw, ls=ls, label=lab)
            ax.scatter(pm[:, 0], pm[:, 1], color=col, s=8, zorder=3); ax.scatter(pm[-1, 0], pm[-1, 1], color=col, marker="x", s=60, lw=2, zorder=4)
            cc = np.arange(N); dT = np.array(v["dT_mean"]); dG = np.array(v["dG_mean"]); eT = np.array(v["dT_se"]); eG = np.array(v["dG_se"]); sel = cc > k
            envT = np.array(v["env_T"])
            if arm == "alone": ax2.errorbar(cc, dT, eT, color=col, lw=2, label="alone: d_T"); ax3.plot(cc, envT, color=col, lw=2.2, label="G alone"); continue
            ax3.plot(cc[sel], envT[sel], color=col, lw=1.6, ls=ls, label=arm)
            if k >= 0: ax3.axvline(k + 0.5, color=col, lw=0.8, alpha=0.5)
            ax2.errorbar(cc[sel], dT[sel], eT[sel], color=col, lw=1.6, ls=ls, label="%s: d_T" % arm); ax2.errorbar(cc[sel], dG[sel], eG[sel], color=col, lw=1.0, ls=":", label="%s: d_G" % arm)
            if k >= 0: ax2.axvline(k + 0.5, color=col, lw=0.8, alpha=0.5)
        ax.set_title("%s — %s\nk*_pca = %s, k*_lat = %s; PC1 %.0f %%, PC2 %.0f %%" % (task, tag.replace("_", " "), rt["kstar_pca"], rt["kstar_lat"], 100 * rt["explained"][0], 100 * rt["explained"][1]), fontsize=10); ax.grid(alpha=0.3); ax.set_xlabel("PC 1 (frame latents, fit on T + G-alone)")
        ax2.axhline(1.0, color="k", lw=0.8, ls="--"); ax2.set_xlabel("chunk"); ax2.set_ylabel("distance / T seed-to-seed distance (1 = independent seeds)"); ax2.set_ylim(0, 1.6); ax2.grid(alpha=0.3); ax2.set_title("same-seed latent dist. to T (solid) / to G-alone (dotted)", fontsize=9)
        ax3.axhline(1.0, color="k", lw=0.8, ls="--"); ax3.set_xlabel("chunk"); ax3.set_ylabel("|mean path - T mean path| / T spread (PCA)"); ax3.grid(alpha=0.3); ax3.set_yscale("log"); ax3.set_title("envelope ratio of the seed-mean path (< 1 = inside T's envelope)", fontsize=9)
        if ci == 0: ax.legend(fontsize=7, loc="best"); ax2.legend(fontsize=6, loc="best", ncol=2); ax3.legend(fontsize=7, loc="best")
fig.suptitle("Divergence injection (§52): G's context replaced by the 14B 4-step latents at chunk k (diamond), then G continues alone. Thin = seeds, thick = seed mean, X = last chunk", fontsize=11)
fig.tight_layout(); os.makedirs(os.path.dirname(a.fig), exist_ok=True); fig.savefig(a.fig, dpi=100); print("saved", a.fig)
# ---------------- tables (markdown) ----------------
def fmt(rows, key, hi):   # bold best / underline 2nd per row
    out = []
    for name, vals in rows:
        vv = [v for v in vals if v is not None]; srt = sorted(vv, reverse=hi); cells = []
        for v in vals:
            if v is None: cells.append("-"); continue
            s = "%.2f" % v
            if len(srt) > 0 and v == srt[0]: s = "**%s**" % s
            elif len(srt) > 1 and v == srt[1]: s = "<u>%s</u>" % s
            cells.append(s)
        out.append("| %s | %s |" % (name, " | ".join(cells)))
    return out
MD = []
for tag in tags:
    res = OUT[tag]; arms_t = [arm for arm in ["once@1", "once@3", "once@6", "once@9", "every", "self@3"] if any(arm in res[t]["arms"] for t in res)]
    MD.append("\n**%s <- 14B 4-step: follow ratio (higher = follows the injected 14B context; 1 = on T's same-seed path, 0 = back on its own path), mean over chunks after the injection of d_G/(d_T+d_G) with same-seed latent distances; bold best, underline 2nd; self@3 = re-commit of G's OWN context (numerical floor). s.e. over seeds in the data json.**\n" % tag.replace("_", " "))
    MD.append("| scene (k*_pca / k*_lat) | n seeds | " + " | ".join("%s ↑" % x for x in arms_t) + " |"); MD.append("|" + "---|" * (len(arms_t) + 2))
    for t in res:
        r = res[t]; f = [(r["arms"][x]["follow"] if x in r["arms"] else None) for x in arms_t[:-1] if x != "self@3"]; cells = fmt([("", f)], None, True)[0].split("|")[2:-1]
        if "self@3" in arms_t: cells.append(" %.2f " % r["arms"]["self@3"]["follow"] if "self@3" in r["arms"] else " - ")
        MD.append("| %s (%s / %s) | %d |%s|" % (t, r["kstar_pca"], r["kstar_lat"], len(r["seeds"]), "|".join(cells)))
    MD.append("\n**Envelope distance ↓ (f20 PCA space): |arm seed-mean path - T seed-mean path| / T seed spread, mean over chunks after the injection; in parentheses G-alone over the SAME chunks; < 1 = inside the 14B envelope. Bold = best arm per scene.**\n")
    MD.append("| scene | " + " | ".join("%s ↓ (alone)" % x for x in arms_t) + " |"); MD.append("|" + "---|" * (len(arms_t) + 1))
    for t in res:
        r = res[t]; e = [(r["arms"][x]["env_T_after"] if x in r["arms"] else None) for x in arms_t]; cells = fmt([("", e)], None, False)[0].split("|")[2:-1]
        cells = ["%s (%.2f)" % (c.strip(), r["arms"][x]["env_T_alone_same"]) if x in r["arms"] else "-" for c, x in zip(cells, arms_t)]
        MD.append("| %s | %s |" % (t, " | ".join(cells)))
    MD.append("\n**Distance to G-alone's own mean path ↑ (same units; large = the arm left the generator's own path):**\n")
    MD.append("| scene | " + " | ".join("%s" % x for x in arms_t) + " |"); MD.append("|" + "---|" * (len(arms_t) + 1))
    for t in res:
        r = res[t]; MD.append("| %s | %s |" % (t, " | ".join(("%.2f" % r["arms"][x]["env_G_after"]) if x in r["arms"] else "-" for x in arms_t)))
    MD.append("\n**First 4 chunks after the injection (same-seed latent distances in units of the T seed-to-seed distance at that chunk; mean over seeds):**\n")
    MD.append("| scene | arm | d_T k+1 | d_T k+2 | d_T k+3 | d_T k+4 | d_G k+1 | d_G k+2 | d_G k+3 | d_G k+4 | follow (first 4) |"); MD.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for t in res:
        for x in arms_t:
            if x not in res[t]["arms"]: continue
            v = res[t]["arms"][x]; q4 = lambda l: " | ".join(["%.2f" % q for q in l] + ["-"] * (4 - len(l))); MD.append("| %s | %s | %s | %s | %.2f |" % (t, x, q4(v["dT_first4"]), q4(v["dG_first4"]), v["follow4"]))
open(R + "/tables.md", "w").write("\n".join(MD) + "\n"); print("\n".join(MD))
