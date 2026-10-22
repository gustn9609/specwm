"""[10/20] HANDOFF arms of the divergence-injection probe (specwm/inject_probe.py, facts §52): the 1.3B generates on its own, at chunk k the 14B
4-step takes over for m chunks ON THE 1.3B's OWN CONTEXT (its chunks 0..k re-committed into the 14B's KV through the clean pass), then the 1.3B
re-commits chunks 0..k+m (its own 0..k + the 14B's k+1..k+m) and continues k+m+1..N-1.  Does the 1.3B stay in the 14B distribution afterwards?
Two sequential passes with saved latents (the probe's solo AR loop, same noise rule gen(seed, c), so chunks 0..k are the archived `alone` latents):
  --phase 14b  : per (task, seed, spec k:m[:steps]) -> <out>/<task>/seg14b_k<k>m<m>[_<steps>step]_s<seed>/lat.pt = chunks 0..k+m (alone 0..k + 14B k+1..k+m)
  --phase 1p3b : per arm hand@<k>m<m>[_2step]: KV reset, seg chunks 0..k+m re-committed, 1.3B continues -> <out>/<task>/<arm>_s<seed>/{lat.pt, run.json}
                 (hand@3m1 uses chunk 4 of seg k3m3: identical by construction, same context + same noise); checks that the loaded 1.3B reproduces `alone` chunk 0
  --phase analyze : inject_analysis.py definitions (latent d_T / d_G in units of the T seed pairwise distance, follow = d_G/(d_T+d_G); f20 PCA on
                 T + alone frames, envelope ratio = |seed-mean - T seed-mean| / 90th-pct T spread) for the chunks AFTER the handoff and for the 14B
                 segment itself -> paper/1020/data/f22_handoff.json + handoff_tables.md
usage: CUDA_VISIBLE_DEVICES=5 envrun python -m specwm.inject_handoff --phase 14b --specs 3:3,6:2 --tasks sandcastle_wave_impact --seeds 1234,4321"""
import argparse, os, sys, json, time, numpy as np, torch
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--phase", default="14b", choices=["14b", "1p3b", "analyze", "reject"])
ap.add_argument("--drift-head", default=S + "/lingbot-world-v2/specwm_out/drift_head/head_L20_all.pt"); ap.add_argument("--tau", default="-3.25", help="reject phase: drift-head logit > tau -> the 14B regenerates the chunk; comma list = several variants; 'p25' = the tau that flags 25 %% of the alone chunks of the run's scenes")
ap.add_argument("--tasks", default="sandcastle_wave_impact,easy_stonehenge_sunset,sand_filling_up_jar,easy_lake,pouring_water_into_cup"); ap.add_argument("--seeds", default="1234,4321,9999,11,22,33,44,55,66,77")
ap.add_argument("--specs", default="3:3,6:2", help="14b phase: k:m[:steps] segments to generate (steps default 4)"); ap.add_argument("--arms", default="hand@3m3,hand@3m1,hand@6m2", help="1p3b phase")
ap.add_argument("--out", default=S + "/lingbot-world-v2/specwm_out/inject/handoff"); ap.add_argument("--alone", default=S + "/lingbot-world-v2/specwm_out/inject/1p3b_4step"); ap.add_argument("--tdir", default=S + "/lingbot-world-v2/specwm_out/paths2")
ap.add_argument("--stream", default=S + "/data/actstream/tau2_s0"); ap.add_argument("--frames", type=int, default=161); ap.add_argument("--area", type=int, default=480 * 832); ap.add_argument("--window", type=int, default=18)
ap.add_argument("--data", default=S + "/paper/1020/data/f22_handoff.json"); ap.add_argument("--tables", default=S + "/paper/1020/data/handoff_tables.md"); a = ap.parse_args()
OUT = a.out; os.makedirs(OUT, exist_ok=True); PROG = OUT + "/PROGRESS.md"; TASKS = a.tasks.split(","); SEEDS = [int(x) for x in a.seeds.split(",")]
def log(*x):
    s = " ".join(str(v) for v in x); print("[handoff] " + s, flush=True)
    with open(PROG, "a") as f: f.write("- %s %s %s\n" % (time.strftime("%m/%d %H:%M:%S"), a.phase, s))
def apath(task, seed): return "%s/%s/alone_s%d/lat.pt" % (a.alone, task, seed)
def tpath(task, seed): return "%s/%s/tau2_s0/K0_uniform4P_fx_s%d/lat.pt" % (a.tdir, task, seed)
def segname(k, m, steps): return "seg14b_k%dm%d%s" % (k, m, "" if steps == 4 else "_%dstep" % steps)
def parse_arm(arm):   # hand@3m3 / hand@3m1 / hand@6m2 / hand@3m3_2step -> (k, m, steps, segment dir name)
    body = arm.split("@")[1]; steps = 2 if body.endswith("_2step") else 4; body = body.replace("_2step", ""); k, m = int(body.split("m")[0]), int(body.split("m")[1])
    seg_m = {(3, 1): 3}.get((k, m), m)   # hand@3m1 reads chunk 4 of the k3m3 segment
    return k, m, steps, segname(k, seg_m, steps)
# ======================================================================================================================= analysis
if a.phase == "analyze":
    torch.set_num_threads(8); ALL_SEEDS = [1234, 4321, 9999, 11, 22, 33, 44, 55, 66, 77]
    def L(p): return torch.load(p, map_location="cpu").float() if os.path.exists(p) else None
    def chunks(x): c, f, h, w = x.shape; return x.view(c, f // 3, 3, h, w).permute(1, 0, 2, 3, 4).reshape(f // 3, -1)
    def frames(x): return x.permute(1, 0, 2, 3).reshape(x.shape[1], -1)
    def se(v): v = np.asarray(v, dtype=np.float64); return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
    ARMS = a.arms.split(","); REF = ["once@3", "once@6"]; RES = {}
    for task in TASKS:
        seeds = [s for s in ALL_SEEDS if os.path.exists(apath(task, s))]
        if not seeds: continue
        T = {s: L(tpath(task, s)) for s in seeds}; AL = {s: L(apath(task, s)) for s in seeds}; N = T[seeds[0]].shape[1] // 3
        CT = torch.stack([chunks(T[s]) for s in seeds]); mT = CT.mean(0); pw = torch.stack([(CT[i] - CT[j]).norm(dim=1) for i in range(len(seeds)) for j in range(i + 1, len(seeds))]).mean(0)
        X = torch.cat([frames(T[s]) for s in seeds] + [frames(AL[s]) for s in seeds], 0); mu = X.mean(0); Xc = X - mu; U, Sv, V = torch.pca_lowrank(Xc, q=8, center=False); V2 = V[:, :2]; ev = (Sv[:2] ** 2 / (Xc ** 2).sum()).tolist()
        proj = lambda x: ((frames(x) - mu) @ V2).view(N, 3, 2).mean(1)
        PT = torch.stack([proj(T[s]) for s in seeds]); mTp = PT.mean(0); sT = torch.quantile((PT - mTp).norm(dim=2), 0.9, dim=0); mAp = torch.stack([proj(AL[s]) for s in seeds]).mean(0)
        rt = dict(seeds=seeds, N=N, explained=ev, T_pairwise=pw.tolist(), alone_env_T=((mAp - mTp).norm(dim=1) / sT).tolist(), arms={}); RES[task] = rt
        for arm in ARMS + REF:
            hand = arm.startswith("hand"); root = OUT if hand else a.alone
            d = {s: L("%s/%s/%s_s%d/lat.pt" % (root, task, arm, s)) for s in seeds}; d = {s: x for s, x in d.items() if x is not None}; ss = list(d)
            if not ss: continue
            rej = arm.startswith("hand@reject")
            if rej:   # verifier-driven: per-seed replaced chunks; "after" = chunks after the seed's first replacement (seed-mean quantities use the median first replacement)
                RJ = {s: json.load(open("%s/%s/%s_s%d/run.json" % (root, task, arm, s))) for s in ss}; firsts = [RJ[s]["first_replaced"] for s in ss if RJ[s]["first_replaced"] is not None]
                kmed = int(np.median(firsts)) - 1 if firsts else N - 1; k = kmed; m = 0; seg = sorted({c for s in ss for c in RJ[s]["replaced_chunks"]}); cs = list(range(kmed + 1, N))
            elif hand: k, m, steps, _ = parse_arm(arm); seg = list(range(k + 1, k + m + 1)); cs = list(range(k + m + 1, N))
            else: k = int(arm.split("@")[1]); m = 0; seg = []; cs = list(range(k + 1, N))
            dT = torch.stack([(chunks(d[s]) - chunks(T[s])).norm(dim=1) for s in ss]) / pw; dG = torch.stack([(chunks(d[s]) - chunks(AL[s])).norm(dim=1) for s in ss]) / pw
            Pm = torch.stack([proj(d[s]) for s in ss]).mean(0); eT = (Pm - mTp).norm(dim=1) / sT; eG = (Pm - mAp).norm(dim=1) / sT
            if rej:
                def msk(s, after):
                    f = RJ[s]["first_replaced"]; return torch.tensor([(c > f if f is not None else False) if after else c >= 1 for c in range(N)])
                fr = torch.stack([(dG[i][msk(s, True)] / (dT[i][msk(s, True)] + dG[i][msk(s, True)]).clamp_min(1e-6)).mean() if msk(s, True).any() else torch.tensor(float("nan")) for i, s in enumerate(ss)])
                frw = torch.stack([(dG[i][msk(s, False)] / (dT[i][msk(s, False)] + dG[i][msk(s, False)]).clamp_min(1e-6)).mean() for i, s in enumerate(ss)]); fr4 = frw
                segP = torch.stack([((proj(d[s]) - mTp).norm(dim=1) / sT)[RJ[s]["replaced_chunks"]].mean() for s in ss if RJ[s]["replaced_chunks"]]) if any(RJ[s]["replaced_chunks"] for s in ss) else torch.tensor([float("nan")])
                segd = [(dT[i][RJ[s]["replaced_chunks"]].mean(), dG[i][RJ[s]["replaced_chunks"]].mean()) for i, s in enumerate(ss) if RJ[s]["replaced_chunks"]]
                extra = dict(reject_rate=float(np.mean([np.mean(RJ[s]["replaced"]) for s in ss])), share_14b=float(np.mean([RJ[s]["share_14b"] for s in ss])), first_replaced=[RJ[s]["first_replaced"] for s in ss], first_median=float(np.median(firsts)) if firsts else None,
                             n_never=int(sum(RJ[s]["first_replaced"] is None for s in ss)), follow_whole=float(frw.mean()), follow_whole_se=se(frw.numpy()), env_T_whole=float(eT[1:].mean()), env_G_whole=float(eG[1:].mean()), env_T_alone_whole=float(((mAp - mTp).norm(dim=1) / sT)[1:].mean()),
                             seg_env_perseed=float(segP[~torch.isnan(segP)].mean()) if (~torch.isnan(segP)).any() else None, seg_dT_perseed=float(np.mean([v[0] for v in segd])) if segd else None, seg_dG_perseed=float(np.mean([v[1] for v in segd])) if segd else None,
                             replaced_per_chunk=[float(np.mean([RJ[s]["replaced"][c] for s in ss])) for c in range(N)], tau=RJ[ss[0]]["tau"], scores_mean=[float(np.mean([RJ[s]["scores"][c] for s in ss])) for c in range(N)])
                fr = fr[~torch.isnan(fr)] if (~torch.isnan(fr)).any() else torch.tensor([float("nan")])
            else:
                fr = (dG[:, cs] / (dT[:, cs] + dG[:, cs]).clamp_min(1e-6)).mean(1); fr4 = (dG[:, cs[:4]] / (dT[:, cs[:4]] + dG[:, cs[:4]]).clamp_min(1e-6)).mean(1); extra = {}
            # same-seed check of the pre-handoff chunks (must be the alone latents) and seed-mean of per-seed env ratio for the segment
            pre = (float(max(RJ[s]["pre_maxabs"] for s in ss)) if rej else float(max((d[s][:, :3 * (k + 1)] - AL[s][:, :3 * (k + 1)]).abs().max() for s in ss))) if hand else None
            rt["arms"][arm] = dict(k=k, m=m, n=len(ss), seeds=ss, seg=seg, after=cs, dT_mean=dT.mean(0).tolist(), dG_mean=dG.mean(0).tolist(), dT_se=[se(dT[:, c].numpy()) for c in range(N)], dG_se=[se(dG[:, c].numpy()) for c in range(N)],
                                   env_T=eT.tolist(), env_G=eG.tolist(), env_T_after=float(eT[cs].mean()), env_G_after=float(eG[cs].mean()), env_T_alone_same=float(((mAp - mTp).norm(dim=1) / sT)[cs].mean()),
                                   env_T_seg=float(eT[seg].mean()) if seg else None, env_T_alone_seg=float(((mAp - mTp).norm(dim=1) / sT)[seg].mean()) if seg else None, dT_seg=float(dT[:, seg].mean()) if seg else None, dG_seg=float(dG[:, seg].mean()) if seg else None,
                                   follow=float(fr.mean()), follow_se=se(fr.numpy()), follow4=float(fr4.mean()), follow4_se=se(fr4.numpy()), dT_after=float(dT[:, cs].mean()), dG_after=float(dG[:, cs].mean()),
                                   dT_first4=dT[:, cs[:4]].mean(0).tolist(), dG_first4=dG[:, cs[:4]].mean(0).tolist(), path_mean=Pm.tolist(), pre_maxabs=pre, **extra)
            print("[%s] %-10s n=%d k=%d m=%d seg env_T %s | after: env_T %.2f (alone %.2f) env_G %.2f follow %.2f±%.2f first4 %.2f dT %.2f dG %.2f" % (task, arm, len(ss), k, m, "%.2f" % rt["arms"][arm]["env_T_seg"] if seg else "-", rt["arms"][arm]["env_T_after"], rt["arms"][arm]["env_T_alone_same"], rt["arms"][arm]["env_G_after"], fr.mean(), se(fr.numpy()), fr4.mean(), dT[:, cs].mean(), dG[:, cs].mean()))
    os.makedirs(os.path.dirname(a.data), exist_ok=True); json.dump(RES, open(a.data, "w"), indent=1)
    # ---------------- tables (English; direction stated; bold best, underline 2nd among the HANDOFF arms) ----------------
    arms_all = [x for x in ARMS + REF if any(x in RES[t]["arms"] for t in RES)]
    def rank(vals, hi):
        xs = [(v, i) for i, v in enumerate(vals) if v is not None and np.isfinite(v)]; xs.sort(key=lambda t: -t[0] if hi else t[0]); r = {}
        if len(xs) > 0: r[xs[0][1]] = "b"
        if len(xs) > 1: r[xs[1][1]] = "u"
        return r
    def row(vals, hi, fmt="%.2f", cand=None):
        cand = cand if cand is not None else list(range(len(vals))); r = rank([v if i in cand else None for i, v in enumerate(vals)], hi); out = []
        for i, v in enumerate(vals):
            s = "-" if v is None or not np.isfinite(v) else fmt % v; out.append("**%s**" % s if r.get(i) == "b" else "<u>%s</u>" % s if r.get(i) == "u" else s)
        return out
    hand_idx = [i for i, x in enumerate(arms_all) if x.startswith("hand")]; md = []
    md.append("# Handoff arms (1.3B -> 14B for m chunks on the 1.3B's own context -> 1.3B continues); 14B = 4-step FP8, window 18, stream tau2_s0, same noise rule as `alone`\n")
    md.append("Definitions = inject_analysis.py (§52): latent distances in units of the T seed-pairwise distance at that chunk; follow = mean over post-handoff chunks of d_G/(d_T+d_G) (1 = on T's same-seed path, 0 = on the 1.3B's own path); envelope ratio in the scene's f20 PCA (fit on T + alone frames, all seeds) = |arm seed-mean - T seed-mean| / 90th-pct T seed spread (< 1 = inside the 14B envelope). `once@k` = the probe's context-replacement arms (14B latents of the SAME seed injected) for reference. Bold = best, underline = 2nd among the handoff arms.\n")
    md.append("**Table 1. Post-handoff chunks (c > k+m): envelope ratio to T ↓ (in parentheses: 1.3B alone over the same chunks), distance to the 1.3B's own alone path ↑, follow ratio ↑ (± s.e. over seeds).**\n")
    md.append("| scene | n seeds (%s) | " % "/".join(arms_all) + " | ".join("%s env_T ↓ (alone)" % x for x in arms_all) + " | " + " | ".join("%s env_G ↑" % x for x in arms_all) + " | " + " | ".join("%s follow ↑" % x for x in arms_all) + " |")
    md.append("|---|---|" + "---|" * (3 * len(arms_all)))
    for t, rt in RES.items():
        A = rt["arms"]; g = lambda x, key: A[x][key] if x in A else None
        e = row([g(x, "env_T_after") for x in arms_all], False, cand=hand_idx); ea = ["%s (%.2f)" % (e[i], A[x]["env_T_alone_same"]) if x in A else "-" for i, x in enumerate(arms_all)]
        gg = row([g(x, "env_G_after") for x in arms_all], True, cand=hand_idx); f = row([g(x, "follow") for x in arms_all], True, cand=hand_idx)
        f = ["%s ± %.2f" % (f[i], A[x]["follow_se"]) if x in A else "-" for i, x in enumerate(arms_all)]
        md.append("| %s | %s | %s | %s | %s |" % (t, "/".join(str(A[x]["n"]) for x in arms_all if x in A), " | ".join(ea), " | ".join(gg), " | ".join(f)))
    md.append("\n**Table 2. The 14B segment itself (chunks k+1..k+m generated by the 14B on the 1.3B's context): envelope ratio to T ↓ (alone over the same chunks), same-seed d_T ↓ and d_G ↑ (latent units).**\n")
    hands = [x for x in arms_all if x.startswith("hand") and not x.startswith("hand@reject")]
    md.append("| scene | " + " | ".join("%s seg env_T ↓ (alone)" % x for x in hands) + " | " + " | ".join("%s seg d_T ↓" % x for x in hands) + " | " + " | ".join("%s seg d_G ↑" % x for x in hands) + " |"); md.append("|---|" + "---|" * (3 * len(hands)))
    for t, rt in RES.items():
        A = rt["arms"]; e = row([A[x]["env_T_seg"] if x in A else None for x in hands], False); e = ["%s (%.2f)" % (e[i], A[x]["env_T_alone_seg"]) if x in A else "-" for i, x in enumerate(hands)]
        md.append("| %s | %s | %s | %s |" % (t, " | ".join(e), " | ".join(row([A[x]["dT_seg"] if x in A else None for x in hands], False)), " | ".join(row([A[x]["dG_seg"] if x in A else None for x in hands], True))))
    md.append("\n**Table 3. Envelope ratio to T per chunk ↓ (seed-mean path; = : context chunk (the alone latents for hand@, T's own latents for once@); [..] = the 14B segment generated on the 1.3B context; < 1 = inside the 14B seed spread).**\n")
    md.append("| scene | arm | " + " | ".join("c%d" % c for c in range(13)) + " |"); md.append("|---|---|" + "---|" * 13)
    for t, rt in RES.items():
        md.append("| %s | alone | %s |" % (t, " | ".join("%.2f" % v for v in rt["alone_env_T"])))
        for x in arms_all:
            if x not in rt["arms"]: continue
            v = rt["arms"][x]; md.append("| %s | %s | %s |" % (t, x, " | ".join(("[%.2f]" % v["env_T"][c]) if c in v["seg"] else ("%.2f" % v["env_T"][c]) if c > v["k"] else "=" for c in range(rt["N"]))))
    md.append("\n**Table 4. First 4 chunks after the handoff: same-seed d_T ↓ / d_G ↑ (latent units), follow over those 4 ↑.**\n")
    md.append("| scene | arm | d_T +1 | d_T +2 | d_T +3 | d_T +4 | d_G +1 | d_G +2 | d_G +3 | d_G +4 | follow (first 4) ↑ |"); md.append("|---|---|" + "---|" * 9)
    for t, rt in RES.items():
        for x in arms_all:
            if x not in rt["arms"]: continue
            v = rt["arms"][x]; f4 = lambda l: " | ".join("%.2f" % q for q in l) + " | -" * (4 - len(l)); md.append("| %s | %s | %s | %s | %.2f ± %.2f |" % (t, x, f4(v["dT_first4"]), f4(v["dG_first4"]), v["follow4"], v["follow4_se"]))
    rejs = [x for x in arms_all if x.startswith("hand@reject")]
    if rejs:
        md.append("\n**Table 5. Verifier-driven handoff `hand@reject` (drift head on the 14B's layer-20 commit features, logit > tau -> the 14B regenerates that chunk on the shared context; both models commit the 14B's chunk). reject rate = share of chunks replaced (= 14B share); first = median first replaced chunk (n seeds never flagged in parentheses); whole = chunks 1-12, after = chunks after each seed's first replacement; envelope ↓ (alone same chunks), env_G ↑, follow ↑; 14B-chunk fit = per-seed envelope / d_T / d_G of the replaced chunks.**\n")
        md.append("| scene | arm (tau) | n | reject rate | first (never) | env_T whole ↓ (alone) | env_T after ↓ (alone) | env_G whole ↑ | env_G after ↑ | follow whole ↑ | follow after ↑ | 14B-chunk env ↓ | 14B-chunk d_T ↓ | 14B-chunk d_G ↑ | replaced per chunk c0..c12 |"); md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for t, rt in RES.items():
            for x in rejs:
                if x not in rt["arms"]: continue
                v = rt["arms"][x]; f_ = lambda q: "-" if q is None or not np.isfinite(q) else "%.2f" % q
                md.append("| %s | %s (%.2f) | %d | %.2f | %s (%d) | %s (%s) | %s (%s) | %s | %s | %s ± %s | %s ± %s | %s | %s | %s | %s |" % (t, x, v["tau"], v["n"], v["reject_rate"], f_(v["first_median"]), v["n_never"], f_(v["env_T_whole"]), f_(v["env_T_alone_whole"]), f_(v["env_T_after"]), f_(v["env_T_alone_same"]), f_(v["env_G_whole"]), f_(v["env_G_after"]), f_(v["follow_whole"]), f_(v["follow_whole_se"]), f_(v["follow"]), f_(v["follow_se"]), f_(v["seg_env_perseed"]), f_(v["seg_dT_perseed"]), f_(v["seg_dG_perseed"]), " ".join("%.1f" % q for q in v["replaced_per_chunk"])))
    pre = {t: {x: v["pre_maxabs"] for x, v in rt["arms"].items() if v["pre_maxabs"] is not None} for t, rt in RES.items()}
    md.append("\nPre-handoff check (max |hand - alone| over chunks 0..k, fp16 latents): %s\n" % json.dumps(pre))
    open(a.tables, "w").write("\n".join(md) + "\n"); print("wrote", a.data, a.tables); sys.exit(0)
# ======================================================================================================================= generation (copied from inject_probe.py)
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"; dev = "cuda:0"; sync = torch.cuda.synchronize
def mk14(): return V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=a.window, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
def mk1(): return V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=a.window, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), cfg_name="i2v-1.3B", assets_dir=FAST)
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 64; _dyn.config.accumulated_recompile_limit = 4096
B14 = None
if a.phase == "14b": B = mk14(); FP.fp8_swap(B.target_model, True, keep=False)
elif a.phase == "reject": B = mk1(); B14 = mk14(); FP.fp8_swap(B14.target_model, True, keep=False)
else: B = mk1()
for bb in ([B] + ([B14] if B14 is not None else [])):
    for blk in bb.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=True)
# text encoder stays on the CPU (binding default: prepare() encodes on 'cpu'); saves ~11 GiB on the shared GPU
def gen(seed, c, k=0): return torch.Generator().manual_seed(seed * 1009 + c * 7 + k)   # = inject_probe.gen = act_spec2.gen for fixed 3-frame units
SIG = None; NCH = None
@torch.no_grad()
def denoise(c, seed, K, M=None):
    M = B if M is None else M; g1 = gen(seed, c, 1); z = M.chunk_noise(c, gen(seed, c, 0))
    for i in range(K):
        x = M.target_x0(z, i, c)
        if i < K - 1: z = (1 - SIG[i + 1]) * x + SIG[i + 1] * M.randn_like(x, g1)
    return x
@torch.no_grad()
def roll(seed, ctx, k, stop=None, K=4):
    """KV reset; ctx chunks 0..k re-committed; generate k+1..stop-1 (stop = NCH) with K steps. returns (chunks [16,3,h,w] fp32 GPU incl. the ctx ones, per-chunk s, re-commit s)"""
    stop = NCH if stop is None else stop; B.reset_caches(); xs = []; tc = []; sync(); t0 = time.time()
    for c in range(k + 1): xc = ctx[:, 3 * c:3 * c + 3].float().to(dev); B.commit(xc, c); xs.append(xc)
    sync(); t_re = time.time() - t0
    for c in range(k + 1, stop):
        sync(); t1 = time.time(); x = denoise(c, seed, K); B.commit(x, c); sync(); tc.append(round(time.time() - t1, 4)); xs.append(x)
    return xs, tc, t_re
def save(xs, od, meta):
    os.makedirs(od, exist_ok=True); torch.save(torch.cat(xs, 1).half().cpu(), od + "/lat.pt"); json.dump(meta, open(od + "/run.json", "w"), indent=1)
def cd(x, y, c): return float((x[:, 3 * c:3 * c + 3].float() - y[:, 3 * c:3 * c + 3].float()).norm() / y[:, 3 * c:3 * c + 3].float().norm())
# ---------------- reject phase: drift head on the 14B's layer-20 hidden of its clean commit pass (two_machine.py --gate drift) ----------------
HH = {}
if a.phase == "reject":
    import torch.nn as nn
    class DHead(nn.Module):
        def __init__(s, C, hid):
            super().__init__(); s.mu = nn.Parameter(torch.zeros(C), requires_grad=False); s.sd = nn.Parameter(torch.ones(C), requires_grad=False)
            s.tok = nn.Sequential(nn.Linear(C, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU()); s.pos = nn.Parameter(torch.zeros(16, hid)); s.out = nn.Sequential(nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))
        def chunk_logits(s, x): h = s.tok((x - s.mu) / s.sd) + s.pos; return s.out(h.mean(1)).squeeze(-1)
    hk = torch.load(a.drift_head, map_location="cpu"); DH = DHead(5120, hk["hid"]).to(dev); DH.load_state_dict(hk["state"]); DH.eval(); log("drift head", a.drift_head, "clean-clip percentiles", hk.get("thresholds"))
    def _h20(m, i, o): HH["h20"] = (o[0] if isinstance(o, tuple) else o)
    B14.target_model.blocks[19].register_forward_hook(_h20)
    @torch.no_grad()
    def score():   # logit of the chunk whose clean commit pass just ran on B14 (HH["h20"]: [1, L, 5120])
        f_, h2, w2 = B14._chunk_size, B14._chunk_shape[2] // 2, B14._chunk_shape[3] // 2; hx = HH["h20"][0].float().view(f_, h2, w2, -1).permute(3, 0, 1, 2)
        g = torch.nn.functional.adaptive_avg_pool3d(hx, (1, 3, 5)).reshape(hx.shape[0], 15).t(); return float(DH.chunk_logits(torch.cat([hx.mean((1, 2, 3))[None], g])[None])[0])
    @torch.no_grad()
    def alone_scores(AL):   # teacher-force the alone clip through the 14B commit pass -> per-chunk logits (= what the online loop sees before its first replacement)
        B14.reset_caches(); sc = []
        for c in range(NCH): B14.commit(AL[:, 3 * c:3 * c + 3].float().to(dev), c); sc.append(score())
        return sc
    @torch.no_grad()
    def roll_reject(seed, tau):
        """1.3B generates chunk c; the 14B commits it (clean pass) and the drift head scores it; if logit > tau the 14B regenerates chunk c (4-step, same noise rule) on the
        shared context and both models commit the 14B's chunk instead. returns (chunks, scores, replaced flags, per-chunk s)"""
        B.reset_caches(); B14.reset_caches(); xs = []; sc = []; rep = []; tc = []
        for c in range(NCH):
            sync(); t1 = time.time(); x = denoise(c, seed, 4); B14.commit(x, c); s_ = score(); r = s_ > tau
            if r: x = denoise(c, seed, 4, M=B14); B14.commit(x, c); B14._committed_frames -= B14._chunk_size   # = two_machine.recommit: the 14B's chunk overwrites the slot
            B.commit(x, c); sync(); xs.append(x); sc.append(s_); rep.append(bool(r)); tc.append(round(time.time() - t1, 3))
        return xs, sc, rep, tc
SPECS = [tuple(int(v) for v in (s.split(":") + ["4"])[:3]) for s in a.specs.split(",")] if a.phase == "14b" else []; ARMS = a.arms.split(",") if a.phase == "1p3b" else []
log("start phase=%s tasks=%s seeds=%s specs=%s arms=%s" % (a.phase, TASKS, SEEDS, SPECS, ARMS))
warm = False
for ti, task in enumerate(TASKS):
    T = ALL[task]; sync(); t0 = time.time(); B._caches = {}; import gc; gc.collect(); torch.cuda.empty_cache()
    cond = B.prepare(T["image"], T.get("camera_WM", ""), a.stream, frame_num=a.frames, max_area=a.area, chunk_size=3); NCH = cond["num_chunks"]; SIG = [float(s) for s in B.node_sigmas()]; sync()
    if B14 is not None: B14._caches = {}; gc.collect(); torch.cuda.empty_cache(); B14.prepare(T["image"], T.get("camera_WM", ""), a.stream, frame_num=a.frames, max_area=a.area, chunk_size=3); sync()
    if a.phase == "reject":   # calibration: alone-chunk logits of this task's seeds -> flag rate at each tau; 'p25' = 75th percentile of the alone logits over the run's seeds
        ASC = {sd: alone_scores(torch.load(apath(task, sd), map_location="cpu")) for sd in SEEDS if os.path.exists(apath(task, sd))}; allsc = np.array(sum(ASC.values(), []))
        TAUS = {}
        for tv in a.tau.split(","):
            if tv == "p25": TAUS["reject25"] = float(np.percentile(allsc, 75))
            elif tv == "auto":   # the p25 variant only if the flag rate at -3.25 is < 5 % or > 80 %
                if not 0.05 <= float((allsc > -3.25).mean()) <= 0.8: TAUS["reject25"] = float(np.percentile(allsc, 75))
            else: TAUS["reject" if float(tv) == -3.25 else "reject%s" % tv] = float(tv)
        log("%s alone-chunk drift logits: n=%d mean %.2f p50 %.2f p75 %.2f p95 %.2f max %.2f | flag rate %s" % (task, len(allsc), allsc.mean(), np.median(allsc), np.percentile(allsc, 75), np.percentile(allsc, 95), allsc.max(), " ".join("%s(tau %.2f)=%.2f" % (k, v, float((allsc > v).mean())) for k, v in TAUS.items())))
        json.dump(dict(task=task, taus=TAUS, alone_scores={str(k): v for k, v in ASC.items()}), open("%s/%s/alone_drift_scores.json" % (OUT, task), "w"), indent=1)
    log("prepare %s: %d chunks, shape %s, %.1f s, GPU alloc %.1f GiB" % (task, NCH, tuple(cond["shape"]), time.time() - t0, torch.cuda.memory_allocated() / 2 ** 30))
    if a.phase == "1p3b":   # the loaded 1.3B must reproduce the archived alone chunk 0 (same noise rule / model); logged, not asserted
        s0 = SEEDS[0]; AL0 = torch.load(apath(task, s0), map_location="cpu"); xs, _, _ = roll(s0, AL0, -1, stop=1); warm = True
        log("%s check: regenerated chunk 0 (s%d) vs alone: rel L2 %.4f, max abs %.4f" % (task, s0, cd(torch.cat(xs, 1).cpu(), AL0, 0), float((xs[0].half().cpu() - AL0[:, :3]).abs().max())))
    for seed in SEEDS:
        ap_ = apath(task, seed)
        if not os.path.exists(ap_): log("skip %s s%d: no alone latents" % (task, seed)); continue
        AL = torch.load(ap_, map_location="cpu"); Tl = torch.load(tpath(task, seed), map_location="cpu"); assert AL.shape[1] == 3 * NCH
        if a.phase == "reject":
            for arm, tau in TAUS.items():
                od = "%s/%s/hand@%s_s%d" % (OUT, task, arm, seed)
                if os.path.exists(od + "/lat.pt"): continue
                if not warm: t0 = time.time(); roll_reject(seed, 1e9); sync(); log("warm-up (compile) %.0f s" % (time.time() - t0)); warm = True
                xs, sc, rep, tc = roll_reject(seed, tau); full = torch.cat(xs, 1).half().cpu(); R = [c for c in range(NCH) if rep[c]]; first = R[0] if R else None
                chk = [c for c in range(NCH) if c < (first if first is not None else NCH)]; pre = float((full[:, :3 * len(chk)] - AL[:, :3 * len(chk)]).abs().max()) if chk else 0.0
                dT = [cd(full, Tl, c) for c in range(NCH)]; dA = [cd(full, AL, c) for c in range(NCH)]
                save(xs, od, dict(task=task, seed=seed, arm="hand@" + arm, tau=tau, scores=sc, replaced=rep, replaced_chunks=R, first_replaced=first, share_14b=len(R) / NCH, chunk_s=tc, pre_maxabs=pre, alone_scores=ASC.get(seed), relL2_to_T=dT, relL2_to_alone=dA))
                log("%s s%d hand@%s (tau %.2f): replaced %s (%d/%d = %.0f %% 14B), %.1f s; pre-first-replacement max|diff| to alone %.1e; logits %s" % (task, seed, arm, tau, R, len(R), NCH, 100 * len(R) / NCH, sum(tc), pre, " ".join("%.1f" % v for v in sc)))
        elif a.phase == "14b":
            for (k, m, K) in SPECS:
                od = "%s/%s/%s_s%d" % (OUT, task, segname(k, m, K), seed)
                if os.path.exists(od + "/lat.pt"): continue
                if not warm: t0 = time.time(); roll(seed, AL, 1, stop=3); sync(); log("warm-up (compile) %.0f s" % (time.time() - t0)); warm = True
                xs, tc, tr = roll(seed, AL, k, stop=k + m + 1, K=K); seg = torch.cat(xs, 1).half().cpu()
                dT = [cd(seg, Tl, c) for c in range(k + 1, k + m + 1)]; dA = [cd(seg, AL, c) for c in range(k + 1, k + m + 1)]
                save(xs, od, dict(task=task, seed=seed, seg=segname(k, m, K), k=k, m=m, steps=K, chunk_s=tc, recommit_s=tr, relL2_to_T=dT, relL2_to_alone=dA, pre_maxabs=float((seg[:, :3 * (k + 1)] - AL[:, :3 * (k + 1)]).abs().max())))
                log("%s s%d %s: 14B gen %.1f s (%.2f s/chunk, re-commit %.2f s); rel L2 of the 14B chunks to T %s / to alone %s" % (task, seed, segname(k, m, K), sum(tc), float(np.mean(tc)), tr, " ".join("%.2f" % v for v in dT), " ".join("%.2f" % v for v in dA)))
        else:
            for arm in ARMS:
                k, m, K, sg = parse_arm(arm); od = "%s/%s/%s_s%d" % (OUT, task, arm, seed); sp = "%s/%s/%s_s%d/lat.pt" % (OUT, task, sg, seed)
                if os.path.exists(od + "/lat.pt"): continue
                if not os.path.exists(sp): log("skip %s s%d %s: segment %s missing" % (task, seed, arm, sg)); continue
                seg = torch.load(sp, map_location="cpu"); ctx = seg[:, :3 * (k + m + 1)]
                if not warm: t0 = time.time(); roll(seed, AL, 1, stop=3); sync(); log("warm-up (compile) %.0f s" % (time.time() - t0)); warm = True
                xs, tc, tr = roll(seed, ctx, k + m); full = torch.cat(xs, 1).half().cpu(); pre = float((full[:, :3 * (k + 1)] - AL[:, :3 * (k + 1)]).abs().max())
                dT = [cd(full, Tl, c) for c in range(k + m + 1, NCH)]; dA = [cd(full, AL, c) for c in range(k + m + 1, NCH)]
                save(xs, od, dict(task=task, seed=seed, arm=arm, k=k, m=m, steps_14b=K, segment=sg, chunk_s=tc, recommit_s=tr, pre_maxabs=pre, relL2_to_T=dT, relL2_to_alone=dA))
                log("%s s%d %s: 1.3B %.1f s (re-commit %d chunks %.2f s); pre-handoff max|diff| to alone %.1e; post rel L2 to T %s | to alone %s" % (task, seed, arm, sum(tc), k + m + 1, tr, pre, " ".join("%.2f" % v for v in dT), " ".join("%.2f" % v for v in dA)))
        torch.cuda.empty_cache()
log("ALL_DONE phase=%s" % a.phase)
