"""Item 3 (exact AR speculative sampler) videos for the 0928 folder: every specstep / target-only run + same-seed Base-70 and Fast references,
side-by-side comparisons (Base | Fast | exact spec | null arm), per-chunk acceptance table, README."""
import os, json, shutil, subprocess, numpy as np
S = "/home/hk28965/specwm"; V = S + "/lingbot-world-v2/specwm_out/spec_v0"; D = S + "/paper/0928/item3_exact_ar_speculative/videos"; os.makedirs(D + "/side_by_side", exist_ok=True)
T = {"sand_filling_up_jar": "jar", "sandcastle_wave_impact": "castle"}
RUNS = [("specstep_n012c9_never", "exactspec_cfg5"), ("specstep_n012c9g2p5_never", "exactspec_cfg2p5"), ("specstep_n012c9g1_never", "exactspec_cfg1"),
        ("tgtonly_n012c9_never", "nullarm_cfg5"), ("tgtonly_n012c9g1_never", "nullarm_cfg1"), ("specstep_n012_never", "firstversion_biased_rejection"),
        ("base70", "ref_base70_target"), ("never", "ref_fast_alone")]
have = {}; acc = []
for t, tn in T.items():
    for run, name in RUNS:
        for s in (1234, 4321, 9999):
            src = "%s/%s/%s_s%d" % (V, t, run, s)
            if not os.path.exists(src + "/video.mp4"): continue
            if run in ("base70", "never") and not any(os.path.exists("%s/%s/%s_s%d/video.mp4" % (V, t, r, s)) for r, _ in RUNS[:6]): continue
            dst = "%s/%s_s%d_%s.mp4" % (D, tn, s, name); shutil.copy2(src + "/video.mp4", dst); have[(tn, s, name)] = dst
            j = src + "/spec_v0.json"
            if os.path.exists(j) and run.startswith("specstep"):
                d = json.load(open(j)); shutil.copy2(j, "%s/%s_s%d_%s.json" % (D, tn, s, name))
                for l in d["log"]:
                    if "spec_acc_0" in l: acc.append(dict(case="%s_s%d" % (tn, s), arm=name, chunk=l["chunk"], **{k: l.get(k) for k in ("spec_acc_0", "spec_acc_1", "spec_acc_2", "spec_dmean_0", "spec_dmean_1", "spec_dmean_2")}, gen_s=d["gen_s"]))
json.dump(acc, open(D + "/per_chunk_acceptance.json", "w"), indent=1)
# side-by-side: Base | Fast | exact spec CFG5 | null arm CFG5 (labels burned in), same seed
def sbs(tn, s, cols, out):
    ins = [have.get((tn, s, c)) for c in cols]
    if not all(ins): return False
    labels = {"ref_base70_target": "Base-70 (target)", "ref_fast_alone": "Fast alone (draft)", "exactspec_cfg5": "Exact spec, CFG-5 target", "nullarm_cfg5": "Target only, CFG 5",
              "exactspec_cfg2p5": "Exact spec, CFG-2.5 target", "exactspec_cfg1": "Exact spec, CFG-1 target", "nullarm_cfg1": "Target only, CFG 1"}
    fc = ";".join("[%d:v]scale=424:-2,fps=16,trim=end_frame=153,setpts=PTS-STARTPTS,drawtext=text='%s':x=8:y=8:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.55[v%d]" % (i, labels[c], i) for i, c in enumerate(cols))
    fc += ";" + "".join("[v%d]" % i for i in range(len(cols))) + "hstack=inputs=%d[o]" % len(cols)
    cmd = ["ffmpeg", "-y", "-loglevel", "error"] + sum([["-i", x] for x in ins], []) + ["-filter_complex", fc, "-map", "[o]", "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", out]
    return subprocess.run(cmd).returncode == 0
made = []
for tn in ("jar", "castle"):
    for s in (1234, 4321, 9999):
        if sbs(tn, s, ["ref_base70_target", "ref_fast_alone", "exactspec_cfg5", "nullarm_cfg5"], "%s/side_by_side/%s_s%d_base_fast_exactspec_nullarm.mp4" % (D, tn, s)): made.append("%s_s%d cfg5" % (tn, s))
    if sbs(tn, 1234, ["ref_base70_target", "exactspec_cfg5", "exactspec_cfg2p5", "exactspec_cfg1"], "%s/side_by_side/%s_s1234_target_cfg_sweep.mp4" % (D, tn)): made.append("%s_s1234 cfg sweep" % tn)
# per-node acceptance summary
import collections
g = collections.defaultdict(list)
for r in acc: g[(r["case"], r["arm"])].append(r)
lines = ["| case | arm | chunks | accept node 0 | accept node 1 | accept node 2 | time per clip |", "|---|---|---|---|---|---|---|"]
for (c, a), L in sorted(g.items()):
    m = lambda k: np.mean([x[k] for x in L if x[k] is not None])
    lines.append("| %s | %s | %d | %.2f | %.2f | %.2f | %.0f s |" % (c, a, len(L), m("spec_acc_0"), m("spec_acc_1"), m("spec_acc_2"), L[0]["gen_s"]))
readme = """# Item 3 — exact frame-level (AR) speculative sampler: videos

File names: `<task>_s<seed>_<arm>.mp4`. All clips: same start image, same camera trajectory (back 3 steps, 360° right turn, then left), empty text prompt.

| arm | what it is |
|---|---|
| exactspec_cfg5 | `spec_v0 --specstep 1 --spec-nodes 0,1,2 --ctx 9`: at Fast nodes 0/1/2 the draft transition is kept or reflected per latent position (reflection maximal coupling, De Bortoli et al. ICML 2025), so each transition is exactly the target's; target = Base one-step CFG-5 x0 re-noised, with 9 prefix latents re-noised to the same σ (RePaint); chunk 0 and node 3 = Fast |
| exactspec_cfg2p5, exactspec_cfg1 | same sampler, target built with Base CFG 2.5 / 1 (seed 1234 only) |
| nullarm_cfg5, nullarm_cfg1 | the same target kernel without any draft (`--bguide 1 --bguide-nodes 0,1,2 --ctx 9`); an exact sampler must match it in distribution |
| firstversion_biased_rejection | the first implementation (≤ 4 rejection retries, then fall back to a target sample): biased, kept for the record |
| ref_base70_target, ref_fast_alone | Base-70 and Fast alone with the same seed, for comparison |

`side_by_side/`: Base | Fast | exact spec (CFG 5) | null arm (CFG 5) per seed, and the target-CFG sweep (Base | CFG 5 | CFG 2.5 | CFG 1) for seed 1234.
`per_chunk_acceptance.json` and the per-run `.json` files hold the per-chunk acceptance rate at each node (spec_acc_k) and the mean draft–target gap (spec_dmean_k).

## Per-position acceptance (mean over chunks)

%s

Whole-chunk acceptance is the product over ~295k latent positions, ≤ 10^-25 in every case, so drafts can never be accepted in blocks and there is no speed-up. What to look for in the videos: the exact sampler and the null arm wash out to white/green the same way at CFG 5 (the target, not the coupling, is the problem); CFG 2.5 over-exposes; CFG 1 keeps colour but the camera stays on the jar (jar) or the image turns hazy and grainy (castle).
""" % "\n".join(lines)
open(D + "/README.md", "w").write(readme); print("videos", len(have), "side-by-side", made)
