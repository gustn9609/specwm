"""Meeting material 01: SDVG as published (Self-Forcing 1.3B -> Krea 14B, T2V) vs the same SDVG loop on the action-conditioned LingBot-World v2
(1.3B -> 14B causal-fast, image + camera poses). Writes paper/1005/meeting/01_sdvg_to_action/: inputs (images, prompts, camera path plot),
side-by-side videos, e2e time chart (PNG), README.md with the tables. usage: python scripts/meeting_01.py"""
import os, re, json, shutil, glob, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = "/home/hk28965/specwm"; P = S + "/paper/1005"; O = P + "/meeting/01_sdvg_to_action"; os.makedirs(O + "/videos", exist_ok=True); os.makedirs(O + "/inputs", exist_ok=True)
T = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; PR = json.load(open(S + "/exp/prompts_1005.json"))
SC = [("sand_filling_up_jar", "jar"), ("sandcastle_wave_impact", "castle")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e4e3dd", "grid.linewidth": 0.8, "axes.axisbelow": True})
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # validated categorical slots 1-4 (fixed order)

# ---- inputs
for task, sn in SC:
    shutil.copy(T[task]["image"], "%s/inputs/%s_input.png" % (O, sn))
    open("%s/inputs/%s_prompts.txt" % (O, sn), "w").write(
        "SDVG original (text-to-video) prompt:\n%s\n\nLingBot-World v2 (image + camera action) text prompt: %s\n\nImageReward prompt used by the SDVG verifier on LingBot:\n%s\n\nCamera action: %s (%s/poses.npy)\n"
        % (PR[task], repr(T[task].get("camera_WM", "")) + " (empty: the task gives no camera_WM text)", open(S + "/exp/irprompt_%s.txt" % task).read().strip(), T[task]["pose"], T[task]["action"]))
# camera path: yaw / pitch relative to frame 0 and forward translation, per video frame
c2w = np.load(T["sand_filling_up_jar"]["action"] + "/poses.npy")[:161]; R0 = c2w[0, :3, :3]
rel = np.einsum("ij,njk->nik", R0.T, c2w[:, :3, :3]); fwd = rel[:, :, 2]
yaw = np.degrees(np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 2]))); pitch = np.degrees(np.arcsin(np.clip(-fwd[:, 1], -1, 1)))
tr = np.einsum("ij,nj->ni", R0.T, c2w[:, :3, 3] - c2w[0, :3, 3])
fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=200)
ax.plot(yaw, color=COL[0], lw=2, label="yaw (deg, + = right)"); ax.plot(pitch, color=COL[1], lw=2, label="pitch (deg)"); ax.set_yticks([-90, 0, 90, 180, 270, 360])
ax.set_xlabel("video frame (16 fps)"); ax.set_ylabel("degrees from frame 0"); ax.set_title("Camera action  's-3, right-30, left-30'  (jar and castle use the same path)", fontsize=11, loc="left")
ax.legend(frameon=False, loc="upper left"); fig.tight_layout(); fig.savefig(O + "/inputs/camera_action.png"); plt.close(fig)
json.dump(dict(yaw_deg=[round(float(v), 2) for v in yaw], pitch_deg=[round(float(v), 2) for v in pitch], translation=[[round(float(u), 4) for u in r] for r in tr]), open(O + "/inputs/camera_action.json", "w"))

# ---- timing tables from the 10/2 clean re-runs
def md_rows(path):
    rows = []
    for line in open(path):
        if line.startswith("|") and not line.startswith("|---") and "scene" not in line and "prompt" not in line: rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    return rows
s1 = md_rows(P + "/step1_sdvg_selfforcing/README.md"); s2 = md_rows(P + "/step2_lingbot_imagereward/README.md")
A1 = ["drafter only", "SDVG tau -0.7", "SDVG tau 1.3", "target only"]; A2 = ["drafter only", "SDVG-IR tau -0.7", "SDVG-IR tau -2.2", "target only"]
t1 = {a: [float(r[3]) for r in s1 if r[2] == a] for a in A1}; acc1 = {a: [int(r[4].split("/")[0]) for r in s1 if r[2] == a] for a in A1}
# LingBot times + prepare (T5 + image VAE encode, 11.3 s), since the SDVG-original times also include text encoding
t2 = {a: [float(r[5]) + 11.3 for r in s2 if r[2] == a] for a in A2}; acc2 = {a: [int(r[6].split("/")[0]) for r in s2 if r[2] == a] for a in A2}
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200); w = 0.19; xs = np.arange(2)
labels = ["drafter only", "SDVG τ −0.7 (paper)", "SDVG looser τ (1.3 / −2.2)", "target only"]
for k in range(4):
    vals = [np.mean(t1[A1[k]]), np.mean(t2[A2[k]])]; bars = ax.bar(xs + (k - 1.5) * w, vals, w * 0.92, color=COL[k], label=labels[k])
    for b_, v in zip(bars, vals): ax.text(b_.get_x() + b_.get_width() / 2, v + 1.2, "%.0f" % v, ha="center", va="bottom", fontsize=9, color="#0b0b0b")
ax.set_xticks(xs); ax.set_xticklabels(["SDVG as published\nSelf-Forcing 1.3B → Krea 14B (T2V)", "SDVG on action-conditioned\nLingBot v2 1.3B → 14B (image + camera)"])
ax.set_ylabel("seconds per clip ↓"); ax.set_title("End-to-end time per 161-frame clip (mean of jar/castle × 3 seeds)", fontsize=11, loc="left")
ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper left"); ax.set_ylim(0, 118); ax.grid(axis="x", visible=False); fig.tight_layout(); fig.savefig(O + "/e2e_time.png"); plt.close(fig)

# ---- videos
for task, sn in SC:
    for sd in (1234, 4321, 9999):
        for src, tag in ((P + "/step1_sdvg_selfforcing/%s__seed%d__drafter-SDVG0.7-SDVG1.3-target.mp4" % (sn, sd), "A_sdvg_original_T2V"),
                         (P + "/step2_lingbot_imagereward/%s__seed%d__none-ir_-0.7-ir_-2.2-target.mp4" % (sn, sd), "B_sdvg_on_lingbot_action")):
            if os.path.exists(src): shutil.copy(src, "%s/videos/%s__%s__seed%d.mp4" % (O, tag, sn, sd))

f = lambda v: "%.1f" % np.mean(v)
md = ["# 01 SDVG as published → SDVG on the action-conditioned world model", "",
      "Both: 161 frames at 832x480, 16 fps, 13 chunks of 3 latent frames, chunk 0 forced to the target, rejected chunks regenerated by the target from the same noise; one RTX PRO 6000 per run; model loading excluded.", "",
      "| | SDVG as published | SDVG on action-conditioned model |", "|---|---|---|",
      "| Draft | Self-Forcing 1.3B (Wan2.1-T2V-1.3B, causal, 4 steps) | LingBot-World v2 1.3B causal-fast (4 steps) |",
      "| Target | Krea Realtime 14B (Wan2.1-T2V-14B, causal) | LingBot-World v2 14B causal-fast (4 steps) |",
      "| Conditioning | text prompt only | input image + camera poses (action); text prompt empty for these tasks |",
      "| Verifier | ImageReward, worst frame of the decoded chunk ≥ τ | same (VAE decode + ImageReward per chunk) |",
      "| Time counted | whole sample_videos call: text encoding + generation + VAE decode + mp4 write | prepare (T5 + image VAE encode, 11.3 s) + latent generation + final VAE decode |", "",
      "## End-to-end time per clip (mean of jar/castle × seeds 1234/4321/9999)", "",
      "| Arm | SDVG as published (s) ↓ | accepted chunks (of 13) | SDVG on LingBot (s) ↓ | accepted chunks (of 13) |", "|---|---|---|---|---|"]
for k, nm in enumerate(["drafter only", "SDVG τ −0.7 (paper)", "SDVG looser τ (1.3 / −2.2)", "target only"]):
    md.append("| %s | %s | %s | %s | %s |" % (nm, f(t1[A1[k]]), f(acc1[A1[k]]), f(t2[A2[k]]), f(acc2[A2[k]])))
md += ["", "![e2e](e2e_time.png)", "", "## Inputs", "", "- jar: `inputs/jar_input.png`, castle: `inputs/castle_input.png` (LingBot only; SDVG original is text-to-video)",
       "- prompts: `inputs/*_prompts.txt`; camera path: `inputs/camera_action.png`", "", "## Videos (`videos/`)", "",
       "- `A_sdvg_original_T2V__<scene>__seed<k>.mp4`: drafter | SDVG τ −0.7 | SDVG τ 1.3 | target", "- `B_sdvg_on_lingbot_action__<scene>__seed<k>.mp4`: drafter | SDVG τ −0.7 | SDVG τ −2.2 | target"]
open(O + "/README.md", "w").write("\n".join(md) + "\n")
json.dump(dict(t1={k: v for k, v in t1.items()}, acc1=acc1, t2=t2, acc2=acc2, yaw_range=[float(yaw.min()), float(yaw.max())]), open(O + "/numbers.json", "w"), indent=1)
print("\n".join(md)); print("yaw range %.1f .. %.1f deg, pitch %.1f .. %.1f" % (yaw.min(), yaw.max(), pitch.min(), pitch.max()))
