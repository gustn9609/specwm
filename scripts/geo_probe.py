"""Step 1a of the geometry verifier: VGGT on every blind-labelled clip (pixels), per-frame geometry signals, clip/chunk scores.
Frames: one per latent (every 4th frame, 39 frames = 13 chunks x 3). Signals per adjacent frame pair (i -> i+1):
  rot_jump  : relative camera rotation angle (deg) between VGGT poses
  reproj    : photometric L1 after warping frame i into i+1 with VGGT depth_i + poses + intrinsics (valid, confident pixels only)
  valid     : fraction of frame-i pixels that land inside frame i+1 with positive depth
  conf      : mean VGGT depth confidence of frame i+1
Camera following: VGGT relative rotations vs the commanded relative rotations (poses.npy of the task action), angle error per frame.
Loop closure: commanded frame (after half the clip) whose rotation is closest to frame 0 -> DINO distance of the generated frame to frame 0.
Writes exp/geo/<task>__<arm>__s<seed>.json. usage: CUDA_VISIBLE_DEVICES=g python scripts/geo_probe.py --shard i/n"""
import os, sys, json, glob, argparse, numpy as np, torch, torch.nn.functional as F, imageio
S = "/home/hk28965/specwm"; OUT = S + "/exp/geo"; os.makedirs(OUT, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument("--shard", default="0/1"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--only", default="", help="file with <task>__<arm>__s<seed> lines"); a = ap.parse_args(); si, sn = (int(x) for x in a.shard.split("/"))
sys.path.insert(0, S + "/b1/vggt")
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
dev = torch.device("cuda:0"); model = VGGT.from_pretrained(S + "/hf/VGGT-1B").to(dev).eval()
torch.hub.set_dir(S + "/.torch_hub"); dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval().to(dev)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
TT = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}
V0 = S + "/lingbot-world-v2/specwm_out/spec_v0"; ROOTS = [V0, S + "/lingbot-world-v2/specwm_out/loop", S + "/lingbot-world-v2/specwm_out/loop_mix"]
clips = []
for f in sorted(glob.glob(S + "/exp/drift_lat/*__s*.pt")):
    t, arm, s = os.path.basename(f)[:-3].split("__"); clips.append((t, arm, int(s[1:])))
if a.only: keep = set(open(a.only).read().split()); clips = [c for c in clips if "%s__%s__s%d" % c in keep]
clips = clips[si::sn][: a.limit or None]; print("clips this shard", len(clips), flush=True)
def rot_angle(R): return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
@torch.no_grad()
def dino_feat(x):  # x [N,3,H,W] in [0,1]
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False); return F.normalize(dino((x - MEAN) / STD).float(), dim=-1)
for n, (task, arm, seed) in enumerate(clips):
    out = "%s/%s__%s__s%d.json" % (OUT, task, arm, seed)
    if os.path.exists(out): continue
    vp = next((p for p in ("%s/%s/%s_s%d/video.mp4" % (r, task, arm, seed) for r in ROOTS) if os.path.exists(p)), None)
    if vp is None: continue
    try: fr = np.stack([np.asarray(x) for x in imageio.get_reader(vp)])[:153]
    except Exception as ex: print("[read fail]", vp, ex, flush=True); continue
    idx = list(range(0, len(fr), 4))[:39]; V = torch.from_numpy(fr[idx]).float().div(255).permute(0, 3, 1, 2).to(dev)   # [T,3,H,W]
    H, W = V.shape[2:]; nw = 518; nh = int(round(H * nw / W / 14)) * 14; img = F.interpolate(V, size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16): pred = model(img[None])
    E, K = pose_encoding_to_extri_intri(pred["pose_enc"], img.shape[-2:]); E = E[0].float(); K = K[0].float()                # [T,3,4] world->cam, [T,3,3]
    D = pred["depth"][0, ..., 0].float(); C = pred["depth_conf"][0].float()                                                   # [T,h,w]
    T_ = len(idx); R = E[:, :, :3]; t = E[:, :, 3]
    ys, xs = torch.meshgrid(torch.arange(nh, device=dev).float() + 0.5, torch.arange(nw, device=dev).float() + 0.5, indexing="ij"); pix = torch.stack([xs, ys, torch.ones_like(xs)], -1).view(-1, 3)
    rot_jump, reproj, valid, conf = [], [], [], []
    for i in range(T_ - 1):
        j = i + 1; Rr = (R[j] @ R[i].T).cpu().numpy(); rot_jump.append(rot_angle(Rr))
        Xc = (torch.linalg.inv(K[i]) @ pix.T).T * D[i].reshape(-1, 1); Xw = (R[i].T @ (Xc - t[i]).T).T; Xj = (R[j] @ Xw.T).T + t[j]
        z = Xj[:, 2:3].clamp(min=1e-6); uv = (K[j] @ (Xj / z).T).T[:, :2]; ok = (Xj[:, 2] > 1e-4) & (uv[:, 0] >= 0) & (uv[:, 0] < nw) & (uv[:, 1] >= 0) & (uv[:, 1] < nh)
        ci = C[i].reshape(-1); ok = ok & (ci > torch.quantile(ci, 0.3))
        grid = torch.stack([uv[:, 0] / nw * 2 - 1, uv[:, 1] / nh * 2 - 1], -1).view(1, nh, nw, 2)
        warped = F.grid_sample(img[j:j + 1], grid, align_corners=False).view(3, -1).T; src = img[i].view(3, -1).T
        err = (warped - src).abs().mean(1); valid.append(float(ok.float().mean())); reproj.append(float(err[ok].mean()) if ok.any() else 1.0); conf.append(float(C[j].mean()))
    # camera following vs commanded poses
    P = np.load(os.path.join(TT[task]["action"], "poses.npy")); P = P.reshape(len(P), 4, 4) if P.ndim == 2 else P
    pidx = [min(k, len(P) - 1) for k in idx]; Rc = [P[k][:3, :3] for k in pidx]                                               # commanded cam->world
    Rg = [r.cpu().numpy() for r in R]                                                                                          # generated world->cam
    cam_err = [rot_angle((Rg[k] @ Rg[0].T) @ (Rc[k].T @ Rc[0]).T) for k in range(T_)]                                          # rel. rotation mismatch (world->cam convention for both)
    # loop closure: commanded frame closest in rotation to frame 0 within the second half
    half = T_ // 2; kc = min(range(half, T_), key=lambda k: rot_angle(Rc[k].T @ Rc[0])); kc_ang = rot_angle(Rc[kc].T @ Rc[0])
    fe = dino_feat(V[[0, kc]]); loop = float(1 - (fe[0] @ fe[1]))
    rec = dict(task=task, arm=arm, seed=seed, frames_idx=idx, rot_jump=[round(x, 3) for x in rot_jump], reproj=[round(x, 4) for x in reproj], valid=[round(x, 3) for x in valid],
               conf=[round(x, 3) for x in conf], cam_err=[round(x, 2) for x in cam_err], loop_frame=idx[kc], loop_cmd_angle=round(kc_ang, 2), loop_dino=round(loop, 4))
    json.dump(rec, open(out, "w"))
    if n % 25 == 0: print("[%d/%d] %s %s s%d rotjump max %.1f reproj max %.3f cam_err max %.1f loop %.3f" % (n, len(clips), task[:6], arm[:20], seed, max(rot_jump), max(reproj), max(cam_err), loop), flush=True)
print("[GEO_DONE] shard", a.shard, flush=True)
