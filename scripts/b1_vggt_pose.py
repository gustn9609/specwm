#!/usr/bin/env python3
"""B1 ② VGGT: 생성 프레임(4 fps)에서 카메라 extrinsics/포인트맵 복원 → (a) 복원 카메라 경로, (b) 명령 포즈(rt2 poses.npy, 16fps)와의 정렬 후 오차,
(c) 프레임 간 회전 점프. 출력: b1/<name>_vggt.json (poses [T,3,4], per-frame rot jump, aligned error)"""
import argparse, json, os, numpy as np, torch, imageio
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--out", required=True); ap.add_argument("--fps", type=float, default=4.0); ap.add_argument("--poses", default=None); ap.add_argument("--resize", type=int, default=0, help="V1: 프레임 짧은 변을 이 픽셀로 다운샘플(정보 손실 시뮬) 후 VGGT"); ap.add_argument("--max-frames", type=int, default=64)
a = ap.parse_args(); dev = "cuda"; S = "/home/hk28965/specwm"
model = VGGT.from_pretrained(S + "/hf/VGGT-1B").to(dev).eval()
rd = imageio.get_reader(a.video); fps = rd.get_meta_data().get("fps", 16); step = max(int(round(fps / a.fps)), 1); V = np.stack([f for i, f in enumerate(rd) if i % step == 0])[: a.max_frames]
if a.resize:
    from PIL import Image; H, W = V.shape[1:3]; nh = a.resize; nw = int(round(W * nh / H / 2)) * 2
    V = np.stack([np.asarray(Image.fromarray(f).resize((nw, nh), Image.BICUBIC)) for f in V]); print("[resize] -> %dx%d" % (nh, nw), flush=True)
import tempfile; td = tempfile.mkdtemp(); paths = []
for i, f in enumerate(V): p = os.path.join(td, "%03d.png" % i); imageio.imwrite(p, f); paths.append(p)
images = load_and_preprocess_images(paths).to(dev)
with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
    pred = model(images[None])
extri, intri = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:]); E = extri[0].float().cpu().numpy()      # [T,3,4] world->cam
R = E[:, :3, :3]; t = E[:, :3, 3]; C = np.einsum("nij,nj->ni", R.transpose(0, 2, 1), -t)                                  # camera centers
fwd = R[:, 2, :]                                                                                                            # camera z axis in world
rot_jump = [0.0] + [float(np.degrees(np.arccos(np.clip(np.trace(R[i] @ R[i - 1].T) / 2 - 0.5, -1, 1)))) for i in range(1, len(R))]
rep = dict(video=a.video, fps=a.fps, n=int(len(V)), centers=C.tolist(), forward=fwd.tolist(), rot_jump_deg=rot_jump, conf=float(pred["depth_conf"].float().mean()) if "depth_conf" in pred else None)
if a.poses and os.path.exists(a.poses):
    P = np.load(a.poses); P = P.reshape(len(P), 4, 4) if P.ndim == 2 else P; idx = np.linspace(0, len(P) - 1, len(V)).round().astype(int); Pc = P[idx][:, :3, 3]; Pf = P[idx][:, :3, 2]
    # Umeyama (similarity) align recovered centers to commanded centers
    def umeyama(X, Y):
        mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my; U, D, Vt = np.linalg.svd(Xc.T @ Yc); d = np.sign(np.linalg.det(U @ Vt)); Dm = np.diag([1, 1, d]); Rr = (U @ Dm @ Vt).T; s = (D * np.diag(Dm)).sum() / (Xc ** 2).sum(); return s, Rr, my - s * Rr @ mx
    s, Rr, tt = umeyama(C, Pc); Ca = (s * (Rr @ C.T)).T + tt; err = np.linalg.norm(Ca - Pc, axis=1); ext = np.linalg.norm(Pc.max(0) - Pc.min(0)) + 1e-9
    fa = (Rr @ fwd.T).T; ang = np.degrees(np.arccos(np.clip((fa * Pf).sum(1) / (np.linalg.norm(fa, axis=1) * np.linalg.norm(Pf, axis=1) + 1e-9), -1, 1)))
    rep.update(cmd_centers=Pc.tolist(), aligned_centers=Ca.tolist(), pos_err_rel=(err / ext).tolist(), fwd_err_deg=ang.tolist(), mean_pos_err_rel=float((err / ext).mean()), mean_fwd_err_deg=float(ang.mean()))
json.dump(rep, open(a.out, "w")); print("[VGGT_DONE]", a.out, "rot_jump max %.1f deg" % max(rot_jump), ("pos_err %.3f fwd_err %.1f" % (rep["mean_pos_err_rel"], rep["mean_fwd_err_deg"]) if "mean_pos_err_rel" in rep else ""))
