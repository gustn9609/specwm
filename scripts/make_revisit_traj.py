#!/usr/bin/env python3
"""재방문 횟수를 통제한 55초 카메라 궤적 생성 (§4.3). 
기본 세그먼트 = traj55_action_<SC>/poses.npy 의 처음 --seg 프레임 (재방문 전 자연 운동).
R 회 재방문: (R+1) 패스로 분할, 세그먼트를 패스 길이로 리샘플링(위치 선형 + 회전 slerp) 후 정방향/역방향 교대 -> 같은 pose 를 R 번 다시 봄.
R = 0: 세그먼트의 상대 운동을 이어붙여(P_end · P_0^-1 · P_i) 계속 멀어지는 경로 (재방문 없음).
출력: data/rev_action_<SC>_r<R>/{poses.npy, intrinsics.npy, action.npy}"""
import argparse, os, numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp
ap = argparse.ArgumentParser(); ap.add_argument("--scene", default="05"); ap.add_argument("--seg", type=int, default=145); ap.add_argument("--frames", type=int, default=873); ap.add_argument("--revisits", default="0,1,4,8"); ap.add_argument("--src", default=None, help="세그먼트 소스 action dir 이름 (기본 traj55_action_<scene>)"); ap.add_argument("--r0-speed", type=float, default=2.0, help="R=0 체인 경로의 병진 배율 (느린 구간의 유사-재방문 방지)")
a = ap.parse_args(); S = os.environ.get("SCRATCH", "/home/hk28965/specwm"); src = "%s/data/%s" % (S, a.src) if a.src else "%s/data/traj55_action_%s" % (S, a.scene)
P = np.load(src + "/poses.npy"); seg = P[:min(a.seg, len(P))]

def resample(seg, n):
    t_src = np.linspace(0, 1, len(seg)); t = np.linspace(0, 1, n)
    pos = np.stack([np.interp(t, t_src, seg[:, i, 3]) for i in range(3)], 1)
    R = Slerp(t_src, Rot.from_matrix(seg[:, :3, :3]))(t).as_matrix()
    out = np.tile(np.eye(4), (n, 1, 1)); out[:, :3, :3] = R; out[:, :3, 3] = pos; return out

for R_ in [int(r) for r in a.revisits.split(",")]:
    if R_ == 0:
        # 회전은 세그먼트를 반복, 병진은 세그먼트 변위를 누적(방향 유지) -> 회전 누적으로 되돌아오는 나선을 막고 계속 멀어짐
        disp = (seg[1:, :3, 3] - seg[:-1, :3, 3]) * a.r0_speed
        out = [seg.copy()]; cur = seg[-1, :3, 3].copy()
        while sum(len(o) for o in out) < a.frames:
            nxt = seg[1:].copy(); nxt[:, :3, 3] = cur + np.cumsum(disp, 0); out.append(nxt); cur = nxt[-1, :3, 3].copy()
        poses = np.concatenate(out)[:a.frames]
    else:
        n_pass = R_ + 1; L = a.frames // n_pass + 1; base = resample(seg, L); parts = []
        for p in range(n_pass):
            s = base if p % 2 == 0 else base[::-1]
            parts.append(s if p == 0 else s[1:])
        poses = np.concatenate(parts)[:a.frames]
        if len(poses) < a.frames: poses = np.concatenate([poses, np.repeat(poses[-1:], a.frames - len(poses), 0)])
    d = "%s/data/rev_action_%s_r%d" % (S, a.scene, R_); os.makedirs(d, exist_ok=True)
    np.save(d + "/poses.npy", poses.astype(P.dtype)); 
    for f in ("intrinsics.npy", "action.npy", "wasd_action.npy"):
        if os.path.exists(src + "/" + f): np.save(d + "/" + f, np.load(src + "/" + f)[:len(poses)] if f != "intrinsics.npy" else np.load(src + "/" + f))
    # 재방문 통계 (4.5s 밖, 위치<5% extent, 각도<10°)
    pos = poses[:, :3, 3]; fwd = poses[:, :3, :3][:, :, 2]; ext = np.linalg.norm(pos.max(0) - pos.min(0)); n = 0; tot = 0
    n_ret = 0
    for i in range(72, len(poses), 4):
        js = np.arange(0, i - 72 + 1); dd = np.linalg.norm(pos[js] - pos[i], axis=1) / ext; ang = np.degrees(np.arccos(np.clip((fwd[js] * fwd[i]).sum(1), -1, 1)))
        k = np.argmin(dd + 0.01 * ang); hit = (dd[k] < 0.05 and ang[k] < 10); n += hit; tot += 1
        if hit:
            exc = np.linalg.norm(pos[js[k]:i] - pos[i], axis=1).max() / ext            # 그 사이 최대 이탈 거리
            n_ret += exc > 0.15
    print("R=%d: %d frames, extent %.1f, near-pose frames %d/%d, true-return frames %d/%d -> %s" % (R_, len(poses), ext, n, tot, n_ret, tot, d), flush=True)
