"""기하 일관성 지표: 명령된 카메라 운동과 생성 영상이 실제로 맞는가.

기존 DINO 전역 유사도의 문제:
  - 전역 특징이 '무슨 장면인가'를 인코딩해 '어디를 보는가' 신호가 묻힌다.
  - 배경이 균질한 장면(S05)에서 신호가 절반으로 약해져 비율 지표가 폭발한다.

대안:
  1) DINOv2 '패치' 토큰으로 두 프레임 간 조밀 대응을 찾고 (전역 X)
  2) 알려진 카메라 상대자세로 만든 기본행렬 F 의 에피폴라 제약을 만족하는
     대응의 비율(inlier ratio)을 측정한다.
  장면 내용과 무관하게 '명령대로 렌더링됐는가'만 잰다.
"""
import numpy as np, torch


def _rel_pose(c2w_i, c2w_j):
    """i -> j 상대 자세 (world-to-cam 규약 변환 포함)."""
    w2c_j = np.linalg.inv(c2w_j)
    T = w2c_j @ c2w_i                 # i 카메라 좌표 -> j 카메라 좌표
    return T[:3, :3], T[:3, 3]


def fundamental(c2w_i, c2w_j, K):
    R, t = _rel_pose(c2w_i, c2w_j)
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E = tx @ R
    Ki = np.linalg.inv(K)
    return Ki.T @ E @ Ki, np.linalg.norm(t)


def sym_epi_dist(F, p1, p2):
    """대칭 에피폴라 거리 (픽셀)."""
    o = np.ones((len(p1), 1))
    x1 = np.hstack([p1, o]); x2 = np.hstack([p2, o])
    l2 = (F @ x1.T).T                 # x1 의 에피폴라선 (이미지2)
    l1 = (F.T @ x2.T).T
    num = np.abs(np.sum(x2 * l2, axis=1))
    d = num / (np.sqrt(l2[:, 0] ** 2 + l2[:, 1] ** 2) + 1e-9) \
      + num / (np.sqrt(l1[:, 0] ** 2 + l1[:, 1] ** 2) + 1e-9)
    return d / 2.0


class PatchMatcher:
    """DINOv2 패치 토큰으로 상호최근접 대응을 뽑는다."""

    def __init__(self, device="cuda", hub=None):
        if hub: torch.hub.set_dir(hub)
        self.dev = device
        self.m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14",
                                trust_repo=True).eval().to(device)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    @torch.no_grad()
    def patches(self, V):
        """V: (T,H,W,3) uint8 -> (T,gh,gw,C) 정규화 패치특징 + 격자크기.
        가로세로비를 유지해야 픽셀좌표 환산이 맞다."""
        T, H, W, _ = V.shape
        gh, gw = H // 14, W // 14           # 14 의 배수로 내림
        X = torch.from_numpy(V).permute(0, 3, 1, 2).float().to(self.dev) / 255
        X = torch.nn.functional.interpolate(X, size=(gh * 14, gw * 14),
                                            mode="bilinear", align_corners=False)
        X = (X - self.mean) / self.std
        out = []
        for i in range(0, T, 4):
            f = self.m.forward_features(X[i:i + 4])["x_norm_patchtokens"]
            f = torch.nn.functional.normalize(f, dim=-1)
            out.append(f.reshape(len(f), gh, gw, -1).cpu())
        return torch.cat(out), gh, gw

    @staticmethod
    def mutual_nn(fi, fj, gh, gw, H, W, min_sim=0.5):
        """상호최근접 + 유사도 하한. 반환: 원본 픽셀좌표 대응."""
        A = fi.reshape(-1, fi.shape[-1]); B = fj.reshape(-1, fj.shape[-1])
        S = A @ B.T
        i2j = S.argmax(1); j2i = S.argmax(0)
        keep = (j2i[i2j] == torch.arange(len(A)))
        sim = S[torch.arange(len(A)), i2j]
        keep &= (sim > min_sim)
        idx = torch.nonzero(keep).squeeze(1)
        if len(idx) == 0:
            return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
        # 패치격자 -> 원본 픽셀 중심좌표
        sy, sx = H / gh, W / gw
        pi = np.stack([((idx % gw).numpy() + 0.5) * sx, ((idx // gw).numpy() + 0.5) * sy], 1)
        jj = i2j[idx]
        pj = np.stack([((jj % gw).numpy() + 0.5) * sx, ((jj // gw).numpy() + 0.5) * sy], 1)
        return pi, pj, sim[idx].numpy()


def geo_consistency(V, poses, K, matcher, thr_px=8.0, min_baseline=1e-3,
                    max_pairs=400, seed=0):
    """반환: 프레임쌍별 (시간간격, 자세각차, inlier비율, 대응수)."""
    T = len(V)
    feats, gh, gw = matcher.patches(V)
    H, W = V.shape[1], V.shape[2]
    fwd = poses[:, :3, :3][:, :, 2]   # OpenCV(+Z 전방), 실측 확인됨
    rng = np.random.default_rng(seed)
    pairs = [(i, j) for i in range(T) for j in range(i + 1, T)]
    if len(pairs) > max_pairs:
        pairs = [pairs[k] for k in rng.choice(len(pairs), max_pairs, replace=False)]
    rec = []
    for i, j in pairs:
        F, base = fundamental(poses[i], poses[j], K)
        if base < min_baseline:          # 병진이 없으면 에피폴라가 정의 안 됨
            continue
        pi, pj, _ = matcher.mutual_nn(feats[i], feats[j], gh, gw, H, W)
        if len(pi) < 10:
            rec.append(((j - i) / 4.0, float(np.degrees(np.arccos(
                np.clip((fwd[i] * fwd[j]).sum(), -1, 1)))), 0.0, len(pi)))
            continue
        d = sym_epi_dist(F, pi, pj)
        rec.append(((j - i) / 4.0,
                    float(np.degrees(np.arccos(np.clip((fwd[i] * fwd[j]).sum(), -1, 1)))),
                    float((d < thr_px).mean()), len(pi)))
    return np.array(rec)
