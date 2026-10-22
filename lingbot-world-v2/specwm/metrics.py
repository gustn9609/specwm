"""Metrics: latent-space MMD (always available) + optional DINOv2 feature MMD
on decoded frames + drift, + video saving."""
import os
import torch


def _rbf_mmd2(X, Y, bw=None):
    X = X.reshape(len(X), -1).float(); Y = Y.reshape(len(Y), -1).float()
    Z = torch.cat([X, Y])
    d2 = torch.cdist(Z, Z).pow(2)
    if bw is None:
        bw = d2[d2 > 0].median().clamp(min=1e-8)
    K = torch.exp(-d2 / bw)
    n, m = len(X), len(Y)
    kxx = (K[:n, :n].sum() - K[:n, :n].diag().sum()) / (n * (n - 1) + 1e-9)
    kyy = (K[n:, n:].sum() - K[n:, n:].diag().sum()) / (m * (m - 1) + 1e-9)
    kxy = K[:n, n:].mean()
    return float(kxx + kyy - 2 * kxy)


def latent_mmd(A, B):
    return _rbf_mmd2(torch.stack([a.float() for a in A]),
                     torch.stack([b.float() for b in B]))


@torch.no_grad()
def dino_mmd(frames_a, frames_b, device="cuda"):
    """frames_*: list of [C,F,H,W] in (-1,1). Uses dinov2_vits14 via torch.hub."""
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device).eval()
    def feats(fr):
        out = []
        for v in fr:
            x = v[:, ::4].permute(1, 0, 2, 3)          # subsample frames
            x = torch.nn.functional.interpolate(x, size=(224, 224))
            x = ((x + 1) / 2 - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) \
                / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            out.append(model(x.to(device)).mean(0).cpu())
        return torch.stack(out)
    return _rbf_mmd2(feats(frames_a), feats(frames_b))


@torch.no_grad()
def save_videos(binding, groups: dict, outdir):
    import imageio, numpy as np
    for name, lats in groups.items():
        for i, lat in enumerate(lats[:2]):
            vid = binding.pipe.vae.decode([lat.to(binding.device)])[0]
            v = ((vid.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 3, 0).cpu().numpy()
            imageio.mimwrite(os.path.join(outdir, f"{name}_{i}.mp4"), list(v), fps=16)
