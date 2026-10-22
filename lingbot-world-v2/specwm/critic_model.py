"""Latent critic shared by scripts/critic_train.py (training) and specwm/spec_ar_loop.py (online verification).
Input per candidate chunk: x [16,3,H,W] latent, prev [16,3,H,W] committed previous chunk (zeros for chunk 0), flag (0 for chunk 0), cam [384,3,h/4,w/4] Plucker emb pooled to the token grid.
Heads: logit 'target-generated' (density-ratio p/q in context) and the ③ contrast log(e_target/e_draft)."""
import torch, torch.nn as nn, torch.nn.functional as F
CH, CW = 56, 104          # canonical latent grid (tasks differ in aspect: 58x106, 58x104, ...); token grid = ((CH+4)//4, (CW+4)//4) = 15 x 27
def canon(x):
    """latent [16,3,H,W] or [B,16,3,H,W] -> bilinear-resized to the canonical grid"""
    b = x.dim() == 5; x = x if b else x.unsqueeze(0); B_, C, T, H, W = x.shape
    if (H, W) != (CH, CW): x = F.interpolate(x.float().permute(0, 2, 1, 3, 4).reshape(B_ * T, C, H, W), size=(CH, CW), mode="bilinear", align_corners=False).reshape(B_, T, C, CH, CW).permute(0, 2, 1, 3, 4)
    return x if b else x.squeeze(0)
class Critic(nn.Module):
    def __init__(self, cin, dim, layers, t, h, w, patch=4, camc=384):
        super().__init__(); self.patch = patch
        self.stem = nn.Conv3d(cin, dim, kernel_size=(1, patch, patch), stride=(1, patch, patch), padding=(0, patch // 2, patch // 2)); self.cam_proj = nn.Linear(camc, dim)
        hh, ww = (h + patch) // patch, (w + patch) // patch; self.pos = nn.Parameter(torch.zeros(1, t * hh * ww, dim)); nn.init.trunc_normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(dim, 8, dim * 4, dropout=0.1, batch_first=True, norm_first=True); self.enc = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(dim); self.head_cls = nn.Linear(dim, 1); self.head_c3 = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
    def forward(self, inp):
        x, cam = inp; z = self.stem(x).flatten(2).transpose(1, 2); c = self.cam_proj(cam.flatten(2).transpose(1, 2))
        z = z + c[:, :z.shape[1]] + self.pos[:, :z.shape[1]]; z = self.enc(z); z = self.norm(z).mean(1)
        return self.head_cls(z).squeeze(-1), self.head_c3(z).squeeze(-1)
def pool_cam(cam):
    """Plucker emb [C,3,H,W] (full-res or already pooled) -> [C,3,15,27] (the critic token grid of the canonical latent)"""
    hh, ww = (CH + 4) // 4, (CW + 4) // 4
    if tuple(cam.shape[-2:]) == (hh, ww): return cam.float()
    return F.adaptive_avg_pool2d(cam.float().permute(1, 0, 2, 3), (hh, ww)).permute(1, 0, 2, 3)
class CriticScorer:
    """load critic.pt and score candidate chunks online"""
    def __init__(self, path, device="cuda"):
        ck = torch.load(path, map_location="cpu"); self.dev = torch.device(device); self.mean = ck["mean"].to(self.dev); self.std = ck["std"].to(self.dev); self.no_prev = ck.get("no_prev", False); self.no_cam = ck.get("no_cam", False); self.kind = ck.get("kind", "generator")
        self.m = Critic(16 + 16 + 1, ck["dim"], ck["layers"], 3, ck["h"], ck["w"], camc=ck["camc"]).to(self.dev); self.m.load_state_dict(ck["state"]); self.m.eval()
    @torch.no_grad()
    def __call__(self, x, prev, cam):
        """x, prev: [16,3,H,W] (prev None for chunk 0); cam: [384,3,H,W] or pooled. returns (logit_target, c3_pred)"""
        x = ((canon(x.float().to(self.dev)) - self.mean[0]) / self.std[0]).unsqueeze(0)
        p = torch.zeros_like(x) if (prev is None or self.no_prev) else ((canon(prev.float().to(self.dev)) - self.mean[0]) / self.std[0]).unsqueeze(0)
        flag = torch.full((1, 1, *x.shape[2:]), 0.0 if prev is None else 1.0, device=self.dev)
        c = pool_cam(cam.to(self.dev)).float().unsqueeze(0); c = torch.zeros_like(c) if self.no_cam else c
        l, r = self.m((torch.cat([x, p, flag], dim=1), c)); return float(l[0]), float(r[0])
