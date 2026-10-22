"""[10/16 trained device refiner] small latent-space residual net: input = (bicubic-upsampled low-res draft chunk xd, the cloud's re-noise draw eps),
output = residual so that xd + net(xd, eps) ~ x = the cloud's one-step refinement at node 750 (sigma 0.625). The target is a deterministic function of
(xd, eps) (two_machine --dump-pairs), so an MSE regression is the right objective (no conditional-mean blur, cf. facts §24).
Shapes: latents [16, F=3, h, w] (704x384 -> 48x88); the net is fully convolutional (works on any grid). Two-level 3D U-Net, GroupNorm + SiLU, ~25M params."""
import torch, torch.nn as nn, torch.nn.functional as F

class Res(nn.Module):
    def __init__(s, c):
        super().__init__(); s.n1 = nn.GroupNorm(8, c); s.c1 = nn.Conv3d(c, c, 3, padding=1); s.n2 = nn.GroupNorm(8, c); s.c2 = nn.Conv3d(c, c, 3, padding=1)
    def forward(s, x): h = s.c1(F.silu(s.n1(x))); h = s.c2(F.silu(s.n2(h))); return x + h

class RefinerNet(nn.Module):
    def __init__(s, cin=32, cout=16, c0=128, c1=256, n0=3, n1=4):
        super().__init__(); s.stem = nn.Conv3d(cin, c0, 3, padding=1); s.enc0 = nn.Sequential(*[Res(c0) for _ in range(n0)])
        s.down = nn.Conv3d(c0, c1, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)); s.mid = nn.Sequential(*[Res(c1) for _ in range(n1)])
        s.up = nn.ConvTranspose3d(c1, c0, (1, 2, 2), stride=(1, 2, 2)); s.dec0 = nn.Sequential(*[Res(c0) for _ in range(n0)]); s.out = nn.Conv3d(c0, cout, 3, padding=1)
        nn.init.zeros_(s.out.weight); nn.init.zeros_(s.out.bias)   # starts as the identity (xd itself)
        s.register_buffer("mu", torch.zeros(16)); s.register_buffer("sd", torch.ones(16))   # per-channel latent stats (set from the data)
    def forward(s, xd, eps):
        """xd, eps: [B, 16, F, h, w] raw latents -> residual in raw latent units"""
        n = lambda t: (t - s.mu.view(1, -1, 1, 1, 1)) / s.sd.view(1, -1, 1, 1, 1)
        x = torch.cat([n(xd), eps], 1); h = s.stem(x); h0 = s.enc0(h); ph = (h0.shape[-2] % 2, h0.shape[-1] % 2)
        hp = F.pad(h0, (0, ph[1], 0, ph[0])) if any(ph) else h0; m = s.mid(s.down(hp)); u = s.up(m)[..., :h0.shape[-2], :h0.shape[-1]]
        return s.out(s.dec0(u + h0)) * s.sd.view(1, -1, 1, 1, 1)
    @torch.no_grad()
    def refine(s, xd, eps): return xd + s.forward(xd, eps)
