import os
import torch, torch.nn as nn, sys
sys.path.insert(0,os.environ['SCRATCH']+'/lingbot-world-v2')
from specwm.quantize import quantize_torchao
m = nn.Sequential(nn.Linear(2048,2048,bias=False), nn.GELU(), nn.Linear(2048,2048,bias=False)).cuda().bfloat16()
x = torch.randn(4,64,2048,device="cuda",dtype=torch.bfloat16)
with torch.no_grad(): ref = m(x)
before = sum(p.numel()*p.element_size() for p in m.parameters())
quantize_torchao(m, spec="int4", group=128, min_features=1024)
after = sum(p.numel()*p.element_size() for p in m.parameters()) + \
        sum(b.numel()*b.element_size() for b in m.buffers())
with torch.no_grad(): got = m(x)
err = (got.float()-ref.float()).norm()/ref.float().norm()
print(f"  weight {before/2**20:.1f} MiB -> {after/2**20:.1f} MiB  ({before/max(after,1):.2f}x)")
print(f"  상대오차 {err:.4f}")
