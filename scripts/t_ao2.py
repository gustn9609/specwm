import torch, torch.nn as nn, traceback
from torchao.quantization import quantize_, Int4WeightOnlyConfig, Int8WeightOnlyConfig

def mk(): return nn.Sequential(nn.Linear(2048,2048,bias=False)).cuda().bfloat16()
def sz(m): return (sum(p.numel()*p.element_size() for p in m.parameters())
                   + sum(b.numel()*b.element_size() for b in m.buffers()))/2**20

x = torch.randn(4,64,2048,device="cuda",dtype=torch.bfloat16)
for name, cfg in [("int4 v1", Int4WeightOnlyConfig(group_size=128, version=1)),
                  ("int8",    Int8WeightOnlyConfig())]:
    m = mk()
    with torch.no_grad(): ref = m(x)
    b0 = sz(m)
    try:
        quantize_(m, cfg)
        with torch.no_grad(): got = m(x)
        e = float((got.float()-ref.float()).norm()/ref.float().norm())
        print(f"[{name}] OK  {b0:.1f} -> {sz(m):.1f} MiB ({b0/sz(m):.2f}x)  상대오차 {e:.4f}")
    except Exception:
        print(f"[{name}] FAILED")
        traceback.print_exc()
