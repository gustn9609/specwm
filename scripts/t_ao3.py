import torch, torch.nn as nn, traceback
from torchao.quantization import quantize_, int4_weight_only, int8_weight_only

x = torch.randn(4, 64, 2048, device="cuda", dtype=torch.bfloat16)
def mk(): return nn.Sequential(nn.Linear(2048, 2048, bias=False)).cuda().bfloat16()

for name, q in [("int4 g128", int4_weight_only(group_size=128)),
                ("int8", int8_weight_only())]:
    m = mk()
    with torch.no_grad(): ref = m(x)
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    b0 = torch.cuda.memory_allocated()
    try:
        quantize_(m, q)
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        b1 = torch.cuda.memory_allocated()
        with torch.no_grad(): got = m(x)
        e = float((got.float() - ref.float()).norm() / ref.float().norm())
        # 속도
        for _ in range(3):
            with torch.no_grad(): m(x)
        torch.cuda.synchronize()
        import time; t = time.perf_counter()
        for _ in range(30):
            with torch.no_grad(): m(x)
        torch.cuda.synchronize(); dt = (time.perf_counter()-t)/30*1e3
        print(f"[{name}] OK  메모리 {b0/2**20:.1f} -> {b1/2**20:.1f} MiB  "
              f"상대오차 {e:.4f}  {dt:.3f} ms/call")
    except Exception:
        print(f"[{name}] FAILED"); traceback.print_exc()

# bf16 기준 속도
m = mk()
for _ in range(3):
    with torch.no_grad(): m(x)
torch.cuda.synchronize()
import time; t=time.perf_counter()
for _ in range(30):
    with torch.no_grad(): m(x)
torch.cuda.synchronize()
print(f"[bf16 기준] {(time.perf_counter()-t)/30*1e3:.3f} ms/call")
