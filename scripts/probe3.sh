#!/bin/bash
true
source $SCRATCH/venv/bin/activate
cd $SCRATCH/lingbot-world-v2
python3 - << 'PY'
import torch
for p in ["specwm_out/revisit/fast_rt2/latent.pt", "specwm_out/revisit/base_rt2/latent.pt",
          "specwm_out/hands/fast_p1/latent.pt", "specwm_out/hands/base_p1/latent.pt"]:
    try:
        t = torch.load(p, map_location="cpu")
        print("  %-42s %s %s" % (p.split("specwm_out/")[1], tuple(t.shape), t.dtype))
    except Exception as e:
        print("  %-42s 없음 (%s)" % (p, str(e)[:40]))
PY
echo
echo "=== Base 스케줄러 sigma/timestep 그리드 (node_grid 70) ==="
cd $SCRATCH/lingbot-world
python3 - << 'PY'
import sys; sys.path.insert(0, ".")
# 모델 로드 없이 스케줄러만
try:
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler as S
    sch = S(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    sch.set_timesteps(70, device="cpu", shift=10.0)
    sig, ts = sch.sigmas.float(), sch.timesteps.float()
    print("  n=%d  첫 sigma %.4f (t=%.0f)  마지막 sigma %.4f (t=%.0f)" % (len(ts), sig[0], ts[0], sig[-1], ts[-1]))
    print("  boundary 근처(t≈947):")
    for i in range(len(ts)):
        if 850 < ts[i] < 1000: print("    i=%2d t=%.0f sigma=%.4f" % (i, ts[i], sig[i]))
    print("  저노이즈 후보:")
    for i in range(len(ts)):
        if ts[i] < 320: print("    i=%2d t=%.0f sigma=%.4f" % (i, ts[i], sig[i]))
except Exception as e:
    print("  스케줄러 로드 실패:", str(e)[:120])
PY
