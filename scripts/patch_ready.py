import os
p = os.environ["SCRATCH"]+"/SolarWM/src/solarwm/backends/wan22/runtime/readiness.py"
s = open(p).read()
old = '''                ReadinessIssue(
                    code="flash_attention_missing",
                    message=(
                        "FlashAttention is required for GPU training; "
                        "the PyTorch attention fallback is for diagnostics only"
                    ),
                )'''
new = '''                ReadinessIssue(
                    code="flash_attention_missing",
                    message=(
                        "FlashAttention is required for GPU training; "
                        "the PyTorch attention fallback is for diagnostics only"
                    ),
                    # [hskim99 local patch] GH200/aarch64 has no flash-attn wheel.
                    # modeling/attention.py falls back to torch SDPA (exact attention),
                    # so downgrade to a warning for inference-only use.
                    severity="warning",
                )'''
assert old in s, "패치 대상 못 찾음"
open(p, "w").write(s.replace(old, new)); print("readiness.py 패치 완료")
