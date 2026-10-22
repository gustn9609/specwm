import os
p = os.environ["SCRATCH"]+"/SolarWM/src/solarwm/backends/wan22/runtime/inference.py"
s = open(p).read()
s = s.replace("from __future__ import traceback\nimport ", "from __future__ import ", 1)
# __future__ 줄 뒤에 안전하게 삽입
lines = s.split("\n"); out = []; done = False
for ln in lines:
    out.append(ln)
    if not done and ln.startswith("from __future__ import"):
        out.append("import traceback  # [hskim99 patch]"); done = True
if not done: out.insert(0, "import traceback  # [hskim99 patch]")
s = "\n".join(out); open(p, "w").write(s)
import py_compile; py_compile.compile(p, doraise=True); print("컴파일 OK"); print("\n".join(l for l in s.split("\n")[:8]))
