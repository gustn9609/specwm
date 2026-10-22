import os
p = os.environ["SCRATCH"]+"/SolarWM/src/solarwm/inference/engine.py"
s = open(p).read()
assert 'case_error = f"{type(exc).__name__}: {exc}"' in s
s = s.replace('case_error = f"{type(exc).__name__}: {exc}"',
              'case_error = f"{type(exc).__name__}: {exc}\\n" + __import__("traceback").format_exc()[-3000:]  # [hskim99 patch]')
open(p, "w").write(s)
import py_compile; py_compile.compile(p, doraise=True); print("engine.py 패치 + 컴파일 OK")
