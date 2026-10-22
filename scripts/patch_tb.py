import os
import re, glob
hits = []
for p in glob.glob(os.environ["SCRATCH"]+"/SolarWM/src/solarwm/backends/wan22/**/*.py", recursive=True):
    s = open(p).read()
    if 'case_error = f"{type(exc).__name__}: {exc}"' in s:
        if "import traceback" not in s:
            s = s.replace("import ", "import traceback\nimport ", 1)
        s = s.replace('case_error = f"{type(exc).__name__}: {exc}"',
                      'case_error = f"{type(exc).__name__}: {exc}\\n" + traceback.format_exc()[-2500:]  # [hskim99 patch] keep traceback')
        open(p, "w").write(s); hits.append(p)
print("패치:", hits)
