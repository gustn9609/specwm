import os
import glob, re
for p in glob.glob(os.environ["SCRATCH"]+"/SolarWM/src/solarwm/**/*.py", recursive=True):
    s = open(p).read()
    if "case wave" in s and "wan22" in p:
        i = s.index("case wave"); pre = s[:i]; ln = pre.count("\n") + 1
        lines = s.split("\n"); print(p, "line", ln); print("\n".join(lines[ln-16:ln+1]))
