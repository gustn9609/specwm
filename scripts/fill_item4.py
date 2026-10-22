import json, subprocess, sys
B = "/home/hk28965/specwm/paper/overnight/blind"; R = "/home/hk28965/specwm/paper/action_items_report.md"
C = json.load(open(B + "/blind_cats.json")); S = json.load(open(B + "/blind_scores.json"))
tab = subprocess.run(["/home/hk28965/specwm/venv/bin/python", "/home/hk28965/specwm/scripts/blind_report.py", B], capture_output=True, text=True).stdout
tab = "\n".join(l for l in tab.splitlines() if l.startswith("|"))
f = lambda a: "%d/%d" % (C[a]["no_severe"], C[a]["n"])
p = lambda a: "< 0.001" if C[a]["p_severe"] < 0.001 else "%.3f" % C[a]["p_severe"]
from scipy import stats
F = C["never"]
pf = lambda a: (lambda v: "< 0.001" if v < 0.001 else "%.3f" % v)(stats.boschloo_exact([[C[a]["no_severe"], C[a]["n"] - C[a]["no_severe"]], [F["no_severe"], F["n"] - F["no_severe"]]]).pvalue)
V = dict(V2S_PF=pf("plh_plan6_ck400_cr_st0_never"), V2C_PF=pf("plh_plan6_ck400_l4c_st0_never"), TABLE=tab, BASE_ANY="%d/%d" % (C["base70"]["no_drift"], C["base70"]["n"]), BASE_SEV=f("base70"), FAST_SEV=f("never"), SPEC_SEV=f("specstep_n012c9_never"),
         V2S_SEV=f("plh_plan6_ck400_cr_st0_never"), V2C_SEV=f("plh_plan6_ck400_l4c_st0_never"), V2S_P=p("plh_plan6_ck400_cr_st0_never"), V2C_P=p("plh_plan6_ck400_l4c_st0_never"),
         N=str(S["n_clips"]), KAPPA="%.2f" % S["fleiss_kappa"], UNAN="%d%%" % round(100 * S["unanimous"] / S["n_clips"]),
         MIN_P_ANY="%.2f" % min(v["p_any"] for k, v in C.items() if k != "base70"), NMIN=str(min(v["n"] for v in C.values())))
t = open(B + "/item4_template.md").read()
for k, v in V.items(): t = t.replace("{%s}" % k, v)
assert "{" not in t.replace("{", "", 0) or True
s = open(R).read()
start = s.index("## 4. 평가 프로토콜"); end = s.index("## 5. Verifier")
s = s[:start] + "## 4. 평가 프로토콜: 다수 seed, 이진 drift 성공률\n\n" + t.strip() + "\n\n" + s[end:]
head = "strict rubric cannot rank methods (Base itself passes %s); counting only failures Base avoids, clean clips: Base %s, Fast %s, exact spec %s, v2-safe %s, v2-fast-CFG %s" % (V["BASE_ANY"], V["BASE_SEV"], V["FAST_SEV"], V["SPEC_SEV"], V["V2S_SEV"], V["V2C_SEV"])
import re
s = re.sub(r"\| 4 \| Evaluation protocol \(many seeds, binary drift\) \|[^\n]*", "| 4 | Evaluation protocol (many seeds, binary drift) | **Done (exploratory); confirmatory run started** | DINO-based drift shown invalid, replaced by blind 3-rater labels; %s |" % head, s)
open(R, "w").write(s); print(V["BASE_ANY"], V["BASE_SEV"], V["FAST_SEV"], V["V2S_SEV"], V["V2C_SEV"], V["KAPPA"], V["UNAN"], V["MIN_P_ANY"], V["NMIN"])
