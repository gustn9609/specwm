"""Distribution-level 'looks like the 14B target' score. Per clip: mean logit of the generator critic (target-vs-draft, nocam variant) over chunks 1..N-1
(prev = previous chunk of the same clip). Per arm: mean ± s.e., and two-sample AUC vs the reference arm's clips (0.5 = indistinguishable from the target,
1.0 = always more draft-like). usage: python scripts/target_likeness.py REF_ARM ARM=glob [ARM=glob ...]   (glob over lat.pt files)"""
import sys, glob, json, numpy as np, torch
sys.path.insert(0, "/home/hk28965/specwm/lingbot-world-v2")
from specwm.critic_model import CriticScorer
from sklearn.metrics import roc_auc_score
C = CriticScorer("/home/hk28965/specwm/lingbot-world-v2/specwm_out/critic/nocam/critic.pt", device="cuda:0"); cam0 = torch.zeros(384, 3, 15, 27)
def clip_score(p):
    lat = torch.load(p).float(); n = lat.shape[1] // 3; ch = [lat[:, 3 * c:3 * c + 3] for c in range(n)]
    return float(np.mean([C(ch[c], ch[c - 1], cam0)[0] for c in range(1, n)]))
ref = sys.argv[1]; S = {}
for it in sys.argv[2:]:
    arm, _, g = it.partition("="); S[arm] = [clip_score(p) for p in sorted(glob.glob(g))]
print("| arm | clips | critic logit ↑ (target-like) | AUC vs %s (0.5 = indistinguishable) ↓ |" % ref); print("|---|---|---|---|")
for arm, v in S.items():
    auc = roc_auc_score([1] * len(S[ref]) + [0] * len(v), S[ref] + v) if arm != ref else 0.5
    print("| %s | %d | %.2f ± %.2f | %.2f |" % (arm, len(v), np.mean(v), np.std(v) / np.sqrt(len(v)), auc))
json.dump(S, open("/home/hk28965/specwm/exp/kveval/target_likeness_last.json", "w"))
