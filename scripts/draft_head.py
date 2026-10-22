"""[10/15, step (b)/(c) of the verified-cheap-forward plan] Draft-verifier head on the device's own features, trained on two-machine runs
(specwm/two_machine.py --dump-feat): per chunk the device saved the last block's hidden after each draft step (standardized, projected to 1536,
2x2 pooled: feat.pt h [N, K, F, h, w, 1536]); the oracle label is what the cloud changed: per token (2x2 latent patch of one latent frame)
rel = ||refined - draft|| / median token norm of the refined chunk, computed from device/<task>/lat.pt (bicubically upsampled to the cloud's grid
when the draft is low-res, exactly as the cloud does) and cloud/<task>/lat.pt, then pooled to the feature grid.
Head = the 10/6 Speculative-Forcing head (LN -> Conv3d 1536->96->48->1, Huber, Adam 1e-4), 5-fold cross-validation over tasks; reports per step
token-level AUROC (label > eps) / Spearman and chunk-level Spearman, and (c) the chunk-selection comparison under a refine budget: which gate
picks the chunks the cloud changes most — learned head vs X-Cache-style feature cosine (chunk c vs c-1) vs latent cosine vs random vs oracle.
usage: CUDA_VISIBLE_DEVICES=g python scripts/draft_head.py --root twomach/a704_43 --arm rt2c1 --out DIR [--epochs 30]"""
import argparse, os, glob, json, time, math, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--arm", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--bs", type=int, default=4); ap.add_argument("--folds", type=int, default=5); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--skip-first", type=int, default=3, help="chunks 0..k-1 excluded (warm-up of the KV window)"); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed); dev = "cuda:0"
def patch_tokens(x):   # [C, F, H, W] -> [F, H/2, W/2, 4C]
    C, F_, H, W = x.shape; h, w = H // 2, W // 2
    return x[:, :, :h * 2, :w * 2].reshape(C, F_, h, 2, w, 2).permute(1, 2, 4, 0, 3, 5).reshape(F_, h, w, C * 4)
# ---- data ----
D = []; tasks = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob("%s/%s/device/*/feat.pt" % (a.root, a.arm)))
for task in tasks:
    pc = "%s/%s/cloud/%s/lat.pt" % (a.root, a.arm, task)
    if not os.path.exists(pc): continue
    fe = torch.load("%s/%s/device/%s/feat.pt" % (a.root, a.arm, task), map_location="cpu"); h = fe["h"]   # [N, K, F, hf, wf, C]
    xd = torch.load("%s/%s/device/%s/lat.pt" % (a.root, a.arm, task), map_location="cpu").float(); xc = torch.load(pc, map_location="cpu").float()
    if xd.shape[2:] != xc.shape[2:]: xd = F.interpolate(xd.permute(1, 0, 2, 3), size=tuple(xc.shape[2:]), mode="bicubic", align_corners=False).permute(1, 0, 2, 3)
    N, K, F_, hf, wf, C = h.shape; assert xd.shape[1] == N * F_ == xc.shape[1], (xd.shape, xc.shape, h.shape)
    td, tc = patch_tokens(xd), patch_tokens(xc)   # [NF, H2, W2, 64]
    for c in range(a.skip_first, N):
        sl = slice(c * F_, (c + 1) * F_); sg = tc[sl].norm(dim=-1).median().clamp_min(1e-6)
        gap = (td[sl] - tc[sl]).norm(dim=-1) / sg                                                              # [F, H2, W2]
        gp = F.adaptive_avg_pool2d(gap[None], (hf, wf))[0]                                                      # to the feature grid
        # cosine gates (open-loop, X-Cache style): feature of chunk c vs c-1 at the last draft step; latent of chunk c vs c-1
        fprev = h[c - 1, -1].float().flatten(); fcur = h[c, -1].float().flatten(); cos_f = float(F.cosine_similarity(fcur[None], fprev[None]))
        cos_l = float(F.cosine_similarity(td[sl].flatten()[None], td[slice((c - 1) * F_, c * F_)].flatten()[None]))
        D.append(dict(task=task, chunk=c, h=h[c], y=gp.half(), y_full=gap.half(), cos_f=cos_f, cos_l=cos_l, rel=float((td[sl] - tc[sl]).norm() / tc[sl].norm())))
print("arm %s: tasks %d chunks %d feature grid %s steps %d; label per-token: median %.3f q10 %.3f q90 %.3f; chunk rel median %.3f" % (
    a.arm, len({d["task"] for d in D}), len(D), tuple(D[0]["y"].shape), D[0]["h"].shape[0], *np.quantile(torch.cat([d["y_full"].flatten() for d in D]).float().numpy(), [0.5, 0.1, 0.9]), np.median([d["rel"] for d in D])), flush=True)
EPS = [float(v) for v in np.quantile(torch.cat([d["y"].flatten() for d in D]).float().numpy(), [0.5, 0.75, 0.9])]   # pooled-label quantiles as thresholds
class Head(nn.Module):
    def __init__(s, cin=1536):
        super().__init__(); s.ln = nn.LayerNorm(cin); s.c1 = nn.Conv3d(cin, 96, 3, padding=1); s.c2 = nn.Conv3d(96, 48, 3, padding=1); s.c3 = nn.Conv3d(48, 1, 1)
    def forward(s, h):   # h [B, F, H, W, C] -> [B, F, H, W]
        x = s.ln(h.float()).permute(0, 4, 1, 2, 3); x = F.gelu(s.c1(x)); x = F.gelu(s.c2(x)); return s.c3(x).squeeze(1)
K = D[0]["h"].shape[0]; tl = sorted({d["task"] for d in D}); random.shuffle(tl); folds = [set(tl[i::a.folds]) for i in range(a.folds)]
PRED = {k: {} for k in range(K)}   # (task, chunk) -> predicted map (test fold)
for fi, ft in enumerate(folds):
    tr = [d for d in D if d["task"] not in ft]; te = [d for d in D if d["task"] in ft]; m = Head(cin=D[0]["h"].shape[-1]).to(dev); opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    steps = a.epochs * math.ceil(len(tr) / a.bs); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps); t0 = time.time()
    for ep in range(a.epochs):
        random.shuffle(tr)
        for i in range(0, len(tr), a.bs):
            b = tr[i:i + a.bs]   # clips differ in aspect ratio (12x22 vs 13x20 grids): one forward per chunk, loss averaged
            loss = sum(F.smooth_l1_loss(m(d["h"].to(dev)), d["y"].float().to(dev)[None].expand(K, -1, -1, -1)) for d in b) / len(b); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    m.eval()
    with torch.no_grad():
        for d in te:
            p = m(d["h"].to(dev)).cpu()   # [K, F, h, w]
            for k in range(K): PRED[k][(d["task"], d["chunk"])] = p[k]
    print("fold %d: train %d test %d chunks, %.0f s" % (fi, len(tr), len(te), time.time() - t0), flush=True)
    torch.save(m.state_dict(), "%s/head_fold%d.pt" % (a.out, fi))
# ---- evaluation (pooled over the 5 test folds) ----
R = dict(arm=a.arm, n_chunks=len(D), n_tasks=len(tl), eps=EPS, label_median=float(np.median(torch.cat([d["y"].flatten() for d in D]).float().numpy())))
Yl = [d["y"].float().flatten().numpy() for d in D]; Y = np.concatenate(Yl); ym = np.array([y.mean() for y in Yl])
for k in range(K):
    Pl = [PRED[k][(d["task"], d["chunk"])].float().flatten().numpy() for d in D]; P = np.concatenate(Pl); pm = np.array([p.mean() for p in Pl])
    r = dict(token_spearman=float(spearmanr(P, Y)[0]), chunk_spearman=float(spearmanr(pm, ym)[0]), chunk_mae=float(np.abs(pm - ym).mean()))
    for e in EPS: r["token_auroc_%.3f" % e] = float(roc_auc_score(Y > e, P))
    for q in (0.5, 0.75): r["chunk_auroc_top%d" % int(100 * (1 - q))] = float(roc_auc_score(ym > np.quantile(ym, q), pm))
    R["step%d" % (k + 1)] = r
# (c) chunk selection under a refine budget: gate score high = refine; metric = mean oracle gap of the chunks left unrefined (lower is better) and fraction of total gap captured
gates = {"head_step%d" % (k + 1): np.array([PRED[k][(d["task"], d["chunk"])].float().mean().item() for d in D]) for k in range(K)}
gates["feat_cos_1-cos"] = np.array([1 - d["cos_f"] for d in D]); gates["latent_cos_1-cos"] = np.array([1 - d["cos_l"] for d in D]); gates["random"] = np.random.rand(len(D)); gates["oracle"] = ym.copy()
R["gates"] = {g: dict(chunk_spearman_vs_oracle=float(spearmanr(s, ym)[0])) for g, s in gates.items()}
for B in (0.25, 0.5, 0.75):
    for g, s in gates.items():
        nsel = int(round(B * len(D))); sel = np.argsort(-s)[:nsel]; mask = np.zeros(len(D), bool); mask[sel] = True
        R["gates"][g]["budget_%.2f" % B] = dict(unrefined_mean_gap=float(ym[~mask].mean()), captured_gap_frac=float(ym[mask].sum() / ym.sum()))
# thresholds for the end-to-end gates (two_machine.py --gate): the (1-B) quantile of each gate's score over the pooled CV predictions = "send B of the chunks"
R["folds"] = [sorted(f) for f in folds]; R["thr"] = {g: {"%.2f" % B: float(np.quantile(s, 1 - B)) for B in (0.25, 0.5, 0.75)} for g, s in gates.items() if g != "random"}
json.dump(R, open(a.out + "/result.json", "w"), indent=1)
print("### %s: verifier head (5-fold CV over %d tasks, %d chunks; label = per-token gap draft -> cloud-refined / median token norm; pooled-label quantiles 50/75/90 %% = %s)" % (a.arm, len(tl), len(D), " / ".join("%.3f" % e for e in EPS)))
print("| features after draft step | token AUROC ↑ (gap > q50 / q75 / q90) | token Spearman ↑ | chunk Spearman ↑ | chunk AUROC ↑ (top-50 % / top-25 % chunks) |"); print("|---|---|---|---|---|")
for k in range(K):
    r = R["step%d" % (k + 1)]; print("| step %d | %s | %.3f | %.3f | %.3f / %.3f |" % (k + 1, " / ".join("%.3f" % r["token_auroc_%.3f" % e] for e in EPS), r["token_spearman"], r["chunk_spearman"], r["chunk_auroc_top50"], r["chunk_auroc_top25"]))
print("\n### (c) which chunks should the cloud refine under a budget? (gate score high = refine; ↓ mean oracle gap of the unrefined chunks, ↑ fraction of the total gap captured)")
print("| gate | Spearman vs oracle chunk gap ↑ | budget 25 %: unrefined gap ↓ / captured ↑ | 50 % | 75 % |"); print("|---|---|---|---|---|")
for g in gates:
    r = R["gates"][g]; print("| %s | %.3f | %s |" % (g, r["chunk_spearman_vs_oracle"], " | ".join("%.3f / %.2f" % (r["budget_%.2f" % B]["unrefined_mean_gap"], r["budget_%.2f" % B]["captured_gap_frac"]) for B in (0.25, 0.5, 0.75))))
print("[DRAFT_HEAD_DONE]", a.out, flush=True)
