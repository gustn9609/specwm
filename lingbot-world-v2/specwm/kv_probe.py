"""Linear probe for the memory translator (option B): how linearly are Base's per-layer K/V related to Fast's?
For clean latent chunks of the training scenes: Fast commits chunk c (t = 0 pass on its cache) -> hook K/V of every block;
Base runs a clean (t = 0) forward on the window [chunks 0..c] -> hook K/V of the same chunk's tokens (low expert).
Fit ridge regression Base -> Fast per layer on train chunks, report R^2 on held-out chunks (per layer, K and V).
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.kv_probe --out DIR [--videos 6 --chunks 2,4,6]"""
import argparse, os, sys, glob, json, torch, numpy as np
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--videos", type=int, default=6); ap.add_argument("--chunks", default="2,4,6")
ap.add_argument("--layers", default="0,5,10,15,20,25,30,35,39"); ap.add_argument("--ridge", type=float, default=1e-2); ap.add_argument("--cache", default=S + "/lingbot-world-v2/specwm_out/mixlora/cache")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dev0 = torch.device("cuda:0"); dev1 = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")   # Fast on GPU 0, Base on GPU 1
os.environ.setdefault("SPECWM_NO_FLASH", "1"); sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from specwm.binding_v2 import V2Binding
b = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
import importlib, importlib.util
os.chdir(S + "/lingbot-world"); sys.path = [S + "/lingbot-world"] + [p for p in sys.path if p != S + "/lingbot-world-v2"]
for k in [k for k in sys.modules if k == "wan" or k.startswith("wan.")]: del sys.modules[k]
spec = importlib.util.spec_from_file_location("specwm_v1", S + "/lingbot-world/specwm/__init__.py", submodule_search_locations=[S + "/lingbot-world/specwm"]); pkg = importlib.util.module_from_spec(spec); sys.modules["specwm_v1"] = pkg; spec.loader.exec_module(pkg)
V1Binding = importlib.import_module("specwm_v1.binding_v1").V1Binding
b1 = V1Binding(S + "/lingbot-world/lingbot-world-base-cam", t5_cpu=True, device=str(dev1))
mB = b1.pipe.low_noise_model.to(dev1).eval(); b1.pipe.high_noise_model = None; torch.cuda.empty_cache()
os.chdir(S + "/lingbot-world-v2")
mF = b.target_model; CS = 3
SC = {sc: dict(img=S + "/lingbot-world-v2/examples/%s/image.jpg" % sc, prompt=open(S + "/p_%s.txt" % sc).read().strip(), act=S + "/rt2_action_%s" % sc) for sc in ("00", "01", "02", "05")}
LAY = [int(x) for x in a.layers.split(",")]; CAP = {}
def hook(tag, li, kind):
    def f(mod, inp, out): CAP[(tag, li, kind)] = out.detach().float()
    return f
HS = []
for li in LAY:
    HS += [mF.blocks[li].self_attn.k.register_forward_hook(hook("F", li, "k")), mF.blocks[li].self_attn.v.register_forward_hook(hook("F", li, "v")),
           mB.blocks[li].self_attn.k.register_forward_hook(hook("B", li, "k")), mB.blocks[li].self_attn.v.register_forward_hook(hook("B", li, "v"))]
files = sorted(glob.glob(a.cache + "/*.pt"))[:a.videos]; CH = [int(x) for x in a.chunks.split(",")]
X = {(li, kind): [] for li in LAY for kind in ("k", "v")}; Y = {(li, kind): [] for li in LAY for kind in ("k", "v")}; TAG = []
with torch.no_grad():
    for fi, f in enumerate(files):
        D = torch.load(f); lat = D["lat"].float(); sc = D["scene"]; print("[probe]", os.path.basename(f), flush=True)
        # Fast: fresh caches, commit chunks 0..max(CH) in order, capture at the chunks of interest
        cond = b.prepare(SC[sc]["img"], SC[sc]["prompt"], SC[sc]["act"], frame_num=161, max_area=480 * 832); b.reset_caches()
        cF = b1.prepare(SC[sc]["img"], SC[sc]["prompt"], SC[sc]["act"], frame_num=4 * ((max(CH) + 1) * CS - 1) + 1, max_area=480 * 832)
        for ci in range(max(CH) + 1):
            x = lat[:, ci * CS:(ci + 1) * CS].to(dev0); b.commit(x, ci)
            if ci in CH:
                fk = {k: v.clone() for k, v in CAP.items() if k[0] == "F"}
                # Base: clean forward over chunks 0..ci (bidirectional), take the last chunk's tokens
                win = lat[:, :(ci + 1) * CS].to(dev1); F = 4 * (win.shape[1] - 1) + 1
                cB = b1.prepare(SC[sc]["img"], SC[sc]["prompt"], SC[sc]["act"], frame_num=F, max_area=480 * 832)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    mB([win.to(torch.bfloat16)], t=torch.full((1,), 0.0, device=dev1), context=[cB["ctx"][0]], seq_len=cB["seq_len"], y=[cB["y"]], dit_cond_dict=cB["dit_cond"])
                ntok = fk[("F", LAY[0], "k")].shape[1]
                for li in LAY:
                    for kind in ("k", "v"):
                        yb = CAP[("B", li, kind)][0, -ntok:]; xf = fk[("F", li, kind)][0]
                        X[(li, kind)].append(yb.cpu()); Y[(li, kind)].append(xf.cpu())
                TAG.append(fi)
for h in HS: h.remove()
TAG = np.array(TAG); test = TAG >= max(1, a.videos - 2)   # last two videos held out
res = {}
for (li, kind), xs in X.items():
    Xa = torch.cat(xs); Ya = torch.cat(Y[(li, kind)]); n = len(xs); per = Xa.shape[0] // n
    m = np.repeat(test, per); Xtr, Ytr, Xte, Yte = Xa[~m].to(dev0), Ya[~m].to(dev0), Xa[m].to(dev0), Ya[m].to(dev0)
    mu_x, mu_y = Xtr.mean(0), Ytr.mean(0); Xc, Yc = Xtr - mu_x, Ytr - mu_y
    W = torch.linalg.solve(Xc.T @ Xc + a.ridge * Xc.shape[0] * torch.eye(Xc.shape[1], device=dev0), Xc.T @ Yc)
    pred = (Xte - mu_x) @ W + mu_y; ss_res = ((Yte - pred) ** 2).sum(); ss_tot = ((Yte - Yte.mean(0)) ** 2).sum(); r2 = float(1 - ss_res / ss_tot)
    cos = float(torch.nn.functional.cosine_similarity(pred, Yte, dim=1).mean()); cos0 = float(torch.nn.functional.cosine_similarity(Xte, Yte, dim=1).mean())
    res["L%02d_%s" % (li, kind)] = dict(r2=round(r2, 4), cos_after=round(cos, 4), cos_raw=round(cos0, 4), n_train=int(Xtr.shape[0]), n_test=int(Xte.shape[0]))
    print("[probe] layer %2d %s  R2 %.3f  cos(raw) %.3f -> cos(linear) %.3f" % (li, kind, r2, cos0, cos), flush=True)
json.dump(res, open(os.path.join(a.out, "kv_probe.json"), "w"), indent=1); print("[probe] done", flush=True)
