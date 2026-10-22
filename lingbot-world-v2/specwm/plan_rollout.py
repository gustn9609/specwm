"""On-policy rollouts of the plan -> render pipeline on TRAINING videos (for the DAgger / texture-LoRA round):
Fast + hint LoRA (nodes t >= lora_tmin see the plan hint in y; the other nodes see the plain y = --hint-low plain), chunk 0 hinted (frame 0 kept),
hint = plandata/hints/<name>.pt hint_pix (the test-time hint). Saves <out>/<name>.pt = dict(lat=[16,39,h,w] committed x0 (bf16), z=[13,4,16,3,h,w] the
state entering each node, x0n=[13,4,...] each node's x0 prediction, scene, seed, lora).
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.plan_rollout --lora DIR/hint_lora.pt --out DIR [--names all --shard k/n]"""
import argparse, os, sys, glob, json, math, numpy as np, torch, torch.nn as nn
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--lora", required=True); ap.add_argument("--out", required=True); ap.add_argument("--names", default="all"); ap.add_argument("--shard", default="0/1")
ap.add_argument("--data", default=S + "/lingbot-world-v2/specwm_out/plandata"); ap.add_argument("--seed-mul", type=int, default=3)
a = ap.parse_args(); a.out = os.path.abspath(a.out); os.makedirs(a.out, exist_ok=True)
os.environ.setdefault("SPECWM_NO_FLASH", "1"); sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from specwm.binding_v2 import V2Binding
dev = torch.device("cuda:0")
b = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), device=str(dev))
m = b.target_model.eval(); CS = 3; NODES = [0, 1, 2, 3]; sig = b.node_sigmas()
HL = {"on": False, "hint": False}
st = torch.load(a.lora); HL["tmin"] = float(st["lora_tmin"]); HL["mask"] = float(st["hint_mask"])
class GL(nn.Module):
    def __init__(self, base, A, B):
        super().__init__(); self.base = base; self.register_buffer("A", A.to(dev, torch.float32), persistent=False); self.register_buffer("B", B.to(dev, torch.float32), persistent=False)
    def forward(self, x):
        y = self.base(x); return y + ((x.float() @ self.A.t()) @ self.B.t()).to(y.dtype) if HL["on"] else y
i = 0
for blk in m.blocks:
    sa = blk.self_attn
    for n_ in ("q", "k", "v", "o"): setattr(sa, n_, GL(getattr(sa, n_), st["A"][i], st["B"][i])); i += 1
    for j in (0, 2): blk.ffn[j] = GL(blk.ffn[j], st["A"][i], st["B"][i]); i += 1
pe = m.patch_embedding; PEW, PEB = st["pe_w"].to(dev, pe.weight.dtype), st["pe_b"].to(dev, pe.bias.dtype)
class GPE(nn.Module):
    def __init__(self, pe): super().__init__(); self.pe = pe
    @property
    def weight(self): return self.pe.weight
    @property
    def bias(self): return self.pe.bias
    def forward(self, x):
        if HL["on"]: return torch.nn.functional.conv3d(x, PEW.to(x.dtype), PEB.to(x.dtype), stride=self.pe.stride, padding=self.pe.padding)
        return self.pe(x)
m.patch_embedding = GPE(pe)
SC = {sc: dict(img=S + "/lingbot-world-v2/examples/%s/image.jpg" % sc, prompt=open(S + "/p_%s.txt" % sc).read().strip(), act=S + "/rt2_action_%s" % sc) for sc in ("00", "01", "02", "05")}
TASKS = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}
def cond_of(scene):
    if scene in SC: return SC[scene]["img"], SC[scene]["prompt"], SC[scene]["act"]
    t = TASKS[scene]; return t["image"], t.get("camera_WM", ""), t["action"]
names = sorted(os.path.basename(f)[:-3] for f in glob.glob(a.data + "/cache/*.pt")) if a.names == "all" else a.names.split(",")
k, n = [int(x) for x in a.shard.split("/")]
for idx, name in enumerate(names):
    if idx % n != k: continue
    op = os.path.join(a.out, name + ".pt")
    if os.path.exists(op): continue
    D = torch.load(os.path.join(a.data, "cache", name + ".pt")); H = torch.load(os.path.join(a.data, "hints", name + ".pt")); hint_all = H["hint_pix"].to(dev).float()
    img, prompt, act = cond_of(D["scene"]); seed = int(D["seed"]) * a.seed_mul + 11
    b.prepare(img, prompt, act, frame_num=161, max_area=480 * 832); b.reset_caches(); g = torch.Generator().manual_seed(seed)
    outs, Z, X0 = [], [], []
    with torch.no_grad():
        for ci in range(13):
            hint = hint_all[:, ci * CS:(ci + 1) * CS]; yy = b._y_chunks[ci]; y_old = yy.clone(); f0 = 1 if ci == 0 else 0
            yy[:4, f0:] = HL["mask"]; yy[4:, f0:] = hint[:, f0:].to(yy.dtype); y_hint = yy.clone(); HL["hint"] = True
            z = b.chunk_noise(ci, g); zs, xs = [], []
            for j, i in enumerate(NODES):
                HL["on"] = float(b._timesteps[i]) >= HL["tmin"]; yy.copy_(y_hint if HL["on"] else y_old)   # hint-low plain
                zs.append(z.clone()); x0 = b.target_x0(z, i, ci); xs.append(x0.clone())
                if j < len(NODES) - 1: s = float(sig[NODES[j + 1]]); z = (1 - s) * x0 + s * b.randn_like(x0, g)
            yy.copy_(y_old); HL["on"] = False; HL["hint"] = False; b.commit(x0, ci); outs.append(x0); Z.append(torch.stack(zs)); X0.append(torch.stack(xs))
    torch.save(dict(lat=torch.cat(outs, 1).to(torch.bfloat16).cpu(), z=torch.stack(Z).to(torch.bfloat16).cpu(), x0n=torch.stack(X0).to(torch.bfloat16).cpu(), scene=D["scene"], seed=seed, lora=a.lora), op)
    print("[rollout] %s: lat %s" % (name, tuple(torch.cat(outs, 1).shape)), flush=True)
print("[rollout] done", flush=True)
