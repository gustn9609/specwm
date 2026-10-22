"""AR speculative pair: draft = Fast 14B (4-step, cuda:0), target = v2 causal-pretrain 14B (40-step + CFG, cuda:1), same latent space / same causal chunking.
Modes: fast_only | pretrain_only (target distribution) | periodic:<k> (target generates every k-th chunk, both KV sets committed with the chosen chunk)
       | verify:<thr> (target predicts x0 for the chunk at sigma of Fast's node 0; accept if rel. distance < thr else target generates the chunk)
usage: CUDA_VISIBLE_DEVICES=a,b python -m specwm.spec_ar --scene 05 --seed 1234 --mode periodic:3 --out <dir> [--max-area 184320]"""
import argparse, os, sys, json, time, numpy as np, torch, imageio
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--scene", required=True); ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--mode", default="periodic:3"); ap.add_argument("--frames", type=int, default=161)
ap.add_argument("--max-area", type=int, default=184320); ap.add_argument("--steps", type=int, default=40); ap.add_argument("--guide", type=float, default=5.0); ap.add_argument("--shift", type=float, default=5.0); ap.add_argument("--start", type=int, default=1); ap.add_argument("--out", required=True)
ap.add_argument("--pretrain", default=S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-pretrain")
ap.add_argument("--draft", default="fast14b", choices=["fast14b", "fast1p3b"]); ap.add_argument("--target", default="pretrain", choices=["pretrain", "fast14b"])
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); mode, _, marg = a.mode.partition(":")
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from specwm.binding_v2 import V2Binding
import wan; from wan.configs import WAN_CONFIGS; from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
dev0, dev1 = torch.device("cuda:0"), torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
prompt = open(S + "/p_%s.txt" % a.scene).read().strip(); act = S + "/rt2_action_" + a.scene; img = S + "/lingbot-world-v2/examples/%s/image.jpg" % a.scene
FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
if a.draft == "fast1p3b": b = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), cfg_name="i2v-1.3B", assets_dir=FAST)
else: b = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
cond = b.prepare(img, prompt, act, frame_num=a.frames, max_area=a.max_area); N = cond["num_chunks"]; sig = b.node_sigmas(); CS = b._chunk_size; fs = b._frame_seqlen
# ---- target: causal-pretrain on dev1
class Target:
    def __init__(self):
        cfg = WAN_CONFIGS["i2v-A14B"]
        self.pipe = wan.WanI2VCausal(config=cfg, checkpoint_dir=a.pretrain, device_id=dev1.index, rank=0, t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True, infer_mode="causal_pretrain", convert_model_dtype=True, assets_dir=S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast")
        self.m = self.pipe.model.to(dev1).eval(); margs = self.m.config; self.dt = torch.bfloat16
        n_prompt = cfg.sample_neg_prompt; te = self.pipe.text_encoder
        self.ctx = [t.to(dev1) for t in te([prompt], torch.device("cpu"))]; self.ctxn = [t.to(dev1) for t in te([n_prompt], torch.device("cpu"))]
        lat_f = N * CS; kv = fs * lat_f; hd = margs.dim // margs.num_heads
        mk = lambda: (self.pipe._initialize_self_kv_cache(num_layers=margs.num_layers, shape=[1, kv, margs.num_heads, hd], dtype=self.dt, device=dev1), self.pipe._initialize_crossattn_cache_pretrain(num_layers=margs.num_layers, shape=[1, 512, margs.num_heads, hd], dtype=self.dt, device=dev1))
        self.kc, self.cc = mk(); self.ku, self.cu = mk(); self.kv_size = kv
        self.sch = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
        print("[target] pretrain on %s, cache %d tokens" % (dev1, kv), flush=True)
    def kw(self, ci, uncond):
        return dict(seq_len=b._seq_len, y=[b._y_chunks[ci].to(dev1)], dit_cond_dict={"c2ws_plucker_emb": b._emb_chunks[ci].to(dev1).chunk(1, dim=0)}, current_start=ci * CS * fs, max_attention_size=self.kv_size,
                    context=self.ctxn if uncond else [self.ctx[0]], kv_cache=self.ku if uncond else self.kc, crossattn_cache=self.cu if uncond else self.cc)
    @torch.no_grad()
    def flow(self, z, t, ci):
        tt = torch.stack([t]).to(dev1) if torch.is_tensor(t) else torch.full((1,), float(t), device=dev1); x = [z.to(dev1, self.dt)]
        with torch.amp.autocast("cuda", dtype=self.dt):
            fc = self.m(x=x, t=tt, **self.kw(ci, False))[0]; fu = self.m(x=x, t=tt, **self.kw(ci, True))[0]
        return (fu + a.guide * (fc - fu)).float()
    @torch.no_grad()
    def gen_chunk(self, ci, g):
        self.sch.set_timesteps(a.steps, device=dev1, shift=a.shift); ts = self.sch.timesteps; sg = torch.Generator(device=dev1); sg.manual_seed(a.seed * 1000 + ci)
        z = b.chunk_noise(ci, g).to(dev1)
        for t in ts:
            f = self.flow(z, t, ci); z = self.sch.step(f.unsqueeze(0), t, z.unsqueeze(0), return_dict=False, generator=sg)[0].squeeze(0)
        return z.float()
    @torch.no_grad()
    def x0_at(self, z, s, ci):
        """single x0-prediction at sigma s (t = 1000 s) with CFG"""
        f = self.flow(z, 1000.0 * s, ci); return (z.to(dev1).float() - s * f)
    @torch.no_grad()
    def commit(self, x0, ci):
        x = [x0.to(dev1, self.dt)]; t0 = torch.zeros(1, device=dev1)
        with torch.amp.autocast("cuda", dtype=self.dt): self.m(x=x, t=t0, **self.kw(ci, False)); self.m(x=x, t=t0, **self.kw(ci, True))
class TargetFast:
    """target = Fast 14B (4-step) on dev1 with its own KV — for the 1.3B-draft pair"""
    def __init__(self):
        self.b = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), device=str(dev1))
        self.b.prepare(img, prompt, act, frame_num=a.frames, max_area=a.max_area); self.b.reset_caches(); self.sig = self.b.node_sigmas(); print("[target] fast14b on %s" % dev1, flush=True)
    def gen_chunk(self, ci, g):
        z = b.chunk_noise(ci, g).to(dev1); x0 = None
        for i in range(len(self.sig)):
            x0 = self.b.target_x0(z, i, ci)
            if i < len(self.sig) - 1: s = float(self.sig[i + 1]); z = (1 - s) * x0 + s * b.randn_like(x0, g)
        return x0.float()
    def x0_at(self, z, s, ci): return self.b.target_x0(z.to(dev1), 0, ci).float()
    def commit(self, x0, ci): self.b.commit(x0.to(dev1), ci)
T = None if mode == "fast_only" else (TargetFast() if a.target == "fast14b" else Target())
def fast_chunk(ci, g, z=None):
    z = b.chunk_noise(ci, g) if z is None else z; x0 = None
    for i in range(len(sig)):
        x0 = b.target_x0(z, i, ci)
        if i < len(sig) - 1: s = float(sig[i + 1]); z = (1 - s) * x0 + s * b.randn_like(x0, g)
    return x0
b.reset_caches(); g = torch.Generator().manual_seed(a.seed); outs = []; who = []; stats = []; t0 = time.time(); tt_target = 0.0
for ci in range(N):
    use_t = False; z0 = None
    if mode == "pretrain_only": use_t = ci >= 1
    elif mode == "periodic": use_t = (ci >= a.start) and ((ci - a.start) % int(marg) == 0)
    elif mode == "verify" and ci >= a.start:
        z0 = b.chunk_noise(ci, g); xf = b.target_x0(z0, 0, ci); tv = time.time(); xt = T.x0_at(z0, float(sig[0]), ci).to(dev0); tt_target += time.time() - tv
        d = float((xf - xt).norm() / (xf.norm() + 1e-6)); use_t = d > float(marg); stats.append(dict(chunk=ci, rel=d, rejected=use_t))
    if use_t:
        tv = time.time(); x0 = T.gen_chunk(ci, g).to(dev0); tt_target += time.time() - tv; who.append("T")
    else:
        if z0 is not None:   # verify accepted: continue Fast from node 0's x0 with the same draws
            x0 = xf
            for i in range(len(sig)):
                if i == 0: continue
                s = float(sig[i]); z = (1 - s) * x0 + s * b.randn_like(x0, g); x0 = b.target_x0(z, i, ci)
        else: x0 = fast_chunk(ci, g)
        who.append("F")
    b.commit(x0, ci)
    if T is not None: T.commit(x0, ci)
    outs.append(x0.detach().float().cpu())
gen_s = time.time() - t0; lat = torch.cat(outs, dim=1)
with torch.no_grad(): im = b.pipe.vae.decode([lat.to(dev0)])[0].detach().float().cpu()
v = ((im.clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy(); v = (v * 255).astype("uint8")
np.save(os.path.join(a.out, "frames_all.npy"), v[::4]); imageio.mimwrite(os.path.join(a.out, "video.mp4"), list(v), fps=16, quality=8)
json.dump(dict(scene=a.scene, seed=a.seed, mode=a.mode, plan="".join(who), gen_s=round(gen_s, 1), target_s=round(tt_target, 1), target_chunks=who.count("T"), steps=a.steps, guide=a.guide, verify=stats), open(os.path.join(a.out, "spec_ar.json"), "w"), indent=1)
print("[SPEC_AR_DONE] %s s%d %s plan=%s gen %.0fs (target %.0fs, %d chunks)" % (a.scene, a.seed, a.mode, "".join(who), gen_s, tt_target, who.count("T")), flush=True)
