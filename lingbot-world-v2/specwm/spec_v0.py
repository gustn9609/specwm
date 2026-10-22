"""Speculative decoding v0 (offline, biased baseline; two GPUs): draft = Fast 14B (cuda:0), corrector/verifier = Base v1 (cuda:1).
Per chunk i: draft proposes x0_i (few-step). Policy decides accept/reject. On reject, Base regenerates chunk i by RePaint (16-step, prefix = accepted chunks kept clean
via per-step replacement, only chunk i free), the corrected chunk is committed to the draft KV and the draft continues. Corrections happen before display -> no cut.
policies: never | always | every:k | verify:<thr>  (verify: rel. distance between draft x0 and Base's 1-step x0 at sigma=1 given the prefix (mask cond.) > thr -> reject)
usage: CUDA_VISIBLE_DEVICES=a,b python -m specwm.spec_v0 --task sand_filling_up_jar --seed 1234 --policy every:2 --out DIR [--max-area 399360] [--frames 161] [--steps 16]"""
import argparse, os, sys, json, time, numpy as np, torch, imageio
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--policy", default="never"); ap.add_argument("--frames", type=int, default=161)
ap.add_argument("--max-area", type=int, default=480 * 832); ap.add_argument("--steps", type=int, default=16); ap.add_argument("--shift", type=float, default=10.0); ap.add_argument("--guide", type=float, default=5.0); ap.add_argument("--nodes", default="0,1,2,3")
ap.add_argument("--fast-ckpt", default="", help="Fast (causal) checkpoint dir; default = lingbot-world-v2-14b-causal-fast, e.g. .../lingbot-world-v2-1.3b-causal-fast")
ap.add_argument("--plan-hint-mode", default="pix", help="pix: VAE-encode the plan video cubic-upscaled to the draft res | lo: VAE-encode it at its own resolution and bilinearly upsample the latents"); ap.add_argument("--specstep", type=int, default=0, help="unbiased per-token speculative sampling at the draft nodes: after node k the draft proposes z_{k+1} ~ N((1-s)x0_D, s^2); Base gives x0_T at the same state; each latent position is accepted with min(1, p_T/p_D) and rejected positions are resampled from the residual of the target kernel"); ap.add_argument("--spec-nodes", default="0,1,2", help="draft nodes after which the transition is verified"); ap.add_argument("--band-probe", type=int, default=0, help="with --specstep: log how much of the draft-target gap (muD-muT)/s lies in low-frequency subspaces (k x k block means per channel and latent frame, k = 2,4,8,16, whole frame) and the implied exact acceptance of a whole frame / whole chunk tested only in that subspace (log10 of 2*Phi(-|P gap|/2)); no change to sampling"); ap.add_argument("--spec-tries", type=int, default=4); ap.add_argument("--kv-mem", default="", help="kv_lora.pt (kvlora_train.py): Base low expert reads the low-res plan once, the translator turns its K/V into Fast plan memory; its LoRA is gated by t >= lora_tmin"); ap.add_argument("--kv-low", default="mem", help="mem: the plan memory stays attached for every forward (as trained) | plain: attached only while the KV LoRA is on (t >= tmin); nodes below and the commit pass run plain Fast"); ap.add_argument("--plan-lat", default="", help="plan_lat.pt of the plan (plan_gen.py) for --kv-mem"); ap.add_argument("--tex-scale", type=float, default=1.0, help="strength of the texture LoRA delta"); ap.add_argument("--tex-lora", default="", help="texture LoRA (texlora_train.py): active at the nodes with t < its lora_tmax, independent of the hint"); ap.add_argument("--hint-low", default="hint", help="what the nodes below the LoRA gate (plain Fast, t < lora_tmin) see in y: hint (the plan hint, as before) | plain (the original y; hint visible only while the LoRA is on, as in training)"); ap.add_argument("--no-base", type=int, default=0, help="1: do not load Base (single GPU; only for --policy never renders)"); ap.add_argument("--start", type=int, default=1, help="first chunk eligible for correction (chunk 0 = image chunk)"); ap.add_argument("--out", required=True)
ap.add_argument("--ctx", type=int, default=0, help="Base sees only the last CTX prefix latents (0 = full prefix); window image = decoded first latent, poses offset accordingly")
ap.add_argument("--base-lora", default="", help="merge a Wan2.2 I2V A14B few-step LoRA into both Base experts: 1022 (lightx2v 4-step 1022) | seko (Lightning Seko-V1)")
ap.add_argument("--lora-scale", type=float, default=1.0)
ap.add_argument("--base-sampler", default="unipc", help="Base correction sampler: unipc | euler (few-step LoRA)")
ap.add_argument("--corr-kv", type=int, default=0, help="RePaint with block-causal Base: prefix encoded once per expert into a KV cache (sink chunk 0 + last --kv-rp-win latents), each step runs only the new chunk")
ap.add_argument("--kv-rp-win", type=int, default=12)
ap.add_argument("--corr-mixed", default="", help="K:R — Base RePaint with a mixed-resolution prefix: sink chunk 0 and the last R prefix latents at full res, the old part in between downsampled K x (K^2 fewer tokens)")
ap.add_argument("--fast-lora", default="", help="fast_lora.pt from fastlora_train.py: LoRA merged into the Fast (draft) model at start")
ap.add_argument("--plan-video", default="", help="with --hint-lora: a low-res whole-clip Base plan (video.mp4); its frames are resized to the draft resolution, VAE-encoded once, and each chunk's plan latents are the hint (no per-chunk Base forward)")
ap.add_argument("--hint-lora", default="", help="hint_lora.pt from hintlora_train.py: Fast LoRA (active at t >= its lora_tmin) that reads Base's one-step x0 hint through the y channel of the current chunk; one Base forward (CFG) per chunk at sigma~1")
ap.add_argument("--fast-win", type=int, default=18, help="Fast KV cache window in latent frames (18 = model default, 4.5 s; 41 = whole clip, nothing evicted)")
ap.add_argument("--bguide", type=float, default=0.0, help="Base-guided Fast: at the Fast nodes in --bguide-nodes, blend Base's x0 for the same chunk state (prefix re-noised to that sigma) into Fast's: x0 = x0_fast + w (x0_base - x0_fast). Fast keeps drawing on its own cache (no seam); 0 = off")
ap.add_argument("--bguide-nodes", default="0,1"); ap.add_argument("--bguide-cfg", type=float, default=-1, help="CFG scale for the Base guidance forward (-1 = --guide; 1 = conditional only, one forward)")
ap.add_argument("--noise-corr", type=float, default=0.0, help="C: one AR(1)-correlated noise field over all latent frames (rho = this), shared by Fast's chunk noise and Base's correction (chunk init + prefix re-noising) so consecutive chunks start from continuous noise; 0 = off")
ap.add_argument("--seam", default="", help="A: S:N — after each chunk (c>=1), the last latent of chunk c-1 and chunk c are re-noised to sigma S and re-denoised jointly by Base (N-step schedule, steps <= S, no CFG), then both chunks are re-committed; '' = off")
ap.add_argument("--ir-prompt", default="", help="ir policy (SDVG router, Hu & Zhang 2026): text prompt (or a file) for ImageReward; the draft chunk is decoded, every frame scored, and the chunk is rejected if min_f reward < tau (policy ir:TAU); chunk 0 is the image chunk, use --start 1 --warmup 1 to force the first generated chunk to Base as in the paper")
ap.add_argument("--base-ckpt", default="", help="target checkpoint dir for the V1Binding (default: LingBot v1 base-cam); e.g. lingbot-world-v2/bid_as_v1_cam = LingBot v2 14B bid (Fast's own teacher) through the v1 camera pipeline")
ap.add_argument("--pq-tau", type=float, default=1e9, help="pq policy: reject a drafted chunk if d = err_Base - err_Fast (one-step re-denoising of the partly noised draft, same noise) > tau")
ap.add_argument("--pq-node", type=int, default=2, help="pq policy: Fast node whose sigma is used for the check (2 = 0.8331)"); ap.add_argument("--pq-ctx", type=int, default=18, help="pq policy: prefix latents Base sees in the check")
ap.add_argument("--pq-retries", type=int, default=3, help="pq policy: on reject, Fast redraws the chunk with fresh noise up to R times; the lowest-d draft is kept (no seam: every frame comes from Fast's own rollout)")
ap.add_argument("--pq-fallback", default="best", help="pq policy, when all redraws fail: best (keep the lowest-d draft) | base (Base 16-step RePaint) | base_low4 (Base RePaint, low frequencies only, Fast keeps the high frequencies)")
ap.add_argument("--pq-verifier", default="pq", help="pq policy scorer: pq (Base minus Fast one-step error, needs Base) | critic:<critic.pt> (latent drift critic logit, higher = more drift; no Base needed)")
ap.add_argument("--pq-teacher", default="", help="pq calibration: score the chunks of this video (teacher-forced) instead of drafting")
ap.add_argument("--corr-blend", default="", help="sem/ir policies (Guandao 9/28, frame-jump fix): how a Base-corrected chunk replaces Fast's draft. '' = replace (default) | soft:A = A*base + (1-A)*draft (soft update) | low:K = draft + lowpass_K(base - draft), lowpass = KxK latent block mean upsampled bilinearly (Base updates only low frequencies, Fast keeps the high frequencies). Warm-up chunks get a draft from a separate RNG stream so the main stream stays paired with the replace baseline")
ap.add_argument("--polish", default="", help="sem policy: S:N — every ACCEPTED Fast chunk is re-noised to sigma S and refined by Base with the steps of an N-step (shift) schedule at or below S (low expert, no CFG). Keeps the draft's content, removes Fast's temporal jitter. '' = off")
ap.add_argument("--corr-from-draft", type=int, default=-1, help="sem policy: Base correction starts from the Fast draft's state entering node K (K=1: sigma .94) and runs only the remaining (lower-noise) steps of its schedule, instead of from pure noise; -1 = off. Parallel-friendly: Base can start as soon as Fast passes node K.")
ap.add_argument("--autoguide", type=float, default=0.0, help="Base correction: replace the CFG uncond branch by the Fast model's velocity on the new chunk (autoguidance with Fast as the 'bad' model): v = v_fast + w (v_base_cond - v_fast); 0 = off")
ap.add_argument("--histguide", type=float, default=0.0, help="Fast drafting: history guidance v = v_full + w (v_full - v_nomem), v_nomem = Fast attending only to the current chunk; 0 = off")
ap.add_argument("--act-theta", type=float, default=0.0, help="action-aware scheduling (sem policy): force a Base correction (no draft check) when the camera yaw at the chunk's end differs by >= THETA deg from the yaw at the last Base-corrected chunk (a new viewing sector gets a Base anchor); 0 = off")
ap.add_argument("--mixed-lora", default="", help="dir with lora_high.pt / lora_low.pt from mixlora_train.py: mixed-memory LoRA applied to the merged old-prefix tokens (--corr-mixed)")
ap.add_argument("--handoff", type=int, default=-1, help="Base plans, Fast renders: Base RePaint runs only its high-expert steps (t >= boundary; layout), its x0 prediction is re-noised to Fast node HANDOFF (1: s=.94, 2: .83, 3: .63) and Fast finishes the remaining nodes. -1 = off")
ap.add_argument("--cfg-steps", type=int, default=-1, help="Base correction: CFG only on the first K steps, conditional-only afterwards (-1 = all steps)")
ap.add_argument("--flow-reuse", default="", help="Base correction: comma list of step indices whose Base forward is skipped and the previous flow reused (feature/flow caching)")
ap.add_argument("--kv-rp-tlow", type=float, default=600.0, help="--corr-kv: noise level (t) of the prefix cache used for all low-expert steps; high-expert steps rebuild the cache at their own t (a clean t=0 prefix breaks the high expert)")
ap.add_argument("--save-drafts", type=int, default=0, help="sem policy: save per-chunk draft frames, Base-corrected frames (incl. resample candidates) and scores to <out>/viz/")
ap.add_argument("--sem-ref", default="start", help="sem policy reference for the colour term: start (start frame) | recent (the most recent Base-corrected chunk, falling back to the start frame)")
ap.add_argument("--sem-retry", type=int, default=0, help="sem policy: re-check Base's corrected chunk with the same score; if still > thr, resample (new noise) up to this many times and keep the lowest-score candidate")
ap.add_argument("--sem-colour", type=float, default=0.0, help="sem policy: add w * (|dS| + |dV|)/80 (mean HSV saturation/brightness shift vs the start frame, 0-255 scale)")
ap.add_argument("--sem-bank", type=int, default=1, help="sem policy: also compare against the DINO features of chunks Base has corrected (1) or only the start frame (0)")
ap.add_argument("--end-pad", type=int, default=1, help="when Base regenerates the last chunk, generate this many extra latent frames past the clip end and drop them (Base fades to black at the end of its window)")
ap.add_argument("--corr-mode", default="repaint", help="Base correction conditioning: repaint (prefix re-noised & replaced each step) | mask (prefix given clean through the I2V mask/y channel) | both")
ap.add_argument("--chunk0-from", default="", help="take chunk 0 from this video (e.g. a Base rollout) instead of the draft: first 4*(CS-1)+1 frames VAE-encoded")
ap.add_argument("--kv-win", type=int, default=6, help="kv check presets: latent frames of recent context kept in the Base KV cache (plus the first chunk as sink)")
ap.add_argument("--mc-k", type=int, default=2, help="mc policy: noise samples per check noise level")
ap.add_argument("--mc-nodes", default="1,2,3", help="mc policy: draft node indices whose sigmas are used to score the finished chunk")
ap.add_argument("--beta", type=float, default=0.0, help="mc policy: acceptance a = min(1, exp(-beta * dL)); 0 = log only (probe)")
ap.add_argument("--mc-chk", default="kv", help="mc policy: target preset (kv / kvg1 / any CHK preset)")
ap.add_argument("--check-at", type=int, default=1, help="early/window: index into --nodes of the draft state sent to the verifier (1 = after the first step)")
ap.add_argument("--corr-area", type=int, default=0, help="Base correction resolution (0 = same as --max-area); latents are resized between draft and Base resolution")
ap.add_argument("--chk", default="full", help="verify_node: check preset (see CHK_PRESETS): guide / prefix window / resolution of the Base check forward")
ap.add_argument("--chk-list", default="full,g1,c9,c3,a360,a240,a360g1,g1c9,a240g1c3", help="probe2: presets evaluated at the check node every chunk")
ap.add_argument("--warmup", type=int, default=0, help="verify_node: chunks start..start+WARMUP-1 are always corrected (no check), then gating")
ap.add_argument("--run", type=int, default=1, help="verify_node: once a chunk is rejected, also correct the next RUN-1 chunks without re-checking (hysteresis)")
ap.add_argument("--check-nodes", default="1,2", help="draft nodes at which the target (Base) evaluates the draft state (probe / spec / verify_node policies)")
ap.add_argument("--tok", type=int, default=4, help="token = (latent frame, tok x tok latent patch) for per-token acceptance (spec policy)")
ap.add_argument("--tau", type=float, default=0.5, help="spec policy, final node only: per-token rel. distance above which the token takes the target x0 (biased last step)")
ap.add_argument("--resident", action="store_true", help="keep both Base experts on the corrector GPU (default: one resident, swap at the t boundary; late chunks at 480p OOM with both)")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); pol, _, parg = a.policy.partition(":")
T = [t for t in json.load(open(S + "/data/stevo/tasks_camera.json")) if t["id"] == a.task][0]; img, act, prompt = T["image"], T["action"], T.get("camera_WM", "")
dev0, dev1 = torch.device("cuda:0"), torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from specwm.binding_v2 import V2Binding
b = V2Binding(a.fast_ckpt or (S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"), target_mode="fast_bf16", quant=None, sink=6, local_attn=a.fast_win, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), **({"cfg_name": "i2v-1.3B", "assets_dir": S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"} if "1.3b" in a.fast_ckpt else {}))
HL = {}
if a.hint_lora:   # Base-hint LoRA: gated (by timestep) LoRA modules + fine-tuned patch embedding
    import torch.nn as _nn, math as _math
    _st = torch.load(a.hint_lora); HL["tmin"] = float(_st["lora_tmin"]); HL["mask"] = float(_st["hint_mask"]); HL["on"] = False
    class _GL(_nn.Module):
        def __init__(self, base, A, B):
            super().__init__(); self.base = base; self.register_buffer("A", A.to(base.weight.device, torch.float32), persistent=False); self.register_buffer("B", B.to(base.weight.device, torch.float32), persistent=False)
        def forward(self, x):
            y = self.base(x)
            return y + ((x.float() @ self.A.t()) @ self.B.t()).to(y.dtype) if HL["on"] else y
    _i = 0
    for _blk in b.target_model.blocks:
        _sa = _blk.self_attn
        for _n in ("q", "k", "v", "o"): setattr(_sa, _n, _GL(getattr(_sa, _n), _st["A"][_i], _st["B"][_i])); _i += 1
        for _j in (0, 2): _blk.ffn[_j] = _GL(_blk.ffn[_j], _st["A"][_i], _st["B"][_i]); _i += 1
    _pe = b.target_model.patch_embedding; HL["pe_w"], HL["pe_b"] = _st["pe_w"].to(_pe.weight.device, _pe.weight.dtype), _st["pe_b"].to(_pe.bias.device, _pe.bias.dtype)
    class _GPE(_nn.Module):   # fine-tuned patch embedding only while the hint is active
        def __init__(self, pe): super().__init__(); self.pe = pe
        @property
        def weight(self): return self.pe.weight
        @property
        def bias(self): return self.pe.bias
        def forward(self, x):
            if HL["on"]: return torch.nn.functional.conv3d(x, HL["pe_w"].to(x.dtype), HL["pe_b"].to(x.dtype), stride=self.pe.stride, padding=self.pe.padding)
            return self.pe(x)
    b.target_model.patch_embedding = _GPE(_pe)
    print("[hint-lora] %d gated LoRA modules, tmin %.0f, mask %.2f (step %s)" % (_i, HL["tmin"], HL["mask"], _st.get("step")), flush=True)
KVM = {}
if a.kv_mem:   # KV-translator route: gated LoRA (timestep only) + plan memory attached to every Fast attention block before the rollout
    import torch.nn as _nn
    _st = torch.load(a.kv_mem); KVM["tmin"] = float(_st["lora_tmin"]); KVM["on"] = False; KVM["st"] = _st
    _wrap = a.kv_mem != a.hint_lora   # the same checkpoint as --hint-lora: its LoRA/PE are already applied by the hint route (y-hint + memory), only the memory is added here
    class _KL(_nn.Module):
        def __init__(self, base, A, B):
            super().__init__(); self.base = base; self.register_buffer("A", A.to(dev0, torch.float32), persistent=False); self.register_buffer("B", B.to(dev0, torch.float32), persistent=False)
        def forward(self, x):
            y = self.base(x)
            return y + ((x.float() @ self.A.t()) @ self.B.t()).to(y.dtype) if KVM["on"] else y
    _i = 0
    for _blk in (b.target_model.blocks if _wrap else []):
        _sa = _blk.self_attn
        for _n in ("q", "k", "v", "o"): setattr(_sa, _n, _KL(getattr(_sa, _n), _st["A"][_i], _st["B"][_i])); _i += 1
        for _j in (0, 2): _blk.ffn[_j] = _KL(_blk.ffn[_j], _st["A"][_i], _st["B"][_i]); _i += 1
    print("[kv-mem] %d gated LoRA modules, tmin %.0f, layers %s, kv_t %.0f, stride %d (step %s)" % (_i, KVM["tmin"], _st["layers"], _st["kv_t"], _st.get("mem_stride", 2), _st.get("step")), flush=True)
TX = {}
if a.tex_lora:   # texture LoRA: gated by timestep only (t < tmax), on top of the hint LoRA modules
    import torch.nn as _nn
    _st = torch.load(a.tex_lora); TX["tmax"] = float(_st["lora_tmax"]); TX["on"] = False
    class _TL(_nn.Module):
        def __init__(self, base, A, B):
            super().__init__(); self.base = base; self.register_buffer("A", A.to(dev0, torch.float32), persistent=False); self.register_buffer("B", B.to(dev0, torch.float32), persistent=False)
        def forward(self, x):
            y = self.base(x)
            return y + (a.tex_scale * ((x.float() @ self.A.t()) @ self.B.t())).to(y.dtype) if TX["on"] else y
    _i = 0
    for _blk in b.target_model.blocks:
        _sa = _blk.self_attn
        for _n in ("q", "k", "v", "o"): setattr(_sa, _n, _TL(getattr(_sa, _n), _st["A"][_i], _st["B"][_i])); _i += 1
        for _j in (0, 2): _blk.ffn[_j] = _TL(_blk.ffn[_j], _st["A"][_i], _st["B"][_i]); _i += 1
    print("[tex-lora] %d gated LoRA modules, tmax %.0f (step %s)" % (_i, TX["tmax"], _st.get("step")), flush=True)
if a.fast_lora:   # merge the trained LoRA deltas into the Fast weights (same modules/order as fastlora_train.py)
    _st = torch.load(a.fast_lora); _i = 0
    with torch.no_grad():
        for _blk in b.target_model.blocks:
            _mods = [getattr(_blk.self_attn, _n) for _n in ("q", "k", "v", "o")] + ([_blk.ffn[0], _blk.ffn[2]] if _st.get("ffn", 1) else [])
            for _m in _mods:
                _dW = (_st["B"][_i].float() @ _st["A"][_i].float()).to(_m.weight.device); _m.weight.add_(_dW.to(_m.weight.dtype)); _i += 1
    print("[fast-lora] merged %d LoRA deltas from %s (step %s)" % (_i, a.fast_lora, _st.get("step")), flush=True)
cond = b.prepare(img, prompt, act, frame_num=a.frames, max_area=a.max_area); N = cond["num_chunks"]; sig = b.node_sigmas(); NODES = [int(x) for x in a.nodes.split(",")]; CS = 3
# ---- Base v1 on dev1 (isolated package name; both experts resident)
import importlib, importlib.util
os.chdir(S + "/lingbot-world"); sys.path = [S + "/lingbot-world"] + [p for p in sys.path if p != S + "/lingbot-world-v2"]
for k in [k for k in sys.modules if k == "wan" or k.startswith("wan.")]: del sys.modules[k]
spec = importlib.util.spec_from_file_location("specwm_v1", S + "/lingbot-world/specwm/__init__.py", submodule_search_locations=[S + "/lingbot-world/specwm"]); pkg = importlib.util.module_from_spec(spec); sys.modules["specwm_v1"] = pkg; spec.loader.exec_module(pkg)
V1Binding = importlib.import_module("specwm_v1.binding_v1").V1Binding
b1 = None if a.no_base else V1Binding(a.base_ckpt or (S + "/lingbot-world/lingbot-world-base-cam"), t5_cpu=True, device=str(dev1))
if a.no_base: assert (a.policy == "never" or (a.policy == "pq" and a.pq_verifier.startswith("critic:"))) and a.bguide <= 0, "--no-base needs --policy never (or pq with a critic verifier)"; live = None
cur_expert = ["high"]; n_swap = [0]; TM = dict(swap=0.0, cond=0.0, denoise=0.0); W = {}
# expert swap: ONE live model on dev1 (high's module), both weight sets pinned on the host; swap = 28 GB H2D copy into the live parameters (~2-3 s).
if not a.no_base: live = b1.pipe.high_noise_model.to(dev1); b1.device = dev1
if a.no_base: pass
elif not a.resident:
    W["high"] = {n: torch.empty(p.shape, dtype=p.dtype, device="cpu", pin_memory=True).copy_(p) for n, p in live.named_parameters()}
    W["low"] = {n: torch.empty(p.shape, dtype=p.dtype, device="cpu", pin_memory=True).copy_(p) for n, p in b1.pipe.low_noise_model.named_parameters()}
    assert set(W["high"]) == set(W["low"]); b1.pipe.low_noise_model = None; torch.cuda.empty_cache()
else: b1.pipe.low_noise_model.to(dev1)
LORA = {"1022": ("/home/hk28965/specwm/hf/lightx2v/Wan2.2-Distill-Loras/wan2.2_i2v_A14b_%s_noise_lora_rank64_lightx2v_4step_1022.safetensors", False),
        "seko": ("/home/hk28965/specwm/hf/lightx2v/Wan2.2-Lightning/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/%s_noise_model.safetensors", True)}
def merge_lora(Wd, path, use_alpha, scale):
    """merge lightx2v-format LoRA (lora_down/up [+alpha], diff_b bias deltas, diff weight deltas) into the host weight dict"""
    from safetensors import safe_open
    n_lin = n_b = n_d = 0
    with safe_open(path, "pt") as fh:
        keys = set(fh.keys())
        for k in keys:
            if not k.endswith(".lora_down.weight"): continue
            base = k[len("diffusion_model."):-len(".lora_down.weight")]; wk = base + ".weight"
            if wk not in Wd: continue
            dn = fh.get_tensor(k).float(); up = fh.get_tensor("diffusion_model." + base + ".lora_up.weight").float()
            sc = scale * (float(fh.get_tensor("diffusion_model." + base + ".alpha")) / dn.shape[0] if use_alpha and ("diffusion_model." + base + ".alpha") in keys else 1.0)
            dW = (up.flatten(1) @ dn.flatten(1)).reshape(Wd[wk].shape) * sc; Wd[wk].copy_((Wd[wk].float() + dW).to(Wd[wk].dtype)); n_lin += 1
        for k in keys:
            if k.endswith(".diff_b") or k.endswith(".diff"):
                base = k[len("diffusion_model."):].rsplit(".", 1)[0]; wk = base + (".bias" if k.endswith(".diff_b") else ".weight")
                if wk in Wd and Wd[wk].shape == fh.get_tensor(k).shape:
                    Wd[wk].copy_((Wd[wk].float() + scale * fh.get_tensor(k).float()).to(Wd[wk].dtype)); n_b += k.endswith(".diff_b"); n_d += k.endswith(".diff")
    print("[lora] %s: merged %d low-rank, %d bias, %d weight deltas" % (os.path.basename(path), n_lin, n_b, n_d), flush=True)
if a.base_lora:
    assert not a.resident, "--base-lora needs the default (non-resident) expert swap"
    pth, ua = LORA[a.base_lora]
    for ex in ("high", "low"): merge_lora(W[ex], pth % ex, ua, a.lora_scale)
    cur_expert[0] = None   # force the merged weights to be copied into the live model on first use
MLORA = {}
if a.mixed_lora:
    from specwm.base_mixed import attach_lora, load_lora
    MLORA["st"] = {ex: torch.load(os.path.join(a.mixed_lora, "lora_%s.pt" % ex)) for ex in ("high", "low")}
    MLORA["mods"] = attach_lora(live, MLORA["st"]["high"]); MLORA["cur"] = "high"
    print("[mixed-lora] attached (steps high %d, low %d)" % (MLORA["st"]["high"]["step"], MLORA["st"]["low"]["step"]), flush=True)
def expert(name):
    if a.resident: return b1.pipe.high_noise_model if name == "high" else b1.pipe.low_noise_model
    if cur_expert[0] != name:
        t = time.time()
        with torch.no_grad():
            for n, p in live.named_parameters(): p.copy_(W[name][n.replace(".base.", ".")], non_blocking=True)
            if MLORA: load_lora(MLORA["mods"], MLORA["st"][name])
        torch.cuda.synchronize(dev1); cur_expert[0] = name; n_swap[0] += 1; TM["swap"] += time.time() - t
    return live
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
os.chdir(S + "/lingbot-world-v2"); vae = b.pipe.vae; cond_cache = {}
def base_cond(F, j=0, prefix_lat=None, area=None):
    """conditioning for a Base window of F frames starting at prefix latent j (j=0: start image; j>0: decoded latent j as the window's first image, poses from frame 4j)."""
    area = area or a.max_area
    if (F, j, area) not in cond_cache:
        if j == 0: cond_cache[(F, j, area)] = b1.prepare(img, prompt, act, frame_num=F, max_area=area)
        else:
            with torch.no_grad(): im0 = vae.decode([prefix_lat[:, j:j + 1].to(dev0)])[0][:, 0].clamp(-1, 1).float()
            cond_cache[(F, j, area)] = b1.prepare(im0, prompt, act, frame_num=F, max_area=area, pose_offset=4 * j)
    return cond_cache[(F, j, area)]
def base_flow(z, t_val, c, guide):
    m = expert("high" if t_val >= b1.boundary_t else "low"); tt = torch.full((1,), float(t_val), device=dev1); zin = [z.to(dev1, b1.param_dtype)]
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=b1.param_dtype):
        fc = m(zin, t=tt, context=[c["ctx"][0]], seq_len=c["seq_len"], y=[c["y"]], dit_cond_dict=c["dit_cond"])[0]
        if guide == 1.0: return fc.float()   # no CFG: skip the unconditional forward
        fu = m(zin, t=tt, context=c["ctxn"], seq_len=c["seq_len"], y=[c["y"]], dit_cond_dict=c["dit_cond"])[0]
    return (fu + guide * (fc - fu)).float()
@torch.no_grad()
def _down(x, hw):
    if tuple(x.shape[2:]) == tuple(hw): return x
    return torch.nn.functional.interpolate(x.permute(1, 0, 2, 3).float(), size=hw, mode="bilinear", antialias=True, align_corners=False).permute(1, 0, 2, 3)
def base_repaint_chunk(prefix_lat, ci, seed, n=1, init=None, nsteps=None):
    """Base regenerates chunks ci..ci+n-1 jointly given the clean prefix (RePaint: prefix re-noised & replaced every step; chunks free). Returns [16,3n,h,w] on dev0 (draft resolution)."""
    hw0 = tuple(prefix_lat.shape[2:]); area = a.corr_area or a.max_area; nout = int(round(n * CS)); pad = a.end_pad if ci + n >= N else 0
    n = n + pad / CS   # extra latent frames, dropped below
    t = time.time(); j = max(0, prefix_lat.shape[1] - a.ctx) if a.ctx > 0 else 0; c = dict(base_cond(int(4 * (prefix_lat.shape[1] - j + n * CS - 1) + 1), j, prefix_lat, area)); prefix_lat = _down(prefix_lat[:, j:].to(dev1), tuple(c["lat"][1:])); L = prefix_lat.shape[1]; TM["cond"] += time.time() - t
    if a.corr_mixed and a.ctx == 0 and not a.corr_area and a.corr_mode == "repaint" and a.handoff < 0 and L - int(a.corr_mixed.split(":")[1]) > CS: return base_repaint_mixed(prefix_lat, ci, seed, hw0, c, nout, int(round(n * CS)), init, nsteps)
    if a.corr_kv and n == 1 and a.ctx == 0 and not a.corr_area and a.corr_mode == "repaint": return base_repaint_kv(prefix_lat, ci, seed, hw0)
    sch = FlowUniPCMultistepScheduler(num_train_timesteps=b1.pipe.num_train_timesteps, shift=1, use_dynamic_shifting=False); sch.set_timesteps(nsteps or a.steps, device=dev1, shift=a.shift); ts, sg = sch.timesteps, sch.sigmas
    if a.base_sampler == "euler": sg = euler_sigmas(); ts = [x * 1000.0 for x in sg[:-1]]
    if a.corr_mode in ("mask", "both"):   # I2V-style conditioning: decoded prefix frames in the y channel with mask = 1 (VAE work on the draft GPU)
        t = time.time(); F = int(4 * (L + n * CS - 1) + 1); h, w = c["hw"]; lat_h, lat_w = c["lat"][1:]
        with torch.no_grad():
            px = vae.decode([prefix_lat.to(dev0)])[0].clamp(-1, 1); Pf = px.shape[1]
            msk = torch.ones(1, F, lat_h, lat_w, device=dev1); msk[:, Pf:] = 0
            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1); msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]
            y_lat = vae.encode([torch.concat([px, torch.zeros(3, F - Pf, h, w, device=dev0)], dim=1)])[0].to(dev1); del px; torch.cuda.empty_cache()
        c["y"] = torch.concat([msk, y_lat]); TM["cond"] += time.time() - t
    g = torch.Generator(device=dev1); g.manual_seed(seed * 7919 + ci); pref = prefix_lat.to(dev1); t0c = time.time(); flow = None; reuse = {int(x) for x in a.flow_reuse.split(",") if x} if not nsteps else set()
    lat = torch.randn((16, L + int(round(n * CS))) + tuple(prefix_lat.shape[2:]), generator=g, device=dev1); i0 = 0; nfw = None
    if a.noise_corr > 0:   # C: continuous noise field for the whole window (prefix part re-used for the per-step re-noising)
        nfw = _down(noise_field()[:, j:j + lat.shape[1]].to(dev1), tuple(lat.shape[2:])); lat = nfw.clone()
    if init is not None:   # start from the draft's state at sigma init[1]: only the steps at or below that noise level
        lat[:, L:L + init[0].shape[1]] = _down(init[0].to(dev1), tuple(lat.shape[2:])); i0 = next(i for i in range(len(ts)) if float(sg[i]) <= init[1] + 1e-4)
    for i, t in enumerate(ts):
        if i < i0: continue
        s = float(sg[i])
        if a.corr_mode in ("repaint", "both"): lat[:, :L] = (1 - s) * pref + s * (nfw[:, :L] if nfw is not None else torch.randn(pref.shape, generator=g, device=dev1))
        if not (i in reuse and flow is not None):
            gd = a.guide if (a.cfg_steps < 0 or i < a.cfg_steps) else 1.0
            if a.autoguide > 0 and gd != 1.0:   # autoguidance: Fast replaces the uncond forward on the new chunk
                flow = base_flow(lat, float(t), c, 1.0); vf = fast_vel(lat[:, L:L + CS], float(t), ci, hw0)
                flow[:, L:L + CS] = vf + a.autoguide * (flow[:, L:L + CS] - vf)
            else: flow = base_flow(lat, float(t), c, gd)
        if a.handoff >= 0 and (i + 1 == len(ts) or float(ts[i + 1]) < b1.boundary_t):   # last high-expert step: hand the layout to Fast
            x0b = (lat - s * flow)[:, L:L + nout].float(); torch.cuda.synchronize(dev1); TM["denoise"] += time.time() - t0c
            return fast_finish(_down(x0b, hw0).to(dev0), ci, seed)
        if a.base_sampler == "euler": lat = lat + (float(sg[i + 1]) - s) * flow
        else: lat = sch.step(flow.unsqueeze(0), t, lat.unsqueeze(0), return_dict=False, generator=g)[0].squeeze(0)
    torch.cuda.synchronize(dev1); TM["denoise"] += time.time() - t0c; return _down(lat[:, L:L + nout].float(), hw0).to(dev0)
@torch.no_grad()
def base_repaint_mixed(pref, ci, seed, hw0, c, nout, nn, init=None, nsteps=None):
    """RePaint of the nn latents after pref (base res, dev1) with the old prefix part at 1/K resolution (see base_mixed.py)."""
    from specwm.base_mixed import mixed_forward, down, down_cam
    K, R = [int(x) for x in a.corr_mixed.split(":")]; L = pref.shape[1]; hi = max(CS, L - R); has_old = hi > CS
    cam = c["dit_cond"]["c2ws_plucker_emb"][0] if c["dit_cond"] else None; y = c["y"]
    cs = lambda lo, h_, k=1: None if cam is None else down_cam(cam[:, :, lo:h_], k)
    old = down(pref[:, CS:hi], K) if has_old else None
    segs0 = [dict(y=y[:, :CS], f0=0, k=1, cam=cs(0, CS))] + ([dict(y=down(y[:, CS:hi], K), f0=CS, k=K, cam=cs(CS, hi, K), lora=bool(MLORA))] if has_old else []) + [dict(y=y[:, hi:L + nn], f0=hi, k=1, cam=cs(hi, L + nn))]
    sch = FlowUniPCMultistepScheduler(num_train_timesteps=b1.pipe.num_train_timesteps, shift=1, use_dynamic_shifting=False); sch.set_timesteps(nsteps or a.steps, device=dev1, shift=a.shift)
    g = torch.Generator(device=dev1); g.manual_seed(seed * 7919 + ci); t0c = time.time(); reuse = {int(x) for x in a.flow_reuse.split(",") if x} if not nsteps else set(); flow = None
    z = torch.randn((16, nn) + tuple(pref.shape[2:]), generator=g, device=dev1); i0 = 0
    if init is not None:
        z[:, :init[0].shape[1]] = _down(init[0].to(dev1), tuple(z.shape[2:])); i0 = next(i for i in range(len(sch.timesteps)) if float(sch.sigmas[i]) <= init[1] + 1e-4)
    for i, t in enumerate(sch.timesteps):
        if i < i0: continue
        s = float(sch.sigmas[i]); tv = float(t); rn = lambda x: (1 - s) * x + s * torch.randn(x.shape, generator=g, device=dev1)
        xs = [rn(pref[:, :CS])] + ([rn(old)] if has_old else []) + [torch.cat([rn(pref[:, hi:]), z], 1)]
        segs = [dict(sg, x=x) for sg, x in zip(segs0, xs)]
        if not (i in reuse and flow is not None):
            m = expert("high" if tv >= b1.boundary_t else "low"); gd = a.guide if (a.cfg_steps < 0 or i < a.cfg_steps) else 1.0
            fc = mixed_forward(m, segs, tv, [c["ctx"][0]], [len(segs) - 1])[0][:, -nn:]
            flow = fc if gd == 1.0 else (lambda fu: fu + gd * (fc - fu))(mixed_forward(m, segs, tv, c["ctxn"], [len(segs) - 1])[0][:, -nn:])
        z = sch.step(flow.unsqueeze(0), t, z.unsqueeze(0), return_dict=False, generator=g)[0].squeeze(0)
    torch.cuda.synchronize(dev1); TM["denoise"] += time.time() - t0c; return _down(z[:, :nout].float(), hw0).to(dev0)
@torch.no_grad()
def fast_finish(x0b, ci, seed):
    """re-noise Base's x0 to Fast node a.handoff and let Fast (its own KV cache = committed prefix) finish the remaining nodes."""
    t = time.time(); gg = torch.Generator().manual_seed(seed * 31 + ci); k = NODES.index(a.handoff) if a.handoff in NODES else a.handoff
    s0 = float(sig[NODES[k]]); z = (1 - s0) * x0b + s0 * b.randn_like(x0b, gg)
    for j in range(k, len(NODES)):
        x0 = b.target_x0(z, NODES[j], ci)
        if j < len(NODES) - 1: s_n = float(sig[NODES[j + 1]]); z = (1 - s_n) * x0 + s_n * b.randn_like(x0, gg)
    TM["fast_finish"] = TM.get("fast_finish", 0.0) + time.time() - t; return x0
def euler_sigmas():
    s = torch.linspace(1, 0, a.steps + 1); return (a.shift * s / (1 + (a.shift - 1) * s)).tolist()
CURM = [None]
@torch.no_grad()
def base_repaint_kv(prefix_lat, ci, seed, hw0):
    """RePaint of chunk ci with a block-causal Base: for each expert used, the clean prefix (sink chunk 0 + last kv_rp_win latents) is written once
    into a KV cache (t=0, attending causally chunk by chunk); every denoising step then runs only the 3 new latents against that cache."""
    from specwm.base_cache import BaseKV
    t = time.time(); c = kv_full_cond(); cam = c["dit_cond"]["c2ws_plucker_emb"][0] if c["dit_cond"] else None; TM["cond"] += time.time() - t
    f0 = ci * CS; yy = c["y"][:, f0:f0 + CS]; cm = None if cam is None else cam[:, :, f0:f0 + CS]
    if a.base_sampler == "euler": sgl = euler_sigmas(); ts = [x * 1000.0 for x in sgl[:-1]]; sch = None
    else:
        sch = FlowUniPCMultistepScheduler(num_train_timesteps=b1.pipe.num_train_timesteps, shift=1, use_dynamic_shifting=False); sch.set_timesteps(a.steps, device=dev1, shift=a.shift); ts = [float(x) for x in sch.timesteps]
    g = torch.Generator(device=dev1); g.manual_seed(seed * 7919 + ci); t0c = time.time()
    z = torch.randn((16, CS) + tuple(prefix_lat.shape[2:]), generator=g, device=dev1)
    kvr = BaseKV(lambda: CURM[0], win=10 ** 6); built = None; nc = prefix_lat.shape[1] // CS
    keep = [k for k in range(nc) if k == 0 or k >= nc - a.kv_rp_win // CS]
    eps = torch.randn(prefix_lat.shape, generator=g, device=dev1)
    for i, tv in enumerate(ts):
        ex = "high" if tv >= b1.boundary_t else "low"; tp = tv if ex == "high" else a.kv_rp_tlow
        if built != (ex, tp):
            CURM[0] = expert(ex); kvr.reset(); torch.cuda.empty_cache(); sp = tp / 1000.0
            for k in keep:
                xk = ((1 - sp) * prefix_lat[:, k * CS:(k + 1) * CS].to(dev1) + sp * eps[:, k * CS:(k + 1) * CS]); yk = c["y"][:, k * CS:(k + 1) * CS]; ck = None if cam is None else cam[:, :, k * CS:(k + 1) * CS]
                kvr.forward("c", xk, yk, k * CS, tp, [c["ctx"][0]], ck, write=True)
                if a.guide != 1.0: kvr.forward("u", xk, yk, k * CS, tp, c["ctxn"], ck, write=True)
            built = (ex, tp)
        fc = kvr.forward("c", z, yy, f0, tv, [c["ctx"][0]], cm)
        flow = fc if a.guide == 1.0 else (lambda fu: fu + a.guide * (fc - fu))(kvr.forward("u", z, yy, f0, tv, c["ctxn"], cm))
        if os.environ.get("KVDBG"): print("[kvdbg] ci %d i %d t %.0f %s z %.3f flow %.3f fc %.3f shapes z%s y%s pref%s" % (ci, i, tv, ex, float(z.std()), float(flow.std()), float(fc.std()), tuple(z.shape), tuple(yy.shape), tuple(prefix_lat.shape)), flush=True)
        if sch is None: z = z + (sgl[i + 1] - sgl[i]) * flow
        else: z = sch.step(flow.unsqueeze(0), sch.timesteps[i], z.unsqueeze(0), return_dict=False, generator=g)[0].squeeze(0)
    kvr.reset(); torch.cuda.synchronize(dev1); TM["denoise"] += time.time() - t0c; return _down(z.float(), hw0).to(dev0)
@torch.no_grad()
def base_x0_at_sigma1(prefix_lat, z_chunk):
    """verifier signal: Base's one-step x0 for the chunk at sigma=1 given the prefix through the y (mask) channel."""
    L = prefix_lat.shape[1]; F = 4 * (L + CS - 1) + 1; c = dict(base_cond(F)); h, w = c["hw"]; lat_h, lat_w = c["lat"][1:]
    px = vae.decode([prefix_lat.to(dev0)])[0]; Pf = px.shape[1]; msk = torch.ones(1, F, lat_h, lat_w, device=dev1); msk[:, Pf:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1); msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]
    y_lat = vae.encode([torch.concat([px, torch.zeros(3, F - Pf, h, w, device=dev0)], dim=1)])[0].to(dev1); del px; torch.cuda.empty_cache(); c["y"] = torch.concat([msk, y_lat])   # VAE work on the draft GPU (same Wan2.1 VAE)
    z = torch.cat([torch.randn_like(prefix_lat).to(dev1), z_chunk.to(dev1)], 1); flow = base_flow(z, 1000.0, c, a.guide); return (z - flow)[:, L:].float().to(dev0)
@torch.no_grad()
def base_x0_at(prefix_lat, z_chunk, s):
    """Base's x0 prediction for the chunk state z (at sigma s) with the prefix re-noised to the same sigma (RePaint-style context, --ctx window). One CFG forward."""
    t = time.time(); j = max(0, prefix_lat.shape[1] - a.ctx) if a.ctx > 0 else 0; c = dict(base_cond(4 * (prefix_lat.shape[1] - j + CS - 1) + 1, j, prefix_lat)); pref = prefix_lat[:, j:].to(dev1); L = pref.shape[1]; TM["cond"] += time.time() - t; t0c = time.time()
    lat = torch.cat([(1 - s) * pref + s * torch.randn_like(pref), z_chunk.to(dev1)], 1); flow = base_flow(lat, s * 1000.0, c, a.bguide_cfg if a.bguide_cfg > 0 else a.guide); x0 = (lat - s * flow)[:, L:].float().to(dev0)
    torch.cuda.synchronize(dev1); TM["denoise"] += time.time() - t; return x0
CHK_PRESETS = {   # guide (1 = no uncond forward), ctx (prefix latents seen; 0 = all), area (Base check resolution)
    "full": (a.guide, 0, a.max_area), "g1": (1.0, 0, a.max_area), "c9": (a.guide, 9, a.max_area), "c3": (a.guide, 3, a.max_area),
    "a360": (a.guide, 0, 184320), "a240": (a.guide, 0, 99840), "a360g1": (1.0, 0, 184320), "g1c9": (1.0, 9, a.max_area), "a240g1c3": (1.0, 3, 99840),
    "kv": (a.guide, -1, a.max_area), "kvg1": (1.0, -1, a.max_area)}   # ctx -1: block-causal Base with a prefix KV cache (base_cache.py)
from specwm.base_cache import BaseKV
KV = BaseKV(lambda: expert("low"), win=a.kv_win); KVS = dict(n=0)
def kv_full_cond():
    return base_cond(4 * (N * CS - 1) + 1)
def kv_sync(prefix_lat, guide):
    """make the Base KV cache hold exactly the committed chunks of prefix_lat (clean, t=0); rebuild after a rollback."""
    nc = prefix_lat.shape[1] // CS; c = kv_full_cond(); cam = c["dit_cond"]["c2ws_plucker_emb"][0] if c["dit_cond"] else None
    if nc < KVS["n"]: KV.reset(); KVS["n"] = 0
    while KVS["n"] < nc:
        i = KVS["n"]; f0 = i * CS; x = prefix_lat[:, f0:f0 + CS].to(dev1); yy = c["y"][:, f0:f0 + CS]; cm = None if cam is None else cam[:, :, f0:f0 + CS]
        KV.forward("c", x, yy, f0, 0.0, [c["ctx"][0]], cm, write=True)
        if guide != 1.0: KV.forward("u", x, yy, f0, 0.0, c["ctxn"], cm, write=True)
        KVS["n"] += 1
    return c, cam
@torch.no_grad()
def kv_x0(prefix_lat, z, s, guide):
    """Base x0 for the next chunk state z at sigma s, attending to the cached (clean) prefix."""
    c, cam = kv_sync(prefix_lat, guide); f0 = KVS["n"] * CS; yy = c["y"][:, f0:f0 + CS]; cm = None if cam is None else cam[:, :, f0:f0 + CS]; zz = z.to(dev1)
    fc = KV.forward("c", zz, yy, f0, s * 1000.0, [c["ctx"][0]], cm)
    flow = fc if guide == 1.0 else (lambda fu: fu + guide * (fc - fu))(KV.forward("u", zz, yy, f0, s * 1000.0, c["ctxn"], cm))
    return (zz - s * flow).float()
@torch.no_grad()
def base_check(prefix_lat, x0_prev, z, s, x0_draft, name):
    """cheap verifier: rel. distance between the draft x0 and Base's x0 for the same chunk state at sigma s, under preset `name`.
    Lower resolution: the latents are resized (bilinear, antialias) and the chunk state is rebuilt as (1-s)*down(x0_prev)+s*eps so the noise level stays exact."""
    guide, ctx, area = CHK_PRESETS[name]; t = time.time()
    if ctx == -1:
        x0t = kv_x0(prefix_lat, z, s, guide); xd = x0_draft.to(dev1).float(); d = float((xd - x0t).norm() / (xd.norm() + 1e-6)); torch.cuda.synchronize(dev1); tf = time.time() - t
        TM["denoise"] += tf; return d, 0.0, tf
    j = max(0, prefix_lat.shape[1] - ctx) if ctx > 0 else 0; c = dict(base_cond(4 * (prefix_lat.shape[1] - j + CS - 1) + 1, j, prefix_lat, area)); hw = tuple(c["lat"][1:])
    pref = _down(prefix_lat[:, j:].to(dev1), hw); L = pref.shape[1]
    zc = z.to(dev1) if hw == tuple(z.shape[2:]) else (1 - s) * _down(x0_prev.to(dev1), hw) + s * torch.randn((16, CS) + hw, device=dev1)
    torch.cuda.synchronize(dev1); tc = time.time() - t; t = time.time()
    lat = torch.cat([(1 - s) * pref + s * torch.randn_like(pref), zc], 1); flow = base_flow(lat, s * 1000.0, c, guide); x0t = (lat - s * flow)[:, L:].float()
    xd = _down(x0_draft.to(dev1), hw).float(); d = float((xd - x0t).norm() / (xd.norm() + 1e-6)); torch.cuda.synchronize(dev1); tf = time.time() - t
    TM["cond"] += tc; TM["denoise"] += tf; return d, tc, tf
@torch.no_grad()
def base_check_window(prefix_lat, sts, s, name):
    """one Base forward over [prefix @ sigma s, k draft chunk states @ sigma s]: per-chunk rel. distance of the draft x0 (at the check node) to Base's x0.
    sts: list of (z_state, x0_prev, x0_draft) per chunk (draft resolution)."""
    guide, ctx, area = CHK_PRESETS[name]; t = time.time(); k = len(sts)
    j = max(0, prefix_lat.shape[1] - ctx) if ctx > 0 else 0; c = dict(base_cond(4 * (prefix_lat.shape[1] - j + k * CS - 1) + 1, j, prefix_lat, area)); hw = tuple(c["lat"][1:])
    pref = _down(prefix_lat[:, j:].to(dev1), hw); L = pref.shape[1]
    zs = [z.to(dev1) if hw == tuple(z.shape[2:]) else (1 - s) * _down(xp.to(dev1), hw) + s * torch.randn((16, CS) + hw, device=dev1) for z, xp, _ in sts]
    torch.cuda.synchronize(dev1); tc = time.time() - t; t = time.time()
    lat = torch.cat([(1 - s) * pref + s * torch.randn_like(pref)] + zs, 1); flow = base_flow(lat, s * 1000.0, c, guide); x0t = (lat - s * flow)[:, L:].float()
    ds = []
    for q, (_, _, xd) in enumerate(sts):
        xd = _down(xd.to(dev1), hw).float(); xt = x0t[:, q * CS:(q + 1) * CS]; ds.append(float((xd - xt).norm() / (xd.norm() + 1e-6)))
    torch.cuda.synchronize(dev1); tf = time.time() - t; TM["cond"] += tc; TM["denoise"] += tf; return ds, tc, tf
def _blk(n, P):
    while n % P: P -= 1
    return P
def tok_sum(x):
    """[16,3,H,W] -> per-token sum [3,H/bh,W/bw]"""
    C, Fn, H, Wd = x.shape; bh, bw = _blk(H, a.tok), _blk(Wd, a.tok)
    return x.view(C, Fn, H // bh, bh, Wd // bw, bw).sum((0, 3, 5))
def tok_expand(m, like):
    C, Fn, H, Wd = like.shape; bh, bw = _blk(H, a.tok), _blk(Wd, a.tok)
    return m.repeat_interleave(bh, 1).repeat_interleave(bw, 2).unsqueeze(0).expand(C, -1, -1, -1)
def tok_dim(like):
    C, Fn, H, Wd = like.shape; return C * _blk(H, a.tok) * _blk(Wd, a.tok)
C0 = {}
NF = {}
def noise_field():
    """AR(1)-correlated Gaussian noise over all latent frames (--noise-corr), unit variance per frame."""
    if "f" not in NF:
        gg = torch.Generator().manual_seed(a.seed * 7 + 3); h, w = b._chunk_shape[2:]; T = N * CS + 4; r = a.noise_corr
        e = torch.randn((16, T, h, w), generator=gg); f = e.clone()
        for t in range(1, T): f[:, t] = r * f[:, t - 1] + (1 - r * r) ** 0.5 * e[:, t]
        NF["f"] = f.to(dev0)
    return NF["f"]
def chunk_noise(ci, g):
    z = b.chunk_noise(ci, g)   # keeps the RNG stream aligned with the plain runs
    return noise_field()[:, ci * CS:(ci + 1) * CS].clone() if a.noise_corr > 0 else z
def fast_x0(z, i, ci):
    """Fast x0 at node i; with --histguide: amplify what the memory (KV cache) contributes."""
    if TX: TX["on"] = float(b._timesteps[i]) < TX["tmax"]
    if KVM:
        KVM["on"] = float(b._timesteps[i]) >= KVM["tmin"]
        if a.kv_low == "plain" and "mem" in KVM: KVM["set_mem"](b.target_model, KVM["mem"] if KVM["on"] else None)
    if HL:
        HL["on"] = bool(HL.get("hint")) and float(b._timesteps[i]) >= HL["tmin"]
        if HL.get("hint") and a.hint_low == "plain" and "y_hint" in HL:   # plain-Fast nodes see the original y, LoRA nodes the hint
            b._y_chunks[ci].copy_(HL["y_hint"] if HL["on"] else HL["y_old"])
    if a.histguide <= 0: return b.target_x0(z, i, ci)
    s = float(sig[i]); ff = b.target_flow(z, s, ci); kv = b._kv_size; b._kv_size = CS * b._frame_seqlen
    try: fs = b.target_flow(z, s, ci)
    finally: b._kv_size = kv
    return z - s * (ff + a.histguide * (ff - fs))
@torch.no_grad()
def fast_vel(zb, t_val, ci, hw0):
    """Fast velocity of chunk ci's state zb (Base res) at timestep t_val, returned at Base res."""
    zd = _down(zb, hw0).to(dev0); b.set_phase("target")
    return _down(b._call(b.target_model, "target", zd, t_val, ci), tuple(zb.shape[2:])).to(zb.device)
def draft_chunk(ci, g):
    if ci == 0 and a.chunk0_from:
        if "x" not in C0:
            z = b.chunk_noise(ci, g)   # keep the draft RNG stream aligned
            fr = np.stack([np.asarray(f) for f in imageio.get_reader(a.chunk0_from)][:4 * (CS - 1) + 1]).astype(np.float32) / 127.5 - 1
            v = torch.from_numpy(fr).permute(3, 0, 1, 2).to(dev0); hw = tuple(z.shape[2:]); H, W = hw[0] * 8, hw[1] * 8
            v = torch.nn.functional.interpolate(v.permute(1, 0, 2, 3), size=(H, W), mode="bicubic", align_corners=False).permute(1, 0, 2, 3).clamp(-1, 1)
            with torch.no_grad(): C0["x"] = vae.encode([v])[0].float()
            print("[chunk0] from %s -> latent %s" % (a.chunk0_from, tuple(C0["x"].shape)), flush=True)
        return C0["x"].clone(), None
    z = chunk_noise(ci, g); x0 = None; z0 = z.clone(); ZS[ci] = dict(z=[], x0=[]); BGN = {int(x) for x in a.bguide_nodes.split(",")} if a.bguide > 0 else set()
    if KVM and "update" in KVM: KVM["update"](ci)   # kv4: time-local plan memory for this chunk
    y_old = None
    if HL and ci >= a.start and (outs or a.plan_video):   # hint -> y channel of the chunk: the low-res plan's latents, or Base's one-step x0 for this chunk's noise
        th = time.time()
        if a.plan_video:
            if "lat" not in PLAN:
                import cv2 as _cv2
                fr = [np.asarray(f) for f in imageio.get_reader(a.plan_video)]; hw = b._chunk_shape[2:]
                if a.plan_hint_mode == "lo":
                    v = torch.from_numpy(np.stack(fr)).float().div(127.5).sub(1).permute(3, 0, 1, 2).to(dev0)
                    with torch.no_grad(): PLAN["lat"] = torch.nn.functional.interpolate(vae.encode([v])[0].float().permute(1, 0, 2, 3), size=tuple(hw), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
                else:
                    v = torch.from_numpy(np.stack([_cv2.resize(f, (hw[1] * 8, hw[0] * 8), interpolation=_cv2.INTER_CUBIC) for f in fr])).float().div(127.5).sub(1).permute(3, 0, 1, 2).to(dev0)
                    with torch.no_grad(): PLAN["lat"] = vae.encode([v])[0].float()
                print("[plan] %s -> latents %s" % (a.plan_video, tuple(PLAN["lat"].shape)), flush=True)
            hint = PLAN["lat"][:, ci * CS:(ci + 1) * CS]
        else:
            with DLOCK: hint = base_x0_at(torch.cat(outs, 1), z, 1.0)
        yy = b._y_chunks[ci]; y_old = yy.clone(); f0 = 1 if ci == 0 else 0   # chunk 0: keep the image latent (frame 0) conditioning, hint only the remaining latent frames
        yy[:4, f0:] = HL["mask"]; yy[4:, f0:] = hint[:, f0:].to(yy.dtype); HL["hint"] = True; HL["y_old"] = y_old; HL["y_hint"] = yy.clone(); VZ.setdefault(ci, {})["t_hint"] = round(time.time() - th, 2)
    for j, i in enumerate(NODES):
        ZS[ci]["z"].append(z.clone()); x0 = fast_x0(z, i, ci)
        if j in BGN and ci >= a.start and outs:   # Base-guided Fast: pull this node's x0 toward Base's x0 for the same state
            tb_ = time.time()
            with DLOCK: x0b = base_x0_at(torch.cat(outs, 1), z, float(sig[i]))
            x0 = x0 + a.bguide * (x0b - x0); VZ.setdefault(ci, {})["t_bguide"] = round(VZ.get(ci, {}).get("t_bguide", 0.0) + time.time() - tb_, 2)
        ZS[ci]["x0"].append(x0.clone())
        if j < len(NODES) - 1:
            s = float(sig[NODES[j + 1]]); z = (1 - s) * x0 + s * b.randn_like(x0, g)
            if a.specstep and ci >= a.start and outs and i in SPEC_NODES:   # speculative sampling of the transition z_k -> z_{k+1} (per latent position)
                ts_ = time.time()
                with DLOCK: x0T = base_x0_at(torch.cat(outs, 1), ZS[ci]["z"][-1], float(sig[i]))
                muD, muT = (1 - s) * x0, (1 - s) * x0T
                # exact speculative step (reflection maximal coupling, per latent position; both kernels are N(mu, s^2 I_16) and factorise over positions):
                # draft sample z = muD + s*eps is kept with prob min(1, N(z; muT)/N(z; muD)); otherwise it is replaced by the reflection
                # muT + s*(eps - 2 (e.eps) e), e = (muD - muT)/|muD - muT|, which makes the output exactly N(muT, s^2 I) (no rejection loop, no bias)
                eps = (z - muD) / s
                logr = -(((z - muT) ** 2).sum(0, keepdim=True) - ((z - muD) ** 2).sum(0, keepdim=True)) / (2 * s * s)   # log p_T/p_D per position [1,3,h,w]
                acc = torch.rand_like(logr) < logr.clamp(max=0).exp(); rej = ~acc
                if rej.any():
                    dlt = (muD - muT) / s; e = dlt / dlt.norm(dim=0, keepdim=True).clamp_min(1e-12)
                    y = muT + s * (eps - 2 * (e * eps).sum(0, keepdim=True) * e)
                    z = torch.where(rej.expand_as(z), y, z)
                VZ.setdefault(ci, {})["spec_dmean_%d" % i] = round(float(((muD - muT) / s).norm(dim=0).mean()), 4)   # mean per-position |muD - muT| / s
                if a.band_probe:   # frequency split of the gap: exact test restricted to the block-mean subspace accepts a whole unit w.p. 2*Phi(-|P gap|/2)
                    import math
                    def _l10acc(x):   # log10(2*Phi(-x/2)) = log10(erfc(x / (2*sqrt 2))), asymptotic form for large x
                        y = x / (2 * math.sqrt(2)); return math.log10(math.erfc(y)) if y < 20 else (-y * y - math.log(y * math.sqrt(math.pi))) / math.log(10)
                    gap_ = ((muD - muT) / s).float()   # [16, F, h, w]  (do not name this g: g is the sampler's RNG)
                    rec = dict(full=[float(gap_[:, f].norm()) for f in range(gap_.shape[1])] + [float(gap_.norm())])
                    for kk in (2, 4, 8, 16, 0):
                        x_ = gap_.permute(1, 0, 2, 3)   # [F, 16, h, w]
                        if kk == 0: sq = (x_.sum((2, 3)) ** 2 / (x_.shape[2] * x_.shape[3])).sum(1)   # whole-frame mean per channel
                        else:
                            sm = torch.nn.functional.avg_pool2d(x_, kk, stride=kk, ceil_mode=True, divisor_override=1); cn = torch.nn.functional.avg_pool2d(torch.ones_like(x_[:, :1]), kk, stride=kk, ceil_mode=True, divisor_override=1)
                            sq = (sm ** 2 / cn).sum((1, 2, 3))   # |P g|^2 per frame
                        rec["k%d" % kk] = [float(v.sqrt()) for v in sq] + [float(sq.sum().sqrt())]
                    VZ[ci].setdefault("band", {})[str(i)] = {kk: dict(norm=[round(v, 3) for v in vv], log10acc_frame=[round(_l10acc(v), 2) for v in vv[:-1]], log10acc_chunk=round(_l10acc(vv[-1]), 2)) for kk, vv in rec.items()}
                ar = acc.float().mean().item(); VZ.setdefault(ci, {})["spec_acc_%d" % i] = round(ar, 4); VZ[ci]["t_spec"] = round(VZ[ci].get("t_spec", 0.0) + time.time() - ts_, 2); TM["spec"] = TM.get("spec", 0.0) + time.time() - ts_
                if j == 0: print("[spec] chunk %d node %d: accept %.3f" % (ci, i, ar), end="", flush=True)
                else: print("  node %d: %.3f" % (i, ar), end="" if j < len(NODES) - 2 else "\n", flush=True)
    if TX: TX["on"] = False   # the commit pass is plain Fast
    if KVM:
        KVM["on"] = False
        if a.kv_low == "plain" and "mem" in KVM: KVM["set_mem"](b.target_model, None)   # commit pass without the memory
    if y_old is not None: b._y_chunks[ci].copy_(y_old); HL["on"] = False; HL["hint"] = False   # the commit pass must see the plain y
    return x0, z0
ZS = {}; PLAN = {}; SPEC_NODES = {int(x) for x in a.spec_nodes.split(",")}
@torch.no_grad()
def polish(x0, ci):
    """Base refinement of an accepted Fast chunk: re-noise to sigma S, denoise with the low-noise part of an N-step schedule (--polish S:N)."""
    S_, N_ = a.polish.split(":"); S_ = float(S_); gg = torch.Generator().manual_seed(a.seed * 23 + ci * 7 + 5)
    z = (1 - S_) * x0 + S_ * b.randn_like(x0, gg); t = time.time()
    with DLOCK: y = base_repaint_chunk(torch.cat(outs, 1), ci, a.seed, init=(z, S_), nsteps=int(N_))
    VZ.setdefault(ci, {})["t_polish"] = round(time.time() - t, 2); TM["polish"] = TM.get("polish", 0.0) + time.time() - t; return y
@torch.no_grad()
def seam(x0, ci):
    """A: joint low-noise re-denoise of [last latent of chunk ci-1, chunk ci] (--seam S:N); updates outs[-1] in place and returns the new chunk."""
    S_, N_ = a.seam.split(":"); S_ = float(S_); prev = outs[-1]; free = torch.cat([prev[:, -1:].to(dev0), x0], 1)
    gg = torch.Generator().manual_seed(a.seed * 29 + ci * 11 + 1); z = (1 - S_) * free + S_ * b.randn_like(free, gg); t = time.time()
    pre = torch.cat(outs[:-1] + [prev[:, :-1]], 1)
    with DLOCK: y = base_repaint_chunk(pre, ci, a.seed, n=4 / CS, init=(z, S_), nsteps=int(N_))
    outs[-1] = torch.cat([prev[:, :-1], y[:, :1].detach().float().cpu()], 1); VZ.setdefault(ci, {})["t_seam"] = round(time.time() - t, 2); TM["seam"] = TM.get("seam", 0.0) + time.time() - t
    return y[:, 1:]
def fd_init(ci, tries=0):
    """Base start state for --corr-from-draft: the draft's state entering node K (tries=0) or a fresh re-noising of the draft's x0 at that level (retries)."""
    if a.corr_from_draft < 0 or ci not in ZS: return None
    k = a.corr_from_draft; s = float(sig[NODES[k]])
    if tries == 0: return (ZS[ci]["z"][k], s)
    gg = torch.Generator().manual_seed(a.seed * 17 + ci * 131 + tries); x0 = ZS[ci]["x0"][k - 1] if k > 0 else ZS[ci]["x0"][0]
    return ((1 - s) * x0 + s * b.randn_like(x0, gg), s)
if a.kv_mem:   # plan memory: Base low expert on dev1 reads the plan latents, translator -> Fast K/V (kept for the whole rollout, all nodes and commit passes)
    from specwm.plan_kv import capture_base_kv, plan_rope, Translator, Translator2, set_mem, subsample_frames, window_tokens
    _st = KVM["st"]; _t = time.time(); _lo = torch.load(a.plan_lat)["lat"].to(dev1).float(); _c_lo = b1.prepare(img, prompt, act, frame_num=a.frames, max_area=int(_st.get("area_lo", 184320)))
    if tuple(_c_lo["lat"][1:]) != tuple(_lo.shape[2:]): _lo = torch.nn.functional.interpolate(_lo.permute(1, 0, 2, 3), size=tuple(_c_lo["lat"][1:]), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
    _KVb = capture_base_kv(expert("low"), _lo, _c_lo, float(_st["kv_t"]), list(_st["layers"]), dev1)
    _grid_lo = (_lo.shape[1], _lo.shape[2] // 2, _lo.shape[3] // 2); _hw = b._chunk_shape[2:]; _grid_hi = (_lo.shape[1], _hw[0] // 2, _hw[1] // 2); _stride = int(_st.get("mem_stride", 2)); _W = int(_st.get("mem_window", -1))
    _TR = Translator2(list(_st["layers"]), rank=int(_st.get("mem_rank", 256)), dev=dev0) if _st.get("translator") == "v2" else Translator(list(_st["layers"]), dev=dev0)
    _TR.load_state_dict({k: v.to(dev0) for k, v in _st.get("translator_sd", _st.get("translator")).items()}); _freqs = b.target_model.freqs.to(dev0)
    KVM["set_mem"] = set_mem
    if _W < 0:   # kv1: whole clip, frame stride, fixed for the rollout
        _KV = {l: (k.to(dev0), vv.to(dev0)) for l, (k, vv) in _KVb.items()}; _KV, _ = subsample_frames(_KV, _grid_lo, _stride); _mult = plan_rope(_freqs, _grid_lo, _grid_hi, frame_ids=list(range(0, _grid_lo[0], _stride)))
        with torch.no_grad(): KVM["mem"] = _TR(_KV, _mult)
        set_mem(b.target_model, KVM["mem"]); del _TR, _KV, _KVb
    else:   # kv4: time-local memory, recomputed per chunk (Base K/V of the whole plan stays on dev1)
        _P, _F = int(_st.get("mem_past", _W)), int(_st.get("mem_future", _W))
        def _update(ci, P=_P, F=_F, KVb=_KVb, TR=_TR, freqs=_freqs, glo=_grid_lo, ghi=_grid_hi):
            kvw, fids = window_tokens(KVb, glo, CS * (ci - P), CS * (ci + 1 + F), dev=dev0, stride=int(_st.get("mem_wstride", 1)))
            with torch.no_grad(): KVM["mem"] = TR(kvw, plan_rope(freqs, glo, ghi, frame_ids=fids))
            set_mem(b.target_model, KVM["mem"])
        KVM["update"] = _update; _update(0)
    del _lo; torch.cuda.empty_cache(); TM["kvmem"] = time.time() - _t; print("[kv-mem] plan memory attached (%s, window %d): %.1f s" % (a.kv_low, _W, TM["kvmem"]), flush=True)
carry = [0]; b.reset_caches(); g = torch.Generator().manual_seed(a.seed); outs = []; plan = []; log = []; t0 = time.time(); t_base = 0.0
CHECK = [int(x) for x in a.check_nodes.split(",")] if a.check_nodes else []
def spec_chunk(ci, g, mode):
    """draft with target evaluation at CHECK nodes. mode 'probe': log only. mode 'spec': per-token speculative acceptance of the draft's next-noise-level sample
    (Gaussian kernels: q = N((1-s')x0_draft, s'^2), p = N((1-s')x0_target, s'^2); accept token w.p. min(1, p/q); rejected tokens resample from p (approximation of the residual)).
    At the final node the draft step is deterministic, so if it is in CHECK the token takes the target x0 when rel. distance > --tau (biased)."""
    z = b.chunk_noise(ci, g); pref = torch.cat(outs, 1); rec = []
    for j, i in enumerate(NODES):
        x0 = b.target_x0(z, i, ci); s_k = float(sig[i])
        if j < len(NODES) - 1:
            s_n = float(sig[NODES[j + 1]]); eps = b.randn_like(x0, g); z_n = (1 - s_n) * x0 + s_n * eps
            if j in CHECK:
                x0t = base_x0_at(pref, z, s_k); delta = (x0 - x0t)
                lr = -(2 * s_n * (1 - s_n) * tok_sum(eps * delta) + (1 - s_n) ** 2 * tok_sum(delta * delta)) / (2 * s_n ** 2)   # log p/q per token
                acc = torch.rand_like(lr) < torch.exp(lr.clamp(max=0)); rate = float(acc.float().mean()); rel = float(delta.norm() / (x0.norm() + 1e-6))
                rec.append(dict(node=j, sigma=round(s_k, 3), rel=round(rel, 4), acc_rate=round(rate, 4), lr_mean=float(lr.mean()), lr_med=float(lr.median())))
                if mode == "spec":
                    m = tok_expand(acc, x0); z_n = torch.where(m, z_n, (1 - s_n) * x0t + s_n * b.randn_like(x0, g))
            z = z_n
        elif j in CHECK:
            x0t = base_x0_at(pref, z, s_k); delta = x0 - x0t; rel_tok = (tok_sum(delta * delta) / (tok_sum(x0t * x0t) + 1e-6)).sqrt(); acc = rel_tok < a.tau
            rec.append(dict(node=j, sigma=round(s_k, 3), rel=round(float(delta.norm() / (x0.norm() + 1e-6)), 4), acc_rate=round(float(acc.float().mean()), 4), tau=a.tau))
            if mode == "spec": x0 = torch.where(tok_expand(acc, x0), x0, x0t)
    return x0, rec
import threading
DLOCK = threading.Lock(); AS = dict(wait=0.0, rollback=0.0, n_rollback=0, t_draft=0.0)
def draft_keep(ci, g):
    """draft chunk ci; returns final x0 and the state at the check node NODES[1]: (z, x0_prev, x0_draft)."""
    z = b.chunk_noise(ci, g); x0 = None; st = None; x0p = None
    for j, i in enumerate(NODES):
        x0 = b.target_x0(z, i, ci)
        if j == a.check_at: st = (z.clone(), x0p.clone(), x0.clone())
        if j < len(NODES) - 1: s_n = float(sig[NODES[j + 1]]); z = (1 - s_n) * x0 + s_n * b.randn_like(x0, g); x0p = x0
    return x0, st
DS = []
def run_early(thr):
    """thr: fixed threshold, or 'q<frac>[:<init>]' = adaptive: reject if d > the <frac>-quantile of the distances seen so far in this rollout
    (the first check uses <init>); targets a rejection rate of about 1-<frac> in the corrected regime."""
    """send the draft's check-node state to the Base verifier (other GPU, thread) and keep drafting the remaining steps; commit only after the verdict."""
    global t_base
    for ci in range(N):
        d = None; reject = False; tq = time.time()
        if ci < a.start: x0 = draft_chunk(ci, g)[0]
        elif ci < a.start + a.warmup or carry[0] > 0:
            if ci >= a.start + a.warmup: carry[0] -= 1
            reject = True
        else:
            pref = torch.cat(outs, 1); z = b.chunk_noise(ci, g); x0p = None; res = {}; th = None
            for j, i in enumerate(NODES):
                x0 = b.target_x0(z, i, ci)
                if j == 1:
                    st = (z.clone(), x0p.clone(), x0.clone())
                    def work(st=st, pref=pref):
                        with DLOCK: tb = time.time(); res["d"] = base_check(pref, st[1], st[0], float(sig[NODES[1]]), st[2], a.chk)[0]; res["t"] = time.time() - tb
                    th = threading.Thread(target=work); th.start()
                if j < len(NODES) - 1: s_n = float(sig[NODES[j + 1]]); z = (1 - s_n) * x0 + s_n * b.randn_like(x0, g); x0p = x0
            tw = time.time(); th.join(); AS["wait"] += time.time() - tw; t_base += res["t"]; d = res["d"]
            if isinstance(thr, str):
                q, _, init = thr[1:].partition(":"); th_now = float(np.quantile(DS, float(q))) if len(DS) >= 2 else float(init or "inf"); DS.append(d); reject = d > th_now
            else: reject = d > thr
            if reject: carry[0] = a.run - 1
        if reject:
            with DLOCK: tb = time.time(); x0 = base_repaint_chunk(torch.cat(outs, 1), ci, a.seed); t_base += time.time() - tb
        b.commit(x0, ci); outs.append(x0.detach().float().cpu()); plan.append("C" if reject else "D"); log.append(dict(chunk=ci, rejected=reject, verify_dist=d, t_done=round(time.time() - t0, 2)))
        print("[chunk %d] %s%s  wait %.1fs" % (ci, "CORRECT" if reject else "draft", "" if d is None else " d=%.3f" % d, AS["wait"]), flush=True)
def run_window(k, thr):
    """draft commits optimistically; every k chunks the Base verifier checks the window (one forward, thread) while the draft continues.
    On a reject at chunk i: roll the draft back to i (reset + recommit accepted chunks), Base re-generates i..end of that window jointly (RePaint), draft resumes.
    At most one window is outstanding (the draft waits if it gets k chunks ahead of an unfinished check)."""
    global t_base
    states = {}; verified = {}; ci = 0; pend = []; job = None
    def finish(job):
        global t_base
        nonlocal ci, pend
        th, win, res = job; tw = time.time(); th.join(); AS["wait"] += time.time() - tw; t_base += res["t"]
        bad = [q for q, dd in enumerate(res["d"]) if dd > thr]
        for q, c in enumerate(win): log[c]["verify_dist"] = res["d"][q]
        if not bad:
            for c in win: verified[c] = time.time() - t0
            return False
        i0 = win[bad[0]]; end = win[-1]; tr = time.time()
        del outs[i0:]; del plan[i0:]; del log[i0:]; pend = []
        b.reset_caches()
        for c, x in enumerate(outs): b.commit(x.to(b.device), c)
        AS["rollback"] += time.time() - tr; AS["n_rollback"] += 1
        with DLOCK: tb = time.time(); xs = base_repaint_chunk(torch.cat(outs, 1), i0, a.seed, n=end - i0 + 1); t_base += time.time() - tb
        for q, c in enumerate(range(i0, end + 1)):
            x0 = xs[:, q * CS:(q + 1) * CS]; b.commit(x0, c); outs.append(x0.detach().float().cpu()); plan.append("C"); verified[c] = time.time() - t0
            log.append(dict(chunk=c, rejected=True, verify_dist=res["d"][win.index(c)] if c in win else None, t_done=round(time.time() - t0, 2)))
        print("[window %s] reject at chunk %d -> rollback + Base re-gen %d..%d" % (win, i0, i0, end), flush=True)
        ci = end + 1; return True
    while ci < N:
        if ci < a.start + a.warmup:
            if ci < a.start: x0 = draft_chunk(ci, g)[0]; r = False
            else:
                with DLOCK: tb = time.time(); x0 = base_repaint_chunk(torch.cat(outs, 1), ci, a.seed); t_base += time.time() - tb
                r = True
            b.commit(x0, ci); outs.append(x0.detach().float().cpu()); plan.append("C" if r else "D"); verified[ci] = time.time() - t0
            log.append(dict(chunk=ci, rejected=r, verify_dist=None, t_done=round(time.time() - t0, 2))); ci += 1; continue
        td = time.time(); x0, st = draft_keep(ci, g); b.commit(x0, ci); AS["t_draft"] += time.time() - td
        outs.append(x0.detach().float().cpu()); plan.append("D"); log.append(dict(chunk=ci, rejected=False, verify_dist=None, t_done=round(time.time() - t0, 2)))
        states[ci] = st; pend.append(ci); ci += 1
        if len(pend) == k or ci == N:
            if job is not None:
                job_, job = job, None
                if finish(job_): continue
            win = list(pend); pend = []; pref = torch.cat(outs[:win[0]], 1); sts = [states[c] for c in win]; res = {}
            def work(pref=pref, sts=sts, res=res):
                with DLOCK: tb = time.time(); res["d"] = base_check_window(pref, sts, float(sig[NODES[a.check_at]]), a.chk)[0]; res["t"] = time.time() - tb
            th = threading.Thread(target=work); th.start(); job = (th, win, res)
            if ci == N: job_, job = job, None; finish(job_)
    if job is not None: finish(job)
    for c, rec in enumerate(log): rec["t_verified"] = round(verified.get(c, time.time() - t0), 2)
def run_mc():
    """whiteboard loop: draft finishes chunk x -> re-noise x at several sigmas (mc-nodes) x mc-k samples -> target (Base, cached) and draft
    both predict x0 -> dL = mean[(|x0_T - x|^2 - |x0_D - x|^2) / |x|^2] (denoising-loss difference ~ log q/p up to scale) ->
    accept with prob a = min(1, exp(-beta*dL)) against u ~ U(0,1); reject -> target resamples the chunk (16-step RePaint)."""
    global t_base
    gu = torch.Generator().manual_seed(a.seed + 777); MN = [int(x) for x in a.mc_nodes.split(",")]
    for ci in range(N):
        d = None; reject = False; acc = None; uu = None
        if ci < a.start: x0 = draft_chunk(ci, g)[0]
        elif ci < a.start + a.warmup: reject = True
        else:
            x0 = draft_chunk(ci, g)[0]; pref = torch.cat(outs, 1); dls = []
            for jn in MN:
                s_ = float(sig[NODES[jn]])
                for _ in range(a.mc_k):
                    zz = (1 - s_) * x0 + s_ * torch.randn_like(x0)
                    xd = b.target_x0(zz, NODES[jn], ci).float()
                    with DLOCK:
                        tb = time.time()
                        if CHK_PRESETS[a.mc_chk][1] == -1: xt = kv_x0(pref, zz, s_, CHK_PRESETS[a.mc_chk][0]).to(dev0)
                        else:
                            _, _, _ = (None, None, None); gd, cx, ar = CHK_PRESETS[a.mc_chk]
                            j0 = max(0, pref.shape[1] - cx) if cx > 0 else 0; c = dict(base_cond(4 * (pref.shape[1] - j0 + CS - 1) + 1, j0, pref, ar)); hw = tuple(c["lat"][1:])
                            pp = _down(pref[:, j0:].to(dev1), hw); L = pp.shape[1]; zc = _down(zz.to(dev1), hw) if hw != tuple(zz.shape[2:]) else zz.to(dev1)
                            lat = torch.cat([(1 - s_) * pp + s_ * torch.randn_like(pp), zc], 1); fl = base_flow(lat, s_ * 1000.0, c, gd); xt = _down((lat - s_ * fl)[:, L:].float(), tuple(zz.shape[2:])).to(dev0)
                        torch.cuda.synchronize(dev1); t_base += time.time() - tb
                    xn = x0.float().pow(2).sum(); dls.append(float(((xt - x0).pow(2).sum() - (xd - x0.float()).pow(2).sum()) / xn))
            d = float(np.mean(dls)); acc = 1.0 if a.beta == 0 else float(min(1.0, np.exp(-a.beta * d))); uu = float(torch.rand(1, generator=gu)); reject = a.beta > 0 and uu > acc
        if reject:
            with DLOCK: tb = time.time(); x0 = base_repaint_chunk(torch.cat(outs, 1), ci, a.seed); t_base += time.time() - tb
        b.commit(x0, ci); outs.append(x0.detach().float().cpu()); plan.append("C" if reject else "D")
        log.append(dict(chunk=ci, rejected=reject, verify_dist=d, accept_prob=acc, u=uu, t_done=round(time.time() - t0, 2)))
        print("[chunk %d] %s%s" % (ci, "CORRECT" if reject else "draft", "" if d is None else " dL=%.4f a=%.3f u=%.3f" % (d, acc, uu)), flush=True)
SEM = {}
@torch.no_grad()
def sem_feat(frames):
    """frames: [3,T,H,W] in [-1,1] (dev0) -> mean DINOv2 ViT-B/14 CLS (unit norm)"""
    if "m" not in SEM:
        torch.hub.set_dir(S + "/.torch_hub"); SEM["m"] = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval().to(dev0)
        SEM["mean"] = torch.tensor([0.485, 0.456, 0.406], device=dev0).view(1, 3, 1, 1); SEM["std"] = torch.tensor([0.229, 0.224, 0.225], device=dev0).view(1, 3, 1, 1)
    x = ((frames.permute(1, 0, 2, 3).float() + 1) / 2).clamp(0, 1); x = torch.nn.functional.interpolate(x, size=(224, 392), mode="bilinear", antialias=True)
    f = torch.nn.functional.normalize(SEM["m"]((x - SEM["mean"]) / SEM["std"]).float(), dim=-1).mean(0); return torch.nn.functional.normalize(f, dim=0)
def viz_save(tag, ci, fr):
    """fr: [3,T,H,W] in [-1,1] -> <out>/viz/c<ci>_<tag>_<k>.jpg (every 2nd frame, 256 px wide)"""
    if not a.save_drafts: return
    import cv2
    os.makedirs(os.path.join(a.out, "viz"), exist_ok=True)
    v = ((fr.permute(1, 2, 3, 0).float().cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    for k in range(0, v.shape[0], 2):
        im = cv2.resize(v[k], (256, int(256 * v.shape[1] / v.shape[2]))); cv2.imwrite(os.path.join(a.out, "viz", "c%02d_%s_%d.jpg" % (ci, tag, k)), cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
VZ = {}
GBL = torch.Generator().manual_seed(a.seed * 7919 + 17) if a.corr_blend else None   # separate RNG stream for warm-up drafts used only by --corr-blend
def blend_corr(xb, xd):
    """Base-corrected chunk xb vs Fast's draft xd, [16,F,h,w]: soft:A or low:K (see --corr-blend)."""
    mode, _, val = a.corr_blend.partition(":"); xd = xd.to(xb.device, xb.dtype)
    if mode == "soft": w = float(val); return w * xb + (1 - w) * xd
    if mode == "low":
        k = int(val); df = (xb - xd).float().permute(1, 0, 2, 3)   # [F,16,h,w]
        lo = torch.nn.functional.interpolate(torch.nn.functional.avg_pool2d(df, k, stride=k, ceil_mode=True), size=df.shape[2:], mode="bilinear", align_corners=False)
        return xd + lo.permute(1, 0, 2, 3).to(xb.dtype)
    raise ValueError(a.corr_blend)
def pq_score(x0, ci, pref):
    """d = err_Base - err_Fast for chunk ci: both models re-denoise the same partly noised chunk (sigma of Fast node --pq-node) in one step;
    Fast sees its committed prefix through its KV cache, Base the last --pq-ctx prefix latents re-noised to the same sigma (no CFG)."""
    s = float(sig[a.pq_node]); gg = torch.Generator().manual_seed(a.seed * 1009 + ci * 13 + 7)
    eps = torch.randn(x0.shape, generator=gg).to(x0.device, torch.float32); x0f = x0.float(); z = (1 - s) * x0f + s * eps
    with torch.no_grad(): xf = b.target_x0(z, a.pq_node, ci).float()
    eF = ((xf - x0f) ** 2).mean(dim=(0, 2, 3))
    j = max(0, pref.shape[1] - a.pq_ctx); c = base_cond(4 * (pref.shape[1] - j + CS - 1) + 1, j, pref); pp = pref[:, j:].to(dev1).float(); L = pp.shape[1]
    ge = torch.Generator().manual_seed(a.seed * 2003 + ci * 17 + 3); pn = torch.randn(pp.shape, generator=ge).to(dev1)
    lat = torch.cat([(1 - s) * pp + s * pn, z.to(dev1)], 1)
    with DLOCK: tb = time.time(); flow = base_flow(lat, s * 1000.0, c, 1.0); TM["pq_base"] = TM.get("pq_base", 0.0) + time.time() - tb
    xb = (lat - s * flow)[:, L:].float().to(x0.device); eB = ((xb - x0f) ** 2).mean(dim=(0, 2, 3))
    vd = float(((xb - xf) ** 2).mean() / (s * s))   # |v_Base - v_Fast|^2 on the same input (x0_hat = z - s v): the whiteboard's score difference, no ground truth needed
    PQV.setdefault(ci, []).append(round(vd, 5))
    return float((eB - eF).mean()), [round(float(v), 5) for v in eB], [round(float(v), 5) for v in eF]
PQV = {}
def low_blend(xb, xd, k=4):
    df = (xb - xd.to(xb.device, xb.dtype)).float().permute(1, 0, 2, 3)
    lo = torch.nn.functional.interpolate(torch.nn.functional.avg_pool2d(df, k, stride=k, ceil_mode=True), size=df.shape[2:], mode="bilinear", align_corners=False)
    return xd.to(xb.device, xb.dtype) + lo.permute(1, 0, 2, 3).to(xb.dtype)
CRITIC = {}
def chunk_score(x0, ci, pref):
    """verifier for the pq policy: p/q (Base) or the latent drift critic (no Base): returns (score, errB-or-None, errF-or-None); lower = better"""
    if not a.pq_verifier.startswith("critic:"): return pq_score(x0, ci, pref)
    if "sc" not in CRITIC:
        from specwm.critic_model import CriticScorer; CRITIC["sc"] = CriticScorer(a.pq_verifier.split(":", 1)[1], device=str(dev0)); print("[critic] loaded", a.pq_verifier, flush=True)
    prev = pref[:, -CS:].to(dev0) if pref is not None and pref.shape[1] >= CS else None
    with torch.no_grad(): l, _ = CRITIC["sc"](x0.float(), prev, b._emb_chunks[ci].squeeze(0))
    return float(l), None, None
def run_pq():
    """draft -> verify (p/q one-step re-denoising) -> on reject Fast redraws with fresh noise (smooth: no Base frames spliced in); --pq-teacher: calibration."""
    global t_base
    TL = None
    if a.pq_teacher:
        fr = np.stack([np.asarray(f) for f in imageio.get_reader(a.pq_teacher)][:4 * (N * CS - 1) + 1]).astype(np.float32) / 127.5 - 1
        hw = tuple(b._chunk_shape[2:]); v = torch.from_numpy(fr).permute(3, 0, 1, 2)
        if v.shape[2:] != (hw[0] * 8, hw[1] * 8): v = torch.nn.functional.interpolate(v.permute(1, 0, 2, 3), size=(hw[0] * 8, hw[1] * 8), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
        with torch.no_grad(): TL = vae.encode([v.to(dev0)])[0].float()
        print("[pq] teacher %s -> latents %s" % (a.pq_teacher, tuple(TL.shape)), flush=True)
    for ci in range(N):
        d = None; tries = 0; kind = "D"; cands = []
        if TL is not None:
            x0 = TL[:, ci * CS:(ci + 1) * CS].to(dev0)
            if ci >= 1: d, eB, eF = chunk_score(x0, ci, torch.cat(outs, 1)); cands = [(d, eB, eF)]
        elif ci < max(1, a.start): x0 = draft_chunk(ci, g)[0]
        else:
            pref = torch.cat(outs, 1)
            for r in range(a.pq_retries + 1):
                gr = g if r == 0 else torch.Generator().manual_seed(a.seed * 7919 + ci * 101 + r)
                xr = draft_chunk(ci, gr)[0]; dr, eB, eF = chunk_score(xr, ci, pref); cands.append((dr, eB, eF, xr))
                if dr <= a.pq_tau: break
            best = min(cands, key=lambda c: c[0]); d, x0 = best[0], best[3]; tries = len(cands) - 1
            kind = "D" if tries == 0 else "R"
            if d > a.pq_tau:
                kind = "F"
                if a.pq_fallback in ("base", "base_low4"):
                    with DLOCK: tb = time.time(); xb = base_repaint_chunk(pref, ci, a.seed); t_base += time.time() - tb
                    x0 = xb if a.pq_fallback == "base" else low_blend(xb, x0, 4); kind = "B"
        b.commit(x0, ci); outs.append(x0.detach().float().cpu()); plan.append(kind)
        log.append(dict(chunk=ci, pq_d=None if d is None else round(d, 5), pq_cands=[round(c[0], 5) for c in cands], pq_errB=cands[0][1] if cands else None, pq_errF=cands[0][2] if cands else None, tries=tries, kind=kind, pq_vdiff=PQV.get(ci), **VZ.get(ci, {})))
        print("[chunk %d] %s d=%s tries %d" % (ci, kind, "-" if d is None else "%.4f" % d, tries), flush=True)
def run_sem(thr):
    """semantic gate (no Base forward for the check): the draft finishes chunk ci -> decode it alone -> DINO feature ->
    d = 1 - max cos(feature, bank) with bank = {start frame} (+ Base-corrected chunks if --sem-bank) -> d > thr: Base 16-step RePaint of the chunk."""
    global t_base
    from PIL import Image
    import cv2
    def sv(fr_uint8):   # mean HSV saturation and value over frames [T,H,W,3] uint8
        hs = [cv2.cvtColor(f, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32) for f in fr_uint8]; return np.array([np.mean([h[:, 1].mean() for h in hs]), np.mean([h[:, 2].mean() for h in hs])])
    im0u = np.asarray(Image.open(img).convert("RGB")); im0 = im0u.astype(np.float32) / 127.5 - 1; bank = [] if pol == "ir" else [sem_feat(torch.from_numpy(im0).permute(2, 0, 1)[:, None].to(dev0))]; sv0 = sv([im0u]); SEMCI = [0]
    SVREF = []
    if pol == "ir":
        import ImageReward as RM
        IRM = RM.load("ImageReward-v1.0", device=dev0); IRM.eval(); IRM.requires_grad_(False)
        irp = a.ir_prompt if not os.path.isfile(a.ir_prompt) else open(a.ir_prompt).read().strip(); assert irp, "--ir-prompt needed"; print("[ir] prompt: %s" % irp, flush=True)
    def score(fr):
        if pol == "ir":   # SDVG worst-frame router: negated min ImageReward so that 'd > thr' <=> min reward < tau
            fu = ((fr.permute(1, 2, 3, 0).float() + 1) * 127.5).clamp(0, 255).byte().cpu().numpy()
            with torch.no_grad(): _, rw = IRM.inference_rank(irp, [Image.fromarray(x) for x in fu])
            VZ.setdefault(SEMCI[0], {}).update(ir_min=round(float(min(rw)), 4), ir_avg=round(float(np.mean(rw)), 4)); return -float(min(rw))
        f = sem_feat(fr); dd = float(1 - max(float(f @ bb) for bb in bank))
        if a.sem_colour > 0:
            fu = ((fr.permute(1, 2, 3, 0).float() + 1) * 127.5).clamp(0, 255).byte().cpu().numpy(); ref = SVREF[-1] if (a.sem_ref == "recent" and SVREF) else sv0
            dd += a.sem_colour * float(np.abs(sv(fu) - ref).sum()) / 80
        return dd
    P_ = np.load(os.path.join(act, "poses.npy")); fwd_ = P_[:, :3, 2] / np.linalg.norm(P_[:, :3, 2], axis=1, keepdims=True)
    YAW = [float(np.degrees(np.arctan2(fwd_[min(12 * c + 8, len(P_) - 1), 0], fwd_[min(12 * c + 8, len(P_) - 1), 2]))) for c in range(N)]
    ANCH = [0]; angd = lambda x, y: abs((x - y + 180) % 360 - 180)
    for ci in range(N):
        d = None; reject = False; SEMCI[0] = ci
        if ci < a.start: x0 = draft_chunk(ci, g)[0]
        elif ci < a.start + a.warmup:
            reject = True
            if a.corr_from_draft >= 0: td0 = time.time(); draft_chunk(ci, g); VZ.setdefault(ci, {})["t_draft"] = round(time.time() - td0, 2)
        elif a.act_theta > 0 and angd(YAW[ci], YAW[ANCH[0]]) >= a.act_theta:
            reject = True; VZ.setdefault(ci, {})["forced"] = round(angd(YAW[ci], YAW[ANCH[0]]), 1)   # action says: new viewing sector -> Base anchor
            if a.corr_from_draft >= 0: draft_chunk(ci, g)
        else:
            td0 = time.time(); x0 = draft_chunk(ci, g)[0]; t_dr = time.time() - td0; tb = time.time()
            with torch.no_grad(): fr = vae.decode([x0.to(dev0)])[0].clamp(-1, 1)
            viz_save("draft", ci, fr); VZ.setdefault(ci, {})["t_draft"] = round(t_dr, 2)
            d = score(fr); reject = d > thr; AS["t_sem"] = AS.get("t_sem", 0.0) + time.time() - tb
        if reject:
            ANCH[0] = ci
            if a.corr_blend:   # keep Fast's draft of this chunk for the blend
                if d is None: x0_draft = draft_chunk(ci, GBL)[0]
                else: x0_draft = x0.clone()
            with DLOCK: tb = time.time(); x0 = base_repaint_chunk(torch.cat(outs, 1), ci, a.seed, init=fd_init(ci)); t_base += time.time() - tb
            VZ.setdefault(ci, {})["t_base"] = round(time.time() - tb, 2)
            if a.save_drafts and not (ci >= a.start + a.warmup and a.sem_retry > 0):
                with torch.no_grad(): viz_save("base0", ci, vae.decode([x0.to(dev0)])[0].clamp(-1, 1))
            if ci >= a.start + a.warmup and a.sem_retry > 0:
                with torch.no_grad(): frc0 = vae.decode([x0.to(dev0)])[0].clamp(-1, 1)
                dc = score(frc0); viz_save("base0", ci, frc0)
                cands = [(dc, x0)]; tries = 0
                while dc > thr and tries < a.sem_retry:
                    tries += 1
                    with DLOCK: tb = time.time(); xr = base_repaint_chunk(torch.cat(outs, 1), ci, a.seed + 100003 * tries, init=fd_init(ci, tries)); t_base += time.time() - tb
                    with torch.no_grad(): frr = vae.decode([xr.to(dev0)])[0].clamp(-1, 1)
                    dc = score(frr); viz_save("base%d" % tries, ci, frr)
                    cands.append((dc, xr))
                dbest, x0 = min(cands, key=lambda c: c[0]); AS["retries"] = AS.get("retries", 0) + tries; AS.setdefault("retry_scores", {})[ci] = [round(c[0], 4) for c in cands]
                print("[chunk %d] corrected-chunk score %s -> kept %.3f" % (ci, [round(c[0], 3) for c in cands], dbest), flush=True)
            if a.corr_blend:
                x0 = blend_corr(x0, x0_draft); VZ.setdefault(ci, {})["blend"] = a.corr_blend
            if a.sem_bank or a.sem_ref == "recent":
                with torch.no_grad(): frc = vae.decode([x0.to(dev0)])[0].clamp(-1, 1)
                if a.sem_bank and pol != "ir": bank.append(sem_feat(frc))
                if a.sem_ref == "recent": SVREF.append(sv(((frc.permute(1, 2, 3, 0).float() + 1) * 127.5).clamp(0, 255).byte().cpu().numpy()))
        if a.save_drafts and ci < a.start:
            with torch.no_grad(): viz_save("draft", ci, vae.decode([x0.to(dev0)])[0].clamp(-1, 1))
        if a.polish and not reject and ci >= a.start: x0 = polish(x0, ci)
        if a.seam and ci >= 1: x0 = seam(x0, ci); b.commit(outs[-1].to(dev0), ci - 1)   # re-commit the modified last latent of chunk ci-1 before committing ci
        b.commit(x0, ci); outs.append(x0.detach().float().cpu()); plan.append("C" if reject else "D")
        log.append(dict(chunk=ci, rejected=reject, verify_dist=d, corrected_scores=AS.get("retry_scores", {}).get(ci), t_done=round(time.time() - t0, 2), **VZ.get(ci, {})))
        print("[chunk %d] %s%s" % (ci, "CORRECT" if reject else "draft", "" if d is None else " dsem=%.3f" % d), flush=True)
if pol == "sem": run_sem(float(parg))
if pol == "pq": run_pq()
if pol == "ir": run_sem(-float(parg))   # SDVG router: reject if min ImageReward < tau
if pol == "mc": run_mc()
if pol == "early": run_early(parg if parg.startswith("q") else float(parg))
if pol == "window": run_window(int(parg.split(":")[0]), float(parg.split(":")[1]))
for ci in range(0 if pol in ("early", "window", "mc", "sem", "ir", "pq") else N):
    reject = False; d = None; rec = None
    if pol in ("probe", "spec") and ci >= a.start:
        tb = time.time(); x0, rec = spec_chunk(ci, g, pol); t_base += time.time() - tb; reject = pol == "spec" and any(r["acc_rate"] < 1 for r in rec)
    elif pol == "verify_node" and ci >= a.start:
        k, thr = int(parg.split(":")[0]), float(parg.split(":")[1]); z = b.chunk_noise(ci, g); pref = torch.cat(outs, 1)
        if ci < a.start + a.warmup: reject = True; x0 = b.target_x0(z, NODES[0], ci)   # warm-up: always correct the first chunks
        elif carry[0] > 0: carry[0] -= 1; reject = True; x0 = b.target_x0(z, NODES[0], ci)   # inside a correction run: skip the check
        else:
          x0p = None
          for j, i in enumerate(NODES):
            x0 = b.target_x0(z, i, ci)
            if j == k:
                tb = time.time(); d, _, _ = base_check(pref, x0p, z, float(sig[i]), x0, a.chk); t_base += time.time() - tb; reject = d > thr
                if reject: carry[0] = a.run - 1
            if reject: break
            if j < len(NODES) - 1: s_n = float(sig[NODES[j + 1]]); z = (1 - s_n) * x0 + s_n * b.randn_like(x0, g); x0p = x0
    elif pol == "probe2" and ci >= a.start:
        k = int(parg or 1); z = b.chunk_noise(ci, g); pref = torch.cat(outs, 1); x0p = None; rec = []
        for j, i in enumerate(NODES):
            x0 = b.target_x0(z, i, ci)
            if j == k:
                tb = time.time()
                for nm in a.chk_list.split(","):
                    dd, tc, tf = base_check(pref, x0p, z, float(sig[i]), x0, nm); rec.append(dict(preset=nm, rel=round(dd, 4), t_cond=round(tc, 2), t_fwd=round(tf, 2)))
                t_base += time.time() - tb
            if j < len(NODES) - 1: s_n = float(sig[NODES[j + 1]]); z = (1 - s_n) * x0 + s_n * b.randn_like(x0, g); x0p = x0
        print("[chunk %d] probe2 %s" % (ci, " | ".join("%s %.3f (%.1f+%.1fs)" % (r["preset"], r["rel"], r["t_cond"], r["t_fwd"]) for r in rec)), flush=True)
    else: x0, z0 = draft_chunk(ci, g)
    if pol in ("always", "every", "verify", "sched") and ci >= a.start:
        if pol == "always": reject = True
        elif pol == "sched": reject = ci in {int(x) for x in parg.split(",")}   # fixed list of chunks corrected by Base
        elif pol == "every": reject = ((ci - a.start) % int(parg) == 0)
        elif pol == "verify":
            tb = time.time(); xt = base_x0_at_sigma1(torch.cat(outs, 1), z0); t_base += time.time() - tb
            d = float((x0 - xt).norm() / (x0.norm() + 1e-6)); reject = d > float(parg)
    if reject and pol != "spec":
        tb = time.time(); x0 = base_repaint_chunk(torch.cat(outs, 1), ci, a.seed); t_base += time.time() - tb
    if a.polish and not reject and ci >= a.start: x0 = polish(x0, ci)
    if a.seam and ci >= 1: x0 = seam(x0, ci); b.commit(outs[-1].to(dev0), ci - 1)
    b.commit(x0, ci); outs.append(x0.detach().float().cpu()); plan.append("C" if reject else "D"); log.append(dict(chunk=ci, rejected=reject, verify_dist=d, nodes=rec, **VZ.get(ci, {})))
    if rec and pol != "probe2": print("[chunk %d] probe %s" % (ci, " | ".join("n%d s%.2f rel %.3f acc %.3f" % (r["node"], r["sigma"], r["rel"], r["acc_rate"]) for r in rec)), flush=True)
    print("[chunk %d] %s%s  mem1 %.1f GB (peak %.1f) mem0 peak %.1f swaps %d" % (ci, "CORRECT" if reject else "draft", "" if d is None else " d=%.3f" % d, torch.cuda.memory_allocated(dev1) / 2**30, torch.cuda.max_memory_allocated(dev1) / 2**30, torch.cuda.max_memory_allocated(dev0) / 2**30, n_swap[0]), flush=True)
gen_s = time.time() - t0; lat = torch.cat(outs, 1)
with torch.no_grad(): im = vae.decode([lat.to(dev0)])[0].detach().float().cpu()
v = ((im.clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy(); v = (v * 255).astype("uint8")
np.save(os.path.join(a.out, "frames_all.npy"), v[::4]); imageio.mimwrite(os.path.join(a.out, "video.mp4"), list(v), fps=16, quality=8)
json.dump(dict(task=a.task, seed=a.seed, policy=a.policy, plan="".join(plan), corrected=plan.count("C"), gen_s=round(gen_s, 1), base_s=round(t_base, 1), swaps=n_swap[0], peak_mem1_gb=round(torch.cuda.max_memory_allocated(dev1) / 2**30, 1), steps=a.steps, max_area=a.max_area, guide=a.guide, ctx=a.ctx, run=a.run, warmup=a.warmup, chk=a.chk, check_at=a.check_at, chunk0_from=a.chunk0_from, corr_mode=a.corr_mode, end_pad=a.end_pad, sem_bank=a.sem_bank, sem_colour=a.sem_colour, sem_retry=a.sem_retry, sem_ref=a.sem_ref, save_drafts=a.save_drafts, base_lora=a.base_lora, base_sampler=a.base_sampler, corr_kv=a.corr_kv, kv_rp_win=a.kv_rp_win, kv_rp_tlow=a.kv_rp_tlow, handoff=a.handoff, corr_mixed=a.corr_mixed, mixed_lora=a.mixed_lora, act_theta=a.act_theta, autoguide=a.autoguide, histguide=a.histguide, corr_from_draft=a.corr_from_draft, polish=a.polish, noise_corr=a.noise_corr, seam=a.seam, bguide=a.bguide, fast_win=a.fast_win, bguide_nodes=a.bguide_nodes, bguide_cfg=a.bguide_cfg, fast_lora=a.fast_lora, cfg_steps=a.cfg_steps, flow_reuse=a.flow_reuse, local_attn=18, sink=6, retries=AS.get("retries", 0), t_sem=round(AS.get("t_sem", 0.0), 1), kv_win=a.kv_win, mc_k=a.mc_k, mc_nodes=a.mc_nodes, beta=a.beta, mc_chk=a.mc_chk, corr_area=a.corr_area, nodes=a.nodes, t_wait=round(AS["wait"], 1), t_rollback=round(AS["rollback"], 1), n_rollback=AS["n_rollback"], t_denoise=round(TM["denoise"], 1), t_swap=round(TM["swap"], 1), t_cond=round(TM["cond"], 1), log=log), open(os.path.join(a.out, "spec_v0.json"), "w"), indent=1)
print("[SPEC_V0_DONE] %s s%d %s plan=%s corrected=%d gen %.0fs (base %.0fs: denoise %.0f swap %.0f cond %.0f)" % (a.task, a.seed, a.policy, "".join(plan), plan.count("C"), gen_s, t_base, TM["denoise"], TM["swap"], TM["cond"]), flush=True)
