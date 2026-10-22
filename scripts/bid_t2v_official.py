"""Bidirectional Wan2.1 T2V with the OFFICIAL Wan2.1 code (github Wan-Video/Wan2.1, cloned to specwm/wan21_official), same prompts/seed/settings
as archive/bid_t2v.py (SDVG repo's modified wan package: bf16 residual/modulation/RoPE). Only patch: flash_attention -> the official attention()
SDPA fallback (flash-attn is not installed on these Blackwell GPUs; same math for batch 1).
usage: CUDA_VISIBLE_DEVICES=g archive/.venv/bin/python scripts/bid_t2v_official.py --size 14B --names jar,castle [--width 832 --height 480] [--shift 5 --guide 5]"""
import argparse, os, sys, json, time, torch, imageio
S = "/home/hk28965/specwm"; sys.path.insert(0, S + "/wan21_official")
ap = argparse.ArgumentParser(); ap.add_argument("--size", default="14B"); ap.add_argument("--names", default="jar,castle,tokyo_street,bigsur_drone"); ap.add_argument("--frames", type=int, default=81)
ap.add_argument("--steps", type=int, default=50); ap.add_argument("--guide", type=float, default=5.0); ap.add_argument("--shift", type=float, default=5.0); ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--width", type=int, default=832); ap.add_argument("--height", type=int, default=480); ap.add_argument("--tag", default="off"); a = ap.parse_args()
import wan, wan.modules.model as M, wan.modules.attention as A
from wan.configs import WAN_CONFIGS
M.flash_attention = A.attention
cfg = WAN_CONFIGS["t2v-1.3B" if a.size == "1.3B" else "t2v-14B"]
m = wan.WanT2V(config=cfg, checkpoint_dir=S + "/archive/wan_models/Wan2.1-T2V-" + a.size, device_id=0, rank=0, t5_cpu=False)
O = S + "/lingbot-world-v2/specwm_out/bid_t2v"
for name in a.names.split(","):
    prompt = json.load(open("%s/%s/bid%s_s1234/bid.json" % (O, name, a.size.replace(".", "p"))))["prompt"]
    d = "%s/%s/bid%s%s_s%d" % (O, name, a.size.replace(".", "p"), a.tag, a.seed); os.makedirs(d, exist_ok=True)
    if os.path.exists(d + "/video.mp4"): print("[skip]", d); continue
    torch.cuda.synchronize(); t0 = time.time()
    v = m.generate(prompt, size=(a.width, a.height), frame_num=a.frames, shift=a.shift, sample_solver="unipc", sampling_steps=a.steps, guide_scale=a.guide, seed=a.seed, offload_model=False)
    torch.cuda.synchronize(); dt = time.time() - t0
    fr = ((v.clamp(-1, 1) + 1) * 127.5).permute(1, 2, 3, 0).byte().cpu().numpy(); imageio.mimwrite(d + "/video.mp4", list(fr), fps=16, quality=8)
    json.dump(dict(code="official Wan2.1 9737cba", prompt=prompt, size=a.size, frames=a.frames, steps=a.steps, guide=a.guide, shift=a.shift, seed=a.seed, res="%dx%d" % (a.width, a.height), gen_s=round(dt, 1), fps=16), open(d + "/bid.json", "w"), indent=1)
    print("[BIDOFF_DONE] %s %s %.0f s" % (name, a.size, dt), flush=True)
