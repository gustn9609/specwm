"""Official LingBot-World bidirectional generation (lingbot-world/generate.py, WanI2V A14B MoE: UniPC 70 steps, guide 5, shift 10) with the
v2 bidirectional checkpoint in v1 layout (lingbot-world-v2/bid_as_v1_cam). flash-attn is not installed, so attention falls back to SDPA
(specwm/sdpa_patch). Times WanI2V.generate (prompt encoding + 70 sampling steps + VAE decode; model loading excluded) into <save_file>.json.
usage: CUDA_VISIBLE_DEVICES=g python lingbot-world-v2/specwm/run_bid_official.py <generate.py args incl. --save_file X.mp4>"""
import os, sys, time, json, importlib.util
V1 = "/home/hk28965/specwm/lingbot-world"; V2 = "/home/hk28965/specwm/lingbot-world-v2"
sys.path.insert(0, V1); os.chdir(V1)
spec = importlib.util.spec_from_file_location("sdpa_patch", V2 + "/specwm/sdpa_patch.py"); sp = importlib.util.module_from_spec(spec); spec.loader.exec_module(sp)
import wan.modules.attention, wan.modules.model   # noqa: F401  (load the v1 modules first so the patch rebinds them)
sp.apply()
import wan
T = {}
_gen = wan.WanI2V.generate
def timed(self, *a, **k):
    k.setdefault("vis_ui", False)   # no WASD/IJKL overlay on the saved video
    import torch; torch.cuda.synchronize(); t = time.time(); r = _gen(self, *a, **k); torch.cuda.synchronize(); T["generate_s"] = round(time.time() - t, 1); return r
wan.WanI2V.generate = timed
sys.argv = [V1 + "/generate.py"] + sys.argv[1:]
spec = importlib.util.spec_from_file_location("v1generate", V1 + "/generate.py"); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
t0 = time.time(); args = g._parse_args(); g.generate(args); T["process_s"] = round(time.time() - t0, 1)
T.update(dict(ckpt=args.ckpt_dir, steps=args.sample_steps, shift=args.sample_shift, guide=args.sample_guide_scale, seed=args.base_seed, size=args.size, frames=args.frame_num, image=args.image, action=args.action_path))
json.dump(T, open(os.path.splitext(args.save_file)[0] + ".json", "w"), indent=1); print("[BID_DONE]", T, flush=True)
