"""Official LingBot-World v2 causal generation (generate.py; optional Ulysses sequence parallel under torchrun) with the generate call timed.
Writes <save_file>.json on rank 0: generate_s (encoding + all chunks + decode; model loading excluded), world size.
usage: [torchrun --nproc_per_node 2] python specwm/run_v2_official.py <generate.py args> --save_file X.mp4"""
import os, sys, time, json, importlib.util
V2 = "/home/hk28965/specwm/lingbot-world-v2"; sys.path.insert(0, V2); os.chdir(V2); os.environ.setdefault("SPECWM_NO_FLASH", "1")
import wan, torch
T = {}
for cls in [getattr(wan, n) for n in dir(wan) if n.startswith("WanI2V")]:
    _g = cls.generate
    def timed(self, *a, _g=_g, **k):
        torch.cuda.synchronize(); t = time.time(); r = _g(self, *a, **k); torch.cuda.synchronize(); T["generate_s"] = round(time.time() - t, 2); return r
    cls.generate = timed
spec = importlib.util.spec_from_file_location("v2generate", V2 + "/generate.py"); g = importlib.util.module_from_spec(spec); sys.argv = [V2 + "/generate.py"] + sys.argv[1:]; spec.loader.exec_module(g)
args = g._parse_args(); t0 = time.time(); g.generate(args)
if int(os.environ.get("RANK", "0")) == 0:
    T.update(process_s=round(time.time() - t0, 1), world=int(os.environ.get("WORLD_SIZE", "1")), ulysses=args.ulysses_size, infer_mode=args.infer_mode, frames=args.frame_num, size=args.size)
    json.dump(T, open(os.path.splitext(args.save_file)[0] + ".json", "w"), indent=1); print("[V2OFF_DONE]", T, flush=True)
