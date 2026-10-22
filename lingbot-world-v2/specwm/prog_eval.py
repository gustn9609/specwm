"""[10/22] Solo rollouts of a prog_dmd14 student (or the stock model with --student '') for a task json, in the device config, with per-chunk timing.
Writes <out>/<task_id>/{lat.pt, timing.json, video.mp4} in the layout baselines/ours_to_timing.py expects for a solo arm, so the frozen scorers apply.
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.prog_eval --student specwm_out/prog_dmd14/S0/student_last.pt --tasks /home/hk28965/specwm/baselines/tasks/mg2_long.json --out /home/hk28965/specwm/baselines/raw/ours_prog_mg2_long [--window 12 --prog 1 --frames 969]"""
import argparse, os, sys, json, time, numpy as np, torch
S = os.environ.get("SPECWM_ROOT", "/home/hk28965/specwm")
ap = argparse.ArgumentParser(); ap.add_argument("--student", default=""); ap.add_argument("--tasks", required=True); ap.add_argument("--out", required=True); ap.add_argument("--ids", default="", help="comma list of task ids (default all)")
ap.add_argument("--prog", type=int, default=1); ap.add_argument("--window", type=int, default=12); ap.add_argument("--frames", type=int, default=0, help="0 = the task's frames"); ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--nodes", default="0,500"); ap.add_argument("--wq", default="fp8", choices=["bf16", "fp8"]); ap.add_argument("--decode", type=int, default=1)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
os.environ.setdefault("SPECWM_NO_FLASH", "1"); sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
from specwm import prog_student as PS
dev = "cuda:0"; NODES = tuple(int(v) for v in a.nodes.split(","))
b = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=a.window, swap_models=False, build_draft=False, timesteps_index=NODES)
gates = None; info = {}
if a.student: ck = torch.load(a.student, map_location="cpu"); gates, info = PS.load(b.target_model, ck, torch.device(dev)); a.prog = int(ck.get("prog", a.prog)); print("[prog_eval] student", a.student, info, "prog", a.prog, flush=True)
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True)
if a.wq == "fp8": FP.fp8_swap(b.target_model, True, keep=False)
b.pipe.text_encoder.model.to(dev); _te = b.pipe.text_encoder
class _TE:
    def __init__(s, te): s.te = te; s.model = te.model
    def __call__(s, texts, device): return s.te(texts, torch.device(dev))
b.pipe.text_encoder = _TE(_te); sampler = PS.ProgSampler(b, gates, prog=a.prog)
tasks = json.load(open(a.tasks)); ids = set(a.ids.split(",")) if a.ids else None
for T in tasks:
    if ids and T["id"] not in ids: continue
    od = os.path.join(a.out, T["id"])
    if os.path.exists(od + "/timing.json"): continue
    frames = a.frames or int(T.get("frames", 161)); t0 = time.time(); N = sampler.prepare(T["image"], T.get("camera_WM", T.get("caption", "")), T["action"], frames, int(T["area"]))
    g = torch.Generator().manual_seed(a.seed); xs = []; rows = []
    for c in range(N):
        tm = {}; torch.cuda.synchronize(); t1 = time.time(); x = sampler.chunk(c, g, tm); sampler.commit(x, c); torch.cuda.synchronize(); tm["total_s"] = time.time() - t1; xs.append(x); rows.append({k: round(v, 4) for k, v in tm.items()})
    lat = torch.cat(xs, 1); os.makedirs(od, exist_ok=True); torch.save(lat.half().cpu(), od + "/lat.pt")
    steady = [r["total_s"] for r in rows[3:]] or [r["total_s"] for r in rows]
    timing = dict(task=T["id"], arm="prog_student" if a.student else "prog_stock", student=a.student, prog=a.prog, window=a.window, wq=a.wq, nodes=list(NODES), frames=frames, chunks=rows,
                  steady_fps_gen=round(12.0 / float(np.mean(steady)), 2), gen_s=round(sum(r["total_s"] for r in rows), 3), peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2), hi_grid=list(sampler.HW), lo_grid=list(sampler.LW) if a.prog else None)
    if a.decode:
        sampler._use(sampler.HI); b.decode_and_save([lat[:, 3 * c:3 * c + 3].to(dev) for c in range(N)], od)
        if os.path.exists(od + "/spec.mp4"): os.replace(od + "/spec.mp4", od + "/video.mp4")
    json.dump(timing, open(od + "/timing.json", "w"), indent=1); print("[prog_eval] %s: %d chunks, %.2f fps steady, lo %.3f hi %.3f s, %.0f s" % (T["id"], N, timing["steady_fps_gen"], np.mean([r.get("lo_s", 0) for r in rows]), np.mean([r.get("hi_s", 0) for r in rows]), time.time() - t0), flush=True)
print("[PROG_EVAL_DONE]", flush=True)
