"""Build the drift-critic dataset from ALL blind-labelled clips: VAE-encode each clip (first 153 frames) to 13 chunk latents, store fp16 + labels +
cheap pixel features (HSV saturation / value drift per chunk, the earlier plan-feature baseline). One record per clip:
exp/drift_lat/<task>__<arm>__s<seed>.pt = {task, arm, seed, severe, drift, cats, lat [16,39,h,w] fp16, hsv [13,2]}; cams: exp/drift_lat/cam_<task>.pt
usage: CUDA_VISIBLE_DEVICES=g python scripts/drift_dataset.py --shard i/n"""
import os, sys, json, glob, argparse, collections, numpy as np, torch, imageio, cv2
S = "/home/hk28965/specwm"; OUT = S + "/exp/drift_lat"; os.makedirs(OUT, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument("--shard", default="0/1"); a = ap.parse_args(); si, sn = (int(x) for x in a.shard.split("/"))
SEV = {"teleport", "color_collapse", "texture_collapse", "freeze"}; V0 = S + "/lingbot-world-v2/specwm_out/spec_v0"; ROOTS = {"blind_loop": S + "/lingbot-world-v2/specwm_out/loop", "blind_rl": S + "/lingbot-world-v2/specwm_out/draft_rl_vid", "blind_lowres": S + "/lingbot-world-v2/specwm_out/loop_mix"}
def labels_of(d):
    """majority labels per sheet id from r{1,2,3}*.json (parts / extra files included)"""
    R = {r: {x["id"]: x for f in sorted(glob.glob("%s/r%d*.json" % (d, r))) for x in json.load(open(f))} for r in (1, 2, 3)}
    out = {}
    for i in json.load(open(d + "/mapping.json")):
        rs = [r for r in (1, 2, 3) if i in R[r]]   # 3-rater batches need all three; single-rater batches (blind_more) use the one available
        if not rs or (len(rs) < 3 and any(R[r] for r in (1, 2, 3) if r not in rs)): continue
        need = (len(rs) + 1) // 2 if len(rs) > 1 else 1
        votes = [R[r][i]["drift"] and bool(set(R[r][i].get("categories", [])) & SEV) for r in rs]; dv = [R[r][i]["drift"] for r in rs]
        cats = collections.Counter(c for r in rs for c in R[r][i].get("categories", []))
        out[i] = dict(severe=sum(votes) >= max(need, 2 if len(rs) == 3 else need), drift=sum(dv) >= max(need, 2 if len(rs) == 3 else need), cats=[c for c, m in cats.items() if m >= need], batch=os.path.basename(d), n_raters=len(rs))
    return out
clips = {}
for d in sorted(glob.glob(S + "/paper/overnight/blind*")):
    if not os.path.isdir(d) or not os.path.exists(d + "/mapping.json") or not glob.glob(d + "/r1*.json"): continue
    mp = json.load(open(d + "/mapping.json")); lab = labels_of(d); root = ROOTS.get(os.path.basename(d), V0)
    for i, v in mp.items():
        if i not in lab: continue
        f = "%s/%s/%s_s%d/video.mp4" % (root, v["task"], v["arm"], v["seed"])
        if not os.path.exists(f): continue
        key = (v["task"], v["arm"], v["seed"])
        if key in clips: continue   # same clip rated in two batches: keep the first
        clips[key] = dict(task=v["task"], arm=v["arm"], seed=v["seed"], video=f, **lab[i])
keys = sorted(clips)[si::sn]; print("labelled clips %d, this shard %d" % (len(clips), len(keys)), flush=True)
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2")
from specwm.binding_v2 import V2Binding
dev = torch.device("cuda:0"); FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
D = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), cfg_name="i2v-1.3B", assets_dir=FAST)
TT = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; cams = {}
def sv(fr):
    h = [cv2.cvtColor(f, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32) for f in fr]; return float(np.mean([x[:, 1].mean() for x in h])), float(np.mean([x[:, 2].mean() for x in h]))
for n, key in enumerate(keys):
    c = clips[key]; out = "%s/%s__%s__s%d.pt" % (OUT, c["task"], c["arm"], c["seed"])
    if os.path.exists(out): continue
    if c["task"] not in cams:
        T = TT[c["task"]]; D.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832)
        cams[c["task"]] = torch.stack([e.squeeze(0).clone() for e in D._emb_chunks]).half().cpu(); torch.save(dict(cam=cams[c["task"]], shape=list(D._chunk_shape)), "%s/cam_%s.pt" % (OUT, c["task"]))
    try:
        fr = np.stack([np.asarray(x) for x in imageio.get_reader(c["video"])])[:153]
    except Exception as ex: print("[fail read]", c["video"], ex, flush=True); continue
    if fr.shape[0] < 153: print("[short]", c["video"], fr.shape[0], flush=True); continue
    h, w = D._chunk_shape[2] * 8, D._chunk_shape[3] * 8
    v = torch.from_numpy(fr).float().div(127.5).sub(1).permute(3, 0, 1, 2)
    if v.shape[2:] != (h, w): v = torch.nn.functional.interpolate(v.permute(1, 0, 2, 3), size=(h, w), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
    with torch.no_grad(): lat = D.pipe.vae.encode([v.to(dev)])[0].half().cpu()
    s0, v0 = sv(fr[:1]); hsv = [sv([fr[i] for i in range(1 + 12 * k, min(1 + 12 * (k + 1), len(fr)), 3)]) for k in range(13)]
    torch.save(dict(task=c["task"], arm=c["arm"], seed=c["seed"], severe=c["severe"], drift=c["drift"], cats=c["cats"], batch=c["batch"], n_raters=c.get("n_raters", 3), lat=lat, hsv=[(s / s0, vv / v0) for s, vv in hsv]), out)
    if n % 20 == 0: print("[%d/%d] %s %s s%d severe=%d lat %s" % (n, len(keys), c["task"][:6], c["arm"][:24], c["seed"], c["severe"], tuple(lat.shape)), flush=True)
print("[DRIFT_DS_DONE] shard %s" % a.shard, flush=True)
