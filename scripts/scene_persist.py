#!/usr/bin/env python3
"""[10/21] Scene-persistence metric for long rollouts: does the clip still show the scene it started with?
For every video: frames sampled at --fps (default 1 per second), embedded with CLIP ViT-L/14 (image tower, ~/.cache/clip) and DINOv2-large
(hf/dinov2-large, CLS token); the reference = mean embedding of the first --ref-s seconds (default 2 s = the first chunk); sim_t = cosine(frame t, ref).
Written per video (append-only json keyed by the absolute video path, as the other metrics):
  clip_sim, dino_sim        per-second cosine to the reference (lists)
  seg_clip, seg_dino        mean over consecutive --seg-s pieces (10 s -> 6 values for a 60-s clip, 1 for a 10-s clip)
  clip_last, dino_last      mean over the last piece
  clip_auc, dino_auc        mean over all seconds (area under the persistence curve, 1 = never left the first scene)
  persist_s_clip/dino       first second whose similarity is below --tau (clip 0.70 / dino 0.50); = duration when it never drops ("scene kept")
  step_min_clip/dino        min cosine between consecutive sampled frames (a hard scene cut shows as a low value), n_cuts_* = count below --cut
  n_s, native_fps
Reads frames with decord. Plain tools/vbench_env python (clip, transformers 4.33, decord, torch). GPU: --device cuda (default) under the GPU lock.
usage: scene_persist.py OUT.json VIDEO... [--fps 1] [--ref-s 2] [--seg-s 10] [--tau-clip 0.70] [--tau-dino 0.50] [--cut 0.6] [--device cuda] [--force]"""
import os, sys, json, argparse, math, time
import numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("videos", nargs="+"); ap.add_argument("--fps", type=float, default=1.0); ap.add_argument("--ref-s", type=float, default=2.0)
ap.add_argument("--seg-s", type=float, default=10.0); ap.add_argument("--tau-clip", type=float, default=0.70); ap.add_argument("--tau-dino", type=float, default=0.50); ap.add_argument("--cut", type=float, default=0.6)
ap.add_argument("--device", default="cuda"); ap.add_argument("--force", action="store_true"); ap.add_argument("--dino", default="/home/hk28965/specwm/hf/dinov2-large"); ap.add_argument("--clip", default="ViT-L/14")
a = ap.parse_args()
out = json.load(open(a.out)) if os.path.exists(a.out) else {}
vids = [os.path.abspath(v) for v in a.videos]; todo = [v for v in vids if a.force or v not in out]
print("[scene_persist] %d videos, %d to do -> %s" % (len(vids), len(todo), a.out), flush=True)
if not todo: sys.exit(0)
import decord; decord.bridge.set_bridge("native")
import clip as openai_clip
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
dev = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu")
cm, cpre = openai_clip.load(a.clip, device=dev); cm.eval()
dproc = AutoImageProcessor.from_pretrained(a.dino); dm = AutoModel.from_pretrained(a.dino).to(dev).eval()
def cos(a_, b_): return float(np.dot(a_, b_) / (np.linalg.norm(a_) * np.linalg.norm(b_) + 1e-8))
# zero-shot content class per second (CLIP text tower): what the frame shows, independent of the start frame (a moving camera legitimately leaves the start view).
ZS = [("scene", "a landscape, street, room or building with visible ground or floor, from a photo, a film or a video game"), ("sky", "only sky, clouds or fog, no ground or objects"),
      ("abstract", "an abstract blurry smeared texture with no recognizable objects"), ("ui", "a screenshot of a user interface, a poster, a document or a page of text"), ("blank", "a plain black, white or single-color frame")]
with torch.no_grad():
    ZT = cm.encode_text(openai_clip.tokenize([p for _, p in ZS]).to(dev)).float(); ZT = (ZT / ZT.norm(dim=-1, keepdim=True)).cpu().numpy()
@torch.no_grad()
def embed(frames):   # frames: list of HxWx3 uint8
    ims = [Image.fromarray(f) for f in frames]
    c = cm.encode_image(torch.stack([cpre(im) for im in ims]).to(dev)).float(); c = (c / c.norm(dim=-1, keepdim=True)).cpu().numpy()
    d = dm(**{k: v.to(dev) for k, v in dproc(images=ims, return_tensors="pt").items()}).last_hidden_state[:, 0].float(); d = (d / d.norm(dim=-1, keepdim=True)).cpu().numpy()
    return c, d
def stats(sim, tau, n_ref, seg_n):
    sim = np.asarray(sim, dtype=np.float64); n = len(sim)
    segs = [float(sim[i:i + seg_n].mean()) for i in range(0, n, seg_n)]
    below = np.where(sim < tau)[0]; persist = float(below[0]) if len(below) else float(n)
    return dict(seg=[round(x, 4) for x in segs], last=round(segs[-1], 4), auc=round(float(sim.mean()), 4), persist_s=persist)
t_all = time.time()
for k, v in enumerate(todo):
    t0 = time.time()
    try:
        vr = decord.VideoReader(v, num_threads=2); nf = len(vr); nat = float(vr.get_avg_fps()); dur = nf / nat
        idx = [min(nf - 1, int(round(t * nat))) for t in np.arange(0, dur - 1e-6, 1.0 / a.fps)]
        frames = [vr[i].asnumpy() for i in idx]
        C, D = [], []
        for i in range(0, len(frames), 16):
            c, d = embed(frames[i:i + 16]); C.append(c); D.append(d)
        C = np.concatenate(C); D = np.concatenate(D); n_ref = max(1, int(round(a.ref_s * a.fps))); seg_n = max(1, int(round(a.seg_s * a.fps)))
        cref = C[:n_ref].mean(0); dref = D[:n_ref].mean(0)
        cs = [cos(C[i], cref) for i in range(len(C))]; ds = [cos(D[i], dref) for i in range(len(D))]
        cstep = [cos(C[i], C[i - 1]) for i in range(1, len(C))]; dstep = [cos(D[i], D[i - 1]) for i in range(1, len(D))]
        sc = stats(cs, a.tau_clip, n_ref, seg_n); sd = stats(ds, a.tau_dino, n_ref, seg_n)
        zl = (C @ ZT.T) * 100.0; zc = zl.argmax(1); names = [n_ for n_, _ in ZS]
        zs_frac = {n_: round(float((zc == i).mean()), 4) for i, n_ in enumerate(names)}; zs_seq = [names[i] for i in zc]
        zs_seg = [round(float((zc[i:i + seg_n] == 0).mean()), 4) for i in range(0, len(zc), seg_n)]   # per piece: fraction of seconds that still look like a real scene
        rec = dict(zs_frac=zs_frac, zs_scene_seg=zs_seg, zs_scene_last=zs_seg[-1], zs_seq=zs_seq,n_s=len(cs), native_fps=round(nat, 3), frames=nf, sample_fps=a.fps, ref_s=a.ref_s, seg_s=a.seg_s, tau_clip=a.tau_clip, tau_dino=a.tau_dino, cut=a.cut,
                   clip_sim=[round(x, 4) for x in cs], dino_sim=[round(x, 4) for x in ds],
                   seg_clip=sc["seg"], seg_dino=sd["seg"], clip_last=sc["last"], dino_last=sd["last"], clip_auc=sc["auc"], dino_auc=sd["auc"], persist_s_clip=sc["persist_s"], persist_s_dino=sd["persist_s"],
                   step_min_clip=round(min(cstep), 4) if cstep else None, step_min_dino=round(min(dstep), 4) if dstep else None,
                   n_cuts_clip=int(sum(1 for x in cstep if x < a.cut)), n_cuts_dino=int(sum(1 for x in dstep if x < a.cut)), models=dict(clip=a.clip, dino=os.path.basename(a.dino)), sec=round(time.time() - t0, 2))
        out[v] = rec
        print("[scene_persist] %d/%d %s: %d s, clip auc %.3f last %.3f persist %.0f s | dino auc %.3f last %.3f persist %.0f s | cuts %d (%.1f s)" % (k + 1, len(todo), os.path.relpath(v), len(cs), sc["auc"], sc["last"], sc["persist_s"], sd["auc"], sd["last"], sd["persist_s"], rec["n_cuts_dino"], rec["sec"]), flush=True)
    except Exception as e:
        out[v] = dict(error=repr(e)[:300]); print("[scene_persist] ERROR %s: %r" % (v, e), flush=True)
    if (k + 1) % 10 == 0 or k + 1 == len(todo):
        tmp = a.out + ".tmp"; json.dump(out, open(tmp, "w"), indent=0); os.replace(tmp, a.out)
print("[scene_persist] done %d videos in %.0f s" % (len(todo), time.time() - t_all), flush=True)
