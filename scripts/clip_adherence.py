#!/usr/bin/env python3
"""프롬프트 일치도 (CLIP ViT-L/14): 데모 영상이 자기 프롬프트(변형: 폭풍/눈/야간...)를 얼마나 따르는지.
  sim_own     = mean_t cos(CLIP(frame_t), CLIP(own prompt))
  sim_neutral = 같은 씬의 원래(중립) 프롬프트와의 유사도
  delta       = sim_own - sim_neutral   (변형 프롬프트를 실제로 반영했는지; 0 이면 중립 영상과 구분 안 됨)
대상: demo/<name>/{base16,fast_alone,resync_k6}, floor/base|fast (sim_own 만). 출력 specwm_out/clip/clip_adherence.json + .md"""
import os, json, glob, re
import numpy as np, torch
from transformers import CLIPModel, CLIPProcessor
S = os.environ.get("SCRATCH", "/home/hk28965/specwm"); R = S + "/lingbot-world-v2/specwm_out"; OUT = R + "/clip"; os.makedirs(OUT, exist_ok=True)
dev = "cuda"; name = "openai/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(name, torch_dtype=torch.float16).to(dev).eval(); proc = CLIPProcessor.from_pretrained(name)
NEUTRAL = {"03": S + "/data/demo_prompts/p_03_lake.txt", "05": S + "/data/p_05.txt", "01": S + "/data/p_01.txt", "02": S + "/data/p_02.txt", "00": S + "/data/p_00.txt", "04": S + "/data/p_00.txt", "hands": S + "/data/p_hands1.txt"}


@torch.no_grad()
def temb(text):
    t = proc(text=[text[:1000]], return_tensors="pt", padding=True, truncation=True, max_length=77).to(dev)
    e = model.get_text_features(**t); return torch.nn.functional.normalize(e.float(), dim=-1)[0]


@torch.no_grad()
def iemb(V):
    V = V[::4] if len(V) > 100 else V; out = []
    for i in range(0, len(V), 16):
        px = proc(images=list(V[i:i + 16]), return_tensors="pt")["pixel_values"].to(dev, torch.float16)
        out.append(torch.nn.functional.normalize(model.get_image_features(pixel_values=px).float(), dim=-1))
    return torch.cat(out)


rows = []
for p in sorted(glob.glob(R + "/demo/*/*/frames_all.npy")) + sorted(glob.glob(R + "/h3/demo_*/frames_all.npy")):
    if "/h3/" in p: nm, arm = os.path.basename(os.path.dirname(p))[5:], "h3"
    else: nm, arm = p.split("/")[-3], p.split("/")[-2]
    pp = os.path.join(os.path.dirname(p), "prompt.txt")
    if not os.path.exists(pp): continue
    sc = "hands" if nm.startswith("hands") else nm[:2]; own = open(pp).read().strip(); neu = open(NEUTRAL[sc]).read().strip()
    F = iemb(np.load(p)); so = float((F @ temb(own)).mean()); sn = float((F @ temb(neu)).mean())
    rows.append(dict(group="demo", name=nm, arm=arm, sim_own=round(so, 4), sim_neutral=round(sn, 4), delta=round(so - sn, 4), variant=(own != neu)))
    print("[clip] %-22s %-10s own %.4f neutral %.4f delta %+.4f" % (nm, arm, so, sn, so - sn), flush=True)
for p in sorted(glob.glob(R + "/floor/base/S*/frames_all.npy") + glob.glob(R + "/floor/fast/S*/W18_sink6/fast_alone_s*/frames_all.npy")):
    mm = re.search(r"/S(\d\d|hands)", p); sc = mm.group(1); own = open(NEUTRAL[sc]).read().strip()
    F = iemb(np.load(p)); so = float((F @ temb(own)).mean())
    rows.append(dict(group="floor_base" if "/floor/base/" in p else "floor_fast", name="S" + sc, arm=p.split("/")[-2], sim_own=round(so, 4)))
json.dump(rows, open(OUT + "/clip_adherence.json", "w"), indent=1)
md = ["# CLIP prompt adherence (ViT-L/14, 4fps frames)", "", "## Demo (variant prompts): delta = sim(own) - sim(neutral scene prompt)", "", "| name | arm | sim own | sim neutral | delta |", "|---|---|---|---|---|"]
for r in rows:
    if r["group"] == "demo" and r["variant"]: md.append("| %s | %s | %.4f | %.4f | %+.4f |" % (r["name"], r["arm"], r["sim_own"], r["sim_neutral"], r["delta"]))
md += ["", "## Per-arm mean delta over variant prompts", "", "| arm | mean delta | n |", "|---|---|---|"]
for arm in ("base16", "fast_alone", "resync_k6", "h3"):
    v = [r["delta"] for r in rows if r["group"] == "demo" and r["variant"] and r["arm"] == arm]
    if v: md.append("| %s | %+.4f | %d |" % (arm, np.mean(v), len(v)))
md += ["", "## Floor videos: sim to own prompt (Base 70-step vs Fast)", "", "| scene | Base mean±std | Fast mean±std |", "|---|---|---|"]
for sc in sorted({r["name"] for r in rows if r["group"].startswith("floor")}):
    b = [r["sim_own"] for r in rows if r["group"] == "floor_base" and r["name"] == sc]; f = [r["sim_own"] for r in rows if r["group"] == "floor_fast" and r["name"] == sc]
    md.append("| %s | %s | %s |" % (sc, "%.4f±%.4f (n=%d)" % (np.mean(b), np.std(b), len(b)) if b else "-", "%.4f±%.4f (n=%d)" % (np.mean(f), np.std(f), len(f)) if f else "-"))
open(OUT + "/clip_adherence.md", "w").write("\n".join(md)); print("\n".join(md)); print("[CLIP_DONE]")
