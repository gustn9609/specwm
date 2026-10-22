"""Clip-level VLM judge (Qwen3-VL-32B on :30100): 8 evenly spaced frames per video, two questions -> drift score 0-100 (same prompt as
vlm_drift_judge, AUROC 0.82 vs human severe labels on unseen scenes) and visual quality 1-10 (sharpness, artifacts, realism). Writes a JSON map
{video_path: {drift, quality, raw}}. usage: python scripts/vlm_clip_score.py OUT.json LABEL=GLOB ... """
import sys, glob, json, io, base64, time, os, numpy as np, cv2
from PIL import Image
from openai import OpenAI
client = OpenAI(api_key="local", base_url="http://127.0.0.1:%s/v1" % os.environ.get("JUDGE_PORT", "30100")); out_path = sys.argv[1]; res = json.load(open(out_path)) if os.path.exists(out_path) else {}   # [10/17] JUDGE_PORT env (default 30100 unchanged)
def frames(p, n=8):
    cap = cv2.VideoCapture(p); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    idx = [int(k * (len(fr) - 1) / (n - 1)) for k in range(n)]; return [fr[i] for i in idx]
def b64(im, W=int(os.environ.get("VLM_W", "448"))):
    im = Image.fromarray(im); im = im.resize((W, int(im.height * W / im.width)), Image.BICUBIC); b = io.BytesIO(); im.save(b, format="JPEG", quality=85); return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
def ask(images, prompt):
    r = client.chat.completions.create(model="qwen3-vl-32b", messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": u}} for u in images] + [{"type": "text", "text": prompt}]}], max_tokens=8, temperature=0)
    txt = r.choices[0].message.content.strip()
    try: return float(next(t for t in txt.replace("/", " ").split() if t.replace(".", "").isdigit())), txt
    except StopIteration: return float("nan"), txt
DRIFT = ("These 8 frames are evenly spaced from one 10-second AI-generated video that should keep showing the scene of the first frame while the camera moves. "
         "Rate how severely the video DRIFTS away from the first frame's scene: teleporting to a different place, the main objects vanishing or morphing into other things, colours or textures collapsing, or the picture freezing. "
         "Answer with ONE integer 0-100 (0 = same scene throughout, 100 = completely different scene / collapsed). Only the number.")
QUAL = ("These 8 frames are evenly spaced from one AI-generated video. Rate the VISUAL QUALITY of the frames themselves (sharpness and detail, absence of artifacts such as blur, blocking, smearing or grid noise, and realism of textures). "
        "Ignore whether the scene changes over time. Answer with ONE integer 1-10 (10 = photographic, 1 = unusable). Only the number.")
for spec in sys.argv[2:]:
    lab, pat = spec.split("=", 1); paths = sorted(glob.glob(pat)); t0 = time.time()
    for p in paths:
        if p in res: continue
        ims = [b64(f) for f in frames(p)]; d, rd = ask(ims, DRIFT); q, rq = ask(ims, QUAL); res[p] = dict(label=lab, drift=d, quality=q, raw=[rd, rq])
    json.dump(res, open(out_path, "w"), indent=0)
    R = [res[p] for p in paths if p in res]; print("| %s | %d | %.0f | %.1f |" % (lab, len(R), np.nanmean([r["drift"] for r in R]), np.nanmean([r["quality"] for r in R])), flush=True)
print("[VLM_SCORE_DONE]")
