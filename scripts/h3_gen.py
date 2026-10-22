#!/usr/bin/env python3
"""MiniMax-H3 (sglang serve, fl2va) 생성 드라이버. 항목 = (name, image, prompt_file, seed). 
--floor: 씬 5개 x 시드 (노이즈 플로어)   --demo <list.json>: 데모 프롬프트 목록   결과: specwm_out/h3/<name>/{video.mp4, prompt.txt, meta.json} + h3_log.json"""
import argparse, json, os, time, urllib.request, shutil
S = os.environ.get("SCRATCH", "/home/hk28965/specwm"); OUT = S + "/lingbot-world-v2/specwm_out/h3"; os.makedirs(OUT, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=30010); ap.add_argument("--seconds", type=float, default=10); ap.add_argument("--steps", type=int, default=50); ap.add_argument("--short-edge", type=int, default=768)
ap.add_argument("--floor", default="", help="예: 05,01,00,02,03 (시드는 --seeds)"); ap.add_argument("--seeds", default="1234"); ap.add_argument("--demo", default="", help="demo_list json 경로(콤마로 여러 개)")
a = ap.parse_args(); base = "http://127.0.0.1:%d" % a.port
PROMPT = {"05": S + "/p_05.txt", "01": S + "/p_01.txt", "00": S + "/p_00.txt", "02": S + "/p_02.txt", "03": S + "/data/demo_prompts/p_03_lake.txt", "hands": S + "/p_hands1.txt"}
items = []
for sc in [x for x in a.floor.split(",") if x]:
    for sd in [int(s) for s in a.seeds.split(",")]: items.append(("floor_S%s_s%d" % (sc, sd), "%s/lingbot-world-v2/examples/%s/image.jpg" % (S, sc), PROMPT[sc], sd))
for lp in [x for x in a.demo.split(",") if x]:
    for it in json.load(open(lp)):
        img = it["image"] if it["image"].startswith("/") else "%s/lingbot-world-v2/%s" % (S, it["image"]); items.append(("demo_" + it["name"], img, it["prompt_file"], 1234))

def req(path, data=None):
    r = urllib.request.Request(base + path, data=json.dumps(data).encode() if data is not None else None, headers={"Content-Type": "application/json"}, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(r, timeout=900) as f: return f.read()

lp = os.path.join(OUT, "h3_log.json"); log = json.load(open(lp)) if os.path.exists(lp) else {}
for name, img, pf, sd in items:
    d = os.path.join(OUT, name); out = os.path.join(d, "video.mp4")
    if os.path.exists(out): print("[skip]", name); continue
    prompt = open(pf).read().strip()
    body = {"model": "MiniMaxAI/MiniMax-H3", "task": "fl2va", "prompt": prompt, "conditions": [{"type": "image", "uri": "file://" + img, "role": "keyframe", "frame_index": 0}],
            "target": {"short_edge": a.short_edge, "aspect_ratio": "16:9", "duration_seconds": a.seconds}, "num_inference_steps": a.steps, "seed": sd}
    t0 = time.time()
    try:
        vid = json.loads(req("/v1/videos", body))["id"]; print("[h3] %s submitted id=%s" % (name, vid), flush=True)
        while True:
            st = json.loads(req("/v1/videos/%s" % vid)); s = st.get("status")
            if s in ("completed", "failed", "error", "cancelled"): break
            time.sleep(10)
        dt = time.time() - t0
        if s == "completed":
            os.makedirs(d, exist_ok=True); open(out, "wb").write(req("/v1/videos/%s/content" % vid)); shutil.copy(pf, os.path.join(d, "prompt.txt"))
            json.dump(dict(name=name, image=img, prompt=prompt, seed=sd, wall_s=round(dt, 1), steps=a.steps, seconds=a.seconds, short_edge=a.short_edge, status=st), open(os.path.join(d, "meta.json"), "w"), indent=1)
            print("[h3] %s done %.0fs" % (name, dt), flush=True)
        else: print("[h3] %s FAILED %s: %s" % (name, s, json.dumps(st)[:300]), flush=True)
        log[name] = dict(status=s, wall_s=round(dt, 1), seed=sd)
    except Exception as e:
        print("[h3] %s ERROR %s" % (name, str(e)[:300]), flush=True); log[name] = dict(status="exception", error=str(e)[:300])
    json.dump(log, open(lp, "w"), indent=1)
print("[H3_GEN_DONE]", flush=True)
