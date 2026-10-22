#!/usr/bin/env python3
"""Page 3: what 'control success' is and where Fast+ours flips it vs Fast alone (paired tasks), with judge notes + clips + 1 s frame strips."""
import base64, json, os, glob, subprocess, numpy as np, imageio, cv2
S="/home/hk28965/specwm"; O=S+"/lingbot-world-v2/specwm_out"; ST=O+"/stevo"; R=ST+"/runs"; W=S+"/specwm_out_web"; V=W+"/vid_flips"; os.makedirs(V, exist_ok=True); FF="/home/hk28965/.local/bin/ffmpeg"
def rep(m, t, kind="control"):
    fs=glob.glob(R+"/%s/per_task/%s/%s_report*.json"%(m,t,kind)); return json.load(open(fs[0])) if fs else None
ARMS=[("fast_alone","Fast alone (W18/sink6, seed 1234, 161 f)","lingbot_reground_fast_alone",lambda t: ST+"/lingbot_reground_fast_alone/%s.mp4"%t),("ours","Fast + ours (KV-rewrite from laundered+RePaint target)","lingbot_reground_kvrw_laundered",lambda t: ST+"/lingbot_reground_kvrw_laundered/%s.mp4"%t),
      ("ideal","Fast + ideal regrounding (Base70 ctx 0-5)","lingbot_reground_resync_k6",lambda t: ST+"/lingbot_reground_resync_k6/%s.mp4"%t),("base","Base 70-step (253 f)","lingbot_base_flat",lambda t: ST+"/lingbot_base/%s/video.mp4"%t)]
ctrl=lambda c: bool(c and c.get("occlusion_done") and c.get("trigger_applied"))
fa={os.path.basename(os.path.dirname(p)):json.load(open(p)) for p in glob.glob(R+"/lingbot_reground_fast_alone/per_task/*/control_report*.json")}
ou={os.path.basename(os.path.dirname(p)):json.load(open(p)) for p in glob.glob(R+"/lingbot_reground_kvrw_laundered/per_task/*/control_report*.json")}
wins=sorted(t for t in fa if t in ou and ctrl(ou[t]) and not ctrl(fa[t])); losses=sorted(t for t in fa if t in ou and ctrl(fa[t]) and not ctrl(ou[t]))
def b64(p,m): return "data:%s;base64,%s"%(m,base64.b64encode(open(p,"rb").read()).decode())
def clip(src,dst):
    if not os.path.exists(dst): subprocess.run([FF,"-y","-loglevel","error","-i",src,"-vf","fps=8,scale=360:-2","-c:v","libx264","-preset","slow","-crf","33","-pix_fmt","yuv420p","-movflags","+faststart","-an",dst],check=True)
    return b64(dst,"video/mp4")
def strip(src, n=11, tw=150):
    V_=np.stack([np.asarray(f) for f in imageio.get_reader(src)]); ks=np.linspace(0,len(V_)-1,n).round().astype(int); h,w=V_.shape[1:3]; th=int(h*tw/w); tiles=[]
    for k in ks:
        im=np.ascontiguousarray(cv2.resize(V_[k],(tw,th))); cv2.putText(im,"%.1fs"%(k/16),(4,14),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),2); tiles.append(im)
    im=np.concatenate(tiles,1); ok,buf=cv2.imencode(".jpg",cv2.cvtColor(im,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,72]); return "data:image/jpeg;base64,"+base64.b64encode(buf.tobytes()).decode()
def note(c, key):
    if not c: return ""
    rs=c.get(key+"_responses") or []; return " / ".join((r.get("notes") or "")[:220] for r in rs[:2])
items=[]
for kind,ts in [("win",wins),("loss",losses)]:
    for t in ts:
        arms=[]
        for key,lab,m,fn in ARMS:
            p=fn(t); c=rep(m,t); ph=rep(m,t,"physics")
            if not os.path.exists(p): continue
            arms.append(dict(arm=key,label=lab,src=clip(p,V+"/%s__%s.mp4"%(t,key)),strip=strip(p),occ=bool(c and c.get("occlusion_done")),trig=bool(c and c.get("trigger_applied")),ctrl=ctrl(c),phys=(None if not ph else not ph.get("physical_inaccuracy",True)),occ_note=note(c,"occlusion"),trig_note=note(c,"trigger"),req_trigger=(c or {}).get("requested_trigger","")))
        items.append(dict(kind=kind,task=t,prompt=(fa[t].get("video_WM_prompt","") or "")[:300],arms=arms))
CSS=open(W+"/clip_compare.html").read().split("<style>")[1].split("</style>")[0]
FONT='<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Serif:wght@600&family=IBM+Plex+Mono&display=swap">'
p=["<title>specwm Control Flips</title>",FONT,"<style>",CSS,""".task{border:1px solid var(--line);border-radius:8px;background:var(--card);padding:12px 14px;margin:14px 0}.task h3{margin:0 0 4px;font-size:1rem}.armrow{display:grid;grid-template-columns:minmax(220px,300px) 1fr;gap:12px;align-items:start;border-top:1px solid var(--line);padding:10px 0}
.pill{display:inline-block;padding:1px 8px;border-radius:10px;font-size:.75rem;font-weight:600}.ok{background:#dff3e3;color:#1b6b34}.bad{background:#f9dcd8;color:#9b2c1e}.na{background:var(--chip);color:var(--mute)}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) .ok{background:#1f4a2c;color:#b9efc6}:root:not([data-theme="light"]) .bad{background:#5a221a;color:#f7c2b8}}:root[data-theme="dark"] .ok{background:#1f4a2c;color:#b9efc6}:root[data-theme="dark"] .bad{background:#5a221a;color:#f7c2b8}
.strip{width:100%;display:block;border-radius:4px}.jn{font-size:.8rem;color:var(--mute);margin-top:6px}@media (max-width:700px){.armrow{grid-template-columns:1fr}}</style>""",
   "<h1>specwm Control Flips</h1><p class=sub>What the StEvo <b>control success</b> metric actually measures, and the tasks where Fast&nbsp;+&nbsp;ours changes the verdict relative to Fast alone (same seed, same 161-frame rollout up to the KV rewrite).</p>",
   """<div class=note><b>Definition (StEvo control judge, Qwen3-VL-32B, 3-vote majority; ↑ higher is better).</b> <i>control success</i> = <i>occlusion_done</i> ∧ <i>trigger_applied</i>.
   <b>occlusion_done</b>: the commanded camera look-away actually took the subject fully out of frame at the right moment (not before the dynamic started, not never) — i.e. the model obeyed the camera command. <b>trigger_applied</b>: the task's kickoff action (e.g. pressing the fountain button) happened; for tasks with no trigger it is true by specification, but the judge occasionally votes false when the scene resets or the subject is replaced. So a control flip means: with the Base-derived context in the KV cache, the Fast rollout keeps the scene coherent enough that the camera command reads as executed (subject leaves the frame, then the same scene comes back), instead of drifting into a different scene / never leaving the subject / teleporting.</div>
   <p class=sub>Paired result over 85 tasks: <b>6 wins, 3 losses, 76 ties</b> (McNemar exact p = 0.51 — not significant at n = 85). Fast alone 54.1 % → ours 57.6 % → ideal 62.4 % → Base 65.5 %.</p>""",
   "<h2>Wins (Fast alone ✗ → Fast + ours ✓)</h2><div id=wins></div><h2>Losses (Fast alone ✓ → Fast + ours ✗)</h2><div id=losses></div>",
   "<p class=note>Companion pages: <a href='https://claude.ai/artifact/6E8aFayABD1cpBsfZmbyi3'>specwm Clip Compare</a> · <a href='https://claude.ai/artifact/8sQg23AxUbKKQ6uvY4KccT'>specwm Trajectory Gallery</a>.</p>",
   "<script>const ITEMS=",json.dumps(items),""";
function pill(v,l){return v===null||v===undefined?`<span class="pill na">${l}: n/a</span>`:`<span class="pill ${v?'ok':'bad'}">${l}: ${v?'✓':'✗'}</span>`}
for(const it of ITEMS){const box=document.getElementById(it.kind==='win'?'wins':'losses');const d=document.createElement('div');d.className='task';
 d.innerHTML=`<h3>${it.task.replace(/_/g,' ')}</h3><p class=jn>${it.prompt}</p>`+it.arms.map(a=>`<div class=armrow><div><video muted loop playsinline autoplay src="${a.src}" style="width:100%;border-radius:4px;background:#000"></video><div style="margin-top:6px"><b>${a.label}</b><br>${pill(a.ctrl,'control')} ${pill(a.occ,'occlusion_done')} ${pill(a.trig,'trigger_applied')} ${pill(a.phys,'physics')}</div></div><div><img class=strip src="${a.strip}" alt="frames"><div class=jn><b>judge (occlusion):</b> ${a.occ_note||'—'}</div>${a.req_trigger&&a.req_trigger!=='none'?`<div class=jn><b>judge (trigger = ${a.req_trigger}):</b> ${a.trig_note||'—'}</div>`:''}</div></div>`).join('');
 box.appendChild(d)}
</script>"""]
open(W+"/control_flips.html","w").write("\n".join(p)); print(len(items),"tasks; page %.1f MB"%(os.path.getsize(W+"/control_flips.html")/2**20))
