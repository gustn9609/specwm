#!/usr/bin/env python3
"""Build two artifact pages: (1) clip comparison grid with embedded videos, (2) trajectory-figure gallery."""
import base64, json, os, html
S="/home/hk28965/specwm"; W=S+"/specwm_out_web"; V=W+"/vid"; FIG=S+"/lingbot-world-v2/specwm_out/traj_viz2"
man=json.load(open(V+"/manifest.json")); summ=json.load(open(FIG+"/summary.json"))
META={"base":("0.05 fps","65.5","52.4"),"fast480":("2.4 fps","50.6","32.9"),"fast360_2step":("11.1 fps","55.3","29.4"),"fast360_2step_esrgan":("~10 fps","judging","judging"),"v2_1p3b":("–","32.9","1.2"),"v2_pretrain":("–","61.2","4.7"),
      "h3world":("–","45.9","43.5"),"minimax_h3":("–","86.0 (video protocol)","85.1"),"longlive":("27 fps e2e","23.1 (video protocol)","53.4"),"reground_ours":("2.4 fps + correction","57.6","35.3"),"reground_ideal":("–","62.4","36.5"),
      "anchor_restart":("2.4 fps + 1 image","–","–"),"draft360_ours":("11.1 fps + correction","–","–")}
# numeric metrics for the ranking table (StEvo camera protocol only, 85 tasks; video-protocol rows are not comparable and excluded)
MET={"base":(0.046,65.5,52.4),"fast480":(2.42,50.6,32.9),"fast360_2step":(11.12,55.3,29.4),"v2_1p3b":(None,32.9,1.2),"v2_pretrain":(None,61.2,4.7),"h3world":(None,45.9,43.5),"reground_ours":(2.4,57.6,35.3),"reground_ideal":(None,62.4,36.5)}
LAB={"base":"Base 70-step","fast480":"Fast 480p 4-step","fast360_2step":"Fast 360p 2-step","v2_1p3b":"v2 1.3B","v2_pretrain":"v2 causal-pretrain 14B","h3world":"H3-World","reground_ours":"Fast + ours (KV-rewrite)","reground_ideal":"Fast + ideal regrounding"}
TASKN={"falling_glass":"Falling glass","pouring_water_into_cup":"Pouring water","rocket_launch":"Rocket launch","inflating_balloon":"Inflating balloon","S05":"Dragon ride (S05)","S01":"Stonehenge (S01)","S00":"Great Wall (S00)","S02":"Urban (S02)"}
CSS = """
:root{--bg:#f6f5f1;--ink:#1c1b18;--mute:#6b675e;--line:#dcd8cf;--card:#ffffff;--base:#2d5aa8;--fast:#c03a2b;--gold:#c9a227;--chip:#ece9e1}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#15161a;--ink:#ece9e1;--mute:#a19c90;--line:#33353b;--card:#1e2025;--base:#7fa6e8;--fast:#ef7b6d;--gold:#e0bd4a;--chip:#2a2c33}}
:root[data-theme="dark"]{--bg:#15161a;--ink:#ece9e1;--mute:#a19c90;--line:#33353b;--card:#1e2025;--base:#7fa6e8;--fast:#ef7b6d;--gold:#e0bd4a;--chip:#2a2c33}
body{background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;padding:0 20px;padding-block:20px 48px;max-width:1500px;margin:0 auto}
h1{font-family:"IBM Plex Serif",Georgia,serif;font-weight:600;font-size:1.7rem;margin:0 0 4px;text-wrap:balance}h2{font-size:1.05rem;margin:22px 0 8px;font-weight:600}
.sub{color:var(--mute);margin:0 0 16px;max-width:70ch}.mono{font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}
.tabs,.tasks,.ctrl{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:8px 0}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--ink);padding:6px 12px;border-radius:6px;cursor:pointer}button:focus-visible{outline:2px solid var(--base);outline-offset:2px}
button.on{background:var(--ink);color:var(--bg);border-color:var(--ink)}button.act{border-color:var(--base);color:var(--base)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px;margin-top:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden;display:flex;flex-direction:column}
.card video,.card img{width:100%;display:block;background:#000;aspect-ratio:16/9;object-fit:contain}
.card .hd{display:flex;justify-content:space-between;gap:8px;padding:8px 10px;border-bottom:1px solid var(--line);font-size:.92rem;font-weight:600}
.card .ft{display:flex;flex-wrap:wrap;gap:6px;padding:8px 10px;font-size:.78rem;color:var(--mute)}.chip{background:var(--chip);border-radius:4px;padding:2px 7px}
.tag{font-size:.7rem;letter-spacing:.06em;text-transform:uppercase;padding:2px 7px;border-radius:4px;color:#fff}.tag.base{background:var(--base)}.tag.fast{background:var(--fast)}.tag.oth{background:var(--mute)}
input[type=range]{flex:1;min-width:160px;accent-color:var(--base)}
.note{font-size:.85rem;color:var(--mute);border-left:3px solid var(--gold);padding:6px 12px;margin:14px 0;max-width:90ch}
.figwrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--card)}.figwrap img{display:block;max-width:100%;min-width:900px}
table{border-collapse:collapse;font-size:.85rem;margin-top:8px}th,td{border-bottom:1px solid var(--line);padding:5px 10px;text-align:right}th:first-child,td:first-child{text-align:left}
a{color:var(--base)}
@media (prefers-reduced-motion: reduce){*{transition:none!important}}
"""
FONT='<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Serif:wght@600&family=IBM+Plex+Mono&display=swap">'
def b64(p, mime): return "data:%s;base64,%s" % (mime, base64.b64encode(open(p, "rb").read()).decode())
# ---------- page 1: clips ----------
groups = {}
for it in man: groups.setdefault(it["group"], {}).setdefault(it["task"], []).append(it)
data = {g: {t: [dict(arm=i["arm"], label=i["label"], src=b64(V + "/" + i["file"], "video/mp4"), meta=META.get(i["arm"], ("–","–","–"))) for i in items] for t, items in ts.items()} for g, ts in groups.items()}
p1 = ["<title>specwm Clip Compare</title>", FONT, "<style>", CSS, "</style>",
      "<h1>specwm Clip Compare</h1><p class=sub>Same scene, same commanded camera path (or same StEvo task), every model / arm side by side. Videos are 360 px, 8 fps re-encodes of the evaluation clips; StEvo badges are control-success % / physics-plausible % from the Qwen3-VL judge (85 camera tasks) and gen-only fps on one RTX PRO 6000.</p>",
      "<div class=tabs id=tabs></div><div class=tasks id=tasks></div>",
      "<details id=mtab open><summary style='cursor:pointer;font-weight:600'>StEvo metrics by arm (↑ higher is better; <b>bold</b> = best, <u>underline</u> = second)</summary><div style='overflow-x:auto'><table id=mtable></table></div></details>",
      "<div class=ctrl><button id=play class=act>Play all</button><button id=pause>Pause</button><button id=restart>Restart</button><label><input type=checkbox id=loop checked> loop</label><input type=range id=scrub min=0 max=1000 value=0 aria-label='scrub all clips (fraction of each clip)'><span class=mono id=tval>0.00</span></div>",
      "<div class=grid id=grid></div>",
      "<p class=note>What 'control' means and the tasks where Fast + ours flips it: <a href='https://claude.ai/artifact/1Dsb9JfLrvyL86wywAXdNW'>specwm Control Flips</a>.</p><p class=note>Gold = StEvo camera-lookaway tasks: the camera pans away (~0.75–8 s) and back; the interesting part is whether the object state and layout are still consistent when it returns. Floor scenes: 10 s rollouts, seed 1234; regrounding arms replace the Fast KV for the first 4.5 s with Base-derived context, the C2 arm shows the Fast prefix followed by a restart from one corrected frame. Figures with DINO trajectories for every clip pair are on the companion page <a id=figlink href='https://claude.ai/artifact/8sQg23AxUbKKQ6uvY4KccT'>specwm Trajectory Gallery</a>.</p>",
      "<script>const DATA=", json.dumps(data), ";", """
const MET=%s, LAB=%s; const TASKN=%s; const grid=document.getElementById('grid'), tabs=document.getElementById('tabs'), tasks=document.getElementById('tasks'); let g='stevo', t=null;
function tagOf(a){return a==='base'?'base':(a.startsWith('fast')||a.startsWith('reground')||a.startsWith('anchor')||a.startsWith('draft'))?'fast':'oth'}
function render(){tabs.innerHTML='';for(const k of Object.keys(DATA)){const b=document.createElement('button');b.textContent=k==='stevo'?'StEvo tasks (camera look-away)':'Floor scenes (10 s rollouts)';b.className=k===g?'on':'';b.onclick=()=>{g=k;t=null;render()};tabs.appendChild(b)}
 const ts=Object.keys(DATA[g]); if(!t||!ts.includes(t)) t=ts[0]; tasks.innerHTML='';for(const k of ts){const b=document.createElement('button');b.textContent=TASKN[k]||k;b.className=k===t?'on':'';b.onclick=()=>{t=k;render()};tasks.appendChild(b)}
 grid.innerHTML='';for(const c of DATA[g][t]){const d=document.createElement('div');d.className='card';const tg=tagOf(c.arm);
  d.innerHTML=`<div class=hd><span>${c.label}</span><span class="tag ${tg}">${tg==='base'?'base':tg==='fast'?'fast':'baseline'}</span></div><video muted playsinline preload=auto src="${c.src}"></video><div class=ft><span class=chip>${c.meta[0]}</span><span class=chip>control ${c.meta[1]}</span><span class=chip>physics ${c.meta[2]}</span></div>`;
  grid.appendChild(d)} vids().forEach(v=>{v.loop=document.getElementById('loop').checked}); playAll()}
function vids(){return [...grid.querySelectorAll('video')]}
function playAll(){vids().forEach(v=>{v.play().catch(()=>{})})}
document.getElementById('play').onclick=playAll; document.getElementById('pause').onclick=()=>vids().forEach(v=>v.pause()); document.getElementById('restart').onclick=()=>{vids().forEach(v=>{v.currentTime=0});playAll()};
document.getElementById('loop').onchange=e=>vids().forEach(v=>v.loop=e.target.checked);
const scrub=document.getElementById('scrub'), tval=document.getElementById('tval');
scrub.oninput=()=>{const f=scrub.value/1000;vids().forEach(v=>{if(v.duration)v.currentTime=f*v.duration});tval.textContent=f.toFixed(2)};
setInterval(()=>{const v=vids()[0];if(v&&v.duration&&document.activeElement!==scrub){scrub.value=Math.round(1000*v.currentTime/v.duration);tval.textContent=(v.currentTime/v.duration).toFixed(2)}},250);
(function(){const cols=[["gen fps ↑",0],["control success %% ↑",1],["physics plausible %% ↑",2]];const keys=Object.keys(MET);
 const rank=c=>{const v=keys.map(k=>MET[k][c]).filter(x=>x!==null).sort((a,b)=>b-a);return [v[0],v[1]]};const R=cols.map(([,c])=>rank(c));
 const fmt=(k,c)=>{const v=MET[k][c];if(v===null)return '–';const [b,s2]=R[c];const t=c===0?v.toFixed(2):v.toFixed(1);return v===b?`<b>${t}</b>`:v===s2?`<u>${t}</u>`:t};
 document.getElementById('mtable').innerHTML='<tr><th>arm</th>'+cols.map(([n])=>`<th>${n}</th>`).join('')+'</tr>'+keys.map(k=>`<tr><td>${LAB[k]}</td>`+cols.map(([,c])=>`<td class=mono>${fmt(k,c)}</td>`).join('')+'</tr>').join('')})();
render();
</script>""" % (json.dumps(MET), json.dumps(LAB), json.dumps(TASKN))]
open(W + "/clip_compare.html", "w").write("\n".join(p1)); print("page1 %.1f MB" % (os.path.getsize(W + "/clip_compare.html") / 2**20))
# ---------- page 2: figures ----------
figs = {}
for r in summ: figs.setdefault(r["task"], []).append(dict(arm=r["arm"], label=r["arm_label"], src=b64(FIG + "/" + r["name"] + ".jpg", "image/jpeg"), mean=round(r["mean_divergence"], 3), late=round(r["late_divergence"], 3), jmax=round(r["fast_jump_max"], 3), bjmax=round(r["base_jump_max"], 3), jat=round(r["fast_jump_at_s"], 2)))
p2 = ["<title>specwm Trajectory Gallery</title>", FONT, "<style>", CSS, "</style>",
      "<h1>specwm Trajectory Gallery</h1><p class=sub>Base 70-step vs every model / arm on the same clip, in DINOv2 feature space. Pick a task, then an arm.</p>",
      "<div class=note><b>How to read each figure.</b> Left: both clips' frame features projected with one PCA (light → dark = time; dotted lines join the two models at the same timestamp — the longer the line, the further the compared model has drifted from Base). Middle: black = compared<sub>t</sub> vs Base<sub>t</sub> distance (0.17 = distance between two Base seeds); blue/red = each model's distance from its own first frame (falls back toward 0 when the camera returns to the start). Right: frame-to-frame change — spikes are abrupt object/scene changes (teleports, re-entering a different room); Base is smooth, the fast models spike. Bottom: frames every second; red box = compared model's frame at a top-3 spike; gold border = camera looking away (StEvo camera arms). H3-World uses a keyboard pan of the same look-away; MiniMax-H3 and LongLive have no camera control and are shown under the curtain-occlusion video protocol instead.</div>",
      "<div class=tasks id=tasks></div><div class=tasks id=arms></div><div class=figwrap><img id=fig alt='trajectory figure'></div><h2>Summary for this task (DINO 1−cos, 4 fps; ↓ lower is better for all columns; <b>bold</b> = best, <u>underline</u> = second; Base's own jump given for reference)</h2><div style='overflow-x:auto'><table id=tbl></table></div>",
      "<p class=note>Companion page with the videos: <a id=vidlink href='https://claude.ai/artifact/6E8aFayABD1cpBsfZmbyi3'>specwm Clip Compare</a>.</p>",
      "<script>const FIGS=", json.dumps(figs), ";const TASKN=", json.dumps(TASKN), """;
const tasks=document.getElementById('tasks'),arms=document.getElementById('arms'),fig=document.getElementById('fig'),tbl=document.getElementById('tbl');let t=Object.keys(FIGS)[0],a=null;
function render(){tasks.innerHTML='';for(const k of Object.keys(FIGS)){const b=document.createElement('button');b.textContent=TASKN[k]||k;b.className=k===t?'on':'';b.onclick=()=>{t=k;a=null;render()};tasks.appendChild(b)}
 const L=FIGS[t]; if(!a||!L.find(x=>x.arm===a)) a=L[0].arm; arms.innerHTML='';for(const x of L){const b=document.createElement('button');b.textContent=x.label;b.className=x.arm===a?'on':'';b.onclick=()=>{a=x.arm;render()};arms.appendChild(b)}
 const cur=L.find(x=>x.arm===a); fig.src=cur.src; fig.alt='Base vs '+cur.label+' — '+(TASKN[t]||t);
 const rk=k=>{const v=L.map(x=>x[k]).sort((p,q)=>p-q);return [v[0],v[1]]};const RK={mean:rk('mean'),late:rk('late'),jmax:rk('jmax')};const f=(x,k)=>{const v=x[k],t=v.toFixed(3);return v===RK[k][0]?`<b>${t}</b>`:v===RK[k][1]?`<u>${t}</u>`:t};
 tbl.innerHTML='<tr><th>arm</th><th>mean divergence vs Base ↓</th><th>late-half divergence ↓</th><th>max frame-to-frame jump ↓</th><th>at (s)</th><th>Base max jump (ref)</th></tr>'+L.map(x=>`<tr${x.arm===a?' style="background:var(--chip)"':''}><td>${x.label}</td><td class=mono>${f(x,'mean')}</td><td class=mono>${f(x,'late')}</td><td class=mono>${f(x,'jmax')}</td><td class=mono>${x.jat}</td><td class=mono>${x.bjmax.toFixed(3)}</td></tr>`).join('')}
render();
</script>"""]
open(W + "/traj_gallery.html", "w").write("\n".join(p2)); print("page2 %.1f MB" % (os.path.getsize(W + "/traj_gallery.html") / 2**20))
