"""[10/14] Interactive WASD web demo of the Fast 14B: one server per arm (GPU), the browser page shows one or several arms side by side
driven by the SAME key state. Generation loop = specwm/act_spec2.py (frame-granular K/V ring, per-frame-range model calls, optional
action speculation with the constant-velocity hypothesis + adaptive 2-frame units after a miss). Frames are streamed as JPEG over a
WebSocket as soon as each unit is decoded; the page shows server fps, action->frame latency of the last unit and hit / miss.
Keys: W/S forward/back, A/D strafe, arrows (or J/L, I/K) yaw / pitch. One action per generation unit (the key state when the unit's
action is consumed), in the normalized units of the model (|t| 1 = StEvo key-press speed, yaw 8 deg per latent frame by default).
Conventions checked on data/stevo/actions/s-31_w-32 (10/14): camera +z = forward (W), +x = right, y down; act_stream.se3 puts speed on -z,
so W sends speed -1. The scene prompt drops sentences that mention the camera (the StEvo texts prescribe a camera path).
usage: CUDA_VISIBLE_DEVICES=0 python -m specwm.web_demo --port 7861 --label ours --mode uniform:2 --area 270336 --commit partial1 --spec 1
       CUDA_VISIBLE_DEVICES=1 python -m specwm.web_demo --port 7862 --label teacher --mode uniform:4 --spec 0
then open http://localhost:7861/?servers=7861,7862 (through an ssh tunnel: ssh -L 7861:localhost:7861 -L 7862:localhost:7862 <host>)"""
import argparse, os, sys, json, time, math, threading, collections, asyncio, numpy as np, torch, cv2
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=7861); ap.add_argument("--label", default="ours"); ap.add_argument("--mode", default="uniform:2"); ap.add_argument("--area", type=int, default=270336)
ap.add_argument("--commit", default="partial1", choices=["separate", "partial1"]); ap.add_argument("--spec", type=int, default=1, help="1: action speculation (K = 1, constant velocity) + adaptive 2-frame units after a miss")
ap.add_argument("--task", default="easy_lake"); ap.add_argument("--max-frames", type=int, default=960, help="latent frames per session before an automatic restart (RoPE table = 1024)"); ap.add_argument("--tol-rot", type=float, default=3.0); ap.add_argument("--tol-trans", type=float, default=0.5)
ap.add_argument("--jpeg", type=int, default=85); ap.add_argument("--prompt", default="scene", choices=["scene", "none"], help="scene = the task text without camera sentences; none = empty prompt"); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
from specwm.act_stream import se3
from wan.utils.cam_utils import get_Ks_transformed, get_plucker_embeddings
from einops import rearrange
from aiohttp import web
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"; dev = "cuda:0"; sync = torch.cuda.synchronize
B = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True, ring_frame=True); FP.fp8_swap(B.target_model, True, keep=False)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 512; _dyn.config.accumulated_recompile_limit = 8192
for blk in B.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=True)
B.pipe.text_encoder.model.to(dev); _te = B.pipe.text_encoder
class _TE:
    def __init__(s, te): s.te = te; s.model = te.model
    def __call__(s, texts, device): return s.te(texts, dev)
B.pipe.text_encoder = _TE(_te); sig = [float(s) for s in B.node_sigmas()]; MODE, MVAL = a.mode.split(":"); K_UNI = int(MVAL); VAE = B.pipe.vae; VM = VAE.model; CS = 3
KS = np.array([[502.9, 503.1, 415.8, 239.8]], dtype=np.float32)
# ---------------- shared state (web <-> generation thread)
KEYS = set(); RATES = dict(yaw=8.0, pitch=3.0, speed=1.0, strafe=0.5); CMD = collections.deque(); FRAMES = collections.deque(); CLIENTS = set(); STATUS = dict(label=a.label, mode=a.mode, area=a.area, spec=a.spec, task=a.task, f=0, hits=0, units=0, state="loading")
def key_action():
    """key state -> (yaw deg/frame, pitch, speed, strafe) in the model's normalized units"""
    y = (RATES["yaw"] if ("ArrowRight" in KEYS or "KeyL" in KEYS) else 0.0) - (RATES["yaw"] if ("ArrowLeft" in KEYS or "KeyJ" in KEYS) else 0.0)
    p = (RATES["pitch"] if ("ArrowUp" in KEYS or "KeyI" in KEYS) else 0.0) - (RATES["pitch"] if ("ArrowDown" in KEYS or "KeyK" in KEYS) else 0.0)   # rot('x', +) tilts the forward axis to -y = up (OpenCV y down)
    sp = (RATES["speed"] if "KeyS" in KEYS else 0.0) - (RATES["speed"] if "KeyW" in KEYS else 0.0)   # se3 puts t_z = -speed; StEvo poses: W key = +z (forward), S key = -z
    st = (RATES["strafe"] if "KeyD" in KEYS else 0.0) - (RATES["strafe"] if "KeyA" in KEYS else 0.0)
    return (y, p, sp, st)
# ---------------- model helpers (as act_spec2)
CTX = {}
def emb_frames(rel, Ks, h, w, lat_h, lat_w):
    f = rel.shape[0]; e = get_plucker_embeddings(rel, Ks, h, w); e = rearrange(e, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)', c1=int(h // lat_h), c2=int(w // lat_w))[None, ...]
    return rearrange(e, 'b (f h w) c -> b c f h w', f=f, h=lat_h, w=lat_w).to(B.param_dtype)
def rot_angle(Ra, Rb):
    cosv = ((Ra.transpose(-1, -2) @ Rb).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2; return torch.rad2deg(torch.acos(cosv.clamp(-1, 1)))
def act_err(h, tr):
    dt = float((h[:, :3, 3] - tr[:, :3, 3]).norm(dim=1).sum()); den = max(float(h[:, :3, 3].norm(dim=1).sum()), float(tr[:, :3, 3].norm(dim=1).sum()), 1e-6)
    return float(rot_angle(h[:, :3, :3], tr[:, :3, :3]).sum()), (dt / den if dt > 1e-6 else 0.0)
def y_frames(f0, m):
    Y = CTX["Y"]; n = Y.shape[1]
    if f0 + m <= n: return Y[:, f0:f0 + m]
    return torch.cat([Y[:, f0:n], Y[:, n - 1:n].expand(-1, f0 + m - n, -1, -1)], 1) if f0 < n else Y[:, n - 1:n].expand(-1, m, -1, -1)
def call(z, t_val, f0, m, emb):
    fs = B._frame_seqlen; C = B._caches["target"]
    kw = dict(context=[B._ctx[0]], seq_len=m * fs, y=[y_frames(f0, m)], dit_cond_dict={"c2ws_plucker_emb": emb.chunk(1, dim=0)}, kv_cache=C["kv"], crossattn_cache=C["cross"], current_start=f0 * fs, max_attention_size=B._kv_size, frame_seqlen=fs)
    with torch.amp.autocast("cuda", dtype=B.param_dtype): out = B.target_model(x=[z.to(B.param_dtype)], t=torch.full((1,), float(t_val), device=dev), cross_attn_first_call=not B._cross_init["target"], **kw)[0]
    B._cross_init["target"] = True; return out.float()
def generate(f0, m, rel, g):
    """denoise + commit frames f0..f0+m-1 with relative poses rel [m,4,4]; returns x0"""
    emb = emb_frames(rel, CTX["Ks"][:m], CTX["h"], CTX["w"], CTX["lat_h"], CTX["lat_w"]); z = torch.randn((16, m, CTX["lat_h"], CTX["lat_w"]), generator=g).to(dev)
    for i in range(K_UNI):
        x = z - sig[i] * call(z, B._timesteps[i], f0, m, emb)
        if i < K_UNI - 1: z = (1 - sig[i + 1]) * x + sig[i + 1] * torch.randn(x.shape, generator=g).to(dev)
    if a.commit == "separate" or m == 1: call(x, 0.0, f0, m, emb)
    else:
        fs = B._frame_seqlen; L = m * fs; FP.ACTIVE["a"] = dict(idx=torch.arange(L - fs, L, device=dev), n=fs, L=L)
        try: call(x, 0.0, f0, m, emb)
        finally: FP.ACTIVE["a"] = None
    return x
def kv_pos(f0, m): kv = B._caches["target"]["kv"]; return FP.ring_pos(f0, m, B._frame_seqlen, kv[0]["k"].shape[1], 6, dev)[0]
def kv_snapshot(f0, m): p = kv_pos(f0, m); return [(l["k"][:, p].clone(), l["v"][:, p].clone()) for l in B._caches["target"]["kv"]]
def kv_restore(f0, m, snap):
    p = kv_pos(f0, m)
    for l, (k, v) in zip(B._caches["target"]["kv"], snap): l["k"][:, p] = k; l["v"][:, p] = v
def vae_snapshot(): return [t.clone() if torch.is_tensor(t) else t for t in VM._feat_map]
def vae_restore(snap): VM._feat_map = [t.clone() if torch.is_tensor(t) else t for t in snap]
@torch.no_grad()
def decode(z):
    with torch.amp.autocast("cuda", dtype=VAE.dtype):
        sc = VAE.scale; zz = z.unsqueeze(0).to(VAE.dtype) / sc[1].view(1, -1, 1, 1, 1) + sc[0].view(1, -1, 1, 1, 1); x = VM.conv2(zz); outs = []
        for i in range(x.shape[2]):
            VM._conv_idx = [0]; o = VM.decoder(x[:, :, i:i + 1], feat_cache=VM._feat_map, feat_idx=VM._conv_idx).float().clamp_(-1, 1); outs.append(((o[0] + 1) * 127.5).to(torch.uint8).permute(1, 2, 3, 0))
    fr = torch.cat(outs, 0).cpu().numpy(); sync(); return fr
@torch.no_grad()
def scene_prompt(T):
    if PROMPT["mode"] == "none": return ""
    import re; sents = re.split(r"(?<=[.!?])\s+", (T.get("camera_WM") or T.get("video_WM") or "").strip()); return " ".join(x for x in sents if "camera" not in x.lower())
PROMPT = dict(mode=a.prompt)
def prepare_scene(task):
    T = ALL[task]; pr = scene_prompt(T); STATUS["prompt"] = pr; print("[demo] scene %s prompt: %r" % (task, pr), flush=True)
    cond = B.prepare(T["image"], pr, T["action"], frame_num=161, max_area=a.area, chunk_size=CS); B.reset_caches()
    lat_h, lat_w = B._chunk_shape[2], B._chunk_shape[3]; h, w = lat_h * 8, lat_w * 8
    Ks = get_Ks_transformed(torch.from_numpy(KS).float(), height_org=480, width_org=832, height_resize=h, width_resize=w, height_final=h, width_final=w)[0].repeat(4, 1).to(dev)
    CTX.update(Y=torch.cat(list(B._y_chunks), 1), Ks=Ks, h=h, w=w, lat_h=lat_h, lat_w=lat_w); VM.clear_cache()
def push_frames(fr, meta):
    for i, f in enumerate(fr):
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, a.jpeg]); FRAMES.append((dict(meta, i=i, n=len(fr)), buf.tobytes()))
def gen_loop():
    task = a.task; seed = 7
    while True:
        STATUS.update(state="preparing", task=task); prepare_scene(task); g = torch.Generator().manual_seed(seed); f = 0; mode3 = True; streak = 0; last = None; hits = 0; units = 0; STATUS.update(f=0, hits=0, units=0, state="running")
        while f < a.max_frames:
            if CMD:
                c = CMD.popleft()
                if c.get("cmd") == "reset": task = c.get("task", task); seed += 1; break
            if not CLIENTS: time.sleep(0.05); continue
            m = CS if mode3 else 2; branch = None; t0 = time.time()
            if a.spec and last is not None:
                hyp = torch.from_numpy(se3(*last)).float().to(dev).expand(m, 4, 4).clone(); vs = vae_snapshot(); pre = kv_snapshot(f, m)
                x0 = generate(f, m, hyp, g); kv = kv_snapshot(f, m); fr = decode(x0); branch = dict(hyp=hyp, kv=kv, fr=fr, vae=VM._feat_map, pre=pre); vae_restore(vs)
            t_req = time.time(); act = key_action(); tr = torch.from_numpy(se3(*act)).float().to(dev).expand(m, 4, 4).clone(); hit = False
            if f == 0: tr[0] = torch.eye(4, device=dev)   # frame 0 = conditioning image: its relative pose must be the identity (as prepare() sets it); a non-identity pose here makes the model ignore the camera for the whole session (10/14 debug)
            if branch is not None:
                re, te = act_err(branch["hyp"], tr)
                if re <= a.tol_rot * m / 3.0 and te <= a.tol_trans: kv_restore(f, m, branch["kv"]); VM._feat_map = branch["vae"]; fr = branch["fr"]; hit = True
            if not hit:
                if a.spec and mode3 and last is not None: kv_restore(f, m, branch["pre"]); m = 2; mode3 = False; tr = tr[:m]
                x0 = generate(f, m, tr, g); fr = decode(x0)
            t_ready = time.time(); hits += hit; units += 1; streak = streak + 1 if hit else 0
            if not mode3 and streak >= 3: mode3 = True
            push_frames(fr, dict(f=f, m=m, hit=hit, lat_ms=round(1000 * (t_ready - t_req)), unit_ms=round(1000 * (t_ready - t0)), act=act, mode=3 if mode3 else 2, hits=hits, units=units, t=t_ready))
            STATUS.update(f=f + m, hits=hits, units=units); last = act; f += m
        STATUS.update(state="restarting")
# ---------------- web
HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>world model demo</title>
<style>body{margin:0;background:#111;color:#ddd;font-family:sans-serif}#top{padding:8px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
.panel{display:inline-block;margin:6px;vertical-align:top}.panel canvas{background:#000;width:704px;height:384px;display:block}.stats{font-size:13px;padding:4px 0;white-space:pre}
select,button,input{font-size:14px}</style></head><body>
<div id="top"><b>WASD = move/strafe, arrows (or J/L, I/K) = look. Click the page first.</b>
<label>scene <select id="task"></select></label><button id="reset">restart both</button>
<label>prompt <select id="prompt"><option value="scene">scene text (camera sentences removed)</option><option value="none">none</option></select></label>
<label>yaw deg/frame <input id="yaw" type="number" value="8" step="1" style="width:50px"></label><label>speed <input id="speed" type="number" value="1" step="0.25" style="width:50px"></label>
<span id="keys"></span></div><div id="panels"></div>
<script>
const q=new URLSearchParams(location.search);const servers=(q.get('servers')||location.port).split(',');const keys=new Set();const conns=[];
function sendAll(o){const s=JSON.stringify(o);for(const c of conns)if(c.ws.readyState===1)c.ws.send(s);}
window.addEventListener('keydown',e=>{if(['INPUT','SELECT'].includes(document.activeElement.tagName))return;if(!keys.has(e.code)){keys.add(e.code);sendAll({keys:[...keys]});}e.preventDefault();});
window.addEventListener('keyup',e=>{keys.delete(e.code);sendAll({keys:[...keys]});});
setInterval(()=>{sendAll({keys:[...keys]});document.getElementById('keys').textContent=[...keys].join(' ');},200);
document.getElementById('reset').onclick=()=>sendAll({cmd:'reset',task:document.getElementById('task').value,prompt:document.getElementById('prompt').value});
document.getElementById('yaw').onchange=e=>sendAll({rates:{yaw:+e.target.value}});document.getElementById('speed').onchange=e=>sendAll({rates:{speed:+e.target.value}});
for(const p of servers){const d=document.createElement('div');d.className='panel';d.innerHTML='<div class="stats" id="st'+p+'">connecting '+p+'</div><canvas id="cv'+p+'"></canvas>';document.getElementById('panels').appendChild(d);
 const cv=document.getElementById('cv'+p),ctx=cv.getContext('2d');const st=document.getElementById('st'+p);const queue=[];let recv=[];let info={};
 const ws=new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.hostname+':'+p+'/ws');ws.binaryType='arraybuffer';conns.push({ws});
 ws.onmessage=ev=>{if(typeof ev.data==='string'){const j=JSON.parse(ev.data);if(j.tasks){const s=document.getElementById('task');if(!s.options.length){for(const t of j.tasks){const o=document.createElement('option');o.value=t;o.textContent=t;s.appendChild(o);}s.value=j.task;}}info=Object.assign(info,j);return;}
  const dv=new DataView(ev.data);const n=dv.getUint32(0);const meta=JSON.parse(new TextDecoder().decode(new Uint8Array(ev.data,4,n)));const blob=new Blob([new Uint8Array(ev.data,4+n)],{type:'image/jpeg'});
  createImageBitmap(blob).then(bm=>{queue.push({bm,meta});recv.push(performance.now());});};
 function draw(){const now=performance.now();recv=recv.filter(t=>now-t<2000);if(queue.length){if(queue.length>24)queue.splice(0,queue.length-24);const {bm,meta}=queue.shift();if(cv.width!==bm.width){cv.width=bm.width;cv.height=bm.height;}ctx.drawImage(bm,0,0);
   st.textContent=(info.label||p)+'  '+(info.mode||'')+' area '+(info.area||'')+(info.spec?'  + action speculation':'')+'\nserver fps '+(recv.length/2).toFixed(1)+'   action->frame of last unit '+meta.lat_ms+' ms   unit '+meta.unit_ms+' ms   '+(meta.hit?'HIT':'miss')+'  hits '+meta.hits+'/'+meta.units+'   unit '+meta.m+' frames   latent frame '+meta.f+'   queued '+queue.length+'\\naction used (yaw, pitch, speed, strafe) '+JSON.stringify(meta.act)+'   mode '+meta.mode+'-frame units';}
  setTimeout(draw,1000/16);} draw();}
</script></body></html>"""
async def index(req): return web.Response(text=HTML, content_type="text/html")
async def ws_handler(req):
    ws = web.WebSocketResponse(max_msg_size=16 * 2**20); await ws.prepare(req); CLIENTS.add(ws); await ws.send_str(json.dumps(dict(STATUS, tasks=sorted(ALL))))
    try:
        async for msg in ws:
            if msg.type != web.WSMsgType.TEXT: continue
            j = json.loads(msg.data)
            if "keys" in j: KEYS.clear(); KEYS.update(j["keys"])
            if "rates" in j: RATES.update({k: float(v) for k, v in j["rates"].items()})
            if j.get("cmd") == "reset": PROMPT["mode"] = j.get("prompt", PROMPT["mode"]); CMD.append(j)
    finally: CLIENTS.discard(ws)
    return ws
async def broadcaster(app):
    while True:
        if FRAMES and CLIENTS:
            meta, jpg = FRAMES.popleft(); hdr = json.dumps(meta).encode(); msg = len(hdr).to_bytes(4, "big") + hdr + jpg
            for ws in list(CLIENTS):
                try: await ws.send_bytes(msg)
                except Exception: CLIENTS.discard(ws)
        else: await asyncio.sleep(0.004)
async def on_start(app): app["bc"] = asyncio.create_task(broadcaster(app))
print("[demo] warming up", flush=True); prepare_scene(a.task); g = torch.Generator().manual_seed(1)
with torch.no_grad():
    f = 0
    for m in (3, 2, 3, 2, 3):
        x = generate(f, m, torch.eye(4, device=dev).expand(m, 4, 4).clone(), g); decode(x); f += m
print("[demo] ready on port %d (%s)" % (a.port, a.label), flush=True)
threading.Thread(target=gen_loop, daemon=True).start()
app = web.Application(); app.router.add_get("/", index); app.router.add_get("/ws", ws_handler); app.on_startup.append(on_start); web.run_app(app, host="0.0.0.0", port=a.port, print=None)
