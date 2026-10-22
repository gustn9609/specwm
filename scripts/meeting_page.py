#!/usr/bin/env python3
"""회의용 HTML 페이지 생성: timing_s00.md / lowres_sr.md / stevo_results.md 표를 읽어 한 페이지로."""
import re, os, json, html
S="/home/hk28965/specwm"; O=S+"/lingbot-world-v2/specwm_out"; OUT="/tmp/claude-1741758895/-home-hk28965/2af3c383-3218-4bda-a425-8a247f5f536a/scratchpad/art/specwm_meeting.html"
def md_table(path, drop_cols=(), keep=None):
    rows=[l for l in open(path).read().splitlines() if l.startswith("|")]
    if not rows: return ""
    head=[c.strip() for c in rows[0].strip("|").split("|")]; body=[[c.strip() for c in r.strip("|").split("|")] for r in rows[2:]]
    idx=[i for i,h in enumerate(head) if h not in drop_cols]
    if keep: body=[b for b in body if keep(b)]
    th="".join("<th>%s</th>"%html.escape(head[i]) for i in idx)
    trs=[]
    for b in body:
        cls=' class="ref"' if b[0].startswith("*") else ""
        trs.append("<tr%s>"%cls+"".join("<td>%s</td>"%html.escape(b[i].strip("*")) if i<len(b) else "<td></td>" for i in idx)+"</tr>")
    return "<div class='tw'><table><thead><tr>%s</tr></thead><tbody>%s</tbody></table></div>"%(th,"".join(trs))
T=O+"/timing_s00"
def wc_rows():
    d=json.load(open(T+"/meeting/dino_worldcache.json")); g=lambda n:(json.load(open(T+"/%s/revisit.json"%n)) if os.path.exists(T+"/%s/revisit.json"%n) else None)
    def row(label,res,dirn,bf,dino,visual):
        r=g(dirn); gen="%.0f"%r["gen_s"] if r else "TBD"; sp="%.2fx"%(bf/r["gen_s"]) if r else "TBD"; sk=("%.0f%%"%(100*r["worldcache"]["low"]["skip_rate"])) if r and r.get("worldcache") else ("–" if r else "TBD")
        return (label,res,gen,sp,sk,dino,visual)
    R=[("Base bf16 (reference)","480x832","2787","1.00x","–","seed floor 0.17-0.18","clean"),
       row("WorldCache reimpl k=4, τ₀ 0.08","480x832","base_wc4_480x832",2787,"0.66 / 0.70","mosaic textures"),
       ("Base bf16","360x512","969","1.00x","–","(res-only vs 480p: 0.28)","clean"),
       row("WorldCache official-config port k=3 (thr 0.06, α 0.2, OSI, saliency, decay)","360x512","base_wcoff3_360x512",969,"0.04 / 0.05","clean, indistinguishable"),
       row("official-config, thr 0.10","360x512","base_wcoff3_thr10_360x512",969,"0.06 / 0.07","clean"),
       row("official-config, thr 0.15","360x512","base_wcoff3_thr15_360x512",969,"0.08 / 0.10","clean"),
       row("official-config, thr 0.10, ret 0.1 (best)","360x512","base_wcoff3_thr10_ret10_360x512",969,"0.07 / 0.08","clean"),
       row("official-config, probe 2, thr 0.10","360x512","base_wcoff2_thr10_360x512",969,"0.06 / 0.07","clean"),
       row("WorldCache reimpl k=4, τ₀ 0.08","360x512","base_wc4_360x512",969,"0.68 / 0.68","mosaic"),
       row("official port mis-set (α 5 + AdUC on)","360x512","base_wcoff3_aduc_a5_360x512",969,"0.91 / 0.92","painterly blobs"),
       ("v2-bid bf16","360x512","960","–","–","vs v1 Base same seed: 0.37 / 0.42","different model"),
       row("v2-bid WorldCache reimpl k=4","480x832","v2bid_wc4_480x832",2760,"0.76 / 0.78","mosaic")]
    return "<div class='tw'><table><thead><tr><th>config</th><th>res</th><th>gen (s)</th><th>speedup</th><th>skip (low-noise expert)</th><th>DINO 1−cos vs bf16 (all / late)</th><th>visual</th></tr></thead><tbody>"+"".join("<tr>"+"".join("<td>%s</td>"%html.escape(str(c)) for c in r)+"</tr>" for r in R)+"</tbody></table></div>"
timing=md_table(T+"/timing_s00.md")
lowres=md_table(O+"/lowres_sr/lowres_sr.md")
stevo=md_table(O+"/stevo/stevo_results.md")
fewstep=md_table(O+"/fewstep_s00/fewstep.md") if os.path.exists(O+"/fewstep_s00/fewstep.md") else ""
import re as _re
def md_doc(path):
    txt=open(path).read(); out=[]; buf=[]
    def flush():
        if buf: out.append(md_table_lines(buf)); buf.clear()
    for line in txt.splitlines():
        if line.startswith("|"): buf.append(line)
        else:
            flush()
            if line.startswith("## "): out.append("<h3>%s</h3>"%html.escape(line[3:]))
            elif line.startswith("# "): pass
            elif line.strip(): out.append("<p>%s</p>"%html.escape(line))
    flush(); return "\n".join(out)
def md_table_lines(rows):
    head=[c.strip() for c in rows[0].strip("|").split("|")]; body=[[c.strip() for c in r.strip("|").split("|")] for r in rows[2:]]
    th="".join("<th>%s</th>"%html.escape(h) for h in head); trs="".join("<tr>"+"".join("<td>%s</td>"%html.escape(c.replace("**","")) for c in b)+"</tr>" for b in body)
    return "<div class='tw'><table><thead><tr>%s</tr></thead><tbody>%s</tbody></table></div>"%(th,trs)
followup=md_doc(T+"/meeting/FOLLOWUP_0917.md") if os.path.exists(T+"/meeting/FOLLOWUP_0917.md") else ""
quant=md_table(O+"/quant_s00/quant.md") if os.path.exists(O+"/quant_s00/quant.md") else "<p>running</p>"
page=f"""<title>specwm Sep 15–17 Results</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--bg:#F2F4F1;--surface:#FFFFFF;--ink:#1A1F1C;--muted:#5B655F;--line:#D5DBD6;--accent:#1F6F78;--accent-ink:#134B52;--warn:#9A4A12;--good:#2F6B3A;--tile:#E7ECE9}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#131715;--surface:#1B201D;--ink:#E6EAE6;--muted:#9AA59E;--line:#2E3733;--accent:#63B7BF;--accent-ink:#A9DEE2;--warn:#E09A5A;--good:#8CCB97;--tile:#222926}}}}
:root[data-theme="dark"]{{--bg:#131715;--surface:#1B201D;--ink:#E6EAE6;--muted:#9AA59E;--line:#2E3733;--accent:#63B7BF;--accent-ink:#A9DEE2;--warn:#E09A5A;--good:#8CCB97;--tile:#222926}}
body{{background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:15px;line-height:1.55;padding-block:32px 64px;padding-inline:clamp(16px,4vw,40px)}}
main{{max-width:1080px;margin:0 auto}}
h1{{font-family:"Bricolage Grotesque","IBM Plex Sans",sans-serif;font-weight:700;font-size:clamp(28px,4vw,40px);line-height:1.1;margin:0 0 6px;text-wrap:balance;letter-spacing:-0.01em}}
h2{{font-family:"Bricolage Grotesque","IBM Plex Sans",sans-serif;font-weight:700;font-size:22px;margin:44px 0 10px;text-wrap:balance}}
h3{{font-size:15px;font-weight:600;margin:22px 0 6px;color:var(--accent-ink)}}
.eyebrow{{font-family:"IBM Plex Mono",monospace;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}}
p,li{{max-width:70ch}} p.lede{{font-size:17px;color:var(--muted);margin:0 0 24px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:20px 0 8px}}
.tile{{background:var(--tile);border-radius:6px;padding:14px 16px}} .tile .n{{font-family:"IBM Plex Mono",monospace;font-size:26px;font-weight:500;line-height:1.1}} .tile .l{{font-size:12.5px;color:var(--muted);margin-top:6px}}
.tw{{overflow-x:auto;border:1px solid var(--line);border-radius:6px;background:var(--surface);margin:10px 0 6px}}
table{{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}} th{{text-align:left;font-weight:600;font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);padding:9px 10px;border-bottom:1px solid var(--line);white-space:nowrap}} td{{padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}} tr:last-child td{{border-bottom:0}} tr.ref td{{color:var(--muted);font-style:italic}}
.note{{border-left:3px solid var(--accent);padding:6px 14px;margin:12px 0;color:var(--ink);background:var(--surface);max-width:none}} .note b{{color:var(--accent-ink)}}
.warn{{color:var(--warn)}} .good{{color:var(--good)}}
code{{font-family:"IBM Plex Mono",monospace;font-size:12.5px;background:var(--tile);padding:1px 5px;border-radius:3px}}
ul.paths{{font-family:"IBM Plex Mono",monospace;font-size:12.5px;line-height:1.7;padding-left:18px;max-width:none}}
.two{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}}
</style>
<main>
<div class="eyebrow">specwm · spi0 · Sep 15–17, 2026 · auto-updated draft</div>
<h1>S00 timing matrix, WorldCache, low-res rollout, StEvo-Bench</h1>
<p class="lede">All generation timings include VAE decode. S00 = Great Wall, seed 9999, camera path rt2_action_00, prompt p_00. One RTX PRO 6000 Blackwell 96 GB per run unless noted.</p>
<div class="tiles">
<div class="tile"><div class="n">2.1× → 2.8×</div><div class="l">WorldCache official-config port with the paper's step-index progress: 2.13× at defaults (DINO 0.13), 2.77× at thr 0.10/ret 0.1 (DINO 0.17 = seed floor); high-noise expert now skips 35–62%</div></div>
<div class="tile"><div class="n">65% ± 29%</div><div class="l">KV-rewrite recovery reproduced over 3 scenes × 3 seeds (was 63% on one seed); full-target SDEdit laundering works too (73%) → single-checkpoint recipe</div></div>
<div class="tile"><div class="n">2.2 → 8.4 fps</div><div class="l">Fast 14B, 480×832 → 256×464; low-res changes the trajectory, SR can't recover it</div></div>
<div class="tile"><div class="n">2.4 % / 8.2 %</div><div class="l">StEvo state-progress: LingBot Fast / H3-World (paper LingBot 3.4 %, Veo 3 17.4 %)</div></div>
</div>

<h2>1. WorldCache on LingBot Base (70-step UniPC, CFG 5, MoE 2-expert)</h2>
<p class="note"><b>What was run:</b> not the official script (it replaces WAN2.1's <code>forward</code> and cannot take the camera-plucker/MoE Base forward); the official decision logic (probe 3 blocks → accumulated drift → threshold → residual reuse + OSI) was ported into our block loop, with the official WAN i2v-14B flags (<code>motion_sensitivity 0.2, osi, saliency, dynamic_decay</code>, no AdUC; 1-D flow warp omitted).</p>
{wc_rows()}
<p>Speedup is below the paper's 2.1–2.3× because the high-noise expert (early steps, structure formation) never skips and 70-step UniPC accumulates drift past the threshold more often; the faithful port keeps quality, the aggressive reimpl trades it away. Raising the drift threshold and shortening the no-cache warm-up (ret 0.2 → 0.1) pushes the faithful port to 1.48× at DINO 0.07, still under the seed floor (probe depth 2 vs 3 makes no difference: 1.27×, DINO 0.06). Caveat: the port uses timestep progress (1 − t/1000) for the no-cache warm-up and decay where the official code uses step index; with shift 10 this keeps the whole high-noise expert phase (t ≥ 947, 26 of 70 steps) inside the warm-up, so it never skips. <b>Update (Sep 17):</b> a step-index port (paper definition) lifts this to 2.13× (defaults) / 2.77× (thr 0.10, ret 0.1) at DINO 0.13 / 0.17 — see §5.</p>

<h2>2. Low-res rollout + super-resolution (same S00 / seed / camera / prompt)</h2>
{lowres}
<p>Lowering rollout resolution changes the generated <em>trajectory</em>, not just sharpness: DINO 0.43–0.60 vs Base (seed floor 0.17), and Lanczos vs EDSR ×2/×4 move the number by &lt; 0.02. The 1.3B model is farther from Base (0.54–0.68) at every resolution. Speed gain: 70 s → 26 s / 19 s per 161 frames. Below 256×464 (208×320 … 112×160) the rollout collapses outright (DINO 0.83–0.93 vs Base for both 14B and 1.3B): with ≤ 13×20 latent tokens the camera conditioning and the scene no longer survive, so the useful floor is 256×464.</p>

<h2>2b. Fast 14B few-step rollout (drop denoise nodes, no retraining)</h2>
{fewstep}
<p>1-step = x0 predicted straight from σ=1.0; 2/3-step re-noise to the next kept node. Time does not scale 1:4 because VAE decode (8.6 s) and per-chunk KV commit are fixed costs. DINO vs Base stays in the 0.30–0.40 band for every step count (the 4-step Fast itself is 0.36 from Base), so the metric barely separates them — but the contact sheets do: 1-step collapses scene content (forested hills vanish, the wall turns into a flat grey slab, blue cast), 2-step keeps the scene with softer texture, 3-step ≈ 4-step. Sheets: <code>fewstep_s00/meeting/sheet_fast14b_nodes{{0,03,02,013}}.png</code>, side-by-side <code>S00_fast_1step_vs_2step_vs_4step.mp4</code>.</p>

<h2>2c. Fast 14B quantization ablation (same S00 clip)</h2>
{quant}

<h2>3. StEvo-Bench (judge: Qwen3-VL-32B local, 3-vote; paper judge: Gemini 3.1 Pro)</h2>
{stevo}
<p>N &lt; 85 rows are partial (generation still running). Released eval code merges physics + coherence into one physics judge. Caveat: the Qwen judge sometimes flags a frozen scene as a physics violation, so physics numbers run lower than Gemini's.</p>

<h2>4. Full timing matrix (S00, seed 9999)</h2>
{timing}

<h2>5. Follow-ups (Sep 17): draft landscape, LongLive, KV-rewrite reproduction, low-res Base, stale anchor, step-index WorldCache</h2>
{followup}

<h2>6. Files</h2>
<div class="two"><div><h3>WorldCache videos</h3><ul class="paths">
<li>timing_s00/base_wcoff3_360x512/video.mp4 (official-config)</li><li>timing_s00/base_bf16_360x512/video.mp4</li><li>timing_s00/base_wc4_360x512/video.mp4 (reimpl, mosaic)</li><li>timing_s00/base_wc4_480x832/video.mp4</li><li>timing_s00/meeting/S00_base_bf16_vs_wc4_vs_wcofficial3_360x512.mp4 (side-by-side)</li><li>timing_s00/meeting/sheet_*.png · dino_worldcache.json</li></ul></div>
<div><h3>Low-res / SR · StEvo</h3><ul class="paths"><li>lowres_sr/fast14b_{{399360,184320,122880}}/S00_s9999.mp4</li><li>lowres_sr/v2_1p3b_*/S00_s9999.mp4 · lowres_sr/sr/*.mp4</li><li>stevo/&lt;model&gt;/&lt;task&gt;.mp4 · stevo/runs/&lt;model&gt;/per_task/</li><li>stevo/stevo_results.md · timing_s00/timing_s00.md</li></ul></div></div>
<p class="eyebrow" style="margin-top:36px">root: /home/hk28965/specwm/lingbot-world-v2/specwm_out/</p>
</main>
"""
open(OUT,"w").write(page); print(OUT, len(page))
