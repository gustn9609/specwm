"""Build the ablation HTML page from ablation.json (+ prompt texts). usage: python scripts/ablation_page.py ABLDIR OUT.html"""
import sys, json, html, numpy as np
D, OUT = sys.argv[1], sys.argv[2]
J = json.load(open(D + "/ablation.json")); P = {t["id"]: t.get("camera_WM", "") for t in json.load(open("/home/hk28965/specwm/data/stevo/tasks_camera.json"))}
ORDER = ["easy_urban_rain", "easy_stonehenge_sunset", "domino_falling", "pouring_water_into_cup", "easy_lake"]
NAMES = {"easy_lake": "Lake Wanaka", "easy_urban_rain": "Urban alley in rain", "easy_stonehenge_sunset": "Stonehenge at sunset",
         "domino_falling": "Domino row", "pouring_water_into_cup": "Pouring water"}
ADDS = ["Original LingBot v2 14B causal-fast: 4 sampling steps, bf16, T5 on CPU, full VAE encode, fp32 decode after generation",
        "Time embedding computed once, attention reads the KV cache in place (no copy), RoPE table cached and in fp32",
        "torch.compile on every transformer block (one graph per KV-window length)",
        "FP8 (e4m3) weights and activations for every linear layer in the blocks",
        "2 sampling steps instead of 4 (timesteps 0 and 500)",
        "T5 on the GPU, VAE encode of the first 14 latent frames + cached tail, bf16 VAE decode streamed per chunk"]
QUAL = ["reference", "identical math", "≈ identical", "critic AUC 0.42 vs original (12 clips)", "critic AUC 0.70; blind severe-clean 9/24 vs 3/24, quality 3.79 vs 3.68 (24 clips)",
        "critic AUC 0.66 (12 clips); decode PSNR 54 dB"]
steady = [t for t in J if t not in ("easy_lake", "domino_falling")]
M = [{k: float(np.mean([J[t][s][k] or 0 for t in steady])) for k in ("prepare_s", "denoise_s", "commit_s", "decode_s", "e2e_s", "gen_s")} for s in range(6)]
base = M[0]["e2e_s"]; mx = max(m["e2e_s"] for m in M)
def bar(m, streamed):
    seg = [("prepare", m["prepare_s"]), ("denoise", m["denoise_s"]), ("commit", m["commit_s"]), ("decode", m["decode_s"])]
    other = max(0.0, m["e2e_s"] - sum(v for _, v in seg)); seg.append(("other", other))
    return "".join('<span class="seg %s" style="width:%.2f%%" title="%s %.1f s"></span>' % (k, 100 * v / mx, k, v) for k, v in seg if v > 0.05)
rows = []
for s in range(6):
    m = M[s]; st = J[steady[0]][s]["stage"]
    rows.append('<tr><th scope="row"><span class="code">%s</span><span class="add">%s</span></th><td class="num">%.1f</td><td class="num">%.1f</td><td class="num">%.1f</td><td class="num">%s</td>'
                '<td class="num strong">%.1f</td><td class="num">%.2f×</td><td class="q">%s</td></tr>' % (st.split(" ")[0], html.escape(ADDS[s]), m["prepare_s"], m["denoise_s"], m["commit_s"],
                "streamed" if s == 5 else "%.1f" % m["decode_s"], m["e2e_s"], base / m["e2e_s"], html.escape(QUAL[s])))
bars = "".join('<div class="brow"><span class="blab code">%s</span><div class="track">%s</div><span class="bval num">%.1f s</span></div>' % (J[steady[0]][s]["stage"].split(" ")[0], bar(M[s], s == 5), M[s]["e2e_s"]) for s in range(6))
secs = []
for t in ORDER:
    cells = "".join('<li><span class="code">%s</span><span class="num">%.1f s%s</span></li>' % (r["stage"].split(" ")[0], r["e2e_s"], " *" if r.get("includes_warmup") else "") for r in J[t])
    note = '<p class="fine">* includes the one-time torch.compile warm-up: this was the first prompt at its resolution in each process.</p>' if t in ("easy_lake", "domino_falling") else ""
    secs.append('<section class="prompt" id="%s"><div class="phead"><h3>%s</h3><p class="ptext">%s</p></div>'
                '<video src="videos/%s.mp4" poster="videos/%s_poster.jpg" controls loop muted playsinline preload="metadata"></video>'
                '<ol class="times">%s</ol>%s</section>' % (t, NAMES[t], html.escape(P.get(t, "")), t, t, cells, note))
import os
EXT = ""
if os.path.exists(D + "/ablation_ext.json"):
    JE = json.load(open(D + "/ablation_ext.json")); ME = [{k: float(np.mean([JE[t][s][k] or 0 for t in steady])) for k in ("prepare_s", "denoise_s", "commit_s", "e2e_s")} for s in range(3)]
    QE = [("2 steps (S5)", "critic AUC 0.70", "9/24", "3.79", "0", "first blind batch: original 3/24, 3.68"),
          ("1 step (S6)", "critic AUC 0.91", "4/24", "<span class=bad>2.89</span>", "<span class=bad>15</span>", "quality p &lt; 0.001 vs original: blurred, smeared frames"),
          ("2 steps + SageAttention 2 (S7)", "critic AUC 0.79", "9/24", "3.78", "0", "not worse than original (quality p 0.40)")]
    er = "".join('<tr><th scope="row"><span class="add">%s</span></th><td class="num">%.1f</td><td class="num">%.1f</td><td class="num strong">%.1f</td><td class="num">%.2f×</td><td class="q">%s</td><td class="num">%s</td><td class="num">%s</td><td class="num">%s</td><td class="q">%s</td></tr>' % (QE[s][0], ME[s]["denoise_s"], ME[s]["commit_s"], ME[s]["e2e_s"], base / ME[s]["e2e_s"], QE[s][1], QE[s][2], QE[s][3], QE[s][4], QE[s][5]) for s in range(3))
    vs = "".join('<section class="prompt"><div class="phead"><h3>%s</h3></div><video src="videos/%s_ext.mp4" poster="videos/%s_ext_poster.jpg" controls loop muted playsinline preload="metadata"></video><ol class="times">%s</ol></section>' % (NAMES[t], t, t, "".join('<li><span class="code">%s</span><span class="num">%.1f s%s</span></li>' % (r["stage"], r["e2e_s"], " *" if r.get("includes_warmup") else "") for r in JE[t])) for t in ORDER)
    EXT = ('<section class="block" id="extensions"><h2>Extensions: 1 step and SageAttention</h2><p class="note">Both start from S5. S6 drops to a single sampling step; S7 keeps 2 steps and replaces self-attention with SageAttention 2 (INT8 QK, FP8 PV; 1.45× faster attention kernel). '
           'Blind rating: 3 raters, 24 jar/castle clips per arm, original 14B in the same batch (3/24 severe-clean, quality 3.65).</p>'
           '<div class="tablebox"><table><thead><tr><th>Variant</th><th>Denoise (s)</th><th>KV commit (s)</th><th>End-to-end ↓ (s)</th><th>Speed-up ↑</th><th>Critic</th><th>Severe-clean ↑</th><th>Quality ↑ (1–5)</th><th>Blur (majority)</th><th>Verdict</th></tr></thead><tbody>%s</tbody></table></div>'
           '<p class="note">Each video: S5 | S6 | S7.</p>%s</section>') % (er, vs)
page = open("/home/hk28965/specwm/scripts/ablation_page_template.html").read()
page = page.replace("{{ROWS}}", "".join(rows)).replace("{{BARS}}", bars).replace("{{SECTIONS}}", "".join(secs)).replace("{{EXT}}", EXT).replace("{{BASE}}", "%.0f" % base).replace("{{FINAL}}", "%.0f" % M[5]["e2e_s"]).replace("{{SPEED}}", "%.1f" % (base / M[5]["e2e_s"]))
open(OUT, "w").write(page); print("ok", OUT)
