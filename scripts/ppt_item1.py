from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
BL = prs.slide_layouts[6]; INK = RGBColor(0x1c,0x1b,0x18); MUTE = RGBColor(0x6b,0x67,0x5e); ACC = RGBColor(0x2d,0x5a,0xa8); RED = RGBColor(0xc0,0x3a,0x2b)
def title(s, text, sub=None):
    tb = s.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.9)); tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = text; p.font.size = Pt(28); p.font.bold = True; p.font.color.rgb = INK
    if sub: q = tf.add_paragraph(); q.text = sub; q.font.size = Pt(14); q.font.color.rgb = MUTE
def bullets(s, items, x=0.5, y=1.4, w=12.3, h=5.5, size=16):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); lvl = 0
        while it.startswith("  "): it = it[2:]; lvl += 1
        p.text = ("• " if lvl == 0 else "– ") + it; p.level = lvl; p.font.size = Pt(size - 2 * lvl); p.font.color.rgb = INK; p.space_after = Pt(6)
def table(s, rows, x=0.5, y=1.4, w=12.3, colw=None, size=12, bold_first_col=False):
    nr, nc = len(rows), len(rows[0]); shp = s.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(0.4 * nr)); t = shp.table
    if colw:
        for j, cw in enumerate(colw): t.columns[j].width = Inches(cw)
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            c = t.cell(i, j); c.text = str(v); p = c.text_frame.paragraphs[0]; p.font.size = Pt(size); p.font.bold = (i == 0) or (bold_first_col and j == 0)
            p.font.color.rgb = RGBColor(0xff,0xff,0xff) if i == 0 else INK
            if i == 0: c.fill.solid(); c.fill.fore_color.rgb = ACC
            elif i % 2 == 0: c.fill.solid(); c.fill.fore_color.rgb = RGBColor(0xf1,0xef,0xe9)
            else: c.fill.solid(); c.fill.fore_color.rgb = RGBColor(0xff,0xff,0xff)
    return t
def note(s, text, y=6.7):
    tb = s.shapes.add_textbox(Inches(0.5), Inches(y), Inches(12.3), Inches(0.6)); p = tb.text_frame.paragraphs[0]; p.text = text; p.font.size = Pt(11); p.font.color.rgb = MUTE; tb.text_frame.word_wrap = True
# ---- slide 1
s = prs.slides.add_slide(BL); title(s, "Action item 1 — A real-time draft model (the top risk)", "What we tried: LingBot 1.3B, low-resolution rollout + super-resolution, fewer steps, quantization, cache skipping, other real-time models")
bullets(s, ["Goal: an edge-side draft that runs at ≥10 fps with camera-pose control, close enough to Base for the cloud to correct.",
            "Tried on 1× RTX PRO 6000 (96 GB): LingBot Fast 14B at 480p / 360p / 256p; 2-step vs 4-step; LingBot 1.3B; GPTQ int6 + 4-bit KV; WorldCache (official logic ported); Lanczos / Real-ESRGAN / latent upsampling; LongLive-2.0-5B as a reference.",
            "Metrics: fps (gen-only and end-to-end incl. KV commit + VAE decode), peak GPU memory, StEvo-Bench control / physics (85 camera tasks, Qwen3-VL judge), DINO distance to Base rollouts, pairwise visual-quality judge.",
            "Bottom line: the only viable draft is Fast 14B itself at 360p / 2-step (11 fps gen, 7.6 fps e2e). 1.3B is too weak alone but works as a draft *for the 14B*; super-resolution cannot recover what 360p generation loses; WorldCache and quantization do not help a 4-step model."], size=16)
# ---- slide 2: candidates table
s = prs.slides.add_slide(BL); title(s, "1.1  Draft candidates — speed and quality", "fps on one RTX PRO 6000, 161–253-frame rollouts; StEvo = 85 camera tasks; DINO = 1−cos to the Base 70-step rollout of the same scene (seed floor 0.17)")
rows = [["candidate", "fps gen-only ↑", "fps e2e ↑", "peak GiB ↓", "StEvo control % ↑", "StEvo physics % ↑", "DINO to Base ↓", "verdict"],
        ["Base 70-step 480p (target, reference)", "0.06", "0.06", "64", "65.5", "52.4", "0 (floor 0.17)", "not a draft"],
        ["Fast 14B 480p 4-step (reference draft)", "2.4", "2.0", "75", "50.6", "32.9", "0.36", "too slow"],
        ["LingBot 1.3B 480p 4-step", "4.6", "–", "35", "32.9", "1.2", "0.54", "too weak alone"],
        ["Fast 14B 360p 4-step", "6.9", "5.2", "59", "62.4", "28.2", "0.60 (S00) / 0.43 (S05)", "control ↑, detail ↓"],
        ["Fast 14B 360p 2-step", "11.1 (17.3 profile)", "7.6 (9.1)", "60", "55.3", "29.4", "0.41", "★ draft operating point"],
        ["Fast 14B 256p 4-step", "9.3", "7.0", "54", "–", "–", "0.43", "≤208×320 collapses"],
        ["Fast 14B 360p 2-step + int6 + kv4 + tiled VAE", "5.1 (dequant-bound)", "–", "22.7", "–", "–", "≈ same", "4090: borderline (24 GB)"],
        ["Fast + WorldCache (k3, official thr)", "= Fast", "= Fast", "+0.4", "–", "–", "0 skips / forced: collapse", "does not apply"],
        ["LongLive-2.0-5B (no camera control)", "27 e2e", "27", "–", "23.1 (video protocol)", "53.4", "–", "reference only"]]
table(s, rows, colw=[3.3, 1.3, 1.0, 0.9, 1.3, 1.3, 1.6, 1.6], size=11, bold_first_col=True)
note(s, "e2e = generation + KV commit + VAE decode. WorldCache never fires on the 4-step model (block-3 drift between consecutive steps ≈ 100% vs 2–10% in Base); forced skipping gives DINO ≈ 1.0 (unrelated frames).")
# ---- slide 3: low-res + SR
s = prs.slides.add_slide(BL); title(s, "1.2  Low-resolution rollout + super-resolution", "Draft at 360p, display at 480p — does upscaling recover the loss? (S00 DINO to Base; StEvo 85 tasks; pairwise visual-quality judge vs Fast 480p 4-step, both left/right orders)")
rows = [["condition", "SR cost / frame", "DINO to Base 480p ↓", "StEvo control ↑", "physics ↑", "VQ judge vs 480p (wins : losses : ties)"],
        ["Fast 480p 4-step (display reference)", "–", "0.36", "50.6", "32.9", "–"],
        ["Fast 360p 2-step, as generated", "–", "0.41", "55.3", "29.4", "1 : 77 : 7"],
        ["+ Lanczos 1.5×", "6.9 ms (CPU)", "0.60 → 0.60 (S00 4-step)", "54.1", "34.1", "2 : 77 : 6"],
        ["+ EDSR ×2 (super-image)", "3.7 ms", "0.60 → 0.60", "–", "–", "–"],
        ["+ Real-ESRGAN compact x4v3 (fp16)", "4.1 ms", "–", "60.0", "16.7", "3 : 77 : 5"],
        ["+ latent 1.5× bicubic → VAE decode", "~0", "–", "55.3", "35.3", "0 : 85 : 0 (grid artifacts)"],
        ["judge control: 480p → 360p → Lanczos vs original", "–", "–", "–", "–", "0 : 0 : 85 (cannot tell)"],
        ["judge control: 480p vs itself", "–", "–", "–", "–", "0 : 0 : 85"]]
table(s, rows, colw=[3.6, 1.3, 2.0, 1.3, 1.0, 3.1], size=11, bold_first_col=True)
bullets(s, ["Every upscaler fits the 10 fps budget (9 ms/frame), but none moves the judge; the judge does not even detect a 480p→360p→480p resolution round-trip (85/85 tie).",
            "So the loss is in the content generated at 360p (less detail/texture, different rollout), not in pixel count — no post-hoc SR can add it back.",
            "Consequence: 360p 2-step is a control-side draft, not a display point. Displaying at 480p needs either 360p 4-step (control 62.4, VQ 5:65) or a 480p decode of the corrected stream."], y=5.05, size=13)
# ---- slide 4: profile
s = prs.slides.add_slide(BL); title(s, "1.3  Where the draft's time goes (per 12-frame chunk, RTX PRO 6000)", "CUDA-event hooks on every block: self-attention / cross-attention / FFN / other; plus the KV-commit pass (t=0) and VAE decode")
rows = [["config", "gen s/chunk ↓", "self-attn", "cross", "FFN", "other", "commit ↓", "VAE ↓", "gen fps ↑", "e2e fps ↑"],
        ["480p 4-step", "3.67", "1.83 (50%)", "0.29", "0.53", "1.03", "0.92", "0.61", "3.3", "2.3"],
        ["480p 2-step", "1.84", "0.92 (50%)", "0.14", "0.26", "0.52", "0.92", "0.61", "6.5", "3.6"],
        ["360p 4-step", "1.38", "0.56 (41%)", "0.13", "0.25", "0.44", "0.34", "0.28", "8.7", "6.0"],
        ["360p 2-step", "0.69", "0.28 (41%)", "0.06", "0.12", "0.22", "0.34", "0.28", "17.3", "9.1"]]
table(s, rows, colw=[1.6, 1.3, 1.4, 0.9, 0.9, 0.9, 1.1, 1.0, 1.3, 1.3], size=12, bold_first_col=True)
bullets(s, ["Self-attention is 41–50% of the denoise pass — structural (W18 window: K/V are 6× the query), already ~230 TFLOPS; LIPAR / Monarch-style sparsity caps at ~1.3–1.5× on this half.",
            "At 360p 2-step the denoise pass is only 52% of the chunk: KV commit (an extra full forward, 0.34 s) + VAE decode (0.28 s) cost as much as generation → cheapest levers: streaming/light VAE, torch.compile on 'other' (30%).",
            "Skipping the commit pass (reuse the last denoise step's noisy KV) gives +50% gen fps but DINO 0.41 → 0.52: not acceptable.",
            "4090 projection: int6+kv4 stack peaks at 22.7 GiB (OOM under a strict 24 GB cap → needs ~3 GB headroom), ~2–3 fps gen-only with current dequant kernels."], y=4.0, size=13)
# ---- slide 5: 1.3B as inner draft
s = prs.slides.add_slide(BL); title(s, "1.4  LingBot 1.3B: too weak alone, but a valid draft for the 14B", "S05, one prompt, 24 seeds, 360p; energy distance = distribution-to-distribution gap to the 24 Base seeds on chunks 6–12 (lower is better); 'periodic k' = the 14B Fast generates every k-th chunk, both KV caches committed")
rows = [["arm (360p)", "seeds", "energy distance to Base seeds ↓", "DINO to same-seed Base ↓", "frame-to-frame jump max ↓", "cost"],
        ["Fast 14B 4-step (target of this pair)", "13", "0.199", "0.417", "0.482", "1×"],
        ["LingBot 1.3B 4-step alone", "12", "0.359", "0.538", "0.420", "~¼×"],
        ["1.3B → Fast 14B, periodic every 3 chunks", "11", "0.187", "0.397", "0.383", "≈ ½× (1/3 of chunks by 14B)"],
        ["1.3B → Fast 14B, periodic every 2 chunks", "11", "0.186", "0.401", "0.420", "≈ ⅝×"],
        ["(for scale) Fast 14B 480p → Base KV-rewrite (ours)", "24", "0.135", "0.360", "0.434", "+ cloud correction"]]
table(s, rows, colw=[3.8, 0.8, 2.4, 2.0, 2.0, 1.3], size=12, bold_first_col=True)
bullets(s, ["Alone, the 1.3B is far from Base (0.36) and fails StEvo (control 32.9, physics 1.2).",
            "With the 14B regenerating every third chunk into the shared context, the 1.3B stream lands at the 14B's own distribution (0.187 vs 0.199) with *smaller* temporal jumps than the 14B alone (0.383 vs 0.482): same model family → no visible cut at the hand-over.",
            "This is the LLM-style speculative pair (small draft, same-family target) — it buys speed inside the edge, but it cannot move the stream toward Base (the target is the 14B, not Base)."], y=4.5, size=13)
# ---- slide 6: decision
s = prs.slides.add_slide(BL); title(s, "1.5  Decision", "What we take forward from action item 1")
bullets(s, ["Draft = LingBot Fast 14B at 360p, 2 steps: 11.1 fps gen-only / 7.6 fps e2e on one RTX PRO 6000 (StEvo control 55.3 vs 50.6 at 480p; physics 29.4 vs 32.9). ~2–3 fps projected on a 4090 with the int6/kv4 stack (24 GB borderline).",
            "Do not spend on: super-resolution (cannot recover generated content), WorldCache on Fast (never fires), quantization for speed on this GPU (dequant-bound), 1.3B as a standalone draft.",
            "Optional inner tier: 1.3B → Fast 14B speculative pair (every 3rd chunk by the 14B) reaches the 14B distribution without cuts at ~½ the 14B cost — useful if the edge budget is below the 14B.",
            "Remaining engineering levers for e2e fps (not quality): light/streaming VAE decode (0.28 s/chunk), commit-pass fusion is *not* free (noisy-KV shortcut degrades), torch.compile on the 30% 'other'.",
            "Open risk moved, not removed: the draft is fast enough only at 360p, and 360p is a control point, not a display point — the display path (480p) has to come from the corrected stream or a 4-step 360p draft."], size=16)
note(s, "Sources: specwm_out/timing_s00, profile_fast/PROFILE.md, stevo/stevo_results.md, stevo/SR_results.md, lowres_sr/lowres_sr.md, spec_ar/summary_ar.md, s1_fastwc/S1_fast_worldcache.md (2026-09-15 … 09-21).")
prs.save("/home/hk28965/specwm/specwm_out_web/ppt/item1_realtime_draft.pptx"); print("saved")
