"""paper/1005/bid14B_config_check: side-by-side per prompt of the bidirectional Wan2.1 T2V variants + README with colour stats.
usage: python build_bid_check.py"""
import os, json, subprocess
S = "/home/hk28965/specwm"; O = S + "/lingbot-world-v2/specwm_out/bid_t2v"; P = S + "/paper/1005/bid14B_config_check"; os.makedirs(P, exist_ok=True)
ARMS = [("bid1p3Boff", "1.3B CFG5 sh5"), ("bid1p3Boffrec", "1.3B CFG6 sh8 (rec.)"), ("bid14B", "14B CFG5 sh5 (SDVG code)"), ("bid14Boff", "14B CFG5 sh5 (official)"),
        ("bid14Boffg3", "14B CFG3 sh5"), ("bid14Boffs3", "14B CFG5 sh3"), ("bid14Boff720", "14B 720p CFG5 sh5")]
for n in ("jar", "castle", "tokyo_street", "bigsur_drone"):
    items = []
    for arm, lab in ARMS:
        d = "%s/%s/%s_s1234" % (O, n, arm)
        if os.path.exists(d + "/video.mp4"): items.append("%s %.0fs=%s/video.mp4" % (lab, json.load(open(d + "/bid.json"))["gen_s"], d))
    subprocess.run([S + "/venv/bin/python", S + "/scripts/concat_labeled.py", "%s/%s__seed1234__bid_variants.mp4" % (P, n), "--height", "240"] + items, check=True)
tab = subprocess.run([S + "/venv/bin/python", S + "/scripts/bid_color_stats.py"], capture_output=True, text=True).stdout
open(P + "/README.md", "w").write("""# Bidirectional Wan2.1 T2V 14B: config check (10/2)

All clips: seed 1234, 81 frames, 16 fps, 50 UniPC steps, default negative prompt, no prompt extension, one GPU; 832x480 unless noted.
Panels (left to right): 1.3B official code CFG5/shift5 (what we used) | 1.3B with the README recommendation CFG6/shift8 | 14B SDVG-repo code (the clip shown on 9/30) |
14B official Wan2.1 code, same settings | 14B CFG3 | 14B shift3 | 14B at 1280x720 (its native resolution).

Findings: (1) the SDVG repo's modified wan package (bf16 residual/RoPE) gives the same output as the official code (PSNR 24-26 dB, same colour stats), so the code is not the problem.
(2) The harsh, over-contrasted 14B look comes from CFG 5 at 480p: CFG 3 cuts clipped pixels 65->20 % (bigsur), 55->32 % (tokyo), 29->11 % (castle); 720p does the same (castle 29->8 %) and follows the prompt better ('elaborate sandcastle').
(3) shift 3 (diffusers' 480p hint) does not help.

TABLE
""".replace("TABLE", tab))
print("[BID_CHECK_DONE]", sorted(os.listdir(P)))
