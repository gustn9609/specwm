"""Colour statistics of the bidirectional T2V clips (archive SDVG-repo code vs official Wan2.1 code): mean HSV saturation / value,
fraction of clipped pixels (any channel >= 250 or <= 5), fraction of highly saturated pixels (S > 0.8), saturation change first->last 10 frames.
usage: python bid_color_stats.py  -> prints a markdown table, writes exp/kveval/bid_color_stats.json"""
import os, json, numpy as np, imageio, cv2
O = "/home/hk28965/specwm/lingbot-world-v2/specwm_out/bid_t2v"; R = {}
ARMS = [("bid1p3B", "1.3B archive"), ("bid1p3Boff", "1.3B official"), ("bid14B", "14B archive"), ("bid14Boff", "14B official"), ("bid14Boff720", "14B official 720p"), ("bid14Boffg3", "14B official CFG 3"), ("bid14Boffs3", "14B official shift 3"), ("bid1p3Boffrec", "1.3B official guide 6 shift 8")]
rows = ["| prompt | variant | sat mean ↓ | high-sat px % ↓ | clipped px % ↓ | value mean | sat drift first→last ↓ | gen time (s) |", "|---|---|---|---|---|---|---|---|"]
for name in ("jar", "castle", "tokyo_street", "bigsur_drone"):
    for arm, lab in ARMS:
        p = "%s/%s/%s_s1234/video.mp4" % (O, name, arm)
        if not os.path.exists(p): continue
        V = np.stack([np.asarray(f) for f in imageio.get_reader(p)]); H = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2HSV) for f in V]).astype(np.float32)
        s, v = H[..., 1] / 255, H[..., 2] / 255; clip = ((V >= 250) | (V <= 5)).any(-1).mean()
        d = dict(sat=float(s.mean()), hisat=float((s > 0.8).mean()), clip=float(clip), val=float(v.mean()), sdrift=float(abs(s[-10:].mean() - s[:10].mean())), gen_s=json.load(open(os.path.dirname(p) + "/bid.json")).get("gen_s"))
        R["%s|%s" % (name, arm)] = d
        rows.append("| %s | %s | %.3f | %.1f | %.1f | %.3f | %.3f | %s |" % (name, lab, d["sat"], 100 * d["hisat"], 100 * d["clip"], d["val"], d["sdrift"], d["gen_s"]))
json.dump(R, open("/home/hk28965/specwm/exp/kveval/bid_color_stats.json", "w"), indent=1); print("\n".join(rows))
