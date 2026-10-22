import os
"""2x2 분해용 교정본 조립 + 청크6 상태 잘라내기.
arm  | 앞 2.25초 (청크0-2) | 뒤 2.25초 (청크3-5) | 의미
d0   | Fast 다듬기         | RePaint             | 현행 (이미 있음, 4/10)
LL   | Fast 다듬기         | Fast 자기 것        | 다듬기만의 효과
LB   | Fast 다듬기         | Base16 원본         | RePaint 대신 진짜 Base 를 붙이면
BR   | Base16 원본         | RePaint(Base 조건)  | 다듬기 대신 진짜 Base 앞부분이면
B+B  | Base16              | Base16              | resync_k6 (이미 있음, 8/10)
"""
import sys, os, json, numpy as np, torch, imageio
R = os.environ["SCRATCH"]+"/lingbot-world-v2/specwm_out"
def rd(p, n=None):
    v = imageio.mimread(p, memtest=False); v = np.stack(v)[:, :, :, :3]; return v if n is None else v[:n]
def wr(p, v):
    os.makedirs(os.path.dirname(p), exist_ok=True); imageio.mimwrite(p, list(v), fps=16, quality=9)
def cut_state(src, dst):
    os.makedirs(dst, exist_ok=True); CS = 3
    disp = torch.load(os.path.join(src, "display_lat.pt")); kvl = torch.load(os.path.join(src, "kv_lat.pt"))
    torch.save(disp[:, :6 * CS].clone(), os.path.join(dst, "display_lat.pt")); torch.save(kvl[:, :6 * CS].clone(), os.path.join(dst, "kv_lat.pt"))
    m = json.load(open(os.path.join(src, "meta.json"))); m.update(done=6, corrections=[], anchor_chunks=None)
    json.dump(m, open(os.path.join(dst, "meta.json"), "w"), indent=1)
seed = sys.argv[1]; OUT = "%s/decomp_s%s" % (R, seed); os.makedirs(OUT, exist_ok=True)
d0 = R + ("/lat55/d0" if seed == "1234" else "/lat55_s%s/d0" % seed)
lau = (R + "/e2e55/rolling/launder_k6" if seed == "1234" else d0 + "/launder_k6") + "/REFINE_0.40_0.20_0.08/video.mp4"
ctrl = R + ("/e2e55/control/video.mp4" if seed == "1234" else "/lat55_s%s/control/video.mp4" % seed)
base16 = R + "/revisit/base_official_S05_s16/video.mp4"
L = rd(lau, 33); F = rd(ctrl, 69); B = rd(base16, 69)
assert len(L) == 33 and len(F) == 69 and len(B) == 69, (len(L), len(F), len(B))
pad = lambda v: np.concatenate([v, np.repeat(v[-1:], 73 - len(v), 0)]) if len(v) < 73 else v[:73]
wr(OUT + "/prov_LL.mp4", pad(np.concatenate([L, F[33:69]])))
wr(OUT + "/prov_LB.mp4", pad(np.concatenate([L, B[33:69]])))
wr(OUT + "/base16_prefix33.mp4", B[:33]); imageio.imwrite(OUT + "/base16_first.jpg", B[0], quality=97)
for arm in ["LL", "LB", "BR"]: cut_state(d0, "%s/%s" % (OUT, arm))
print("[mk_decomp] seed %s: providers LL/LB written, base16 prefix written, states cut at chunk 6" % seed)
