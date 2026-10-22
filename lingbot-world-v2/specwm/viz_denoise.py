"""노이즈 -> 완성 시각화용: 한 청크의 각 디노이징 단계 중간상태를 VAE로 디코딩해 저장.
z_k  = 각 단계 입력(노이즈 섞인 잠재)   x0_k = 각 단계에서 모델이 예측한 깨끗한 이미지
"""
import argparse, json, os, time, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--scene", default="00")
ap.add_argument("--steps", type=int, default=4); ap.add_argument("--out", required=True)
ap.add_argument("--prompt", default="A sweeping cinematic journey along the Great Wall of China")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

from specwm.binding_v2 import V2Binding
idx = (0, 250, 500, 750) if a.steps == 4 else tuple(int(round(x)) for x in np.linspace(0, 999, a.steps+1)[:-1])
b = V2Binding(a.ckpt, target_mode="fast_bf16", quant=None, sink=6, local_attn=18,
              swap_models=False, build_draft=False, timesteps_index=idx)
src = os.path.join("examples", a.scene)
cond = b.prepare(os.path.join(src, "image.jpg"), a.prompt, src, frame_num=13)   # 1청크 = 잠재 3프레임 = 픽셀 9프레임
sig = b.node_sigmas(); print("[sigmas]", [round(float(s),4) for s in sig], flush=True)
b.reset_caches()
g = torch.Generator().manual_seed(1234)
z = b.chunk_noise(0, g)
zs, x0s = [z.detach().float().cpu()], []
for i in range(len(sig)):
    x0 = b.target_x0(z, i, 0); x0s.append(x0.detach().float().cpu())
    s = float(sig[i+1]) if i < len(sig)-1 else 0.0
    z = (1-s)*x0 + s*b.randn_like(x0, g) if i < len(sig)-1 else x0
    zs.append(z.detach().float().cpu())
vae = b.pipe.vae; dev = b.device
def dec(lat):
    img = vae.decode([lat.to(dev)])[0].detach().cpu()
    return ((img.clamp(-1,1)+1)/2).permute(1,2,3,0).numpy()[0]   # 첫 프레임 [H,W,3]
for k, zk in enumerate(zs):
    plt.imsave(os.path.join(a.out, f"z{k}.jpg"), dec(zk).clip(0,1))
for k, xk in enumerate(x0s):
    plt.imsave(os.path.join(a.out, f"x0_{k}.jpg"), dec(xk).clip(0,1))
np.save(os.path.join(a.out,"z_latents.npy"), np.stack([z_.numpy() for z_ in zs]))
np.save(os.path.join(a.out,"x0_latents.npy"), np.stack([x_.numpy() for x_ in x0s]))
json.dump(dict(sigmas=[float(s) for s in sig], n_z=len(zs), n_x0=len(x0s)),
          open(os.path.join(a.out, "meta.json"), "w"), indent=1)
print("[done]", os.listdir(a.out), flush=True)
