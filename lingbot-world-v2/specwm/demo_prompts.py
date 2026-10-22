"""다양한 프롬프트/씬 데모 (Fast). 바인딩을 한 번 로드하고 목록(json)의 (scene, image, action, prompt) 마다
  fast_alone  : Fast 단독 생성
  resync_k6   : --base-root/<name>/video.mp4 (Base 참조, 예: 16-step) 가 있으면 청크 0..k-1 를 Base 컨텍스트로 commit 후 생성
출력: <out>/<name>/{fast_alone,resync_k6}/{video.mp4, frames_all.npy, latent.pt, prompt.txt, arm.json}
목록 항목: {"name","image","action","prompt_file","frames"(기본 161),"seed"(기본 1234)}
"""
import argparse, json, os, time, gc, shutil
import numpy as np, torch
from .resync_v4 import read_video, encode_video, CS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--list", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--base-root", default=None); ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--window", type=int, default=18); ap.add_argument("--sink", type=int, default=6)
    ap.add_argument("--only", default="", help="name 필터(콤마)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    items = json.load(open(a.list))
    if a.only: items = [it for it in items if it["name"] in a.only.split(",")]
    from .binding_v2 import V2Binding
    t0 = time.time()
    b = V2Binding(a.ckpt, target_mode="fast_bf16", quant=None, sink=a.sink, local_attn=a.window,
                  swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
    print("[load] %.0fs" % (time.time() - t0), flush=True)

    for it in items:
        name = it["name"]; frames = int(it.get("frames", 161)); seed = int(it.get("seed", 1234))
        prompt = open(it["prompt_file"]).read().strip()
        base_video = os.path.join(a.base_root, name, "video.mp4") if a.base_root else None
        arms = ["fast_alone"] + (["resync_k%d" % a.k] if base_video and os.path.exists(base_video) else [])
        todo = [x for x in arms if not os.path.exists(os.path.join(a.out, name, x, "arm.json"))]
        if not todo: print("[%s] 모두 완료" % name, flush=True); continue
        cond = b.prepare(it["image"], prompt, it["action"], frame_num=frames)
        N = cond["num_chunks"]; sig = [float(s) for s in b.node_sigmas()]; n_draw = len(sig)
        shp = cond["shape"]; hw = (int(shp[2]) * 8, int(shp[3]) * 8)
        print("[%s] %d청크, hw %s, arms %s" % (name, N, hw, todo), flush=True)

        def gen_chunk(ci, g):
            z = b.chunk_noise(ci, g); x0 = None
            for i in range(len(sig)):
                x0 = b.target_x0(z, i, ci)
                if i < len(sig) - 1: z = (1 - sig[i + 1]) * x0 + sig[i + 1] * b.randn_like(x0, g)
            return x0

        def run(plan, ctx):
            b.reset_caches(); torch.cuda.empty_cache(); g = torch.Generator().manual_seed(seed); outs = []; t1 = time.time()
            for ci in range(N):
                if plan[ci] == "G": x0 = gen_chunk(ci, g)
                else:
                    for _ in range(n_draw): torch.randn(b._chunk_shape, generator=g)
                    x0 = ctx[:, ci * CS:(ci + 1) * CS].to(b.device).float()
                b.commit(x0, ci); outs.append(x0.detach().float().cpu())
            lat = torch.cat(outs, dim=1)
            with torch.no_grad(): img = b.pipe.vae.decode([lat.to(b.device)])[0].detach().cpu()
            return lat, img, time.time() - t1

        for arm in todo:
            try:
                if arm == "fast_alone": plan = ["G"] * N; ctx = None
                else:
                    torch.cuda.empty_cache(); ctx = encode_video(b, read_video(base_video), N, hw=hw).cpu(); plan = ["B"] * a.k + ["G"] * (N - a.k)
                lat, img, s = run(plan, ctx)
                v = ((img.clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy(); d = os.path.join(a.out, name, arm); os.makedirs(d, exist_ok=True)
                np.save(os.path.join(d, "frames_all.npy"), (v * 255).astype("uint8")[::4]); torch.save(lat, os.path.join(d, "latent.pt"))
                import imageio; imageio.mimwrite(os.path.join(d, "video.mp4"), list((v * 255).astype("uint8")), fps=16, quality=8)
                shutil.copy(it["prompt_file"], os.path.join(d, "prompt.txt"))
                r = dict(name=name, arm=arm, plan="".join(plan), seed=seed, frames=int(v.shape[0]), gen_s=round(s, 1),
                         flicker=round(float((img[:, 1:] - img[:, :-1]).abs().mean()), 5), image=it["image"], action=it["action"], prompt=prompt)
                json.dump(r, open(os.path.join(d, "arm.json"), "w"), indent=1); print("[%s/%s] gen %.0fs flicker %.4f" % (name, arm, s, r["flicker"]), flush=True)
            except Exception as e:
                print("[%s/%s] 실패: %s" % (name, arm, str(e)[:300]), flush=True)
            gc.collect(); torch.cuda.empty_cache()
    print("[DEMO_PROMPTS_DONE]", flush=True)


if __name__ == "__main__":
    main()
