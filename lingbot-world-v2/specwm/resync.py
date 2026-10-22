"""Re-grounding (resync) 실험: Guandao 의 'cloud target 이 가끔 context 를 교정' 아이디어의 핵심 검증.

질문: draft(Fast) 의 KV 창을 target(Base) 의 프레임으로 teacher-forcing 해서 갈아끼우면
      이후 draft 의 드리프트가 줄어드는가? (줄어들지 않으면 이 설계는 전제가 무너진다)

메커니즘: binding.commit(x0, chunk) 은 clean latent 를 t=0 으로 forward 해 KV 를 쓴다 = teacher forcing.
          Base 의 공식 샘플러 영상을 VAE 로 다시 인코딩해 latent 를 얻고, chunk < k 를 commit 한 뒤
          chunk >= k 를 Fast 가 평소처럼 생성한다. 다른 모델의 KV 를 옮기지 않는다 (프레임 교체 설계).

arm:
  fast_alone            Fast 단독 (기준)
  ctrl_selfcommit_k6    Fast 자신의 latent 를 chunk<6 에 commit 후 계속  -> fast_alone 과 같아야 함 (commit 경로 검증)
  resync_k3 / k6 / k9   Base context 로 단일 재접지 (③ 시점 스윕)
  resync_every3         3,6,9 에서 반복 재접지 (③ 주기)  -- 각 시점마다 Base context 전체(0..k-1)로 다시 씀
저장: 각 arm 의 전체 프레임(uint8, 4프레임 간격) + Base 프레임(같은 간격) → 로컬에서 ②snap, drift-to-teacher 계산.
"""
import argparse, json, os, time, gc
import numpy as np, torch

GiB = 2 ** 30


def read_video(path):
    import cv2; cv2.setNumThreads(1)
    c = cv2.VideoCapture(path); fr = []
    while True:
        ok, f = c.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return np.stack(fr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--action-root", required=True)
    ap.add_argument("--prompt-file", required=True); ap.add_argument("--scene", default="05")
    ap.add_argument("--base-video", required=True, help="Base 공식 샘플러 mp4 (161 프레임)")
    ap.add_argument("--frames", type=int, default=161)
    ap.add_argument("--arms", default="fast_alone,ctrl_selfcommit_k6,resync_k3,resync_k6,resync_k9,resync_every3")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    from .binding_v2 import V2Binding
    prompt = open(a.prompt_file).read().strip()
    b = V2Binding(a.ckpt, target_mode="fast_bf16", quant=None, sink=6, local_attn=18,
                  swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
    cond = b.prepare(os.path.join("examples", a.scene, "image.jpg"), prompt, a.action_root, frame_num=a.frames)
    N = cond["num_chunks"]; sig = b.node_sigmas(); CS = 3
    print("[load] %d chunks" % N, flush=True)

    # --- Base 영상 -> latent (Fast 와 같은 VAE) ---
    V = read_video(a.base_video)                                   # [T,H,W,3] uint8
    T_need = (N * CS - 1) * 4 + 1                                   # 39 latent -> 153 px
    assert len(V) >= T_need, (len(V), T_need)
    Hf, Wf = cond["hw"] if "hw" in cond else (V.shape[1], V.shape[2])
    x = torch.from_numpy(V[:T_need]).permute(3, 0, 1, 2).float() / 127.5 - 1.0   # [3,T,H,W]
    if (x.shape[2], x.shape[3]) != (Hf, Wf):
        x = torch.nn.functional.interpolate(x.permute(1, 0, 2, 3), size=(Hf, Wf), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
    with torch.no_grad():
        base_lat = b.pipe.vae.encode([x.to(b.device)])[0].float()   # [16, 39, h, w]
    print("[base] video %s -> latent %s" % (tuple(V.shape), tuple(base_lat.shape)), flush=True)
    assert base_lat.shape[1] == N * CS, base_lat.shape
    np.save(os.path.join(a.out, "base_frames.npy"), V[:T_need][::4])

    def gen_chunk(ci, g):
        z = b.chunk_noise(ci, g); x0 = None
        for i in range(len(sig)):
            x0 = b.target_x0(z, i, ci)
            if i < len(sig) - 1:
                s = float(sig[i + 1]); z = (1 - s) * x0 + s * b.randn_like(x0, g)
        return x0

    def run(plan, ctx_lat):
        """plan[ci] in {'gen', 'commit'}; 'commit' 은 ctx_lat 의 해당 청크를 teacher-force."""
        b.reset_caches(); torch.cuda.empty_cache()
        g = torch.Generator().manual_seed(1234); outs = []; t0 = time.time()
        for ci in range(N):
            if plan[ci] == "commit":
                _ = b.chunk_noise(ci, g)                          # 노이즈 스트림 정렬 유지
                x0 = ctx_lat[:, ci * CS:(ci + 1) * CS].to(b.device, torch.bfloat16)
                b.commit(x0.float(), ci)
            else:
                x0 = gen_chunk(ci, g); b.commit(x0, ci)
            outs.append(x0.detach().float().cpu())
        lat = torch.cat(outs, dim=1)
        with torch.no_grad():
            img = b.pipe.vae.decode([lat.to(b.device)])[0].detach().cpu()
        return lat, img, time.time() - t0

    def save(tag, lat, img, plan, secs):
        v = ((img.clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy()
        d = os.path.join(a.out, tag); os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "frames_all.npy"), (v * 255).astype("uint8")[::4])
        torch.save(lat, os.path.join(d, "latent.pt"))
        try:
            import imageio
            imageio.mimwrite(os.path.join(d, "video.mp4"), list((v * 255).astype("uint8")), fps=16, quality=8)
        except Exception as e:
            print("[video]", e, flush=True)
        rep = dict(plan="".join("C" if p == "commit" else "G" for p in plan), gen_s=round(secs, 1),
                   flicker=round(float((img[:, 1:] - img[:, :-1]).abs().mean()), 5))
        json.dump(rep, open(os.path.join(d, "arm.json"), "w"), indent=1)
        print("[%s] %s" % (tag, rep), flush=True)

    fast_lat = None
    for tag in a.arms.split(","):
        if os.path.exists(os.path.join(a.out, tag, "arm.json")):
            print("[%s] 이미 있음" % tag, flush=True)
            if tag == "fast_alone": fast_lat = torch.load(os.path.join(a.out, tag, "latent.pt"))
            continue
        try:
            if tag == "fast_alone":
                plan = ["gen"] * N; lat, img, s = run(plan, None); fast_lat = lat
            elif tag.startswith("ctrl_selfcommit_k"):
                k = int(tag.split("k")[-1]); assert fast_lat is not None
                plan = ["commit"] * k + ["gen"] * (N - k); lat, img, s = run(plan, fast_lat)
            elif tag.startswith("resync_k"):
                k = int(tag.split("k")[-1])
                plan = ["commit"] * k + ["gen"] * (N - k); lat, img, s = run(plan, base_lat)
            elif tag == "resync_every3":
                # 3,6,9 에서 재접지: 그 시점까지의 context 가 전부 Base 라는 뜻이므로
                # plan 은 chunk<9 commit(Base), 9.. gen 과 동일하다. 단 '표시 스트림' 은 chunk 3-9 를
                # Fast 가 생성한 것이어야 하므로 별도 arm 으로 뒀다: 각 구간을 Base context 에서 새로 생성.
                lat_parts = []; img_parts = []; tsum = 0.0
                for k0, k1 in [(0, 3), (3, 6), (6, 9), (9, N)]:
                    plan = ["commit"] * k0 + ["gen"] * (k1 - k0) + ["commit"] * (N - k1)
                    plan = [("commit" if i < k0 else "gen" if i < k1 else "skip") for i in range(N)]
                    b.reset_caches(); torch.cuda.empty_cache()
                    g = torch.Generator().manual_seed(1234); t0 = time.time()
                    for ci in range(k1):
                        if ci < k0:
                            _ = b.chunk_noise(ci, g); b.commit(base_lat[:, ci*CS:(ci+1)*CS].to(b.device).float(), ci)
                        else:
                            x0 = gen_chunk(ci, g); b.commit(x0, ci); lat_parts.append(x0.detach().float().cpu())
                    tsum += time.time() - t0
                lat = torch.cat(lat_parts, dim=1)
                with torch.no_grad(): img = b.pipe.vae.decode([lat.to(b.device)])[0].detach().cpu()
                plan = ["gen"] * N; s = tsum
            else:
                raise ValueError(tag)
            save(tag, lat, img, plan, s)
        except Exception as e:
            print("[%s] 실패: %s" % (tag, str(e)[:300]), flush=True)
            json.dump({"error": str(e)[:300]}, open(os.path.join(a.out, tag + "_error.json"), "w"))
        gc.collect(); torch.cuda.empty_cache()
    print("[RESYNC_DONE]", flush=True)


if __name__ == "__main__":
    main()
