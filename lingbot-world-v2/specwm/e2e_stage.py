"""E2E cloud-grounded streaming: v2(Fast) 단계 실행기.
단계 = [상태 복원(kv_lat 0..k0-1 을 commit)] -> [교정 적용: kv_lat[first..first+5] <- 교정 영상; rolling 이면 sink 슬롯도 교정 창으로] -> [청크 k0..gen_until-1 생성] -> [상태 저장 + 다음 교정용 prefix(청크 gen_until-6..gen_until-4 의 display) 저장].
난수 소비는 fast_alone 과 동일(skip_draws) -> 교정 없는 단계 실행은 연속 실행과 비트 동일해야 함(control).
"""
import argparse, json, os, time, numpy as np, torch
CS = 3
def read_video(path):
    import cv2; cv2.setNumThreads(1); c = cv2.VideoCapture(path); fr = []
    while True:
        ok, f = c.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return np.stack(fr)
def encode_frames(b, V, n_lat, hw):
    T = 4 * (n_lat - 1) + 1; assert len(V) >= T, (len(V), T)
    x = torch.from_numpy(np.ascontiguousarray(V[:T])).permute(3, 0, 1, 2).float() / 127.5 - 1.0
    if (x.shape[2], x.shape[3]) != tuple(hw): x = torch.nn.functional.interpolate(x.permute(1, 0, 2, 3), size=tuple(hw), mode="bicubic", align_corners=False).clamp(-1, 1).permute(1, 0, 2, 3)
    with torch.no_grad(): lat = b.pipe.vae.encode([x.to(b.device)])[0].float()
    assert lat.shape[1] == n_lat, lat.shape; return lat.cpu()
def to_u8(img): return (((img.clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy() * 255).astype("uint8")
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--action-root", required=True); ap.add_argument("--prompt-file", required=True); ap.add_argument("--scene", default="05")
    ap.add_argument("--frames", type=int, default=873); ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--window", type=int, default=36); ap.add_argument("--sink", type=int, default=18)
    ap.add_argument("--state", required=True); ap.add_argument("--gen-until", type=int, required=True)
    ap.add_argument("--correction", default="", help="<video.mp4>:<first_chunk> : 청크 first..first+5 의 KV latent 를 교정 영상(>=69프레임)으로 교체")
    ap.add_argument("--anchor", choices=["fixed", "rolling"], default="rolling"); ap.add_argument("--decode", action="store_true")
    a = ap.parse_args(); os.makedirs(a.state, exist_ok=True)
    from .binding_v2 import V2Binding
    import imageio
    prompt = open(a.prompt_file).read().strip(); t0 = time.time()
    b = V2Binding(a.ckpt, target_mode="fast_bf16", quant=None, sink=a.sink, local_attn=a.window, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
    cond = b.prepare(os.path.join("examples", a.scene, "image.jpg"), prompt, a.action_root, frame_num=a.frames)
    N = cond["num_chunks"]; sig = b.node_sigmas(); n_draw = len(sig); hw = tuple(cond["hw"]) if "hw" in cond else (464, 832)
    assert a.gen_until <= N, (a.gen_until, N); print("[e2e] 로드 %.0fs N=%d sigmas %s hw %s" % (time.time() - t0, N, [round(float(x), 3) for x in sig], hw), flush=True)
    meta_p = os.path.join(a.state, "meta.json"); meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {"done": 0, "corrections": [], "anchor_chunks": None, "window": a.window, "sink": a.sink, "anchor_policy": a.anchor}
    k0 = meta["done"]; disp = torch.load(os.path.join(a.state, "display_lat.pt")) if k0 > 0 else None; kvl = torch.load(os.path.join(a.state, "kv_lat.pt")) if k0 > 0 else None
    first = None
    if a.correction:
        vp, first = a.correction.split(":"); first = int(first); assert first + 6 <= k0, (first, k0)
        corr = encode_frames(b, read_video(vp), 6 * CS, hw); kvl[:, first * CS:(first + 6) * CS] = corr
        meta["corrections"].append({"first_chunk": first, "video": vp, "applied_at_chunk": k0}); print("[e2e] 교정: KV 청크 %d..%d <- %s" % (first, first + 5, vp), flush=True)
    def skip_draws(g):
        for _ in range(n_draw): torch.randn(b._chunk_shape, generator=g)
    def gen_chunk(ci, g):
        z = b.chunk_noise(ci, g); x0 = None
        for i in range(len(sig)):
            x0 = b.target_x0(z, i, ci)
            if i < len(sig) - 1:
                s = float(sig[i + 1]); z = (1 - s) * x0 + s * b.randn_like(x0, g)
        return x0
    def write_sink(lat6, first_chunk):
        """sink 슬롯(버퍼 [0:18fs]) <- 청크 first..first+5 (진짜 RoPE). 층별 end index 를 (global=first*3fs, local=0) 으로 두고 6번 commit 후 원복."""
        fs = b._frame_seqlen; L = b._caches["target"]["kv"]; saved = [(c["global_end_index"].clone(), c["local_end_index"].clone()) for c in L]; cf = b._committed_frames
        for c in L: c["global_end_index"].fill_(first_chunk * CS * fs); c["local_end_index"].fill_(0)
        for cj in range(6): b.commit(lat6[:, cj * CS:(cj + 1) * CS].to(b.device).float(), first_chunk + cj)
        for c, (ge, le) in zip(L, saved): c["global_end_index"].copy_(ge); c["local_end_index"].copy_(le)
        b._committed_frames = cf
    b.reset_caches(); torch.cuda.empty_cache(); g = torch.Generator().manual_seed(a.seed); t1 = time.time()
    for ci in range(k0):
        skip_draws(g); b.commit(kvl[:, ci * CS:(ci + 1) * CS].to(b.device).float(), ci)
    if a.correction:
        if a.anchor == "rolling" and first > 0:
            write_sink(kvl[:, first * CS:(first + 6) * CS], first); meta["anchor_chunks"] = [first, first + 5]; print("[e2e] rolling 앵커: sink <- 청크 %d..%d" % (first, first + 5), flush=True)
        elif meta.get("anchor_chunks") is None: meta["anchor_chunks"] = [0, 5]
    print("[e2e] 복원 %d 청크 %.0fs" % (k0, time.time() - t1), flush=True)
    outs = []; t2 = time.time()
    for ci in range(k0, a.gen_until):
        x0 = gen_chunk(ci, g); b.commit(x0, ci); outs.append(x0.detach().float().cpu())
    new = torch.cat(outs, dim=1)
    disp = new if disp is None else torch.cat([disp, new], dim=1); kvl = new.clone() if kvl is None else torch.cat([kvl, new.clone()], dim=1)
    meta["done"] = a.gen_until; torch.save(disp, os.path.join(a.state, "display_lat.pt")); torch.save(kvl, os.path.join(a.state, "kv_lat.pt"))
    k = a.gen_until
    if k >= 6:   # 다음 교정용 prefix = display 청크 k-6..k-4 (클라우드가 받는 2.25 s), 첫 프레임 jpg
        pre = disp[:, (k - 6) * CS:(k - 3) * CS]; torch.save(pre, os.path.join(a.state, "prefix_k%d_lat.pt" % k))
        with torch.no_grad(): v8 = to_u8(b.pipe.vae.decode([pre.to(b.device)])[0].detach().cpu())
        imageio.mimwrite(os.path.join(a.state, "prefix_k%d.mp4" % k), list(v8), fps=16, quality=9); imageio.imwrite(os.path.join(a.state, "prefix_k%d_first.jpg" % k), v8[0], quality=97)
    print("[e2e] 생성 %d..%d %.0fs" % (k0, k - 1, time.time() - t2), flush=True)
    if a.decode:
        with torch.no_grad(): v8 = to_u8(b.pipe.vae.decode([disp.to(b.device)])[0].detach().cpu())
        np.save(os.path.join(a.state, "frames_all.npy"), v8[::4]); imageio.mimwrite(os.path.join(a.state, "video.mp4"), list(v8), fps=16, quality=8); print("[e2e] display 영상 %d 프레임" % len(v8), flush=True)
    json.dump(meta, open(meta_p, "w"), indent=1); print("[E2E_STAGE_DONE] done=%d" % k, flush=True)
if __name__ == "__main__":
    main()
