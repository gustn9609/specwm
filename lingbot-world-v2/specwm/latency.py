"""지연(time-to-first-frame) / 처리량(fps) / 피크 메모리 측정.

Guandao 요구: total seconds 가 아니라
  (1) latency  = 입력 후 첫 프레임까지 걸리는 시간
  (2) throughput = 워밍업 이후 프레임 1장을 더 얻는 데 드는 시간의 역수 (fps)
  (3) peak memory (추론)

정의 (이 스크립트):
  ttff_s          = 청크0 생성 + 청크0 디코드 (3잠재프레임 -> 9픽셀프레임)
  gen_chunk_s     = 청크 2.. 의 생성 시간 중앙값 (워밍업 제외)
  dec_chunk_s     = 청크 2.. 를 '독립적으로' 디코드한 시간 중앙값
                    (VAE 캐시 없이 청크 단위 디코드; 실제 스트리밍 구현보다 약간 비관적)
  throughput_fps  = 12 / (gen_chunk_s + dec_chunk_s)     (청크당 새 픽셀프레임 12장)
  gen_only_fps    = 12 / gen_chunk_s
  peak_alloc_GiB  = 워밍업 후 torch.cuda.max_memory_allocated
모든 시간은 torch.cuda.synchronize() 로 경계를 맞춘다.
"""
import argparse, json, os, time, gc
import numpy as np, torch, torch.nn as nn

GiB = 2 ** 30


def sync(): torch.cuda.synchronize()


def parent_and_leaf(model, name):
    parts = name.split("."); p = model
    for q in parts[:-1]:
        p = p[int(q)] if q.isdigit() else getattr(p, q)
    return p, parts[-1]


def run_arm(a, tag, gptq, kv_bits, la):
    from .binding_v2 import V2Binding
    from .vae_tiled import tiled_decode
    from . import kvstore2, lowbit_alloc
    prompt = open(a.prompt_file).read().strip()
    t0 = time.time()
    b = V2Binding(a.ckpt, target_mode="fast_bf16", quant=None, sink=6, local_attn=la,
                  swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
    if kv_bits: lowbit_alloc.patch(b.pipe, verbose=False)
    cond = b.prepare(os.path.join("examples", a.scene, "image.jpg"), prompt, a.action_root, frame_num=a.frames)
    N = cond["num_chunks"]; sig = b.node_sigmas(); tm = b.target_model
    if kv_bits:
        kvstore2.install(tm, bits=kv_bits, k_axis="token", v_axis="token", asym=True, verbose=False)
    sync(); load_s = time.time() - t0
    print("[%s] 로드 %.0fs, %d청크" % (tag, load_s, N), flush=True)

    def one_chunk(ci, g):
        z = b.chunk_noise(ci, g); x0 = None
        for i in range(len(sig)):
            x0 = b.target_x0(z, i, ci)
            if i < len(sig) - 1:
                s = float(sig[i + 1]); z = (1 - s) * x0 + s * b.randn_like(x0, g)
        b.commit(x0, ci); return x0

    quant_s = 0.0
    if gptq:
        from .gptq import HessianCollector, GPTQLinear
        th = time.time(); hc = HessianCollector(tm, min_features=1024)
        b.reset_caches(); g0 = torch.Generator().manual_seed(7)
        for ci in range(min(3, N)): one_chunk(ci, g0)
        Hs = hc.finish(); b.reset_caches()
        names = [n for n, m in tm.named_modules() if isinstance(m, nn.Linear) and min(m.in_features, m.out_features) >= 1024]
        for n_ in names:
            par, leaf = parent_and_leaf(tm, n_); lin = getattr(par, leaf)
            setattr(par, leaf, GPTQLinear(lin, Hs[n_], bits=6, group=128)); del lin
        del Hs; gc.collect(); torch.cuda.empty_cache(); sync(); quant_s = time.time() - th
        print("[%s] GPTQ %.0fs" % (tag, quant_s), flush=True)

    decode = (lambda lat: tiled_decode(b.pipe.vae, lat, tiles=(2, 2))) if gptq else (lambda lat: b.pipe.vae.decode([lat])[0])

    # ---- 측정 ----
    b.reset_caches(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    g = torch.Generator().manual_seed(1234)
    gen_t, dec_t, outs = [], [], []
    for ci in range(N):
        sync(); t1 = time.time(); x0 = one_chunk(ci, g); sync(); tgc = time.time() - t1
        t2 = time.time(); px = decode(x0.to(b.device)); sync(); tdc = time.time() - t2
        gen_t.append(tgc); dec_t.append(tdc); outs.append(x0.detach().float().cpu())
        if ci == 0: ttff = tgc + tdc; first_px_frames = int(px.shape[1])
        if ci == 1: torch.cuda.reset_peak_memory_stats()   # 워밍업 이후 피크
        if ci % 4 == 0: print("   청크 %d/%d gen %.2fs dec %.2fs" % (ci, N, tgc, tdc), flush=True)
        del px
    peak = torch.cuda.max_memory_allocated() / GiB
    gen_med = float(np.median(gen_t[2:])) if N > 2 else float(np.median(gen_t))
    dec_med = float(np.median(dec_t[2:])) if N > 2 else float(np.median(dec_t))
    rep = dict(tag=tag, gptq=gptq, kv_bits=kv_bits, window=la, chunks=N,
               load_s=round(load_s, 1), quant_s=round(quant_s, 0),
               ttff_s=round(ttff, 3), first_chunk_px_frames=first_px_frames,
               gen_chunk_s=round(gen_med, 3), dec_chunk_s=round(dec_med, 3),
               throughput_fps=round(12.0 / (gen_med + dec_med), 2), gen_only_fps=round(12.0 / gen_med, 2),
               total_gen_s=round(sum(gen_t), 1), total_dec_s=round(sum(dec_t), 1),
               peak_alloc_GiB=round(peak, 2),
               kv_GiB=round(kvstore2.total_bytes(tm) / GiB, 2) if kv_bits else None)
    print("[%s] %s" % (tag, rep), flush=True)
    try: kvstore2.uninstall(tm, verbose=False)
    except Exception: pass
    b._caches = {}; del b; gc.collect(); torch.cuda.empty_cache()
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--action-root", required=True)
    ap.add_argument("--prompt-file", required=True); ap.add_argument("--scene", default="05")
    ap.add_argument("--frames", type=int, default=161)
    ap.add_argument("--arms", default="fast_bf16,ours")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    ARMS = {"fast_bf16": dict(gptq=False, kv_bits=None, la=18),
            "fast_kv4": dict(gptq=False, kv_bits=4, la=18),
            "ours": dict(gptq=True, kv_bits=4, la=18)}
    rp = os.path.join(a.out, "latency.json")
    rep = json.load(open(rp)) if os.path.exists(rp) else {"arms": {}}
    for tag in a.arms.split(","):
        if tag in rep["arms"]: print("[%s] 이미 있음" % tag, flush=True); continue
        try: rep["arms"][tag] = run_arm(a, tag, **ARMS[tag])
        except Exception as e:
            rep["arms"][tag] = {"error": str(e)[:300]}; print("[%s] 실패: %s" % (tag, str(e)[:300]), flush=True)
        json.dump(rep, open(rp, "w"), indent=1)
    print("[LATENCY_DONE]", flush=True)


if __name__ == "__main__":
    main()
