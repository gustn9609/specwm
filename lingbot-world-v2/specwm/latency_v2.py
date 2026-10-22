"""TTFF 와 정상상태 청크 지연을 분리해 보고 (action item #6).

정의:
  ttff_cold_s   = 프로세스 첫 요청: prepare(텍스트/이미지/카메라 인코딩) + 청크0 생성 + 청크0 디코드
                  (CUDA 초기화, 커널 오토튠, 첫 페이지 폴트 포함 -> 사용자가 실제로 겪는 첫 프레임 지연)
  ttff_warm_s   = 모델이 이미 워밍업된 뒤 새 요청(새 prepare, 빈 캐시): prepare + 청크0 생성 + 청크0 디코드
                  (분해: prepare_s / gen0_s / dec0_s)
  steady_*      = 청크 >= steady_from 의 생성/디코드 중앙값 (기본: window 가 가득 차는 청크 6 = 18 잠재프레임)
  early_*       = 청크 2..steady_from-1 (캐시가 아직 덜 찬 구간)
  ingest_ctx_s  = cloud 컨텍스트 k 청크를 commit 하는 edge 비용 (resync 의 첫 프레임 지연에 더해지는 항)
  throughput_fps= 12 / (steady_gen + steady_dec)
모든 경계는 torch.cuda.synchronize().
arm: fast_bf16 | fast_kv4 | ours(GPTQ int6 + kv4)   x  --areas (해상도)
"""
import argparse, json, os, time, gc
import numpy as np, torch, torch.nn as nn
from .latency import parent_and_leaf, sync

GiB = 2 ** 30


def run_arm(a, tag, gptq, kv_bits, la, area, wq=None):
    """wq: None | "gptq"(int6) | "int4"(PackedInt4, weight-only) | "awq"(AWQ int4). gptq=True 는 wq="gptq" 와 동일(하위호환)."""
    wq = wq or ("gptq" if gptq else None); gptq = wq is not None
    from .binding_v2 import V2Binding
    from .vae_tiled import tiled_decode
    from . import kvstore2, lowbit_alloc
    prompt = open(a.prompt_file).read().strip(); image = os.path.join("examples", a.scene, "image.jpg")
    T = {}
    sync(); t0 = time.time()
    b = V2Binding(a.ckpt, target_mode="fast_bf16", quant=None, sink=6, local_attn=la,
                  swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750),
                  cfg_name=a.cfg, assets_dir=(os.path.dirname(a.ckpt.rstrip("/")) + "/lingbot-world-v2-14b-causal-fast") if a.cfg != "i2v-A14B" else None)
    if kv_bits: lowbit_alloc.patch(b.pipe, verbose=False)
    if a.t5_gpu:   # prepare() 는 text_encoder([prompt], cpu) 를 호출 -> GPU 상주 T5 로 우회 (TTFF 의 prepare 항이 T5 위치에 얼마나 의존하는지)
        te = b.pipe.text_encoder; te.model.to(b.device)
        class _TE:
            def __call__(self, texts, device): return te(texts, b.device)
        b.pipe.text_encoder = _TE()
    sync(); T["load_s"] = time.time() - t0
    tm = b.target_model
    if a.fast_wc: from . import worldcache_fast; worldcache_fast.install(tm, k=a.fast_wc, thresh=a.wc_thresh, ret_ratio=a.wc_ret, steps_per_chunk=len(a.nodes.split(","))); print("[fast-wc] installed k=%d" % a.fast_wc, flush=True)
    sig = None

    def prep():
        sync(); t = time.time(); c = b.prepare(image, prompt, a.action_root, frame_num=a.frames, max_area=area, chunk_size=a.chunk_size); sync()
        return c, time.time() - t

    NODES = [int(x) for x in a.nodes.split(",")]
    def one_chunk(ci, g):
        z = b.chunk_noise(ci, g); x0 = None
        for j, i in enumerate(NODES):
            x0 = b.target_x0(z, i, ci)
            if j < len(NODES) - 1: s = float(sig[NODES[j + 1]]); z = (1 - s) * x0 + s * b.randn_like(x0, g)
        b.commit(x0, ci); return x0

    decode = (lambda lat: tiled_decode(b.pipe.vae, lat, tiles=(2, 2))) if gptq else (lambda lat: b.pipe.vae.decode([lat])[0])

    # ---- cold: 첫 요청 ----
    cond, T["prepare_cold_s"] = prep(); N = cond["num_chunks"]; sig = b.node_sigmas()
    if kv_bits: kvstore2.install(tm, bits=kv_bits, k_axis="token", v_axis="token", asym=True, verbose=False)
    if wq == "gptq":
        from .gptq import HessianCollector, GPTQLinear
        th = time.time(); hc = HessianCollector(tm, min_features=1024); b.reset_caches(); g0 = torch.Generator().manual_seed(7)
        for ci in range(min(3, N)): one_chunk(ci, g0)
        Hs = hc.finish(); b.reset_caches()
        for n_ in [n for n, m in tm.named_modules() if isinstance(m, nn.Linear) and min(m.in_features, m.out_features) >= 1024]:
            par, leaf = parent_and_leaf(tm, n_); lin = getattr(par, leaf); setattr(par, leaf, GPTQLinear(lin, Hs[n_], bits=6, group=128)); del lin
        del Hs; gc.collect(); torch.cuda.empty_cache(); sync(); T["quant_s"] = time.time() - th
        T["alloc_after_quant_GiB"] = torch.cuda.memory_allocated() / GiB; print("[quant] resident after quant %.1f GiB" % T["alloc_after_quant_GiB"], flush=True)
        if os.environ.get("SPECWM_MEM_CAP_AFTER_QUANT_GB"):   # 4090 emulation for a pre-quantized deployment: cap the allocator from here on
            cap = float(os.environ["SPECWM_MEM_CAP_AFTER_QUANT_GB"]); tot = torch.cuda.get_device_properties(0).total_memory / GiB; torch.cuda.set_per_process_memory_fraction(min(1.0, cap / tot)); print("[memcap-after-quant] %.0f GB" % cap, flush=True)
    elif wq == "int4":
        from .quantize import quantize_packed_int4
        th = time.time(); quantize_packed_int4(tm, group=128, search=False, verbose=False); gc.collect(); torch.cuda.empty_cache(); sync(); T["quant_s"] = time.time() - th
    elif wq == "awq":
        from .awq import ActStats, quantize_awq
        th = time.time(); st = ActStats(tm, min_features=1024); b.reset_caches(); g0 = torch.Generator().manual_seed(7)
        for ci in range(min(3, N)): one_chunk(ci, g0)
        means = st.finish(); b.reset_caches(); quantize_awq(tm, means, alpha=0.5, group=128, verbose=False); del means; gc.collect(); torch.cuda.empty_cache(); sync(); T["quant_s"] = time.time() - th
    b.reset_caches(); g = torch.Generator().manual_seed(1234)
    sync(); t = time.time(); x0 = one_chunk(0, g); sync(); T["gen0_cold_s"] = time.time() - t
    t = time.time(); px = decode(x0.to(b.device)); sync(); T["dec0_cold_s"] = time.time() - t; del px
    T["ttff_cold_s"] = T["prepare_cold_s"] + T["gen0_cold_s"] + T["dec0_cold_s"] + (T.get("quant_s", 0.0) if a.count_quant_in_cold else 0.0)
    for ci in range(1, min(3, N)): one_chunk(ci, g)          # 워밍업 계속

    # ---- warm: 새 요청 (새 prepare, 빈 캐시), 전체 스트림 ----
    cond, T["prepare_warm_s"] = prep(); N = cond["num_chunks"]
    b.reset_caches(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); g = torch.Generator().manual_seed(1234)
    gen_t, dec_t = [], []
    for ci in range(N):
        sync(); t1 = time.time(); x0 = one_chunk(ci, g); sync(); tg = time.time() - t1
        t2 = time.time(); px = decode(x0.to(b.device)); sync(); td = time.time() - t2
        gen_t.append(tg); dec_t.append(td)
        if ci == 0: first_px = int(px.shape[1])
        if ci == 1: torch.cuda.reset_peak_memory_stats()
        del px
    T["gen0_warm_s"], T["dec0_warm_s"] = gen_t[0], dec_t[0]
    T["ttff_warm_s"] = T["prepare_warm_s"] + gen_t[0] + dec_t[0]
    sf = min(a.steady_from, N - 1)
    early = slice(2, sf) if sf > 2 else slice(1, 2)
    T.update(early_gen_s=float(np.median(gen_t[early])), early_dec_s=float(np.median(dec_t[early])),
             steady_gen_s=float(np.median(gen_t[sf:])), steady_dec_s=float(np.median(dec_t[sf:])),
             steady_gen_p90_s=float(np.percentile(gen_t[sf:], 90)), steady_gen_max_s=float(max(gen_t[sf:])))
    fpc = 4.0 * a.chunk_size; T["throughput_fps"] = fpc / (T["steady_gen_s"] + T["steady_dec_s"]); T["gen_only_fps"] = fpc / T["steady_gen_s"]
    T["peak_alloc_GiB"] = torch.cuda.max_memory_allocated() / GiB
    T["kv_GiB"] = kvstore2.total_bytes(tm) / GiB if kv_bits else None

    # ---- resync ingest: 컨텍스트 k 청크 commit 비용 (자기 latent 로 대체; 비용은 동일) ----
    b.reset_caches(); torch.cuda.empty_cache(); g = torch.Generator().manual_seed(1234)
    x0s = [one_chunk(ci, g) for ci in range(a.k)]
    b.reset_caches(); torch.cuda.empty_cache()
    sync(); t = time.time()
    for ci, x0 in enumerate(x0s): b.commit(x0, ci)
    sync(); T["ingest_ctx_s"] = time.time() - t; T["ingest_ctx_chunks"] = a.k
    from . import worldcache_fast as _wcf
    rep = dict(tag=tag, gptq=gptq, wq=wq, kv_bits=kv_bits, nodes=a.nodes, fast_wc=(_wcf.stats(tm) if a.fast_wc else None), window=la, max_area=area, t5_gpu=a.t5_gpu, chunk_size=a.chunk_size, chunks=N, first_chunk_px_frames=first_px,
               per_chunk_gen_s=[round(x, 3) for x in gen_t], per_chunk_dec_s=[round(x, 3) for x in dec_t],
               **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in T.items()})
    print("[%s] ttff cold %.2fs (prep %.2f gen0 %.2f dec0 %.2f) | warm %.2fs (prep %.2f gen0 %.2f dec0 %.2f) | steady gen %.3f dec %.3f -> %.2f fps | ingest%d %.2fs" % (
        tag, T["ttff_cold_s"], T["prepare_cold_s"], T["gen0_cold_s"], T["dec0_cold_s"], T["ttff_warm_s"], T["prepare_warm_s"], gen_t[0], dec_t[0],
        T["steady_gen_s"], T["steady_dec_s"], T["throughput_fps"], a.k, T["ingest_ctx_s"]), flush=True)
    try: kvstore2.uninstall(tm, verbose=False)
    except Exception: pass
    b._caches = {}; del b; gc.collect(); torch.cuda.empty_cache()
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--action-root", required=True)
    ap.add_argument("--prompt-file", required=True); ap.add_argument("--scene", default="05")
    ap.add_argument("--frames", type=int, default=161); ap.add_argument("--arms", default="fast_bf16,fast_kv4,ours")
    ap.add_argument("--areas", default="399360,184320", help="max_area 목록 (480x832=399360, 360x512=184320)")
    ap.add_argument("--steady-from", type=int, default=6); ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--count-quant-in-cold", action="store_true", help="GPTQ 캘리브레이션 시간을 cold TTFF 에 포함")
    ap.add_argument("--t5-gpu", action="store_true", help="umT5 를 GPU 에 상주시켜 prepare 측정")
    ap.add_argument("--chunk-size", type=int, default=3, help="청크당 잠재프레임 수 (1 = 프레임 단위 디코딩)")
    ap.add_argument("--fast-wc", type=int, default=0, help="Fast 에 WorldCache 적용 (probe k 블록)"); ap.add_argument("--wc-thresh", type=float, default=0.06); ap.add_argument("--wc-ret", type=float, default=0.25)
    ap.add_argument("--cfg", default="i2v-A14B", help="i2v-1.3B for the 1.3B checkpoint (assets from the 14B dir)"); ap.add_argument("--nodes", default="0,1,2,3", help="디노이즈 노드 (few-step: \"0,3\"=2-step)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if os.environ.get("SPECWM_MEM_CAP_GB"):   # 4090 emulation: cap the allocator at N GB (OOM if the stack does not fit)
        cap = float(os.environ["SPECWM_MEM_CAP_GB"]); tot = torch.cuda.get_device_properties(0).total_memory / 2**30; torch.cuda.set_per_process_memory_fraction(min(1.0, cap / tot)); print("[memcap] %.0f GB of %.0f GB" % (cap, tot), flush=True)
    ARMS = {"fast_bf16": dict(gptq=False, kv_bits=None, la=18), "fast_kv4": dict(gptq=False, kv_bits=4, la=18), "ours": dict(gptq=True, kv_bits=4, la=18),
            "int4": dict(gptq=False, kv_bits=4, la=18, wq="int4"), "int4_only": dict(gptq=False, kv_bits=None, la=18, wq="int4"), "awq_only": dict(gptq=False, kv_bits=None, la=18, wq="awq"), "awq": dict(gptq=False, kv_bits=4, la=18, wq="awq"), "gptq6_only": dict(gptq=True, kv_bits=None, la=18),
            "fast_w36": dict(gptq=False, kv_bits=None, la=36)}
    rp = os.path.join(a.out, "latency_v2.json"); rep = json.load(open(rp)) if os.path.exists(rp) else {"arms": {}}
    for area in [int(x) for x in a.areas.split(",")]:
        for tag in a.arms.split(","):
            key = "%s@%d%s%s" % (tag, area, "" if a.chunk_size == 3 else "_cs%d" % a.chunk_size, ("_wc%d_thr%g_ret%g" % (a.fast_wc, a.wc_thresh, a.wc_ret)) if a.fast_wc else "") + ("" if a.nodes == "0,1,2,3" else "_n" + a.nodes.replace(",", ""))
            if key in rep["arms"] and "error" not in rep["arms"][key]: print("[%s] 이미 있음" % key, flush=True); continue
            try: rep["arms"][key] = run_arm(a, key, area=area, **ARMS[tag])
            except Exception as e:
                rep["arms"][key] = {"error": str(e)[:300]}; import traceback; traceback.print_exc(); print("[%s] 실패: %s" % (key, str(e)[:300]), flush=True); gc.collect(); torch.cuda.empty_cache()
            json.dump(rep, open(rp, "w"), indent=1)
    print("\n%-22s %9s %9s %9s %9s %8s %9s" % ("arm", "ttff_cold", "ttff_warm", "steady_g", "steady_d", "fps", "ingest6"), flush=True)
    for k, v in rep["arms"].items():
        if "error" in v: print("%-22s ERROR" % k); continue
        print("%-22s %9.2f %9.2f %9.3f %9.3f %8.2f %9.2f" % (k, v["ttff_cold_s"], v["ttff_warm_s"], v["steady_gen_s"], v["steady_dec_s"], v["throughput_fps"], v["ingest_ctx_s"]), flush=True)
    print("[LATENCY_V2_DONE]", flush=True)


if __name__ == "__main__":
    main()
