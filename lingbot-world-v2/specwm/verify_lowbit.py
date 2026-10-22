"""lowbit_alloc 패치 검증: 소규모(13프레임)에서
  (a) 패치 유무로 출력이 동일한가  (수학 불변)
  (b) 순간 할당이 실제로 줄었는가
  (c) 큰 창(72)이 열리는가
"""
import argparse, time, gc, torch
GiB = 2 ** 30


def run(ckpt, action, prompt, frames, local_attn, use_patch, seed=1234):
    from .binding_v2 import V2Binding
    from . import kvstore2, lowbit_alloc
    b = V2Binding(ckpt, target_mode="fast_bf16", quant=None, sink=6,
                  local_attn=local_attn, swap_models=False, build_draft=False,
                  timesteps_index=(0, 250, 500, 750))
    orig = lowbit_alloc.patch(b.pipe, verbose=False) if use_patch else None
    cond = b.prepare("examples/00/image.jpg", prompt, action, frame_num=frames)
    N = cond["num_chunks"]; sig = b.node_sigmas()
    kvstore2.install(b.target_model, bits=4, k_axis="token", v_axis="token",
                     asym=True, verbose=False)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    b.reset_caches()
    alloc_peak = torch.cuda.max_memory_allocated() / GiB   # 캐시 생성 직후 = 순간 할당
    g = torch.Generator().manual_seed(seed); outs = []
    for ci in range(N):
        z = b.chunk_noise(ci, g); x0 = None
        for i in range(len(sig)):
            x0 = b.target_x0(z, i, ci)
            if i < len(sig) - 1:
                s = float(sig[i + 1]); z = (1 - s) * x0 + s * b.randn_like(x0, g)
        b.commit(x0, ci); outs.append(x0.detach().float().cpu())
    lat = torch.cat(outs, dim=1)
    total_peak = torch.cuda.max_memory_allocated() / GiB
    kvstore2.uninstall(b.target_model, verbose=False)
    if orig is not None:
        lowbit_alloc.unpatch(b.pipe, orig)
    b._caches = {}
    del b; gc.collect(); torch.cuda.empty_cache()
    return lat, alloc_peak, total_peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--action", required=True)
    ap.add_argument("--prompt-file", required=True)
    a = ap.parse_args()
    prompt = open(a.prompt_file).read().strip()

    print("=" * 68, flush=True)
    print("(a) 수학 불변 검증: 창 18, 13프레임, 패치 유무 비교", flush=True)
    l1, ap1, tp1 = run(a.ckpt, a.action, prompt, 13, 18, False)
    print("    패치 없음: 캐시직후 %.2fG 전체피크 %.2fG" % (ap1, tp1), flush=True)
    l2, ap2, tp2 = run(a.ckpt, a.action, prompt, 13, 18, True)
    print("    패치 적용: 캐시직후 %.2fG 전체피크 %.2fG" % (ap2, tp2), flush=True)
    d = (l1 - l2).abs().max().item()
    print("    출력 최대 차이 = %.3e  -> %s" % (d, "동일 (통과)" if d < 1e-5 else "다름 (실패!)"),
          flush=True)
    print("    순간 할당 절감 = %.2f GiB" % (ap1 - ap2), flush=True)

    print("=" * 68, flush=True)
    print("(c) 큰 창 열리는지: 창 72, 13프레임", flush=True)
    try:
        l3, ap3, tp3 = run(a.ckpt, a.action, prompt, 13, 72, True)
        print("    창 72 성공: 캐시직후 %.2fG 전체피크 %.2fG" % (ap3, tp3), flush=True)
    except Exception as e:
        print("    창 72 실패: %s" % str(e)[:200], flush=True)
    print("[VERIFY_DONE]", flush=True)


if __name__ == "__main__":
    main()
