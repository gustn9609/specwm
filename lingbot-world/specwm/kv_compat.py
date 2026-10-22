"""KV 교정 타당성 검사 (3단계, v1/v2 의 wan 패키지가 달라 프로세스를 분리한다).

  --stage fast    : Fast(v2) 로 청크0 잠재 z 와 각 층 self_attn k/v 출력을 덤프
  --stage base    : Base(v1) 고노이즈 expert 에 '같은 z' 를 넣어 k/v 덤프
  --stage compare : 층별 모양/코사인/노름비 비교

판정: 코사인 < 0.5 또는 노름비가 0.5~2.0 밖이면 주입은 OOD → KV 교정 경로 폐기.
      코사인 >= 0.7 이면 실제 주입 구현으로 진행.
"""
import argparse, json, os, numpy as np, torch


def hook_kv(model, call, max_layers):
    store = {}; hooks = []; n = 0
    def mk(name):
        def fn(mod, inp, out):
            if name not in store: store[name] = out.detach().float().cpu()
        return fn
    for name, mod in model.named_modules():
        if "self_attn" in name and (name.endswith(".k") or name.endswith(".v")):
            hooks.append(mod.register_forward_hook(mk(name))); n += 1
            if n >= max_layers * 2: break
    with torch.no_grad(): call()
    for h in hooks: h.remove()
    return store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["fast", "base", "compare"])
    ap.add_argument("--ckpt"); ap.add_argument("--action-root"); ap.add_argument("--prompt-file")
    ap.add_argument("--max-layers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    if a.stage == "fast":
        from .binding_v2 import V2Binding
        prompt = open(a.prompt_file).read().strip()
        fb = V2Binding(a.ckpt, target_mode="fast_bf16", quant=None, sink=6, local_attn=18,
                       swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
        fb.prepare(os.path.join("examples", "00", "image.jpg"), prompt, a.action_root, frame_num=13)
        g = torch.Generator().manual_seed(1234); z = fb.chunk_noise(0, g)
        kv = hook_kv(fb.target_model, lambda: fb.target_x0(z, 0, 0), a.max_layers)
        torch.save({"z": z.detach().cpu(), "kv": kv, "t_node0": float(fb.node_sigmas()[0])}, os.path.join(a.out, "fast_kv.pt"))
        print("[fast] z %s, %d 텐서, 예: %s" % (list(z.shape), len(kv), list(kv)[:2]), flush=True)

    elif a.stage == "base":
        from .binding_v1 import V1Binding
        prompt = open(a.prompt_file).read().strip()
        d = torch.load(os.path.join(a.out, "fast_kv.pt"))
        bb = V1Binding(a.ckpt, t5_cpu=True)
        models = {"high": bb.pipe.high_noise_model, "low": bb.pipe.low_noise_model}
        models["low"].cpu(); models["high"].to(bb.device); torch.cuda.empty_cache()
        # Fast 청크 = 잠재 3프레임. Base 는 frame_num=9 -> 잠재 3프레임 으로 모양을 맞춘다.
        cond = bb.prepare(os.path.join("examples", "00", "image.jpg"), prompt, a.action_root, frame_num=9)
        sig, ts = bb.node_grid(4)
        x0 = bb.make_x0_fn(models, cond, sig, ts)
        z = d["z"].to(bb.device)
        print("[base] z %s, node0 t=%.0f (고노이즈 expert)" % (list(z.shape), float(ts[0])), flush=True)
        kv = hook_kv(models["high"], lambda: x0(z, 0), a.max_layers)
        torch.save({"kv": kv}, os.path.join(a.out, "base_kv.pt"))
        print("[base] %d 텐서, 예: %s" % (len(kv), list(kv)[:2]), flush=True)

    else:
        F = torch.load(os.path.join(a.out, "fast_kv.pt"))["kv"]
        B = torch.load(os.path.join(a.out, "base_kv.pt"))["kv"]
        common = sorted(set(F) & set(B)); rep = {"fast_layers": len(F), "base_layers": len(B), "common": len(common)}
        rep["fast_names"] = list(F)[:4]; rep["base_names"] = list(B)[:4]
        sims, ratios, per = [], [], {}; shape_ok = True
        for n_ in common:
            A, Bt = F[n_], B[n_]
            if A.shape != Bt.shape: shape_ok = False; per[n_] = {"shape": [list(A.shape), list(Bt.shape)]}; continue
            a1, b1 = A.flatten(0, -2), Bt.flatten(0, -2)
            cs = torch.nn.functional.cosine_similarity(a1, b1, dim=-1).mean().item()
            nr = (Bt.norm() / (A.norm() + 1e-9)).item()
            # 무작위 기준선: 토큰을 섞었을 때의 코사인 (구조적 유사도 대비)
            perm = torch.randperm(a1.shape[0]); cs_rand = torch.nn.functional.cosine_similarity(a1, b1[perm], dim=-1).mean().item()
            sims.append(cs); ratios.append(nr); per[n_] = {"cos": round(cs, 4), "cos_shuffled": round(cs_rand, 4), "norm_ratio": round(nr, 4)}
        rep.update(shapes_match=shape_ok, cos_mean=round(float(np.mean(sims)), 4) if sims else None,
                   cos_min=round(float(np.min(sims)), 4) if sims else None,
                   norm_ratio_mean=round(float(np.mean(ratios)), 4) if ratios else None, per_layer=per)
        if rep["cos_mean"] is None: verdict = "비교 불가 (공통 층 없음 또는 모양 불일치)"
        elif rep["cos_mean"] >= 0.7 and 0.5 <= rep["norm_ratio_mean"] <= 2.0: verdict = "GO: 주입 구현 진행"
        elif rep["cos_mean"] < 0.5 or not (0.5 <= rep["norm_ratio_mean"] <= 2.0): verdict = "STOP: OOD, KV 교정 경로 폐기"
        else: verdict = "GRAY: 0.5~0.7, 소규모 주입 실험으로 판정"
        rep["verdict"] = verdict
        json.dump(rep, open(os.path.join(a.out, "kv_compat.json"), "w"), indent=1)
        print(json.dumps({k: v for k, v in rep.items() if k != "per_layer"}, indent=1, ensure_ascii=False), flush=True)
        for n_, v in list(per.items())[:6]: print("  ", n_, v, flush=True)
        print("[KV_COMPAT_DONE]", flush=True)


if __name__ == "__main__":
    main()
