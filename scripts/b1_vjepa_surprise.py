#!/usr/bin/env python3
"""B1 ① V-JEPA 2 surprise: predictor 가 과거 컨텍스트로 예측한 미래 특징 vs 실제 인코딩된 특징의 L1 → 프레임별 '놀람' 곡선.
HF VJEPA2Model: encoder + predictor. 슬라이딩 윈도(16 프레임 @ ~8fps): 앞 12 프레임(컨텍스트) → 뒤 4 프레임 예측. 출력: b1/<name>_surprise.json"""
import argparse, json, os, numpy as np, torch, imageio
from transformers import AutoModel, AutoVideoProcessor
ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--out", required=True); ap.add_argument("--fps", type=float, default=8.0); ap.add_argument("--ctx", type=int, default=12); ap.add_argument("--win", type=int, default=16)
a = ap.parse_args(); dev = "cuda"
S = "/home/hk28965/specwm"; proc = AutoVideoProcessor.from_pretrained(S + "/hf/vjepa2-vitl-fpc64-256"); m = AutoModel.from_pretrained(S + "/hf/vjepa2-vitl-fpc64-256", torch_dtype=torch.float16).to(dev).eval()
rd = imageio.get_reader(a.video); fps = rd.get_meta_data().get("fps", 16); step = max(int(round(fps / a.fps)), 1); V = np.stack([f for i, f in enumerate(rd) if i % step == 0]); T = len(V)
tub = m.config.tubelet_size if hasattr(m.config, "tubelet_size") else 2
out = []; N = None
with torch.no_grad():
    for s in range(0, T - a.win + 1, tub):
        clip = V[s:s + a.win]; x = proc(list(clip), return_tensors="pt")["pixel_values_videos"].to(dev, torch.float16)   # [1, T, C, H, W]
        n_t = a.win // tub
        if N is None:
            N = m.encoder(pixel_values_videos=x).last_hidden_state.shape[1]
        hw = N // n_t; n_ctx = (a.ctx // tub) * hw
        ctx_mask = [torch.arange(n_ctx, device=dev).unsqueeze(0)]; tgt_mask = [torch.arange(n_ctx, N, device=dev).unsqueeze(0)]
        o = m(pixel_values_videos=x, context_mask=ctx_mask, target_mask=tgt_mask)
        pred = o.predictor_output.last_hidden_state; tgt = o.predictor_output.target_hidden_state                      # [1, N_tgt, D]
        tgt_n = torch.nn.functional.layer_norm(tgt.float(), tgt.shape[-1:]); pred_n = torch.nn.functional.layer_norm(pred.float(), pred.shape[-1:])
        err = (pred_n - tgt_n).abs().mean(-1).view(n_t - a.ctx // tub, hw).mean(-1)                                # 예측된 tubelet 별 surprise
        for j, e in enumerate(err.tolist()): out.append(dict(t=(s + a.ctx + j * tub) / a.fps, surprise=e))
json.dump(dict(video=a.video, fps=a.fps, frames=int(T), points=out), open(a.out, "w"), indent=1); print("[VJEPA_DONE]", a.out, "n=%d mean=%.4f max=%.4f" % (len(out), np.mean([o["surprise"] for o in out]), max(o["surprise"] for o in out)))
