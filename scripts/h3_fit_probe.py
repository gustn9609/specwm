#!/usr/bin/env python3
"""MiniMax-H3 적합성 조사 (action item #3). 네트워크만 사용 (가중치 다운로드 없음).
HF API 메타데이터 + config.json 들을 읽어 파라미터 수 / 메모리 / 스트리밍·액션 조건 지원 여부를 정리하고
우리 설정(edge 24GB, cloud 96GB, 청크 스트리밍, 카메라/액션 조건) 과의 적합성을 판정한다."""
import json, os, sys, re, urllib.request

REPO = "MiniMaxAI/MiniMax-H3"
OUT = os.path.join(os.environ.get("SCRATCH", "/home/hk28965/specwm"), "lingbot-world-v2", "specwm_out", "h3_fit")
os.makedirs(OUT, exist_ok=True)


def get(url, binary=False):
    req = urllib.request.Request(url, headers={"User-Agent": "specwm-probe"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = r.read(); return d if binary else d.decode("utf-8", "replace")


def try_json(path):
    try: return json.loads(get(f"https://huggingface.co/{REPO}/raw/main/{path}"))
    except Exception as e: return {"_error": str(e)[:120]}


meta = json.loads(get(f"https://huggingface.co/api/models/{REPO}?blobs=true"))
files = {s["rfilename"]: (s.get("size") or 0) for s in meta["siblings"]}
tops = {}
for f, sz in files.items():
    key = "/".join(f.split("/")[:2]) if "/" in f else f; tops[key] = tops.get(key, 0) + sz
readme = get(f"https://huggingface.co/{REPO}/raw/main/README.md")
cfgs = {p: try_json(p) for p in ["model_index.json", "transformer/config.json", "FL2VA/transformer/config.json", "FL2VA/model_index.json",
                                 "vae/config.json", "FL2VA/video_vae/config.json", "text_encoder/config.json"]}
idx = try_json("transformer/diffusion_pytorch_model.safetensors.index.json")
tf_bytes = (idx.get("metadata", {}) or {}).get("total_size") or tops.get("transformer", 0)
te_bytes = tops.get("text_encoder", 0) or tops.get("FL2VA/text_encoder", 0)
vae_bytes = tops.get("vae", 0) or tops.get("FL2VA/video_vae", 0)
params_tf = tf_bytes / 2                              # bf16 가정


def grep(pat, n=3):
    return [m.strip() for m in re.findall(pat, readme, flags=re.I)][:n]


facts = dict(
    repo=REPO, license=meta.get("cardData", {}).get("license_name") or meta.get("cardData", {}).get("license"),
    total_repo_GiB=round(sum(files.values()) / 2 ** 30, 1), transformer_GiB=round(tf_bytes / 2 ** 30, 1), transformer_params_B=round(params_tf / 1e9, 1),
    text_encoder_GiB=round(te_bytes / 2 ** 30, 1), vae_GiB=round(vae_bytes / 2 ** 30, 1),
    variants=[k for k in tops if k in ("FL2VA", "Ref2VA") or k.startswith("FL2VA/") or k.startswith("Ref2VA/")][:6],
    duration=grep(r"Output duration \|([^|]+)\|"), resolution=grep(r"Output resolution \|([^|]+)\|"), fps=grep(r"Output frame rate \|([^|]+)\|"),
    sparse_attention_released=("full attention only" not in readme.lower()),
    mentions_action_or_camera_control=bool(re.search(r"camera (pose|trajectory|control)|action[- ]condition|plucker|c2w|world model", readme, re.I)),
    mentions_streaming_or_causal=bool(re.search(r"streaming|causal|autoregressive|real[- ]time", readme, re.I)),
    top_dirs_GiB={k: round(v / 2 ** 30, 1) for k, v in sorted(tops.items(), key=lambda x: -x[1])[:10]},
    configs={k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (list, dict))} for k, v in cfgs.items() if isinstance(v, dict)})

edge_int4_GiB = params_tf * 0.5 / 2 ** 30 + 2          # 4bit + 여유
our_fast_params_B = 14.0
verdict = []
verdict.append("transformer %.0fB (bf16 %.0f GiB) vs 우리 Fast 14B: %.1fx 크다 -> INT4 로도 ~%.0f GiB, 24GB 엣지 부적합" % (
    facts["transformer_params_B"], facts["transformer_GiB"], facts["transformer_params_B"] / our_fast_params_B, edge_int4_GiB))
verdict.append("text encoder Qwen3-VL-32B (%.0f GiB bf16) 를 별도로 상주시켜야 함 -> cloud 96GB 1장에 transformer+encoder 동시 상주 불가(오프로드 필요)" % facts["text_encoder_GiB"])
verdict.append("공개 버전은 full attention 만 (sparse 미공개), 4~15s 클립 단위 양방향 생성 -> 청크 스트리밍/causal KV 캐시 구조 없음"
               if not facts["sparse_attention_released"] else "sparse attention 공개됨 (README 재확인 필요)")
verdict.append("카메라/액션 조건 %s -> world-model 로 쓰려면 별도 어댑터 학습 필요" % ("언급 있음(확인)" if facts["mentions_action_or_camera_control"] else "없음"))
verdict.append("가능한 역할: cloud 측 '품질 상한 참조' 또는 오프라인 정제기. edge/draft 후보로는 부적합. 라이선스: %s" % facts["license"])
facts["verdict"] = verdict
json.dump(facts, open(os.path.join(OUT, "h3_fit.json"), "w"), indent=1, ensure_ascii=False)
md = ["# MiniMax-H3 fit check (auto-generated, network metadata only)", "", "| item | value |", "|---|---|"]
for k in ["repo", "license", "total_repo_GiB", "transformer_GiB", "transformer_params_B", "text_encoder_GiB", "vae_GiB", "duration", "resolution", "fps",
          "sparse_attention_released", "mentions_action_or_camera_control", "mentions_streaming_or_causal"]:
    md.append("| %s | %s |" % (k, facts[k]))
md += ["", "## Verdict"] + ["- " + v for v in verdict] + ["", "## Top dirs (GiB)", "```", json.dumps(facts["top_dirs_GiB"], indent=1), "```",
                                                            "", "## Configs (scalars)", "```", json.dumps(facts["configs"], indent=1)[:6000], "```"]
open(os.path.join(OUT, "h3_fit.md"), "w").write("\n".join(md))
print("\n".join(md[:30])); print("[H3_FIT_DONE]", OUT)
