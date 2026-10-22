"""VBench custom-input 평가.

--stage download : 로그인 노드에서 필요한 사전학습 가중치를 캐시에 받는다 (init_submodules 호출).
--stage eval     : gh-dev 에서 local=True 로 평가. 결과 json 을 --out 에 모은다.

차원: VBench 가 custom 입력에 허용하는 6개 (프롬프트 무관 차원)
  subject_consistency, background_consistency, motion_smoothness,
  dynamic_degree, aesthetic_quality, imaging_quality
temporal_flickering 은 VBench 기준 '정적 영상' 전용이라 제외.
"""
import argparse, json, os, sys, glob, shutil, time

DIMS = ["subject_consistency", "background_consistency", "motion_smoothness",
        "dynamic_degree", "aesthetic_quality", "imaging_quality"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["download", "eval"], required=True)
    ap.add_argument("--videos", nargs="*", default=[], help="tag=path.mp4 ...")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dims", default=",".join(DIMS))
    a = ap.parse_args()
    dims = a.dims.split(",")
    import torch
    # VBench 의 사전학습 ckpt(AMT, RAFT 등)는 torch>=2.6 의 weights_only=True 기본값과 충돌한다.
    # VBench 공식 릴리스 파일만 로드하므로 기본값을 되돌린다.
    _orig_load = torch.load
    def _load_compat(*a, **k):
        k.setdefault("weights_only", False); return _orig_load(*a, **k)
    torch.load = _load_compat
    from vbench import VBench
    from vbench.utils import init_submodules, CACHE_DIR
    if a.stage == "download":
        print("[cache]", CACHE_DIR, flush=True)
        sub = init_submodules(dims, local=False, read_frame=False)
        print(json.dumps({k: str(v)[:160] for k, v in sub.items()}, indent=1), flush=True)
        print("[VBENCH_DOWNLOAD_DONE]", flush=True); return

    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for spec in a.videos:
        tag, path = spec.split("=", 1)
        assert os.path.exists(path), path
        work = os.path.join(a.out, "work_" + tag); os.makedirs(work, exist_ok=True)
        vb = VBench(dev, full_info_dir=os.path.join(work, "info.json"), output_path=work)
        t0 = time.time()
        vb.evaluate(videos_path=path, name=tag, dimension_list=dims, local=True, mode="custom_input")
        res_files = sorted(glob.glob(os.path.join(work, "*_eval_results.json")))
        r = json.load(open(res_files[-1])) if res_files else {}
        results[tag] = {d: (r[d][0] if isinstance(r.get(d), list) else r.get(d)) for d in dims}
        results[tag]["_eval_s"] = round(time.time() - t0, 1)
        print("[%s] %s" % (tag, results[tag]), flush=True)
        json.dump(results, open(os.path.join(a.out, "vbench.json"), "w"), indent=1)
    print("[VBENCH_EVAL_DONE]", flush=True)


if __name__ == "__main__":
    main()
