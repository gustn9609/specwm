#!/usr/bin/env python3
"""
Score generated videos with VisionReward-Video in consecutive 6-second windows.

Entry point used (from THUDM/VisionReward, inference-video.py):
    from decord import cpu, VideoReader, bridge
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=..., trust_remote_code=True).eval().to('cuda')
    score(video_path, prompt) -> float   # weighted sum of 29 yes/no VQA dimensions

This script reimplements score()/inference()/load_video() verbatim (same behavior,
same VisionReward_Video/*.txt question set + weight.json) but:
  - points MODEL_PATH at the local snapshot in $SCRATCH/hf/VisionReward-Video
    (downloaded ahead of time; no network access needed on the compute node)
  - cuts every input video into consecutive 6-second windows with ffmpeg (-threads 2)
    before scoring each window
  - records per-window scalar score + per-dimension yes/no answers
  - writes incremental progress to $SCRATCH/vr_scores.json after every window/video
"""
import io
import json
import os
import subprocess
import sys
import time
import traceback

os.environ.setdefault("HF_HOME", os.path.join(os.environ["SCRATCH"], "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

# --- Compat shim -----------------------------------------------------------
# pytorchvideo==0.1.5 (pinned by VisionReward's requirements.txt) imports the
# long-removed `torchvision.transforms.functional_tensor` module (dropped in
# torchvision>=0.17). We had to move to torchvision 0.20.1 because PyPI only
# ships CPU-only aarch64 wheels for torchvision 0.16/torch 2.1 (no CUDA on
# GH200/aarch64) -- see notes in the task report. torchvision still ships the
# same functions privately under `_functional_tensor`, so re-expose that
# module under the old name before anything imports pytorchvideo.
import sys as _sys
import torchvision.transforms._functional_tensor as _ft
_sys.modules.setdefault("torchvision.transforms.functional_tensor", _ft)
# ----------------------------------------------------------------------------

from decord import cpu, VideoReader, bridge
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRATCH = os.environ["SCRATCH"]
VR_REPO = os.path.join(SCRATCH, "VisionReward")
MODEL_PATH = os.environ.get("VR_MODEL_PATH", os.path.join(SCRATCH, "hf", "VisionReward-Video"))
QUESTIONS_PATH = os.path.join(VR_REPO, "VisionReward_Video", "VisionReward_video_qa_select.txt")
WEIGHT_PATH = os.path.join(VR_REPO, "VisionReward_Video", "weight.json")

WINDOW_SEC = 6.0
FFMPEG_BIN = None
try:
    import imageio_ffmpeg
    FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_BIN = "ffmpeg"

R = os.path.join(SCRATCH, "lingbot-world-v2", "specwm_out")
WINDOWS_DIR = os.path.join(SCRATCH, "vr_windows")
OUT_JSON = os.path.join(SCRATCH, "vr_scores.json")

with open(QUESTIONS_PATH, "r") as f:
    questions = f.readlines()
with open(WEIGHT_PATH, "r") as f:
    weight = np.array(json.load(f))

assert len(questions) == len(weight), f"{len(questions)} questions vs {len(weight)} weights"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

print(f"[vr_score] loading tokenizer/model from {MODEL_PATH} on {DEVICE} ({TORCH_TYPE}) ...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=TORCH_TYPE, trust_remote_code=True
).eval().to(DEVICE)
print("[vr_score] model loaded.", flush=True)


def load_video(video_data, strategy="chat"):
    bridge.set_bridge("torch")
    mp4_stream = video_data
    num_frames = 24
    decord_vr = VideoReader(io.BytesIO(mp4_stream), ctx=cpu(0))
    total_frames = len(decord_vr)
    if strategy == "base":
        clip_end_sec = 60
        clip_start_sec = 0
        start_frame = int(clip_start_sec * decord_vr.get_avg_fps())
        end_frame = min(total_frames, int(clip_end_sec * decord_vr.get_avg_fps())) if clip_end_sec is not None else total_frames
        frame_id_list = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    elif strategy == "chat":
        timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
        timestamps = [i[0] for i in timestamps]
        max_second = round(max(timestamps)) + 1
        frame_id_list = []
        for second in range(max_second):
            closest_num = min(timestamps, key=lambda x: abs(x - second))
            index = timestamps.index(closest_num)
            frame_id_list.append(index)
            if len(frame_id_list) >= num_frames:
                break
    video_data = decord_vr.get_batch(frame_id_list)
    video_data = video_data.permute(3, 0, 1, 2)
    return video_data


def inference(video_path, query, temperature=0.1):
    video_data = open(video_path, "rb").read()
    strategy = "chat"
    video = load_video(video_data, strategy=strategy)
    history = []
    inputs = model.build_conversation_input_ids(
        tokenizer=tokenizer, query=query, images=[video], history=history, template_version=strategy
    )
    inputs = {
        "input_ids": inputs["input_ids"].unsqueeze(0).to(DEVICE),
        "token_type_ids": inputs["token_type_ids"].unsqueeze(0).to(DEVICE),
        "attention_mask": inputs["attention_mask"].unsqueeze(0).to(DEVICE),
        "images": [[inputs["images"][0].to(DEVICE).to(TORCH_TYPE)]],
    }
    gen_kwargs = {
        "max_new_tokens": 2048,
        "pad_token_id": 128002,
        "top_k": 1,
        "do_sample": False,
        "top_p": 0.1,
        "temperature": temperature,
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(outputs[0]).strip()


def score_with_dims(video_path, prompt):
    """Same math as upstream score(), but also returns per-dimension yes/no answers."""
    queries = [q.replace("[[prompt]]", prompt) for q in questions]
    raw_answers = []
    for query in queries:
        raw_answers.append(inference(video_path, query))
    binary = np.array([1 if a.lower().startswith("yes") else -1 for a in raw_answers])
    scalar = float(np.mean(binary * weight))
    per_dim = {
        questions[i].strip(): {"answer": raw_answers[i], "weight": float(weight[i]), "signed": int(binary[i])}
        for i in range(len(questions))
    }
    return scalar, per_dim


def get_duration_sec(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    n = len(vr)
    fps = vr.get_avg_fps()
    return n / fps, n, fps


def cut_window(src, start_sec, dur_sec, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = [
        FFMPEG_BIN, "-y",
        "-ss", f"{start_sec:.3f}",
        "-i", src,
        "-t", f"{dur_sec:.3f}",
        "-threads", "2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-an",
        out_path,
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if res.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"ffmpeg failed for {src} [{start_sec},{start_sec+dur_sec}): {res.stdout.decode(errors='replace')[-2000:]}")
    return out_path


def save(results):
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, OUT_JSON)


def score_video(video_path, prompt, tag, results):
    entry = {"tag": tag, "video_path": video_path, "windows": [], "status": "running"}
    results["videos"].append(entry)
    save(results)
    try:
        duration, nframes, fps = get_duration_sec(video_path)
        entry["duration_sec"] = duration
        entry["nframes"] = nframes
        entry["fps"] = fps
        starts = []
        t = 0.0
        while t < duration - 1e-6:
            starts.append(t)
            t += WINDOW_SEC
        safe_tag = tag.replace("/", "__")
        for wi, start in enumerate(starts):
            dur = min(WINDOW_SEC, duration - start)
            clip_path = os.path.join(WINDOWS_DIR, safe_tag, f"win_{wi:03d}.mp4")
            t0 = time.time()
            cut_window(video_path, start, dur, clip_path)
            scalar, per_dim = score_with_dims(clip_path, prompt)
            elapsed = time.time() - t0
            entry["windows"].append({
                "window_index": wi,
                "start_sec": start,
                "duration_sec": dur,
                "clip_path": clip_path,
                "score": scalar,
                "per_dimension": per_dim,
                "elapsed_sec": elapsed,
            })
            print(f"[vr_score] {tag} window {wi} [{start:.2f},{start+dur:.2f}) score={scalar:.4f} ({elapsed:.1f}s)", flush=True)
            save(results)
        scores = [w["score"] for w in entry["windows"]]
        entry["mean_score"] = float(np.mean(scores)) if scores else None
        entry["min_score"] = float(np.min(scores)) if scores else None
        entry["last_window_score"] = scores[-1] if scores else None
        entry["num_windows"] = len(scores)
        entry["status"] = "done"
    except Exception as e:
        entry["status"] = "error"
        entry["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"[vr_score] ERROR on {tag}: {e}", flush=True)
    save(results)
    return entry


def main():
    entries_path = sys.argv[1] if len(sys.argv) > 1 else None
    if entries_path:
        with open(entries_path) as f:
            entries = json.load(f)
    else:
        raise SystemExit("usage: vr_score.py entries.json  (list of {video_path,prompt_path,tag})")

    results = {"model_path": MODEL_PATH, "device": DEVICE, "torch_dtype": str(TORCH_TYPE),
               "window_sec": WINDOW_SEC, "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "videos": [], "missing": []}

    for e in entries:
        if e.get("missing"):
            results["missing"].append(e)
            continue
        with open(e["prompt_path"]) as f:
            prompt = f.read().strip()
        score_video(e["video_path"], prompt, e["tag"], results)

    results["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save(results)
    print(f"[vr_score] DONE. Wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
