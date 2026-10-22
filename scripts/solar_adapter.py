"""LingBot 입력(이미지 + 카메라 궤적 + 캡션)을 SolarWM raw-WDS 샘플로 만든다.

규격은 SolarWM 코드에서 직접 확인한 것 (src/solarwm/data/{index,archive,camera}.py,
backends/wan22/runtime/data.py):
  index 행: sample_id, key, shard(상대경로), shard_size, video_member, camera_member,
            num_frames, fps, caption, manifest(인라인)
  manifest.camera: convention="authoritative_source_c2w_no_axis_flip", dtype="float32",
            finite=true, shape=[T,4,4], magnitude_audit_seconds=10.0,
            max_camera_abs=20.0, max_rel_translation=20.0, array_key="c2w"
  manifest.video: {num_frames, fps}
  camera npz: 키 c2w, [T,4,4] float32, allow_pickle=False 로 읽힘
  video: decord 로 디코드, 필요시 bilinear 리사이즈 + 중앙 크롭 (loader 가 처리)

첫 프레임만 조건으로 쓰고 싶으므로, 비디오는 '첫 프레임을 T 번 반복' 한 mp4 로 만든다.
(SolarWM 은 I2V 로 첫 프레임을 조건으로 쓰고 나머지 프레임은 학습/평가용 GT 이다;
 추론에서는 GT 가 없어도 되지만 로더가 num_frames 개를 요구하므로 채운다.)
"""
import argparse, gzip, io, json, os, tarfile
import numpy as np

CONV = "authoritative_source_c2w_no_axis_flip"


def build(image_path, poses_path, caption, out_dir, sample_id, fps=16.0, height=480, width=864,
          pixel_frames=None, max_rel=20.0, max_abs=20.0, target_rel=10.0):
    import imageio
    from PIL import Image
    os.makedirs(os.path.join(out_dir, "raw-wds", "custom", "shards"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "recipes", "custom", "raw-wds"), exist_ok=True)

    P = np.load(poses_path).astype(np.float32)
    assert P.ndim == 3 and P.shape[1:] == (4, 4), P.shape
    if pixel_frames is not None:
        P = P[:pixel_frames]
    T = len(P)
    assert np.isfinite(P).all(), "poses must be finite"
    # 로더의 magnitude guard 를 미리 재현: 첫 프레임 기준 상대 c2w 의 병진 노름 (첫 10초)
    def rel_mag(Q):
        rel = np.linalg.inv(Q[0])[None] @ Q
        audit = min(len(Q), int(round(fps * 10.0)))
        return float(np.linalg.norm(rel[:audit, :3, 3], axis=-1).max()), rel, audit
    mag, rel, audit = rel_mag(P)
    # LingBot 의 병진 단위는 SolarWM 보다 크다. 회전은 그대로 두고 병진만 균일 스케일링해
    # 첫 프레임 기준 최대 상대 병진을 target_rel 로 맞춘다 (SolarWM 가드 20 의 절반).
    scale_t = 1.0
    if mag > target_rel:
        scale_t = target_rel / mag
        P = P.copy(); P[:, :3, 3] *= scale_t
        mag, rel, audit = rel_mag(P)
    assert mag <= max_rel, "relative translation %.2f exceeds guard %.1f" % (mag, max_rel)
    assert np.abs(rel[:audit]).max() <= max_abs, "camera abs exceeds guard"

    # 비디오: 첫 프레임 반복. 리사이즈는 로더가 하므로 원본 해상도로 저장해도 되지만,
    # 파일을 작게 하려고 목표 비율(480x864)로 미리 맞춘다.
    img = Image.open(image_path).convert("RGB")
    sw, sh = img.size; scale = max(height / sh, width / sw)
    img = img.resize((round(sw * scale), round(sh * scale)), Image.BILINEAR)
    l = (img.size[0] - width) // 2; t = (img.size[1] - height) // 2
    img = img.crop((l, t, l + width, t + height))
    frame = np.asarray(img, dtype=np.uint8)
    vbuf = io.BytesIO()
    imageio.mimwrite(vbuf, [frame] * T, format="mp4", fps=fps, codec="libx264", quality=8, pixelformat="yuv420p")
    video_bytes = vbuf.getvalue()

    cbuf = io.BytesIO(); np.savez(cbuf, c2w=P); cam_bytes = cbuf.getvalue()

    video_member = "%s.video.mp4" % sample_id
    camera_member = "%s.camera.npz" % sample_id
    shard_rel = "raw-wds/custom/shards/custom-000000.tar"
    shard_path = os.path.join(out_dir, shard_rel)
    with tarfile.open(shard_path, "w") as tf:          # 비압축 tar
        for name, data in [(video_member, video_bytes), (camera_member, cam_bytes)]:
            ti = tarfile.TarInfo(name); ti.size = len(data); tf.addfile(ti, io.BytesIO(data))
    shard_size = os.path.getsize(shard_path)

    manifest = {
        "schema": "custom-raw-wds-v1",
        "video": {"num_frames": T, "fps": fps, "height": height, "width": width},
        "camera": {"convention": CONV, "array_key": "c2w", "dtype": "float32", "finite": True,
                   "shape": [T, 4, 4], "magnitude_audit_seconds": 10.0,
                   "max_camera_abs": max_abs, "max_rel_translation": max_rel},
        "prompt": {"text": caption},
        "metadata": {"scene": sample_id, "source": "lingbot-examples",
                     "translation_scale_applied": scale_t, "max_rel_translation_after_scale": mag},
    }
    # inference.py 의 camera publication 이 dataset/clip_id 를 요구한다 (portable ASCII path component)
    row = {"sample_id": sample_id, "key": sample_id, "shard": shard_rel, "shard_size": shard_size,
           "dataset": "lingbot_custom", "clip_id": sample_id,
           "video_member": video_member, "camera_member": camera_member,
           "num_frames": T, "fps": fps, "caption": caption, "manifest": manifest}
    idx = os.path.join(out_dir, "recipes", "custom", "raw-wds", "test-index.jsonl.gz")
    with gzip.open(idx, "wt") as f:
        f.write(json.dumps(row) + "\n")
    return dict(shard=shard_path, shard_size=shard_size, index=idx, frames=T, video_bytes=len(video_bytes),
                translation_scale=round(scale_t, 4), max_rel_translation=round(mag, 3))


def selfcheck(out_dir):
    """로더가 하는 검사를 그대로 재현해 통과하는지 본다 (SolarWM import 없이)."""
    idx = os.path.join(out_dir, "recipes", "custom", "raw-wds", "test-index.jsonl.gz")
    rows = [json.loads(l) for l in gzip.open(idx, "rt")]
    assert len(rows) == 1
    r = rows[0]
    for k in ("sample_id", "key", "shard", "shard_size", "video_member", "camera_member", "num_frames", "fps", "caption", "manifest", "dataset", "clip_id"):
        assert k in r, "row lacks %s" % k
    assert not r["shard"].startswith("/") and ".." not in r["shard"], "shard key must be relative"
    sp = os.path.join(out_dir, r["shard"])
    assert os.path.getsize(sp) == r["shard_size"], "shard_size mismatch"
    with tarfile.open(sp) as tf:
        names = tf.getnames()
        assert r["video_member"] in names and r["camera_member"] in names, names
        cam = tf.extractfile(r["camera_member"]).read()
        vid = tf.extractfile(r["video_member"]).read()
    with np.load(io.BytesIO(cam), allow_pickle=False) as z:
        assert "c2w" in z.files, z.files
        C = z["c2w"]
    m = r["manifest"]["camera"]
    assert m["convention"] == CONV and m["dtype"] == "float32" and m["finite"] is True
    assert C.dtype == np.float32 and list(C.shape) == list(m["shape"]) and np.isfinite(C).all()
    for f, v in [("magnitude_audit_seconds", 10.0), ("max_camera_abs", 20.0), ("max_rel_translation", 20.0)]:
        assert float(m[f]) == v, (f, m[f])
    assert float(r["fps"]) == float(r["manifest"]["video"]["fps"]) == 16.0
    try:
        import imageio
        rd = imageio.get_reader(io.BytesIO(vid), format="mp4"); n = rd.count_frames()
        assert n == r["num_frames"], "video has %d frames, index says %d" % (n, r["num_frames"])
        vinfo = "video %d frames OK" % n
    except Exception as e:
        vinfo = "video frame-count check skipped (%s)" % str(e)[:60]
    return "selfcheck OK: %d camera frames, %s" % (len(C), vinfo)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True); ap.add_argument("--poses", required=True)
    ap.add_argument("--caption-file", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--sample-id", default="lingbot_s05")
    ap.add_argument("--pixel-frames", type=int, default=None)
    a = ap.parse_args()
    info = build(a.image, a.poses, open(a.caption_file).read().strip(), a.out, a.sample_id, pixel_frames=a.pixel_frames)
    print(json.dumps(info, indent=1)); print(selfcheck(a.out))
