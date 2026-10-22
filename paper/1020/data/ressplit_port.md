# [10/20] Resolution-split collaboration baseline ported into the frozen protocol (arms `rs576`, `rs576r`)

Advisor's "strongest collaboration baseline": send the actions to the cloud, the cloud generates at LOW resolution, send it back, the device upsamples.
The one-GPU implementation `lingbot-world-v2/specwm/res_split_infer.py` read `data/stevo/tasks_camera.json`; it is now runnable on the frozen baseline
task sets (PROTOCOL.md §1–§7) as two arms, producing `baselines/runs/ours_<ARM>_<SET>/<task>/{video.mp4, timing.json}` in the `ours_to_timing.py` schema.
Nothing frozen was edited (run_ours.sh, ours_to_timing.py, report.py, runs/metrics/raw of the existing arms untouched).

## What was built (all new files, each with a [10/20] header)
| file | role |
|---|---|
| `lingbot-world-v2/specwm/res_split_baseline.py` | engine: port of res_split_infer's default (non-fused) loop to the baseline task dicts; cloud 14B FP8 2-step (nodes 0,500) at 576x320, device = bilinear latent upsample to the 832x480 grid (+ 1.3B refiner LoRA for rs576r); streamed 832x480 side-stream decode (as `two_machine --timed-decode 1`) for ttff / e2e; writes the frozen video (161 frames, libx264 crf 18 via `ours_to_timing.trim`) + timing.json itself (imports `ours_to_timing` helpers, does not edit it); skip-if-done per task |
| `baselines/gen/run_ressplit.sh SET ARM GPU [CHUNKJSON]` | wrapper with run_ours.sh's CLI shape (chunk json `{"tasks":[...]}` or plain list, copied to `raw/ours_<ARM>_<SET>/chunk_<name>`; skip-if-done on the frozen dir; refuses any area other than 399361 = 832x480; takes NO lock, the caller does); logs `baselines/logs/ours_<ARM>_<SET>.log`, `logs/run_ressplit.out` |
| `baselines/gen/make_queue_ressplit.py` | chunk maker -> PARKED `.sh` chunks (301-style: the chunk takes `flock /tmp/gpu$G.lock` itself, `_gpu_peak_run.sh`, embedded chunk json heredoc -> `raw/ours_<arm>_<set>/chunkdef_<name>.json`) |
| `baselines/gen/queue/park/ressplit/360–377_*.sh` | 18 parked chunks: neutral_832 rs576 (360–366, 7 x 12 clips), neutral_832 rs576r (367–373), astra rs576 (374–375: 12 + 9), astra rs576r (376–377). Not in the live queue. |

## Arms
- `rs576`: cloud = LingBot 14B causal-fast, single-GPU FP8 port, 2-step (t = [1000, 500], separate clean commit), KV window 18, at **576x320** (`--lo-area 193536`, latent 40x72); device = bilinear upsample of each cloud chunk latent to the 832x480 latent grid (60x104), no model; Wan VAE decode at 832x480.
- `rs576r`: the same cloud + device LingBot 1.3B causal-fast (bf16) with the merged refiner LoRA, re-noise to the LoRA's sigma(s) + one 1.3B forward each, committed to the 1.3B's own KV window 18.
- Both parts run serially in ONE process on ONE GPU (no network, no RTT). timing.json records `chunk_s = cloud_chunk_s + device_chunk_s` per chunk (serial; `steady_fps_gen` uses it), plus the `cloud_chunk_s` / `device_chunk_s` lists, `steady_fps_cloud_only` (= 12 / median cloud_chunk_s, the parallel-device bound behind the 9.8 / 15.8 fps of facts.md §12), `ttff_s` / `e2e_s` / `steady_fps_e2e` / `decode_s` from the streamed 832 decode, `peak_alloc_GiB` / `peak_reserved_GiB` (this clip; all loaded weights included), `kv_window` 18, `wq` fp8, `nulls`, `gpus` 1, `transport`, `refiner` (lora path + meta + sigmas), `lo_resolution`, and a `fps_note` spelling all of this out.

## Smoke (GPU 5 under its lock, neutral_832 balloon_block_cut P1 + P3, both arms)
rs576 ran 20:52–20:58 CDT (rc 0; lock hold 357 s for load + warm-up + 2 clips; nvidia-smi peak 51235 MiB incl. the 1.2-GB scorer context). Both frozen videos probe as h264 832x480, 161 frames, 16 fps; timing.json has every schema field, `nulls` = {}.
**rs576r smoke ran 10/21 00:00–00:08 CDT on GPU 6 after the reboot (rc 0, lock hold 480 s, nvidia-smi peak 60313 MiB); its rows are below.** (Earlier text:) rs576r had not obtained the GPU 5 lock when this report was first written (it is queued in `$SC/smoke.sh`, pid 1832712, and runs automatically when the lock frees; result lines go to `baselines/logs/ours_rs576r_neutral_832.log`, status to `/tmp/claude-1741758895/-home-hk28965/76e7a8ef-ae2e-455b-8c73-052f5bde2a30/scratchpad/ressplit/smoke.status`, frozen output to `runs/ours_rs576r_neutral_832/`). Its rows are to be added from that log.

Serial one-GPU numbers; reference arms = the frozen `runs/ours_<arm>_neutral_832` files of the same clips (single GPU, same timing definitions). Best per column in bold, second underlined (over the arms with a value).

| arm (neutral_832, balloon_block_cut) | clip | steady fps gen ↑ | steady fps e2e ↑ | TTFF s ↓ | e2e s ↓ | gen s ↓ | median chunk s ↓ (cloud + device) | decode/chunk s (832 stream) | peak alloc / reserved GiB ↓ |
|---|---|---|---|---|---|---|---|---|---|
| **rs576** (cloud 2-step 576x320 + bilinear) | P1 | **20.54** (cloud-only 20.54) | **9.71** | **3.86** | **20.05** | **7.73** | **0.584** (0.584 + 0.0003) | 0.652 | **39.5 / 49.4** |
| **rs576** | P3 | **20.54** (20.55) | **9.70** | **3.92** | **20.12** | **7.73** | **0.584** (0.584 + 0.0003) | 0.652 | **39.3 / 49.3** |
| rs576r (+ 1.3B refiner r850_930_pix) | P1 | <u>12.52</u> (cloud-only 20.4) | <u>7.45</u> | 10.17 | <u>30.95</u> | <u>12.54</u> | <u>0.955</u> (0.587 + 0.368) | 0.652 | <u>48.1 / 58.0</u> |
| rs576r | P3 | <u>12.60</u> (20.5) | <u>7.48</u> | 10.69 | <u>31.37</u> | <u>12.45</u> | <u>0.952</u> (0.585 + 0.366) | 0.652 | <u>48.1 / 58.0</u> |
| rt2 (14B 2-step at 832) | P1 | <u>9.74</u> | <u>6.37</u> | <u>7.44</u> | <u>31.44</u> | <u>15.90</u> | <u>1.232</u> | | 62.6 / 62.6 |
| adapt25 | P1 | 9.69 | 6.35 | 7.75 | 34.61 | 19.11 | 1.238 | | 62.8 / 62.8 |
| student1p3b (1.3B 4-step) | P1 | 8.99 | 9.00 | 6.77 | 23.94 | 17.64 | 1.335 | | n/a |
| sf24 | P1 | 5.49 | 5.49 | 8.16 | 37.10 | 30.37 | 2.186 | | n/a |
| teacher (14B 4-step) | P1 | 5.32 | 4.13 | 8.12 | 44.71 | 29.15 | 2.256 | | 62.6 / 62.6 |

Readings (timing only; no judge / VBench / RPE yet): the 576x320 cloud at 2 steps is 0.584 s/chunk, 2.1x rt2 at 832 and 3.9x the teacher; the bilinear device step is free (0.3 ms); e2e is bounded by the 832x480 streamed VAE decode on the same GPU (0.652 s/chunk > the 0.584 s DiT chunk, so steady e2e = 12 / 1.236 = 9.7 fps); TTFF 3.9 s vs 7.4–8.1 s (cheaper prepare: the VAE encode and the first chunks at 576). Memory 39.5 GiB allocated (14B FP8 + KV at 576 + two VAEs) vs 62.6 at 832. The frozen frames (frames 0 / 80 / 160 of P3 looked at) show the lattice texture of a bilinear LATENT upsample decoded at 832 and the 576-cloud drift (P3 ends in a different scene) — the quality metrics will carry that; a pixel-space variant (decode at 576 on the device, bicubic to 832) is not built.

## Caveats
1. **The r900 refiner LoRA named in the task does not exist any more**: `specwm_out/reslora/r900/` holds only `log.txt` (purged in the 10/15 disk clean-up; not on lamarr either, `~/specwm_archive/.../reslora` absent). The only refiner LoRA left is `specwm_out/reslora/r850_930_pix/res_lora.pt` (rank 64, ffn, ref_index [850, 930] = TWO refiner forwards at sigma ≈ 0.85 and 0.93, step 2784, 10/10). `rs576r` uses it (env `RS_LORA` overrides) and records it in `timing.json.refiner`. The refiner was retired on 10/11 (−0.50 judge at 704, memory `specwm-draft-finish-1011`), so rs576r is expected to be the weaker of the two arms; rs576 is the clean "cloud low-res + device upsample" baseline.
2. **576x320 is exact now**: V2Binding.prepare's floor formula at the 832x480 aspect gives 576x320 only for max_area in [191424, 195680]; res_split_infer's default `--lo-area 175104` gives 544x304 on an 832x480 image (and 560x304 on the 1408x768 StEvo images of the earlier "576" numbers). The arm uses 193536; `timing.json.lo_resolution` = "576x320".
3. Only the 832x480 sets (area 399361: neutral_832, neutral_832_nocap, astra) are supported; mg2 / wp / abot sets are refused with a message (the cloud low resolution at those aspects is not defined yet; 640x352 would need a ~448x256 cloud).
4. `prepare_s` (inside ttff_s) is the cloud prepare at 576x320, plus the 1.3B's prepare at 832x480 for rs576r (both bindings encode the prompt on the CPU T5).
5. Serial single-GPU measurement: the device upsample / refiner / commit and the streamed 832 decode share the GPU with the cloud. The parallel-device bound is `steady_fps_cloud_only`; a true two-device number (device RTT, device GPU) is not measured here.
6. ours_to_timing.py / report.py were not edited: `report.py` does not know the new arms yet (they are not in `ARMS`); the timing.json files carry the same field names so a one-line arm registration there (or a parked follow-up) is all that is needed to show them in TABLE.md. The metrics scorers (`scripts/vlm_clip_score.py`, `vbench6.py`, `rpe_da3.py`) read `runs/<arm>/<task>/video.mp4` and need no change.

## Exact commands
```
# one chunk json by hand (the caller takes the lock; GPU 5 example; CHUNKJSON = {"tasks":[...]} or a plain task list, e.g. the chunkdef json a parked .sh writes):
flock /tmp/gpu5.lock env CUDA_VISIBLE_DEVICES=5 /home/hk28965/specwm/baselines/_gpu_peak_run.sh 5 /tmp/peak.txt \
  /home/hk28965/specwm/baselines/gen/run_ressplit.sh neutral_832 rs576 5 /home/hk28965/specwm/baselines/raw/ours_rs576_neutral_832/chunkdef_360_neutral_832_rs576.json
# the parked chunk scripts themselves take the lock:  bash /home/hk28965/specwm/baselines/gen/queue/park/ressplit/360_neutral_832_rs576.sh 5
# whole set without chunks:  flock /tmp/gpu5.lock env CUDA_VISIBLE_DEVICES=5 /home/hk28965/specwm/baselines/gen/run_ressplit.sh astra rs576 5
# enqueue for the workers (tier 9 example; check that 360-377 do not collide with names in queue/9, claimed/, done/, failed/; else re-run the maker with --start N):
mv /home/hk28965/specwm/baselines/gen/queue/park/ressplit/36[0-9]_*.sh /home/hk28965/specwm/baselines/gen/queue/park/ressplit/37[0-7]_*.sh /home/hk28965/specwm/baselines/gen/queue/9/
# regenerate the parked chunks (skips tasks already frozen):  python3 /home/hk28965/specwm/baselines/gen/make_queue_ressplit.py [--start 360] [--size 12] [--sets neutral_832,astra] [--arms rs576,rs576r]
```

## rs576r smoke readings (10/21 00:10 CDT; frames of P3 at 0 / 80 / 160 looked at next to rs576 and rt2)
- Timing: the 1.3B refiner (two forwards at sigma 0.85 / 0.93 + its own commit) costs 0.37 s/chunk on top of the 0.59 s cloud chunk → 12.5 fps serial (rs576 20.5, rt2 9.7, teacher 5.3); e2e 7.5 fps (decode-bound as rs576); TTFF 10.2–10.7 s vs rs576 3.9 s because the second binding's prepare (1.3B prompt encode on the CPU T5 + its first-chunk warm-up) sits inside ttff — a measurement artefact of the serial one-GPU port, not a device cost (prompt encode happens once per session on a device). Memory 48.1 / 58.0 GiB = 14B FP8 cloud + 1.3B + two VAEs on ONE GPU; the device half alone is the 1.3B's ~8 GiB.
- Frames: rs576 shows the bilinear-latent lattice texture and blur at every frame; rs576r is sharp (scissors + hand at frame 80 are crisp) — the refiner repairs the per-frame look of the upsampled latent, unlike its 10/11 result on a full-res 704 stream (−0.50). All three arms (rs576, rs576r, rt2) end P3 in different content (the balloon is gone in both res-split arms by frame 80; rt2 keeps balloon + scissors) — the 576 cloud's drift is the content problem the refiner cannot fix; the judge / RPE scores from the scorer loop will carry it.
- Both arms are safe to run in the queue (tier 9 chunks 380–397).


## Scored (10/21 13:20 CDT, frozen 10-s sets; metrics_g1; judge Qwen3-VL-32B, VBench6, RPE DA3; best **bold**, second <u>underlined</u>)

| set | arm | n | judge q ↑ | judge drift ↓ | VBench q ↑ | VBench img ↑ | MUSIQ drift ↓ | RPE rot ↓ | RPE trans ↓ | fps gen ↑ | fps e2e ↑ | TTFF s ↓ | peak GiB ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| neutral_832 | teacher (4-step, 832) | 84 | **7.73** | 38.9 | 74.0 | **69.2** | 4.11 | 0.76 | **0.29** | 5.3 | 4.1 | <u>8.8</u> | 62.6 |
| neutral_832 | rt2 (2-step, 832) | 84 | <u>7.60</u> | 39.2 | 74.1 | <u>68.0</u> | 4.10 | 0.81 | **0.29** | <u>9.6</u> | <u>6.3</u> | **8.6** | 62.6 |
| neutral_832 | rt2c1hyb_dev (2-step + cloud) | 84 | 7.68 | <u>37.9</u> | <u>74.4</u> | 67.8 | <u>3.85</u> | 0.80 | **0.29** | 6.8 | - | - | 62.1 |
| neutral_832 | rs576 (cloud 576 + bilinear) | 84 | 5.42 | 45.0 | 70.8 | 51.7 | 5.68 | 0.76 | 0.50 | **20.5** | **9.7** | **4.6** | **39.3** |
| neutral_832 | rs576r (cloud 576 + 1.3B refiner) | 84 | 7.54 | **36.4** | 74.2 | 67.1 | 4.70 | 0.76 | <u>0.38</u> | 12.6 | 7.5 | 11.6 | <u>48.1</u> |
| neutral_832 | sf24 (Speculative Forcing, 832) | 84 | 7.79 | 35.3 | **74.7** | 69.6 | **3.74** | **0.71** | 0.31 | 5.1 | 4.6 | 9.4 | - |
| astra | teacher | 21 | 7.86 | **10.9** | <u>78.7</u> | <u>74.0</u> | 2.36 | 0.56 | <u>0.19</u> | 5.3 | 4.1 | 9.1 | 62.6 |
| astra | rt2 | 21 | 7.76 | 12.4 | 78.5 | 72.9 | 2.34 | 0.52 | **0.17** | <u>9.6</u> | <u>6.3</u> | <u>8.3</u> | 62.6 |
| astra | rt2c1hyb_dev | 21 | 7.90 | <u>11.4</u> | 78.8 | 73.1 | 2.39 | <u>0.45</u> | **0.17** | 6.8 | - | - | 62.1 |
| astra | rs576 | 21 | 3.67 | 15.7 | 73.7 | 59.5 | 3.17 | <u>0.45</u> | 0.42 | **20.5** | **9.7** | **4.4** | **39.3** |
| astra | rs576r | 21 | <u>8.00</u> | 14.1 | 77.7 | **74.3** | **2.03** | 0.47 | 0.26 | 12.6 | 7.5 | 11.1 | <u>48.1</u> |
| astra | sf24 | 21 | **8.10** | <u>11.4</u> | **79.0** | 72.4 | <u>2.25</u> | **0.44** | 0.13 | 7.2 | 6.3 | 7.9 | - |

Readings: the bilinear arm (rs576) is unusable (judge 5.4 / 3.7: the latent-upsample lattice). With the 1.3B refiner (rs576r) the res-split baseline matches the full-resolution 2-step on judge quality (7.54 vs 7.60; astra 8.00 vs 7.76), VBench (74.2 vs 74.1) and camera following (RPE rot 0.76 = 0.76), and is 1.3x faster in generation (12.6 vs 9.6 fps) at 48 GiB on one GPU (cloud + device serial; the device part alone is the 1.3B at ~25 fps). It loses on translation RPE (0.38 vs 0.29) and TTFF (11.6 s: the refiner's first chunk). This is the strongest "cloud does the content, device does the pixels" baseline and the row the hybrid has to beat on the device-fps axis; its content is decided at 576x320 on the cloud, so it is a cloud-only system with a device post-processor, not an on-device world model (bandwidth = one 576 latent per chunk down, no actions up to the device).
