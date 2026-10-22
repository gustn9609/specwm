"""[10/17 frozen baselines] Convert OUR raw run dirs into the frozen layout baselines/runs/<arm_set>/<task>/{video.mp4 (161 frames = 10.0 s @ 16 fps),
timing.json with every PROTOCOL §4 field}.  Accepts
  (a) a two_machine --role solo task dir  <out>/<task>/            : timing.json + spec.mp4 (+ <out>/summary.json = the run's args: window, wq, nodes, use, area, frames)
  (b) a spec_early task dir               <out>/<task>/<arm>_s<seed>/: loop.json + video.mp4      (sf24 = 14B fp8 token-wise early exit, student1p3b = 1.3B uniform:4,
      sf1p3b_pub / sf1p3b_cal = specwm.spec_early_tf --mode tf --model 1p3b: 1.3B + Speculative Forcing training-free verifier, published / calibrated thresholds)
  (c) [10/17 C] a two_machine HYBRID pair (arms adapt25c1hyb_dev / adapt25c1hyb_cloud): <raw>/{device,cloud}/<task>/ (timing.json + spec.mp4); each
      conversion also reads the PARTNER stream's raw timing.json (<raw>/<other role>/<task>/timing.json, kept by --purge) for the cross-stream fields.
      dev   = the stream the DEVICE shows (its own adapt25 drafts, every chunk sent; corrections re-committed into its KV only, never re-shown):
              chunk_s = per-chunk device wall (draft + send + applying the corrections that arrived) = the shown-frame interval; steady_fps_gen = fps_shown.
      cloud = the cloud-final stream (draft + 1 step at t=750 in the cloud's own context): chunk c is final at (device start of chunk c) + draft_s[c] +
              lat_from_show_s[c] on the device's clock (same host); chunk_s = intervals of these ready times, gen_s = the last one (derived, see notes).
      ttff / e2e / steady_fps_e2e / decode_s / peak_reserved are null for both (two_machine's --timed-decode is solo only), said in "nulls".
The generated clip has 165 frames (--frames 165 = 42 latent frames, 14 chunks of 3); ffmpeg re-encodes the first 161 frames (libx264 crf 18, 10/17 B) so the
written video is exactly 10.0 s; timing covers ALL 14 chunks (gen_s, e2e_s, steady fps use every generated chunk).
[10/19 EXP 1] the 60-s program P6 (task ids <set>__<scene>__P6_long, 969 generated frames = 81 chunks) is trimmed to 961 frames (60.0 s) instead
(LONG_FRAMES_OUT; --frames-out overrides). [10/19 EXP 2 / 3] hybrid arms rt2c1hyb_{dev,cloud} and sf24c1hyb_{dev,cloud} (same layout as adapt25c1hyb).
[10/19 EXP 2 light device] rt2c1hyb32_{dev,cloud} = rt2c1hyb with a 32-GiB device (two_machine --window 12 --kv fp8 --cap 32 --lowmem 1 on the device only); when the
device timing.json has "mem" (any hybrid whose device ran with --kv fp8 / --cap / --lowmem) both streams get device_mem + device_peak_alloc_proc_GiB /
device_peak_reserved_proc_GiB (whole device process: load, warm-up, prepare, generation + re-commits, decode) + device_nvsmi_peak_MiB (that process's nvidia-smi
used memory, polled every 0.5 s) + device_cap_GiB / device_kv_dtype / device_kv_window; the dev stream's peak_reserved_GiB = this clip's reserved peak.
[10/17 B] --batch takes --tasks a,b,c (convert ONLY these task ids: a chunk's own tasks, so a concurrent chunk of the same (set, arm) on another GPU is
never converted half-written) and --purge (after a SUCCESSFUL conversion delete the raw lat.pt + spec.mp4 (two_machine) / lat.pt + video.mp4 (spec_early,
also in the <arm>_warm dir); timing.json / loop.json / summary.json are kept); the final line is "[batch] converted N failed M missing K" and the exit
status is 1 when M + K > 0.
Field definitions (ours): gen_s = sum of per-chunk DiT wall times (denoise + commit, CUDA-synchronised; chunk_s); steady_fps_gen = 12 / median chunk_s over
chunks >= 3; ttff_s = prepare (image + T5 + VAE encode) + time from the start of chunk 0 to the first decoded frame of the side-stream decode
(two_machine --timed-decode 1 / spec_early --stream-decode 1); e2e_s = prepare + time to the last decoded frame (DiT and VAE sequential, as the baselines'
e2e); steady_fps_e2e = 12 / median interval between consecutive chunks' last decoded frame (chunks >= 3); decode_s = sum of the per-chunk side-stream
decode times.  Without the timed decode these four are null (said so in "nulls").  peak_* = torch peak stats since the warm-up reset (weights included;
with the timed decode the streamed VAE decode is included, the final full decode that writes spec.mp4 is not).
[10/19 audit] spec_early / spec_early_tf arms (sf24, student1p3b, sf1p3b_pub, sf1p3b_cal) get three ADDITIVE fields: gen_s_dit = loop.json times.denoise +
commit + verify (CUDA-synced section totals, i.e. gen_s without the host-side side-stream push gaps), steady_fps_gen_dit = steady_fps_gen x gen_s / gen_s_dit
(uniform scaling of the median chunk) and fps_correction (why; see spec_early_dit_fields). gen_s / steady_fps_gen / chunk_s stay as frozen. The
existing frozen files were patched in place by gen/add_dit_fields_1019.py. Node labels of adapt25 / *c1hyb* strings: NODE_LABEL_FIX (gen/fix_node_labels_1019.py).
usage: /home/hk28965/specwm/scripts/envrun.sh python baselines/ours_to_timing.py RUN_TASK_DIR OUT_DIR --arm ARM [--set SET] [--kv-window 18] [--wq fp8]
       ... --batch RAW_OUT_DIR RUNS_DIR --arm ARM --set SET      (every <task> under RAW_OUT_DIR -> RUNS_DIR/<task>/; spec_early: <task>/<arm>_s1234)
   needs ffmpeg/ffprobe on PATH (~/.local/bin)."""
import os, sys, json, glob, shutil, argparse, subprocess, numpy as np
S = "/home/hk28965/specwm"; B = S + "/baselines"; FPC = 12; FRAMES_OUT = 161
CKPT14 = "lingbot-world-v2-14b-causal-fast (HF snapshot 5c33dd40b213598c418fd25bff30fdbd23fd38a7: transformers/ 8 safetensors shards, models_t5_umt5-xxl-enc-bf16.pth, Wan2.1_VAE.pth)"
CKPT1P3 = "lingbot-world-v2-1.3b-causal-fast (HF snapshot 7e36a5f919f86cb4255cc9bfc30adb44963fbde1); T5 / VAE / assets from lingbot-world-v2-14b-causal-fast"
PREC = dict(fp8="FP8 W8A8 e4m3 on every nn.Linear of the DiT blocks (per-output-channel static weight scales, per-token dynamic activation scales, torch._scaled_mm, bf16 out; specwm/fast_patch.FP8Linear); everything else bf16; x0 / scheduler fp32",
            bf16="bf16 weights + activations (no FP8); x0 / scheduler fp32",
            gptq4="INT4 weight-only GPTQ (4-bit, group 128, Hessians from 2 calibration scenes x 3 chunks at 832x480 on the rt2 schedule; specwm.gptq_save, offline file specwm_out/gptq4/14b_g128_832.pt) on every nn.Linear of the DiT blocks with in_features >= 1024, packed uint8 nibbles + bf16 group scales, dequantized to bf16 per forward (fused Triton kernel int4_kernels.PackedInt4LinearT when available, else torch) then a bf16 GEMM; everything else bf16; x0 / scheduler fp32")   # [10/19 thesis]
ATTN = "SageAttention (specwm/fast_patch FLAGS attn='sage', ring KV, fp32 RoPE cache), torch.compile'd blocks"
ARMS = {  # arm -> (model string, steps, notes)
    "teacher": ("LingBot-World 2.0 14B causal-fast, single-GPU FP8 port (specwm.two_machine --role solo), 4-step teacher: nodes t=[1000, 750, 500, 250] (timesteps_index 0,250,500,750)", 4),
    "rt2": ("LingBot-World 2.0 14B causal-fast, single-GPU FP8 port (specwm.two_machine --role solo), realtime 2-step: nodes t=[1000, 500]", 2),
    "solo3": ("LingBot-World 2.0 14B causal-fast, single-GPU FP8 port (specwm.two_machine --role solo), 3-step: nodes t=[1000, 500, 250]", 3),
    "adapt25": ("LingBot-World 2.0 14B causal-fast, single-GPU FP8 port (specwm.two_machine --role solo), adaptive: 2-step draft (t=[1000, 500]) + step-1 verifier head adds a 3rd step at the last node, σ 0.625 (timesteps_index 750 of the 4-node schedule; --adapt-node 2 of nodes 0,500,750 = the same node as the hybrids' cloud step) on 25 % of the chunks (chunks 0-2 always)", "2 + chunk-wise 3rd"),
    "sf24": ("LingBot-World 2.0 14B causal-fast, single-GPU FP8 port (specwm.spec_early --mode early), Speculative Forcing on the 14B: 4-step schedule with token-wise early exit by the learned gap head (eps 0.24)", "4 with token-wise early exit"),
    "student1p3b": ("LingBot-World 2.0 1.3B causal-fast (specwm.spec_early --mode uniform:4 --model 1p3b), 4-step student", 4),
    # [10/17 B] Speculative Forcing in ITS setting (PROTOCOL rule 9) run through run_ours.sh at the baseline's exact 832x480 (specwm.spec_early_tf --mode tf --model 1p3b --stream-decode 1)
    "sf1p3b_pub": ("LingBot 1.3B causal-fast + SF training-free verifier (published): LingBot-World 2.0 1.3B causal-fast with the Speculative Forcing training-free verifier (arXiv 2610.22347 Sec. 3.3.1 / App. A) at the paper's LingBot thresholds tau_red 0.375 / tau_stab 0.300, eps 0.24 (specwm.spec_early_tf --mode tf --model 1p3b)", "4 with token-wise early exit"),
    # [10/17 C] OUR headline hybrid on 2 GPUs (run_ours.sh adapt25c1hyb; PROTOCOL rule 8): two frozen streams of the same run
    # [10/19 audit] node labels: the cloud's ONE step is --use 3 of timesteps_index (0,250,500,750) = σ [1.0, 0.937, 0.833, 0.625], i.e. the LAST node, σ 0.625 (t ≈ 625);
    #   adapt25's --adapt-node 2 of nodes 0,500,750 is that SAME node. The earlier strings said "t=750 (node index 3 of [1000, 750, 500, 250])" / "3rd step (t=250)" (NODE_LABEL_FIX below).
    "adapt25c1hyb_dev": ("LingBot-World 2.0 14B causal-fast, FP8 port, device+cloud hybrid on 2 GPUs (specwm.two_machine --role device/--role cloud, same host, TCP): DEVICE stream shown = adapt25 (2-step draft t=[1000, 500] + step-1 verifier head adds a 3rd step at the last node, σ 0.625 (timesteps_index 750 of the 4-node schedule; --adapt-node 2 of nodes 0,500,750 = the same node as the hybrids' cloud step) on 25 % of the chunks, chunks 0-2 always); EVERY chunk is sent to the cloud 14B FP8 which finishes it with ONE step at the last node, σ 0.625 (timesteps_index 750 of the 4-node schedule 0,250,500,750 = σ [1.0, 0.937, 0.833, 0.625]; --use 3; re-noised draft, --max-lag 99) in its own corrected context and returns it; the device re-commits the returned chunk into its KV window (--thr 0, KV feedback). Device GPU only is counted as 'the device'; 2 GPUs total", "device 2 + chunk-wise 3rd (shown); + 1 cloud step at σ 0.625 (timesteps_index 750, the last node) fed back into the device KV"),
    "adapt25c1hyb_cloud": ("LingBot-World 2.0 14B causal-fast, FP8 port, device+cloud hybrid on 2 GPUs (specwm.two_machine --role device/--role cloud, same host, TCP): CLOUD stream = the device's adapt25 draft (2-step t=[1000, 500] + step-1 verifier head 3rd step at the last node, σ 0.625 (timesteps_index 750) on 25 % of the chunks) finished by ONE cloud 14B FP8 step at the last node, σ 0.625 (timesteps_index 750 of the 4-node schedule 0,250,500,750 = σ [1.0, 0.937, 0.833, 0.625]; --use 3) on EVERY chunk (--max-lag 99), in the cloud's own context; each finished chunk is also fed back into the device KV (--thr 0). 2 GPUs", "device 2 + chunk-wise 3rd, then 1 cloud step at σ 0.625 (timesteps_index 750, the last node) (cloud-final stream)"),
    # [10/19 EXP 2 / EXP 3] more hybrids on 2 GPUs, same pair mechanics / cloud / streams as adapt25c1hyb (run_ours.sh rt2c1hyb | sf24c1hyb)
    "rt2c1hyb_dev": ("LingBot-World 2.0 14B causal-fast, FP8 port, device+cloud hybrid on 2 GPUs (specwm.two_machine --role device/--role cloud, same host, TCP): DEVICE stream shown = rt2 (2-step t=[1000, 500], light device); EVERY chunk is sent to the cloud 14B FP8 which finishes it with ONE step at the last node, σ 0.625 (timesteps_index 750 of the 4-node schedule 0,250,500,750 = σ [1.0, 0.937, 0.833, 0.625]; --use 3; re-noised draft, --max-lag 99) in its own corrected context and returns it; the device re-commits the returned chunk into its KV window (--thr 0, KV feedback). 2 GPUs total", "device 2 (shown); + 1 cloud step at σ 0.625 (timesteps_index 750, the last node) fed back into the device KV"),
    "rt2c1hyb_cloud": ("LingBot-World 2.0 14B causal-fast, FP8 port, device+cloud hybrid on 2 GPUs (specwm.two_machine --role device/--role cloud, same host, TCP): CLOUD stream = the device's rt2 draft (2-step t=[1000, 500]) finished by ONE cloud 14B FP8 step at the last node, σ 0.625 (timesteps_index 750 of the 4-node schedule; --use 3) on EVERY chunk (--max-lag 99), in the cloud's own context; each finished chunk is also fed back into the device KV (--thr 0). 2 GPUs", "device 2, then 1 cloud step at σ 0.625 (timesteps_index 750, the last node) (cloud-final stream)"),
    "sf24c1hyb_dev": ("LingBot-World 2.0 14B causal-fast, FP8 port, device+cloud hybrid on 2 GPUs (specwm.two_machine --role device/--role cloud, same host, TCP): DEVICE stream shown = Speculative Forcing on the 14B as the sf24 arm (4-step schedule t=[1000, 750, 500, 250] with token-wise early exit by the learned gap head head14b_nob43, eps 0.24, dense clean commit; two_machine --ee-head); EVERY chunk is sent to the cloud 14B FP8 which finishes it with ONE step at the last node, σ 0.625 (timesteps_index 750 of the 4-node schedule; --use 3; --max-lag 99) in its own corrected context and returns it; the device re-commits it into its KV window (--thr 0, KV feedback). 2 GPUs total", "device 4 with token-wise early exit (shown); + 1 cloud step at σ 0.625 (timesteps_index 750, the last node) fed back into the device KV"),
    "sf24c1hyb_cloud": ("LingBot-World 2.0 14B causal-fast, FP8 port, device+cloud hybrid on 2 GPUs (specwm.two_machine --role device/--role cloud, same host, TCP): CLOUD stream = the device's sf24 draft (4-step schedule with token-wise early exit, eps 0.24) finished by ONE cloud 14B FP8 step at the last node, σ 0.625 (timesteps_index 750 of the 4-node schedule; --use 3) on EVERY chunk (--max-lag 99), in the cloud's own context; each finished chunk is also fed back into the device KV (--thr 0). 2 GPUs", "device 4 with token-wise early exit, then 1 cloud step at σ 0.625 (timesteps_index 750, the last node) (cloud-final stream)"),
    # [10/19 EXP 2 light device] rt2c1hyb with a 32-GiB device (run_ours.sh rt2c1hyb32; PROTOCOL §7)
    "rt2c1hyb32_dev": ("LingBot-World 2.0 14B causal-fast, FP8 port, device+cloud hybrid on 2 GPUs (specwm.two_machine --role device/--role cloud, same host, TCP), LIGHT DEVICE: the device process runs under a 32-GiB CUDA allocator cap from process start (--cap 32 --lowmem 1: block-wise FP8 load, the bf16 checkpoint never on the GPU; OOM = does not fit), FP8 (e4m3) KV ring buffer, KV window 12 (--kv fp8 --window 12); DEVICE stream shown = rt2 (2-step t=[1000, 500]); EVERY chunk is sent to the cloud 14B FP8 (unchanged from rt2c1hyb: bf16 KV, window 18 / 12 at 1280x704, no cap) which finishes it with ONE step at the last node, σ 0.625 (timesteps_index 750 of the 4-node schedule; --use 3; --max-lag 99) in its own corrected context and returns it; the device re-commits it into its KV window (--thr 0, KV feedback). 2 GPUs total", "device 2 (shown); + 1 cloud step at σ 0.625 (timesteps_index 750, the last node) fed back into the device KV"),
    "rt2c1hyb32_cloud": ("LingBot-World 2.0 14B causal-fast, FP8 port, device+cloud hybrid on 2 GPUs (specwm.two_machine --role device/--role cloud, same host, TCP), LIGHT DEVICE (32-GiB allocator cap, FP8 KV, KV window 12 on the device): CLOUD stream = the device's rt2 draft (2-step t=[1000, 500]) finished by ONE cloud 14B FP8 step at the last node, σ 0.625 (timesteps_index 750 of the 4-node schedule; --use 3) on EVERY chunk (--max-lag 99; cloud unchanged from rt2c1hyb), in the cloud's own context; each finished chunk is also fed back into the device KV (--thr 0). 2 GPUs", "device 2, then 1 cloud step at σ 0.625 (timesteps_index 750, the last node) (cloud-final stream)"),
    "sf1p3b_cal": ("LingBot 1.3B causal-fast + SF training-free verifier (calibrated): LingBot-World 2.0 1.3B causal-fast with the Speculative Forcing training-free verifier (arXiv 2610.22347 Sec. 3.3.1 / App. A) at the 10/17 re-calibrated thresholds from baselines/calib.json['sf1p3b']['calibrated'] (oracle-rate-matched quantiles on the 1.3B's own 4-step output), eps 0.24 (specwm.spec_early_tf --mode tf --model 1p3b)", "4 with token-wise early exit")}
# [10/19 thesis] rt2c1hyb24 (rt2c1hyb32 + GPTQ-4 weights, 24-GiB device) and the MATRIX arms mx<X> (device only, two_machine --role solo on the rt2 schedule) / mx<X>c1hyb (+ the standard cloud): one
#   memory squeeze at a time on top of rt2, then the two bundles (run_ours.sh mx_flags; PROTOCOL §9). Built from one table so the wording stays identical across the 20 entries.
_HYB_PRE = "LingBot-World 2.0 14B causal-fast, FP8 port, device+cloud hybrid on 2 GPUs (specwm.two_machine --role device/--role cloud, same host, TCP)"
_CLOUD = "the cloud 14B FP8 (unchanged from rt2c1hyb: FP8 weights, bf16 KV, window 18 / 12 at 1280x704, no cap) which finishes it with ONE step at the last node, σ 0.625 (timesteps_index 750 of the 4-node schedule 0,250,500,750 = σ [1.0, 0.937, 0.833, 0.625]; --use 3; --max-lag 99) in its own corrected context and returns it; the device re-commits it into its KV window (--thr 0, KV feedback)"
MX = {   # name -> (device description, device squeeze flags)
    "mxW12": ("KV window 12 (--window 12; FP8 weights, bf16 KV, no cap)", "--window 12"),
    "mxW9": ("KV window 9 (--window 9 = sink 6 + the current chunk; FP8 weights, bf16 KV, no cap)", "--window 9"),
    "mxK8": ("FP8 (e4m3) KV ring buffer (--kv fp8; FP8 weights, window 18, no cap)", "--kv fp8"),
    "mxQ4": ("GPTQ-4 g128 packed INT4 block weights from the offline file specwm_out/gptq4/14b_g128_832.pt (--wq gptq4 --wq-file F --lowmem 1: loaded block by block, the bf16 checkpoint never on the GPU; bf16 KV, window 18, no cap)", "--wq gptq4 --wq-file specwm_out/gptq4/14b_g128_832.pt --lowmem 1"),
    "mx32": ("the rt2c1hyb32 device bundle: 32-GiB CUDA allocator cap from process start (--cap 32 --lowmem 1: block-wise FP8 load; OOM = does not fit), FP8 (e4m3) KV, KV window 12, spec.mp4 decoded by a bf16 copy of the VAE in 2x2 tiles with the KV resident (--vae-dtype bf16 --vae-tiles 2,2)", "--window 12 --kv fp8 --cap 32 --lowmem 1 --vae-dtype bf16 --vae-tiles 2,2"),
    "mx24": ("the rt2c1hyb24 device bundle: 24-GiB CUDA allocator cap from process start (--cap 24 --lowmem 1; OOM = does not fit), GPTQ-4 g128 packed INT4 block weights from the offline file specwm_out/gptq4/14b_g128_832.pt (--wq gptq4 --wq-file F), FP8 (e4m3) KV, KV window 12, bf16 2x2-tiled VAE decode with the KV resident", "--window 12 --kv fp8 --cap 24 --lowmem 1 --vae-dtype bf16 --vae-tiles 2,2 --wq gptq4 --wq-file specwm_out/gptq4/14b_g128_832.pt"),
}
for _k, (_d, _f) in MX.items():
    ARMS[_k] = ("LingBot-World 2.0 14B causal-fast, single-GPU FP8 port (specwm.two_machine --role solo), realtime 2-step: nodes t=[1000, 500], MATRIX device squeeze %s: %s (device-only flags %s)" % (_k, _d, _f), 2)
    ARMS[_k + "c1hyb_dev"] = ("%s, MATRIX device squeeze %s: the device runs with %s; DEVICE stream shown = rt2 (2-step t=[1000, 500]); EVERY chunk is sent to %s. 2 GPUs total" % (_HYB_PRE, _k, _d, _CLOUD), "device 2 (shown); + 1 cloud step at σ 0.625 (timesteps_index 750, the last node) fed back into the device KV")
    ARMS[_k + "c1hyb_cloud"] = ("%s, MATRIX device squeeze %s (the device runs with %s): CLOUD stream = the device's rt2 draft (2-step t=[1000, 500]) finished by ONE cloud 14B FP8 step at the last node, σ 0.625 (timesteps_index 750 of the 4-node schedule; --use 3) on EVERY chunk (--max-lag 99; cloud unchanged from rt2c1hyb), in the cloud's own context; each finished chunk is also fed back into the device KV (--thr 0). 2 GPUs" % (_HYB_PRE, _k, _d), "device 2, then 1 cloud step at σ 0.625 (timesteps_index 750, the last node) (cloud-final stream)")
ARMS["rt2c1hyb24_dev"] = ("%s, LIGHT DEVICE 24 GiB: the device process runs under a 24-GiB CUDA allocator cap from process start (--cap 24 --lowmem 1; OOM = does not fit) with GPTQ-4 g128 packed INT4 block weights from the offline file specwm_out/gptq4/14b_g128_832.pt (--wq gptq4 --wq-file F: loaded block by block, the bf16 checkpoint never on the GPU), FP8 (e4m3) KV ring buffer, KV window 12 (--kv fp8 --window 12), bf16 2x2-tiled VAE decode with the KV resident; DEVICE stream shown = rt2 (2-step t=[1000, 500]); EVERY chunk is sent to %s. 2 GPUs total" % (_HYB_PRE, _CLOUD), "device 2 (shown); + 1 cloud step at σ 0.625 (timesteps_index 750, the last node) fed back into the device KV")
ARMS["rt2c1hyb24_cloud"] = ("%s, LIGHT DEVICE 24 GiB (24-GiB allocator cap, GPTQ-4 INT4 block weights, FP8 KV, KV window 12 on the device): CLOUD stream = the device's rt2 draft (2-step t=[1000, 500]) finished by ONE cloud 14B FP8 step at the last node, σ 0.625 (timesteps_index 750 of the 4-node schedule; --use 3) on EVERY chunk (--max-lag 99; cloud unchanged from rt2c1hyb), in the cloud's own context; each finished chunk is also fed back into the device KV (--thr 0). 2 GPUs" % _HYB_PRE, "device 2, then 1 cloud step at σ 0.625 (timesteps_index 750, the last node) (cloud-final stream)")
# [10/21 cloud-fraction sweep, additive] rt2c1hyb32g<B>: rt2c1hyb32 whose DEVICE sends only a fraction of its chunks to the cloud (two_machine --gate random --gate-budget B: chunks 0-2 always sent, every later chunk sent with probability B from a seeded RNG; an unsent chunk is accepted as drafted and the cloud commits it as is, so the cloud's own context still follows the device). B = 0.5 / 0.25 / 0.125 / 0 (g0 = chunks 0-2 only).
for _gb, _gn in (("0.5", "g50"), ("0.25", "g25"), ("0.125", "g12"), ("0", "g0")):
    ARMS["rt2c1hyb32%s_dev" % _gn] = (ARMS["rt2c1hyb32_dev"][0].replace("EVERY chunk is sent to", "chunks 0-2 and then a random fraction %s of the chunks (--gate random --gate-budget %s, seeded; the rest are accepted as drafted and committed as is on both sides) are sent to" % (_gb, _gb)), ARMS["rt2c1hyb32_dev"][1] + " (cloud fraction %s)" % _gb)
    ARMS["rt2c1hyb32%s_cloud" % _gn] = (ARMS["rt2c1hyb32_cloud"][0].replace("on EVERY chunk (--max-lag 99; cloud unchanged from rt2c1hyb)", "on chunks 0-2 and a random fraction %s of the later chunks (--gate random --gate-budget %s on the device; the other chunks are committed as drafted; --max-lag 99; cloud otherwise unchanged from rt2c1hyb)" % (_gb, _gb)), ARMS["rt2c1hyb32_cloud"][1] + " (cloud fraction %s)" % _gb)
    assert "fraction" in ARMS["rt2c1hyb32%s_dev" % _gn][0] and "fraction" in ARMS["rt2c1hyb32%s_cloud" % _gn][0]
HYB_ARMS = ("adapt25c1hyb_dev", "adapt25c1hyb_cloud", "rt2c1hyb_dev", "rt2c1hyb_cloud", "sf24c1hyb_dev", "sf24c1hyb_cloud", "rt2c1hyb32_dev", "rt2c1hyb32_cloud") + ("rt2c1hyb24_dev", "rt2c1hyb24_cloud") + tuple(k + s_ for k in MX for s_ in ("c1hyb_dev", "c1hyb_cloud"))   # [10/17 C] (c) two_machine device/cloud pair; [10/19] + rt2c1hyb, sf24c1hyb; [10/19 thesis] + rt2c1hyb24, mx<X>c1hyb
HYB_ARMS = HYB_ARMS + tuple("rt2c1hyb32%s_%s" % (g, r) for g in ("g50", "g25", "g12", "g0") for r in ("dev", "cloud"))   # [10/21 cloud-fraction sweep]
# [10/22 step 0 + 3, additive] rt2c1hyb32o<P>: ORACLE gate (every chunk sent and refined; the device re-commits only corrections with rel > thr = the 50/75/87.5 % rel percentile of the frozen rt2c1hyb32 runs -> ~50/25/12.5 % applied; cloud cost not saved);
# rt2c1hyb32d: LEARNED drift-head gate (--gate drift --drift-tau -3.25, chunks 0-2 always sent); rt2c1hyb32L1: cloud --max-lag 1 (stale drafts committed as drafted = mode skip) instead of 99.
for _on, _txt in (("o50", "ORACLE gate: every chunk is sent and refined, but the device re-commits only the corrections whose relative change exceeds the 50 % percentile of rel (--thr; ~50 % applied)"),
                  ("o25", "ORACLE gate: every chunk is sent and refined, but the device re-commits only the corrections whose relative change exceeds the 75 % percentile of rel (--thr; ~25 % applied)"),
                  ("o12", "ORACLE gate: every chunk is sent and refined, but the device re-commits only the corrections whose relative change exceeds the 87.5 % percentile of rel (--thr; ~12.5 % applied)"),
                  ("d", "LEARNED gate: chunks 0-2 and then only the chunks flagged by the 10/9 scene-general drift head on the layer-20 hidden of the dense commit pass (--gate drift --drift-tau -3.25) are sent to the cloud; the rest are accepted as drafted"),
                  ("L1", "EVERY chunk is sent, but the cloud keeps a backlog of at most ONE draft (--max-lag 1): a further queued draft is committed as drafted (mode skip) instead of being refined late, so a slow cloud degrades gracefully")):
    ARMS["rt2c1hyb32%s_dev" % _on] = (ARMS["rt2c1hyb32_dev"][0].replace("EVERY chunk is sent to", _txt + "; sent to", 1), ARMS["rt2c1hyb32_dev"][1] + " (%s)" % _on)
    ARMS["rt2c1hyb32%s_cloud" % _on] = (ARMS["rt2c1hyb32_cloud"][0].replace("on EVERY chunk (--max-lag 99; cloud unchanged from rt2c1hyb)", "with the %s variant on the device/cloud (%s)" % (_on, _txt), 1), ARMS["rt2c1hyb32_cloud"][1] + " (%s)" % _on)
    assert _txt in ARMS["rt2c1hyb32%s_dev" % _on][0] and _txt in ARMS["rt2c1hyb32%s_cloud" % _on][0]
HYB_ARMS = HYB_ARMS + tuple("rt2c1hyb32%s_%s" % (g, r) for g in ("o50", "o25", "o12", "d", "L1") for r in ("dev", "cloud"))   # [10/22 step 0 + 3]

# [10/19 audit] node-label fix for the FROZEN timing.json "model" / "steps" strings of adapt25 and every *c1hyb* arm (gen/fix_node_labels_1019.py applies it in place):
#   the cloud step (--use 3 of timesteps_index 0,250,500,750 = σ [1.0, 0.937, 0.833, 0.625]) and adapt25's --adapt-node 2 (of nodes 0,500,750) are the SAME node, σ 0.625 (t ≈ 625).
#   Longest patterns first; applying the table to the ARMS strings above is a no-op (asserted at import).
NODE_LAST = "the last node, σ 0.625 (timesteps_index 750 of the 4-node schedule"
NODE_LABEL_FIX = [
    ("ONE step at t=750 (node index 3 of [1000, 750, 500, 250], re-noised draft, --max-lag 99)", "ONE step at %s 0,250,500,750 = σ [1.0, 0.937, 0.833, 0.625]; --use 3; re-noised draft, --max-lag 99)" % NODE_LAST),
    ("ONE cloud 14B FP8 step at t=750 on EVERY chunk (node index 3 of [1000, 750, 500, 250], --max-lag 99)", "ONE cloud 14B FP8 step at %s 0,250,500,750 = σ [1.0, 0.937, 0.833, 0.625]; --use 3) on EVERY chunk (--max-lag 99)" % NODE_LAST),
    ("ONE cloud 14B FP8 step at t=750 on EVERY chunk (--max-lag 99", "ONE cloud 14B FP8 step at %s; --use 3) on EVERY chunk (--max-lag 99" % NODE_LAST),
    ("ONE step at t=750 (--max-lag 99)", "ONE step at %s; --use 3; --max-lag 99)" % NODE_LAST),
    ("step-1 verifier head adds a 3rd step (t=250) on 25 % of the chunks", "step-1 verifier head adds a 3rd step at %s; --adapt-node 2 of nodes 0,500,750 = the same node as the hybrids' cloud step) on 25 %% of the chunks" % NODE_LAST),
    ("step-1 verifier head adds a 3rd step t=250 on 25 % of the chunks", "step-1 verifier head adds a 3rd step at %s; --adapt-node 2 of nodes 0,500,750 = the same node as the hybrids' cloud step) on 25 %% of the chunks" % NODE_LAST),
    ("step-1 verifier head 3rd step t=250 on 25 % of the chunks", "step-1 verifier head 3rd step at the last node, σ 0.625 (timesteps_index 750) on 25 % of the chunks"),
    ("1 cloud step at t=750", "1 cloud step at σ 0.625 (timesteps_index 750, the last node)")]
def fix_node_labels(s):
    """[10/19 audit] apply NODE_LABEL_FIX to one string (None / non-str returned unchanged)"""
    if not isinstance(s, str): return s
    for old, new in NODE_LABEL_FIX: s = s.replace(old, new)
    return s
for _k, _v in ARMS.items(): assert fix_node_labels(_v[0]) == _v[0] and fix_node_labels(_v[1]) == _v[1], ("ARMS string still carries an old node label", _k)
def spec_early_dit_fields(L, gen_s, steady_fps_gen):
    """[10/19 audit] spec_early / spec_early_tf loop.json: the chunk timer (chunk_start, time.time() with NO cuda sync, spec_early.py line 100) starts right after
    the previous chunk's SDc.push (line 141 / 116): the side-stream VAE decode of chunk c-1 (--stream-decode 1) then runs during chunk c's step 1 and the
    first torch.cuda.synchronize() inside the chunk waits for it, so chunk_s / gen_s contain the decode (two_machine solo syncs before AND after its chunk timer
    and pushes after it: its chunk_s excludes the decode). times = {denoise, commit, verify} are per-section totals (each closed by a sync; start of denoise =
    right after the previous commit's sync + the push's host-side launches): gen_s_dit = their sum = gen_s minus the host-side push / chunk_noise gaps
    (~0.14 s per chunk at 640x352, ~0.24 s at 832x480, ~0.55 s at 1280x704). The GPU time of the overlapped decode is still inside 'denoise' (device-wide
    sync), so gen_s_dit remains CONSERVATIVE vs two_machine's chunk_s; the queued _dit re-measure (--stream-decode 0, gen/queue/8/302) quantifies the rest.
    In --mode early (sf24) 'verify' (~0.05 s/clip) is nested inside 'denoise' and counted twice here (conservative); spec_early_tf subtracts it.
    steady_fps_gen_dit = steady_fps_gen x gen_s / gen_s_dit: the median chunk is scaled uniformly (the per-section totals are not per chunk)."""
    tm = L.get("times") or {}
    if "denoise" not in tm or "commit" not in tm or not gen_s: return {}
    gd = float(tm["denoise"]) + float(tm["commit"]) + float(tm.get("verify") or 0.0)
    if gd <= 0: return {}
    return dict(gen_s_dit=round(gd, 3), steady_fps_gen_dit=round(float(steady_fps_gen) * float(gen_s) / gd, 2),
                fps_correction="[10/19 audit] gen_s / steady_fps_gen / chunk_s are from an unsynced per-chunk wall timer that also contains the previous chunk's side-stream VAE decode (--stream-decode 1); gen_s_dit = times.denoise + commit + verify (CUDA-synced section totals) and steady_fps_gen_dit = steady_fps_gen x gen_s / gen_s_dit (uniform scaling of the median chunk); still conservative (the overlapped decode's GPU time stays inside denoise); frozen values unchanged")
LONG_FRAMES_OUT = {"P6": 961}   # [10/19 EXP 1] long programs (baselines/convert.py LONG_PROGRAMS): frames written = 16 fps x 60 s + 1; every other program keeps FRAMES_OUT = 161
def frames_out_for(task, override=None):
    """[10/19] frames written for a task id <set>__<scene>__<P>[suffix]: LONG_FRAMES_OUT[P] for a long program, else FRAMES_OUT (161)"""
    import re as _re
    if override: return int(override)
    m = _re.match(r"(P\d+p?)", task.split("__")[-1]) if "__" in task else None
    return LONG_FRAMES_OUT.get(m[1], FRAMES_OUT) if m else FRAMES_OUT
ONE_P3 = ("student1p3b", "sf1p3b_pub", "sf1p3b_cal")   # 1.3B arms: bf16, CKPT1P3
SF_ARMS = ("sf24", "sf1p3b_pub", "sf1p3b_cal")        # token-wise early-exit arms: steps string carries mean_steps
ATTN_TF = "SageAttention for dense forwards (step 1 of every chunk, clean t=0 commit); SDPA flash for the active-token (subset) forwards of steps 2-4 (fast_patch.ACTIVE, index padding to a multiple of 128); torch.compile(dynamic=True) per block"
def sh(cmd): return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
def probe(path):
    o = json.loads(sh(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate", "-of", "json", path]))["streams"][0]
    return o["width"], o["height"], int(o["nb_read_frames"]), o["r_frame_rate"]
def trim(src, dst, n=FRAMES_OUT):
    sh(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-frames:v", str(n), "-an", "-c:v", "libx264", "-crf", str(CRF), "-preset", "medium", "-pix_fmt", "yuv420p", "-r", "16", dst])
CRF = 18   # [10/17 B] frozen video quality (was 10; crf 18 is visually lossless and ~half the size; disk)
PURGE_NAMES = ("lat.pt", "spec.mp4", "video.mp4")
def purge(run, kind):
    """[10/17 B] delete the raw latent + raw video of a CONVERTED task (keep timing.json / loop.json); spec_early: also the <arm>_warm sibling's lat.pt / video.mp4"""
    gone = []
    dirs = [run] + ([d for d in glob.glob(os.path.dirname(run.rstrip("/")) + "/*_warm") if os.path.isdir(d)] if kind == "spec_early" else [])
    for d in dirs:
        for n in PURGE_NAMES:
            p = os.path.join(d, n)
            if os.path.isfile(p): gone.append(p); os.remove(p)
    return gone
def median(v): v = sorted(v); return float(v[len(v) // 2]) if v else None
def load_summary(path, tries=4):
    """[10/17 verifier] <out>/summary.json is rewritten by two_machine after EVERY clip; with several workers on chunks of the same (set, arm) a read can hit a
    half-written file. Its fields are informational here (wq / window come from the CLI), so retry briefly and fall back to {} instead of failing the chunk."""
    import time
    for i in range(tries):
        if not os.path.exists(path): return {}
        try: return json.load(open(path))
        except Exception as e:
            if i == tries - 1: print("[warn] unreadable %s after %d tries (%s): summary fields omitted" % (path, tries, e)); return {}
            time.sleep(0.5)
    return {}
def e2e_from_ready(fr, prep_s, t0):
    """fr = [(frame, chunk, ready wall time)] -> (ttff, e2e, steady_fps_e2e, chunk_ready_interval list)"""
    if not fr: return None, None, None, None
    last = {}
    for f, c, t in fr: last[c] = max(last.get(c, 0.0), t)
    ks = sorted(last); iv = [last[ks[i]] - last[ks[i - 1]] for i in range(1, len(ks))]; st = iv[2:] if len(iv) > 3 else iv   # intervals ending at chunks >= 3
    return round(prep_s + fr[0][2] - t0, 4), round(prep_s + last[ks[-1]] - t0, 4), (round(FPC / median(st), 2) if st else None), [round(x, 4) for x in iv]
def convert(run, out, arm, set_=None, kv_window=None, wq=None, label=None, frames_out=None):
    os.makedirs(out, exist_ok=True); nulls = {}; T = dict(model=None, ckpt=None, precision=None, attention=ATTN, vae=None, resolution_out=None, frames=None, frames_generated=None, fps_native=16,
        frames_per_chunk=FPC, frames_per_call_or_chunk="12 (chunk = 3 latent frames; chunk 0 yields 9 = 1 + 4 + 4)", steps=None, gen_s=None, e2e_s=None, ttff_s=None, steady_fps_gen=None, steady_fps_e2e=None,
        peak_alloc_GiB=None, peak_reserved_GiB=None, text_encoder_resident=None, kv_window=None, wq=None, warm=True)
    if os.path.exists(run + "/loop.json"):   # (b) spec_early
        L = json.load(open(run + "/loop.json")); src = run + "/video.mp4"; kind = "spec_early"; is14 = arm not in ONE_P3
        cs = L["chunk_start"] + [L.get("t_gen_end", L["t_gen_start"] + L["gen_s"])]; dt = list(np.diff(cs)); st = dt[3:] if len(dt) > 4 else dt
        prep = L["prepare_s"]; ttff, e2e, fps_e2e, iv = e2e_from_ready(L.get("frame_ready"), prep, L["t_gen_start"])
        if arm in ("sf1p3b_pub", "sf1p3b_cal"):   # [10/17 B] training-free verifier fields (spec_early_tf loop.json)
            T.update(attention=ATTN_TF, tau_red=L.get("tau_red"), tau_stab=L.get("tau_stab"), tf_norm=L.get("tf_norm"), image_ref=L.get("image_ref"), latent_frames_per_chunk=3, kv_sink=6,
                     verifier="training-free (arXiv 2610.22347 Sec. 3.3.1): step 1 redundancy vs the clean tokens in the KV window + image latent, steps 2-3 stability; 9x9 spatial + forward temporal Gaussian smoothing; 64-dim latent tokens, distances / median token norm of the draft (specwm/early_exit_tf.py)",
                     gen_method_s=L.get("gen_method_s"))
        T.update(model=ARMS[arm][0], ckpt=CKPT14 if is14 else CKPT1P3, precision=PREC["fp8" if is14 else "bf16"], steps=("4 with token-wise early exit: mean_steps %.3f" % L["mean_steps"]) if arm in SF_ARMS else ARMS[arm][1],
                 gen_s=round(L["gen_s"], 3), steady_fps_gen=round(FPC / median(st), 2), text_encoder_resident=True, kv_window=kv_window or 18, wq=wq or ("fp8" if is14 else "bf16"), warm=True,
                 vae="Wan2.1_VAE.pth, full decode after generation (video.mp4)" + ("; --stream-decode 1: second VAE instance decodes every committed chunk on a side stream (ttff / e2e)" if L.get("stream_decode") else ""),
                 mean_steps=round(L["mean_steps"], 4), prepare_s=prep, decode_full_s=L.get("decode_s"), chunk_s=[round(float(x), 4) for x in dt], seed=L["seed"], task=L["task"], mode=L["mode"], eps=L.get("eps"), per_chunk_commit_hist=L.get("per_chunk_commit_hist"),
                 ttff_s=ttff, e2e_s=e2e, steady_fps_e2e=fps_e2e, chunk_ready_interval_s=iv, peak_alloc_GiB=L.get("peak_alloc_GiB"), peak_reserved_GiB=L.get("peak_reserved_GiB"))
        if not L.get("frame_ready"): nulls.update(ttff_s="spec_early run without --stream-decode 1", e2e_s="same", steady_fps_e2e="same")
        if T["peak_alloc_GiB"] is None: nulls.update(peak_alloc_GiB="spec_early loop.json has no peak stats", peak_reserved_GiB="same")
        if T["e2e_s"] is None and L.get("decode_s") is not None: T["e2e_s"] = round(prep + L["gen_s"] + L["decode_s"], 3); nulls.pop("e2e_s", None); T["e2e_note"] = "prepare + gen + full decode after generation (not streamed)"
        T.update(spec_early_dit_fields(L, L["gen_s"], T["steady_fps_gen"]))   # [10/19 audit] gen_s_dit / steady_fps_gen_dit / fps_correction (additive; gen_s / steady_fps_gen unchanged)
        task = L["task"]
    elif arm in HYB_ARMS:   # (c) [10/17 C] two_machine hybrid pair
        J = json.load(open(run + "/timing.json")); src = run + "/spec.mp4"; kind = "two_machine_hyb"; task = J["task"]; role = "device" if arm.endswith("_dev") else "cloud"
        root = os.path.dirname(os.path.dirname(run.rstrip("/"))); other = "cloud" if role == "device" else "device"
        A = load_summary(root + "/" + role + "/summary.json"); pp = "%s/%s/%s/timing.json" % (root, other, task); P = json.load(open(pp)) if os.path.exists(pp) else None
        D, C = (J, P) if role == "device" else (P, J)
        T.update(model=ARMS[arm][0], ckpt=CKPT14, precision=PREC[wq or "fp8"], text_encoder_resident=False, gpus=2, role=role, stream=("device-shown" if role == "device" else "cloud-final"),   # [10/19 thesis] precision follows --wq (gptq4 device streams; fp8 as before otherwise)
                 text_encoder="umT5-XXL on the CPU (V2Binding t5_cpu=True) in both processes; the prompt is encoded during prepare, cached per prompt string", kv_window=kv_window or A.get("window", 18), kv_sink=6, wq=wq or "fp8", warm=True,
                 vae="Wan2.1_VAE.pth, full decode after generation (spec.mp4) in each process; no streamed decode (two_machine --timed-decode is solo only)", seed=A.get("seed", 1234), task=task,
                 nodes=A.get("nodes"), use=A.get("use"), area=A.get("area"), frames_arg=A.get("frames"), transport="TCP 127.0.0.1 (both GPUs on one host, no injected delay), length-prefixed pickles; fp16 chunk latents",
                 partner_timing=os.path.abspath(pp) if P else None)
        if D is not None:
            dch = D["chunks"]; tot = [r_["total_s"] for r_ in dch]; ext = [r_.get("ext") for r_ in dch]; n_ext = sum(1 for r_ in dch[3:] if r_.get("ext")); n_ext_all = sum(1 for e_ in ext if e_)
            dev_steps = "2 + chunk-wise 3rd: mean %.3f (chunks 0-2 always, %d of %d later chunks extended)" % (2 + n_ext_all / len(dch), n_ext, len(dch) - 3)
            if arm.startswith("rt2c1hyb") or arm.startswith("mx"): dev_steps = "2 (t=[1000, 500])"   # [10/19] rt2 device; [10/19 thesis] + mx<X>c1hyb (rt2 schedule)
            if arm.startswith("sf24c1hyb"): dev_steps = "4 with token-wise early exit: mean_steps %.3f" % D.get("ee_mean_steps", float("nan")); T.update(mean_steps=D.get("ee_mean_steps"), ee_eps=D.get("ee_eps"), ee_head=D.get("ee_head"), ee_per_chunk=D.get("ee"))   # [10/19] sf24 device
            T.update(device_fps_shown=D.get("fps_shown"), device_total_s_median=D.get("total_s_median"), device_draft_s_median=D.get("draft_s_median"), corr_lag_median_s=D.get("corr_lag_median"),
                     corrections_received=D.get("n_corr_received"), corrections_applied=D.get("n_applied"), chunks_sent=D.get("n_sent"), device_peak_alloc_GiB=D.get("peak_GiB"), device_clip_s=D.get("clip_s"),
                     corr_lag_s=[c_["lag_s"] for c_ in D.get("corrections", [])], corr_applied=[bool(c_["applied"]) for c_ in D.get("corrections", [])], corr_age_chunks=[c_["age_chunks"] for c_ in D.get("corrections", [])],
                     n_ext=n_ext, ext=ext)
        else: dev_steps = None; nulls["device_fields"] = "partner device timing.json missing (%s)" % pp
        if C is not None:
            cch = sorted(C["chunks"], key=lambda r_: r_["c"])
            T.update(cloud_lat_from_show_median_s=C.get("cloud_lat_from_show_median"), cloud_lat_from_show_s=[r_["lat_from_show_s"] for r_ in cch], cloud_refined=C.get("n_refined"), cloud_skipped=C.get("n_skipped"),
                     cloud_refine_s_median=C.get("refine_s_median"), cloud_mode=[r_["mode"] for r_ in cch], cloud_rel_change=[r_["rel"] for r_ in cch], cloud_peak_alloc_GiB=C.get("peak_GiB"), cloud_clip_s=C.get("clip_s"))
        else: nulls["cloud_fields"] = "partner cloud timing.json missing (%s)" % pp
        if role == "device":
            tc = [r_["total_s"] for r_ in J["chunks"]]; st = tc[3:] if len(tc) > 4 else tc
            T.update(steps="%s (shown); every chunk then gets 1 cloud step at σ 0.625 (timesteps_index 750, the last node), fed back into the device KV" % dev_steps,   # [10/19 audit] node label; [10/20] the comment had swallowed chunk_s / draft_s / gen_s (fixed; gen/fix_dev_gen_s_1020.py re-derived the frozen files)
                     chunk_s=tc, draft_s=[r_["draft_s"] for r_ in J["chunks"]], gen_s=round(sum(tc), 3),
                     steady_fps_gen=round(FPC / median(st), 2), peak_alloc_GiB=J.get("peak_GiB"),
                     notes="device-shown stream: steady_fps_gen = 12 / median per-chunk device wall over chunks >= 3 (= two_machine fps_shown); gen_s = sum of the 14 per-chunk device walls; peak_alloc_GiB = DEVICE torch peak (weights included, since the warm-up reset); the cloud GPU's peak is cloud_peak_alloc_GiB. Corrections only change the device's KV (later chunks), the shown frames of a chunk are its draft.")
        else:
            st0 = []; t_ = 0.0
            if D is not None:
                for r_ in D["chunks"]: st0.append(t_); t_ += r_["total_s"]
                dmap = {r_["c"]: r_ for r_ in D["chunks"]}
                ready = [st0[r_["c"]] + dmap[r_["c"]]["draft_s"] + r_["lat_from_show_s"] for r_ in cch]
                iv = [ready[0]] + [ready[i] - ready[i - 1] for i in range(1, len(ready))]; stv = iv[3:] if len(iv) > 4 else iv
                T.update(chunk_s=[round(x, 4) for x in iv], chunk_final_t_s=[round(x, 4) for x in ready], gen_s=round(ready[-1], 3), steady_fps_gen=round(FPC / median(stv), 2))
            else: nulls.update(gen_s="needs the device timing.json", steady_fps_gen="same", chunk_s="same")
            T.update(steps="%s, then 1 cloud step at σ 0.625 (timesteps_index 750, the last node) on %s of %d chunks" % (dev_steps, C.get("n_refined"), len(cch)), peak_alloc_GiB=J.get("peak_GiB"),   # [10/19 audit] node label
                     notes="cloud-final stream: chunk c is final at device-start(c) + draft_s[c] + cloud lat_from_show_s[c] (one host clock; device-start(c) = sum of the earlier device chunk walls, relative to the device's chunk-0 start); chunk_s = intervals of these times (chunk_s[0] = time to chunk 0 final), steady_fps_gen = 12 / their median over chunks >= 3, gen_s = time until the last chunk is final (all excl. prepare); peak_alloc_GiB = CLOUD torch peak; the device's is device_peak_alloc_GiB.")
        nulls.update(ttff_s="hybrid arm: two_machine --timed-decode (streamed decode) exists for --role solo only; no per-frame ready times", e2e_s="same", steady_fps_e2e="same", decode_s="same", peak_reserved_GiB="not recorded by two_machine device/cloud")
        DM = D.get("mem") if D is not None else None
        if DM:   # [10/19 light device] the device process ran with --kv fp8 / --cap / --lowmem: its memory records (two_machine timing.json "mem")
            T.update(device_mem=DM, device_peak_alloc_proc_GiB=DM.get("proc_peak_alloc_GiB"), device_peak_reserved_proc_GiB=DM.get("proc_peak_reserved_GiB"), device_nvsmi_peak_MiB=DM.get("nvsmi_proc_peak_MiB"),
                     device_cap_GiB=DM.get("cap_GiB"), device_kv_dtype=("float8_e4m3fn" if DM.get("kv") == "fp8" else "bfloat16"), device_kv_window=DM.get("window"), device_lowmem=DM.get("lowmem"),
                     device_clip_peak_alloc_GiB=DM.get("clip_peak_alloc_GiB"), device_clip_peak_reserved_GiB=DM.get("clip_peak_reserved_GiB"),
                     device_mem_note="device_peak_*_proc = torch peaks of the whole device process up to the end of this clip (load, warm-up, prepare, generation incl. KV re-commits, the frame-by-frame decode of spec.mp4 with the KV window resident); device_clip_peak_* = this clip only (prepare + generation + decode); device_nvsmi_peak_MiB = that process's nvidia-smi used memory (CUDA context included), polled every 0.5 s; device_cap_GiB = torch.cuda.set_per_process_memory_fraction cap held from process start (--lowmem 1)")
            if role == "device": T["peak_reserved_GiB"] = DM.get("clip_peak_reserved_GiB"); nulls.pop("peak_reserved_GiB", None)
            T.update(device_vae_dtype=DM.get("vae_dtype", "fp32"), device_vae_tiles=DM.get("vae_tiles", "1,1"))   # [10/19 PM] the device's spec.mp4 decode
            T.update(device_wq=DM.get("wq", "fp8"), device_wq_file=DM.get("wq_file"), device_int4_kernel=DM.get("int4_kernel"), device_wq_layers=DM.get("wq_layers"), device_wq_meta=DM.get("wq_meta"), device_compile=DM.get("compile"))   # [10/19 thesis] GPTQ-4 device records (None / fp8 for every older arm)
            if role == "device" and (DM.get("vae_dtype", "fp32") != "fp32" or DM.get("vae_tiles", "1,1") != "1,1"):
                T["vae"] = "Wan2.1_VAE.pth, DEVICE decode after generation (two_machine --vae-dtype %s --vae-tiles %s): %s copy of the VAE (weights + activations, no autocast), latent frame by latent frame with the KV window resident, %s spatial tiles (overlap 8 latents, linear feathering, blended on the host); approximate vs the fp32 decode (10/19: 51-53 dB PSNR on the same latents, PROTOCOL §7); no streamed decode" % (DM.get("vae_dtype"), DM.get("vae_tiles"), DM.get("vae_dtype"), DM.get("vae_tiles", "1,1").replace(",", "x"))
    else:   # (a) two_machine solo
        J = json.load(open(run + "/timing.json")); src = run + "/spec.mp4"; kind = "two_machine"; task = J["task"]
        A = load_summary(os.path.dirname(run.rstrip("/")) + "/summary.json")   # [10/17 verifier] tolerant: a concurrent chunk of the same (set, arm) rewrites summary.json after every clip
        wq_ = wq or A.get("wq", "fp8"); tc = J["chunk_s"]; st = tc[3:] if len(tc) > 4 else tc; n_ext = J.get("n_ext")
        steps = ARMS[arm][1]
        if arm == "adapt25" and J.get("adapt"): steps = "2 + chunk-wise 3rd: mean %.3f (chunks 0-2 always, %d of %d later chunks extended)" % (2 + sum(1 for r in J["adapt"] if r["ext"]) / len(J["adapt"]), n_ext or 0, len(J["adapt"]) - 3)
        T.update(model=ARMS[arm][0], ckpt=CKPT14, precision=PREC[wq_], steps=steps, gen_s=round(sum(tc), 3), steady_fps_gen=round(FPC / median(st), 2), text_encoder_resident=False,
                 text_encoder="umT5-XXL on the CPU (V2Binding t5_cpu=True); the prompt is encoded during prepare (inside prep_s / ttff_s), cached per prompt string", kv_window=kv_window or A.get("window", 18), kv_sink=6, wq=wq_, warm=True,
                 vae="Wan2.1_VAE.pth, full decode after generation (spec.mp4)" + ("; --timed-decode 1: second VAE instance decodes every committed chunk on a side stream right after its commit (ttff / e2e / decode_s)" if J.get("timed_decode") else ""),
                 chunk_s=tc, seed=A.get("seed", 1234), task=task, nodes=A.get("nodes"), use=A.get("use"), area=A.get("area"), frames_arg=A.get("frames"), adapt=J.get("adapt"), n_ext=n_ext, peak_alloc_GiB=J.get("peak_alloc_GiB", J.get("peak_GiB")), peak_reserved_GiB=J.get("peak_reserved_GiB"))
        if J.get("timed_decode"):
            ttff, e2e, fps_e2e, iv = e2e_from_ready(J["frame_ready"], J["prep_s"], J["t_gen_start"])
            T.update(ttff_s=ttff, e2e_s=e2e, steady_fps_e2e=fps_e2e, decode_s=J["decode_s"], prepare_s=J["prep_s"], chunk_decode_s=[r["gpu_ms"] / 1000.0 for r in J["chunk_decode"]], chunk_ready_interval_s=iv, frames_streamed=J.get("frames_streamed"))
        else: nulls.update(ttff_s="two_machine run without --timed-decode 1 (no streamed decode)", e2e_s="same", steady_fps_e2e="same", decode_s="same", peak_reserved_GiB="only recorded with --timed-decode 1")
        DM = J.get("mem")
        if DM:   # [10/19 thesis] solo matrix arms (mx<X>): the process ran with --kv fp8 / --cap / --lowmem / --wq gptq4 / light decode -> the same memory records as the hybrids' device stream, on this (single) GPU
            T.update(device_mem=DM, device_peak_alloc_proc_GiB=DM.get("proc_peak_alloc_GiB"), device_peak_reserved_proc_GiB=DM.get("proc_peak_reserved_GiB"), device_nvsmi_peak_MiB=DM.get("nvsmi_proc_peak_MiB"),
                     device_cap_GiB=DM.get("cap_GiB"), device_kv_dtype=("float8_e4m3fn" if DM.get("kv") == "fp8" else "bfloat16"), device_kv_window=DM.get("window"), device_lowmem=DM.get("lowmem"),
                     device_clip_peak_alloc_GiB=DM.get("clip_peak_alloc_GiB"), device_clip_peak_reserved_GiB=DM.get("clip_peak_reserved_GiB"), device_vae_dtype=DM.get("vae_dtype", "fp32"), device_vae_tiles=DM.get("vae_tiles", "1,1"),
                     device_wq=DM.get("wq", "fp8"), device_wq_file=DM.get("wq_file"), device_int4_kernel=DM.get("int4_kernel"), device_wq_layers=DM.get("wq_layers"), device_wq_meta=DM.get("wq_meta"), device_compile=DM.get("compile"),
                     device_mem_note="solo arm: device_* = this single process (torch peaks since the warm-up reset: load / warm-up / each clip's prepare + generation + decode; nvidia-smi used memory polled every 0.5 s; device_cap_GiB = torch.cuda.set_per_process_memory_fraction cap held from process start with --lowmem 1)")
            if T.get("peak_reserved_GiB") is None and DM.get("clip_peak_reserved_GiB") is not None: T["peak_reserved_GiB"] = DM.get("clip_peak_reserved_GiB"); nulls.pop("peak_reserved_GiB", None)
            if DM.get("vae_dtype", "fp32") != "fp32" or DM.get("vae_tiles", "1,1") != "1,1":
                T["vae"] = "Wan2.1_VAE.pth, decode after generation (two_machine --vae-dtype %s --vae-tiles %s): %s copy of the VAE (weights + activations, no autocast), latent frame by latent frame with the KV window resident, %s spatial tiles (overlap 8 latents, linear feathering, blended on the host); approximate vs the fp32 decode (10/19: 51-53 dB PSNR on the same latents, PROTOCOL §7); no streamed decode" % (DM.get("vae_dtype"), DM.get("vae_tiles"), DM.get("vae_dtype"), DM.get("vae_tiles", "1,1").replace(",", "x"))
    # video: first 161 frames
    assert os.path.exists(src), "no video at " + src
    nout = frames_out_for(task, frames_out)   # [10/19] 161, or 961 for the 60-s program P6
    w0, h0, n0, _ = probe(src); dst = out + "/video.mp4"; trim(src, dst, nout); w, h, n, fr = probe(dst)
    assert n == min(nout, n0) and (w, h) == (w0, h0), (src, (w0, h0, n0), dst, (w, h, n))
    T.update(resolution_out="%dx%d" % (w, h), frames=n, frames_generated=n0, clip_s=round(n / 16.0, 3), video_fps=fr, kind=kind, arm=arm, set=set_ or task.split("__")[0], program=task.split("__")[-1] if "__" in task else None, scene=task.split("__")[1] if task.count("__") >= 2 else None,
             source=os.path.abspath(run), label=label, nulls=nulls, trim="ffmpeg -frames:v %d libx264 crf %d from the %d generated frames" % (nout, CRF, n0))
    if n0 < nout: nulls["frames"] = "source has only %d frames (< %d)" % (n0, nout)
    json.dump(T, open(out + "/timing.json", "w"), indent=1)
    print(json.dumps(dict(task=task, out=out, resolution_out=T["resolution_out"], frames=n, frames_generated=n0, steps=T["steps"], gen_s=T["gen_s"], e2e_s=T["e2e_s"], ttff_s=T["ttff_s"], steady_fps_gen=T["steady_fps_gen"], steady_fps_e2e=T["steady_fps_e2e"], peak_alloc_GiB=T["peak_alloc_GiB"], peak_reserved_GiB=T["peak_reserved_GiB"], mean_steps=T.get("mean_steps"), nulls=nulls)))
    return T, kind
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("out"); ap.add_argument("--arm", required=True, choices=list(ARMS)); ap.add_argument("--set", default=None); ap.add_argument("--kv-window", type=int, default=None); ap.add_argument("--wq", default=None, choices=[None, "fp8", "bf16", "gptq4"])   # [10/19 thesis] + gptq4
    ap.add_argument("--batch", action="store_true", help="RUN = raw out dir with one sub-dir per task; OUT = runs/<arm_set> dir"); ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--label", default=None)
    ap.add_argument("--tasks", default="", help="[10/17 B] --batch: comma list of task ids to convert (default: every sub-dir of RUN); ids without a raw dir are reported as missing")
    ap.add_argument("--frames-out", type=int, default=None, help="[10/19] frames written (default: 161, or 961 for a P6 task)")
    ap.add_argument("--purge", action="store_true", help="[10/17 B] after a successful conversion delete the raw lat.pt + spec.mp4 / video.mp4 of that task (and the spec_early <arm>_warm dir's); timing.json / loop.json stay")
    a = ap.parse_args()
    if not a.batch: convert(a.run, a.out, a.arm, a.set, a.kv_window, a.wq, a.label, a.frames_out); sys.exit(0)
    n = 0; failed = []; missing = []
    want = [t for t in a.tasks.split(",") if t] if a.tasks else None
    dirs = [a.run.rstrip("/") + "/" + t for t in want] if want else [d.rstrip("/") for d in sorted(glob.glob(a.run.rstrip("/") + "/*/"))]
    for r0 in dirs:
        task = os.path.basename(r0); r = r0
        if not os.path.isdir(r0): print("[missing] no raw dir", r0); missing.append(task); continue
        if not os.path.exists(r + "/timing.json"):
            cands = sorted(glob.glob(r + "/*_s%d/loop.json" % a.seed)); cands = [c for c in cands if "_warm" not in c]
            if not cands:
                if want: print("[missing] no timing.json / loop.json in", r); missing.append(task)
                else: print("[skip] no timing.json / loop.json in", r)
                continue
            r = os.path.dirname(cands[0])
        try:
            _, kind = convert(r, a.out.rstrip("/") + "/" + task, a.arm, a.set, a.kv_window, a.wq, a.label, a.frames_out); n += 1
            if a.purge and kind == "two_machine_hyb": gone = purge(r, "two_machine"); print("[purge]", task, len(gone), "files:", " ".join(os.path.relpath(g, a.run) for g in gone))
            elif a.purge:
                gone = purge(r, kind); print("[purge]", task, len(gone), "files:", " ".join(os.path.relpath(g, a.run) for g in gone))
        except Exception as e: print("[fail]", task, repr(e)); failed.append(task)
    print("[batch] converted %d failed %d missing %d -> %s" % (n, len(failed), len(missing), a.out), ("FAILED: " + ",".join(failed)) if failed else "", ("MISSING: " + ",".join(missing)) if missing else "")
    sys.exit(1 if (failed or missing) else 0)
