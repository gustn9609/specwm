import os
S = os.environ["SCRATCH"]+""; R = S + "/lingbot-world-v2/specwm_out"
vids = [("base_S05_s16_seed4321", R + "/revisit/base_official_S05_s16_seed4321/video.mp4"), ("base_S05_s16_seed9999", R + "/revisit/base_official_S05_s16_seed9999/video.mp4"),
        ("base_prefix_S05_L9", R + "/revisit/base_prefix_S05_L9/video.mp4"), ("hands_causal_N3", R + "/hands/base_causal_N3/video.mp4"), ("hands_causal_N6", R + "/hands/base_causal_N6/video.mp4"), ("hands_causal_N12", R + "/hands/base_causal_N12/video.mp4")]
s = open(S + "/vbench2.slurm").read(); head = s.split("--videos")[0]
head = head.replace("--out $SCRATCH/vbench_out2", "--out $SCRATCH/vbench_out5").replace("vbench2.%j.out", "vbench5.%j.out").replace("-p gh-dev", "-p gh")
open(S + "/vbench5.slurm", "w").write(head + "--videos \\\n" + " \\\n".join("  %s=%s" % kv for kv in vids) + "\necho \"[VBENCH_JOB_DONE] exit=$?\"\n"); print("vbench5:", len(vids))
