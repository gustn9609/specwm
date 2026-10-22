import os
S = os.environ["SCRATCH"]+""
vids = [l.strip() for l in open(S + "/vbench_crops/videos.txt") if "=" in l]
s = open(S + "/vbench2.slurm").read(); head = s.split("--videos")[0]
head = head.replace("--out $SCRATCH/vbench_out2", "--out $SCRATCH/vbench_out4").replace("vbench2.%j.out", "vbench4.%j.out").replace("-p gh-dev", "-p gh").replace("-t 01:00:00", "-t 01:30:00")
body = "--videos \\\n" + " \\\n".join("  " + v for v in vids) + "\necho \"[VBENCH_JOB_DONE] exit=$?\"\n"
open(S + "/vbench4.slurm", "w").write(head + body); print("vbench4.slurm:", len(vids), "videos")
