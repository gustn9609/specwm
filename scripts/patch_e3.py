import os; S = os.environ["SCRATCH"]; p = S + "/r4E3.slurm"; s = open(p).read()
add = '''echo "##### mask + Base-refined prefix (S05 s1234): target #####"
cd $V1; PYTHONPATH=$V1 python -m specwm.bench_base_prefix2 --ckpt $CK --image $V2/examples/05/image.jpg --action-path $S/rt2_action_05 --prompt-file $S/p_05.txt --frames 161 --steps 16 --prefix-latents 9 --mode mask --seed 1234 --prefix-video $R/hybrid/s05_launder/REFINE_0.40_0.20_0.08/video.mp4 --out $R/revisit/e3d_mask_refined
echo "##### kvrw arms: repro #####"'''
assert s.count('echo "##### kvrw arms: repro #####"') == 1; s = s.replace('echo "##### kvrw arms: repro #####"', add)
old = 'raw=$R/resync_v2/W18_sink6/fast_alone_s1234/video.mp4"'; assert old in s
s = s.replace(old, 'raw=$R/resync_v2/W18_sink6/fast_alone_s1234/video.mp4,maskref=$R/revisit/e3d_mask_refined/video.mp4,repaintref=$R/revisit/e3c_repaint_laundered/video.mp4"')
s = s.replace("--arms kvrw_l15,kvrw_l25,kvrw_l40,kvrw_l60,kvrw_l80,kvrw_raw", "--arms kvrw_l15,kvrw_l25,kvrw_l40,kvrw_l60,kvrw_l80,kvrw_raw,kvrw_maskref,kvrw_repaintref")
open(p, "w").write(s); print("r4E3.slurm 패치 완료"); print([l for l in s.split("\n") if "e3d_mask_refined" in l or "kvrw_maskref" in l][:3])
