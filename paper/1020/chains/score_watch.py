import json,glob,os,sys
B='/home/hk28965/specwm/baselines'
def cov(mdir,pats):
    miss=[]
    for pat in pats:
        for d in sorted(glob.glob(f'{B}/runs/{pat}')):
            n=len(glob.glob(d+'/*/video.mp4')); r=os.path.basename(d)
            for m in ('vlm','rpe','vbench6'):
                try: k=sum(1 for x in json.load(open(f'{mdir}/{m}.json')) if f'/runs/{r}/' in x)
                except Exception: k=0
                if k<n: miss.append(f'{r}:{m}:{n-k}')
    return miss
g0=cov(B+'/metrics_g0',['ours_mx*_long','ours_rt2c1hyb24_*_long','ours_rt2c1hyb32_*_long'])
g1=cov(B+'/metrics_g1',['ours_rs576*','ours_rt2_*_long','ours_teacher_*_long','ours_rt2c1hyb_dev_neutral_832_long','ours_rt2c1hyb_dev_wp_long'])
print('g0 missing',len(g0),'g1 missing',len(g1)); print(' '.join(g0[:6])); print(' '.join(g1[:6]))
sys.exit(0 if not g0 and not g1 else 1)
