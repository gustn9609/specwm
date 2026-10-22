import os
import subprocess, numpy as np, glob, os
FF=os.environ["SCRATCH"]+"/specwm-venv/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-aarch64-v7.0.2"
D=os.environ["SCRATCH"]+"/lingbot-world-v2/specwm_out/vid_int8"
O=os.environ["SCRATCH"]+"/frames"; os.makedirs(O, exist_ok=True)
W,H=832,480; picks=[10,40,70]
for f in sorted(glob.glob(D+"/*.mp4")):
    k=os.path.basename(f)[:-4]
    p=subprocess.run([FF,"-v","error","-i",f,"-f","rawvideo","-pix_fmt","rgb24","-"],
                     capture_output=True)
    if p.returncode!=0:
        print("FAIL",k,p.stderr[-200:]); continue
    buf=np.frombuffer(p.stdout,dtype=np.uint8)
    n=len(buf)//(W*H*3)
    v=buf[:n*W*H*3].reshape(n,H,W,3)
    sel=np.stack([v[min(i,n-1)] for i in picks])
    np.save(f"{O}/{k}.npy", sel)
    print("ok",k,"frames",n)
