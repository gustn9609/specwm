import numpy as np, os, glob
for ex in sorted(glob.glob("examples/0*")):
    a=np.load(f"{ex}/action.npy"); p=np.load(f"{ex}/poses.npy")
    u=len(np.unique(a,axis=0)); has=os.path.exists(ex+"/image.jpg")
    print(f"{ex}: action {a.shape[0]} frames ({a.shape[0]/16:.1f}s), unique {u}, poses {p.shape[0]}, image={has}")
