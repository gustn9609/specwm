#!/usr/bin/env python3
"""B1 고의 대조군: Base 영상에서 (a) shuffle: 청크(12프레임) 순서 섞기, (b) teleport: 6 s 지점부터 다른 씬(다른 seed/씬 Base)의 프레임으로 교체, (c) freeze: 6 s 이후 마지막 프레임 정지.
검사기가 이걸 못 잡으면 검정력 없음. 출력: b1/controls/<scene>_<kind>.mp4"""
import numpy as np, imageio, os, sys
S="/home/hk28965/specwm"; O=S+"/lingbot-world-v2/specwm_out"; D=O+"/b1/controls"; os.makedirs(D,exist_ok=True)
def rd(p): return np.stack([np.asarray(f) for f in imageio.get_reader(p)])
def wr(p,V): imageio.mimwrite(p,list(V),fps=16,quality=8)
rng=np.random.default_rng(0)
for sc,other in [("05","01"),("01","02"),("00","05"),("02","00")]:
    V=rd(O+"/floor/base/S%s_s1234/video.mp4"%sc); W=rd(O+"/floor/base/S%s_s1234/video.mp4"%other); n=len(V)
    ch=[V[i:i+12] for i in range(0,n-11,12)]; idx=rng.permutation(len(ch)); wr(D+"/S%s_shuffle.mp4"%sc, np.concatenate([ch[i] for i in idx])[:n])
    T=96; Wr=np.stack([np.array(imageio.core.asarray(f)) for f in W[:n-T]]) if W.shape[1:3]==V.shape[1:3] else None
    if Wr is None:
        import cv2; Wr=np.stack([cv2.resize(f,(V.shape[2],V.shape[1])) for f in W[:n-T]])
    wr(D+"/S%s_teleport.mp4"%sc, np.concatenate([V[:T],Wr]))
    wr(D+"/S%s_freeze.mp4"%sc, np.concatenate([V[:T],np.repeat(V[T-1:T],n-T,0)]))
    print(sc,"controls written")
