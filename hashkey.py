import numpy as np
from math import atan2, floor, pi

def hashkey(block, Qangle, W):
    # gradient
    gy, gx = np.gradient(block)

    # 將 2D 矩陣轉換為 1D 數組
    gx = gx.ravel()
    gy = gy.ravel()

    # SVD
    G = np.vstack((gx,gy)).T
    GTWG = G.T.dot(W).dot(G)
    w, v = np.linalg.eig(GTWG);

    # 確保 V 和 D 僅有實數
    nonzerow = np.count_nonzero(np.isreal(w))
    nonzerov = np.count_nonzero(np.isreal(v))
    if nonzerow != 0:
        w = np.real(w)
    if nonzerov != 0:
        v = np.real(v)

    # 根據 w 的降序對 w 和 v 進行排序
    idx = w.argsort()[::-1]
    w = w[idx]
    v = v[:,idx]

    # theta
    theta = atan2(v[1,0], v[0,0])
    if theta < 0:
        theta += pi

    # lamda
    lamda = w[0]

    # u
    sqrtlamda1 = np.sqrt(w[0])
    sqrtlamda2 = np.sqrt(w[1]) # bug
    if sqrtlamda1 + sqrtlamda2 == 0:
        u = 0
    else:
        u = (sqrtlamda1 - sqrtlamda2)/(sqrtlamda1 + sqrtlamda2)

    # 量化
    angle = floor(theta/pi*Qangle)
    if lamda < 0.0001:
        strength = 0
    elif lamda > 0.001:
        strength = 2
    else:
        strength = 1
    if u < 0.25:
        coherence = 0
    elif u > 0.5:
        coherence = 2
    else:
        coherence = 1

    # 將輸出綁定到所需範圍
    if angle > 23:
        angle = 23
    elif angle < 0:
        angle = 0
    return angle, strength, coherence
