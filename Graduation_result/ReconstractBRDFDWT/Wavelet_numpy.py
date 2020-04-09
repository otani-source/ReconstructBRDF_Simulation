from os.path import abspath, dirname
from time import time
from tkinter import Tk, filedialog

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from numba import jit

from PIL import Image

N = 128
mtxpath = 'Documents/python/dwt/wmtx_f4.npz'

@jit('void()')
def main():
    st = time()
    imgor = Image.open('C:/Users/otani/Documents/python/dwt/90x90.png')
    img = np.asarray(imgor.resize([N,N]))[...,:3] / 255
    imgVec = sp.csr_matrix(img.reshape(N**2,3))
    # wNmtx(N)
    wmtx = sp.load_npz(mtxpath)

    wim = wmtx*imgVec
    iim = wmtx.T.tocsr()*wim
    wim = wim.toarray().reshape(N,N,3)
    iim = iim.toarray().reshape(N,N,3)
    iimPIL = Image.fromarray(np.uint8(iim*255)).resize([90,90])

    plt.subplot(131)
    plt.imshow(imgor)
    # plt.imsave('C:/Users/otani/Documents/Python/dwt/img.png',img)
    plt.subplot(132)
    plt.imshow(wim)
    # plt.imsave('C:/Users/otani/Documents/Python/dwt/wim.png',wim)
    plt.subplot(133)
    plt.imshow(iimPIL)
    # plt.imsave('C:/Users/otani/Documents/Python/dwt/iim.png',iim)
    print(time() - st)
    plt.show()

@jit('f4[:,:](u1)')
def wNmtx(N):
    level = int(np.log2(N))
    out = sp.lil_matrix((N**2,N**2),dtype='f4')
    out[0] = 1
    for i in range(level):
        n = 2**i
        Nn = N//n
        val = n
        for j in range(N*n):
            ii = j%n + N*(j//N)
            jn = j * Nn
            out[n+ii,jn:jn+Nn//2] = val
            out[n+ii,jn+Nn//2:jn+Nn] = -val
            out[N*n+ii,jn:jn+Nn] = val*(-1)**(2*j // N)
            out[n+N*n+ii,jn:jn+Nn//2] = val*(-1)**(2*j // N)
            out[n+N*n+ii,jn+Nn//2:jn+Nn] = -val*(-1)**(2*j // N)
    sp.save_npz('C:/Users/otani/Documents/python/dwt/wmtx_f4.npz',out.tocsr()/N)
    return out.tocsr()/N

if __name__ == "__main__":
    main()
