import os.path as op
from time import time
from tkinter import Tk, filedialog

import cvxpy as cv
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from numba import jit

MTXPATH = 'Reconstract_BRDF_DWT/wmtx_f4.npz'

dirPath = 'Google ドライブ/卒論/program/'
filename = 'gold-metallic-paint_m4000'


# @jit('void()')
def main():
    yinVec = read2dBRDF(dirPath + '2dbrdf/' + filename +
                        '.brdf2').reshape(-1, 3)
    start = time()
    N = np.sqrt(yinVec.shape[0]).astype('u1')
    ind = yinVec >= 0
    m = np.sum(ind) // 3
    # S = 128

    y = yinVec[ind].reshape(m, 3)
    RI = resizeMat(128, N)
    A = sp.csr_matrix(createA(m, N, ind[:, 0]))
    Wdwt = sp.load_npz(dirPath + MTXPATH)

    WRx = reconstruct(y, A * RI * Wdwt.T, 128)
    x = (RI * Wdwt.T * WRx).reshape(N, N, 3)

    makefigure(121, x)

    t = time() - start
    print(t, '[sec]')
    plt.show()
    plt.imsave(dirPath + filename + '_rdwt.png', x)
    saveMERLBRDF(dirPath + filename + '_rdwt.binary', x)


def read2dBRDF(filePath):
    try:
        f = open(filePath, 'rb')
        # dims = np.fromfile(f, np.int32, 3)
        vals = np.fromfile(f)  # , np.float64, -1)
        f.close()
        return vals  # .reshape(dims[2], dims[1], dims[0], 3, order='F')
    except IOError:
        print('Cannot read file:', op.basename(filePath))
        exit(-1)


def resizeMat(t, f):
    # R = sp.lil_matrix((t ** 2, f ** 2), dtype='u1')
    out = sp.lil_matrix((t ** 2, f ** 2), dtype='u1')
    block = np.zeros([t, f])
    # block[:, -1] = 1
    I = np.identity(f)
    block[:f] = I
    deff = t - f
    # R[:t * deff, :f] = np.tile(block, [deff, 1])
    for i in range(f):
        # R[(deff + i) * t:(deff + i + 1) * t,
        #   i * f: (i + 1) * f] = block
        out[(deff + i) * t: (deff + i + 1) * t,
            i * f: (i + 1) * f] = block
    # R = R.tocsr()
    # RT = R.T.tocsr()
    # out = sp.csr_matrix(np.matrix((RT * R).toarray()).I) * RT
    return out.T.tocsr()


@jit('u1[:, :](u2, u1, b1[:])')
def createA(m, N, ind):
    mat = np.zeros([m, N ** 2], dtype='u1')
    i = 0
    for j in range(N**2):
        if ind[j]:
            mat[i, j] = 1
            i += 1
    return mat


@jit('f8[:, :](f8[:, :], f8[:, :], u1)')
def reconstruct(y, L, N):
    out = np.zeros([N ** 2, 3])
    for c in range(3):
        print('チャンネル', c+1, 'を計算中')
        Wp = cv.Variable(N**2)
        objective = cv.Minimize(cv.norm1(Wp))
        constraints = [y[:, c] == L*Wp]
        prob = cv.Problem(objective, constraints)
        result = prob.solve()

        out[:, c] = Wp.value
    print('終了')
    return out


# ウェーブレット変換行列を作成
@jit('f4[:,:](u1)')
def makeDWM(size):
    level = int(np.log2(size))
    out = sp.lil_matrix((size**2, size**2), dtype='f4')
    out[0] = 1
    for i in range(level):
        n = 2**i
        Nn = size//n
        val = n
        for j in range(size*n):
            ii = j % n + size*(j//size)
            jn = j * Nn
            out[n+ii, jn: jn+Nn//2] = val
            out[n+ii, jn+Nn//2: jn+Nn] = -val
            out[size*n+ii, jn: jn+Nn] = val*(-1)**(2*j // size)
            out[n+size*n+ii, jn: jn+Nn//2] = val*(-1)**(2*j // size)
            out[n+size*n+ii, jn+Nn//2: jn+Nn] = -val*(-1)**(2*j // size)
    sp.save_npz(
        'C:/Users/otani/Google\ ドライブ/卒論/program/Reconstract_BRDF_DWT/wmtx_f4.npz', out.tocsr()/size)
    return out.tocsr()/size


@jit('f4(f8[:,:,:],f8[:,:,:])')
def RMSE(org, trans):
    return np.sqrt(np.mean((org - trans)**2))


@jit('void(i4, f8[:,:,:])')
def makefigure(args, data):
    plt.subplot(args)
    plt.tick_params(labelbottom=False, labelleft=False,
                    left=False, bottom=False)
    plt.imshow(np.clip(data, 0, 1))


def saveMERLBRDF(filename, vals, shape=(180, 90, 90), toneMap=True):
    # Saves a BRDF to a MERL-type .binary file
    print("Saving MERL-BRDF: ", filename)
    BRDFVals = np.zeros_like(vals)
    for i in range(90):
        ind = i ** 2 // 90
        BRDFVals[:, i] = vals[:, ind]
    # makefigure(133, BRDFVals)
    BRDFVals = BRDFVals.reshape(1, 90, 90, 3).repeat(180, 0)  # Make a copy
    if(BRDFVals.shape != (np.prod(shape), 3) and BRDFVals.shape != shape+(3,)):
        print("Shape of BRDFVals incorrect")
        return

    # Do MERL tonemapping if needed
    if(toneMap):
        BRDFVals /= (1.00/1500, 1.15/1500, 1.66/1500)  # Colorscaling

    # Are the values not mapped in a cube?
    if(BRDFVals.shape[1] == 3):
        BRDFVals = BRDFVals.reshape(shape+(3,))

    # Vectorize:
    vec = BRDFVals.reshape(-1, order='F')
    shape = [shape[2], shape[1], shape[0]]

    try:
        f = open(filename, "wb")
        np.array(shape).astype(np.int32).tofile(f)
        vec.astype(np.float64).tofile(f)
        f.close()
    except IOError:
        print("Cannot write to file:", op.basename(filename))
        return


if __name__ == '__main__':
    main()
