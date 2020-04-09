import os.path as op
from time import time
from tkinter import Tk, filedialog

import cvxpy as cv
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from scipy.fftpack import dct


dirPath = 'Google ドライブ/卒論/program/'
filename = 'orange-aruco_439_89s_m2000'

K = 5


def main():
    yinVec = read2dBRDF(dirPath + '2dbrdf/' +
                        filename + '.brdf2').reshape(-1, 3)
    start = time()
    N = np.sqrt(yinVec.shape[0]).astype('u1')
    ind = yinVec >= 0
    m = np.sum(ind) // 3

    y = yinVec[ind].reshape(m, 3)
    A = createA(m, N, ind[:, 0])
    Wdct = createBDM(N)

    Wx = reconstruct(y, A.dot(Wdct.T), N)
    x = Wdct.T.dot(Wx).reshape(N, N, 3)
    y2d = yinVec.reshape(N, N, 3)

    makefigure(131, x)
    makefigure(132, y2d)
    plt.imsave(dirPath + filename + '_img.png', y2d)
    plt.imsave(dirPath + filename + 'K2_rdct.png', x)
    # saveMERLBRDF(dirPath + filename + '_rdct.binary', x)

    t = time() - start
    print(rmse(y2d, y2d))
    print(439.89 + t, '[sec]')
    plt.show()


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


@jit('u1[:, :](u2, u1, b1[:])')
def createA(m, N, ind):
    mat = np.zeros([m, N ** 2], dtype='u1')
    i = 0
    for j in range(N**2):
        if ind[j]:
            mat[i, j] = 1
            i += 1
    return mat

# ブロック2次元DCT行列を作成
@jit('f8[:, :](u1)')
def createBDM(N):
    B = N // K
    # ブロックDCT行列
    dctmtx = dct(np.identity(B), axis=0, norm='ortho')
    bdm = np.identity(K).repeat(B, 0).repeat(B, 1) * np.tile(dctmtx, [K, K])
    # 2次元ブロックDCT行列
    return bdm.repeat(N, 0).repeat(N, 1) * np.tile(bdm, [N, N])


@jit('f8[:, :](f8[:, :], f8[:, :], u1)')
def reconstruct(y, L, N):
    out = np.zeros([N**2, 3])
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


@jit
def makefigure(args, data):
    plt.subplot(args)
    plt.tick_params(labelbottom=False, labelleft=False,
                    left=False, bottom=False)
    plt.imshow(data)


def rmse(tVal, eVal):
    return np.sqrt(np.mean((tVal - eVal)**2))


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
