import os.path as op
from time import time
from tkinter import Tk, filedialog

import cvxpy as cv
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from scipy.fftpack import dct


dirPath = 'Desktop/Graduation_result/'      # このファイルの場所
filename = 'gold-metallic-paint_40p_17m18s'  # 読み込むデータのファイル名

K = 5  # DCTを行う範囲の分割数


def main():
    print(__file__)
    yinVec = read2dBRDF(dirPath + '2dbrdf/' +
                        filename + '.brdf2').reshape(-1, 3)
    org_data = read2dBRDF(
        dirPath + '2dbrdf/gold-metallic-paint.brdf2')
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

    makefigure(131, y2d, 'observation data')
    makefigure(132, x, 'estimation data')
    makefigure(133, org_data.reshape(N, N, 3), 'original data')

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

# 観測行列を作成
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


# 圧縮センシングによる推定
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


# 画像表示
@jit
def makefigure(args, data, title):
    plt.subplot(args)
    plt.tick_params(labelbottom=False, labelleft=False,
                    left=False, bottom=False)
    plt.title(title)
    plt.imshow(data)


# RMSE算出
def rmse(tVal, eVal):
    return np.sqrt(np.mean((tVal - eVal)**2))


if __name__ == '__main__':
    main()
