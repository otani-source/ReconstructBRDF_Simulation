import os.path as op
from time import time

import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from scipy.fftpack import dct

K = 5
BRDF = 'Desktop/sotsuron/program/2dbrdf/gold-metallic-paint.brdf2'
saveDir = 'Desktop/sotsuron/fig/'


# @jit()
def main():
    yinVec = read2dBRDF()
    N = np.sqrt(yinVec.shape[0]).astype('u1')
    B = N // K
    dctmtx = dct(np.identity(B), axis=1, norm='ortho')

    img = np.ones([N, N, 4])
    img[..., :3] = yinVec.reshape(N, N, 3)

    spl = np.zeros([N + 20, N + 20, 4])

    s = 0
    for i in range(0, N, B):
        t = 0
        for j in range(0, N, B):
            spl[i + s:i + s + B, j + t:j + t + B] = img[i:i + B, j:j + B]
            t += 5
        s += 5

    block = img[36:54, :18, :3]
    x = np.einsum('ij,jkc,kl -> ilc', dctmtx.T, block, dctmtx)
    # y = np.einsum('ij,jkc,kl -> ilc', dctmtx, x, dctmtx.T)

    makefigure(131, spl)
    makefigure(132, block)
    makefigure(133, x)
    plt.imsave(saveDir + 'split.png', spl)
    plt.imsave(saveDir + 'blockbrdf.png', block)
    plt.imsave(saveDir + 'dctbrdf.png', x)
    # saveMERLBRDF(filename, x)
    plt.show()


# ファイル読み込み
def read2dBRDF():
    try:
        f = open(BRDF, 'rb')
        # dims = np.fromfile(f, np.int32, 3)
        vals = np.fromfile(f)  # , np.float64, -1)
        f.close()
        return vals.reshape(-1, 3)
    except IOError:
        print('Cannot read file:', op.basename(BRDF))
        exit(-1)


@jit
def makefigure(args, data):
    plt.subplot(args)
    plt.tick_params(labelbottom=False, labelleft=False,
                    left=False, bottom=False)
    plt.imshow(data)


# ファイル書き出し
def saveMERLBRDF(filename, vals, shape=(180, 90, 90), toneMap=True):

    # Saves a BRDF to a MERL-type .binary file

    # root = Tk()
    # root.withdraw()
    # fTyp = [("すべてのファイル", "*")]
    # iDir = op.abspath(op.dirname(__file__))

    # filename = filedialog.asksaveasfilename(filetypes=fTyp, initialdir=iDir)'''

    print("Saving MERL-BRDF: ", filename)
    plt.imsave(filename + '.png', vals)
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
        f = open(filename + '.binary', "wb")
        np.array(shape).astype(np.int32).tofile(f)
        vec.astype(np.float64).tofile(f)
        f.close()
    except IOError:
        print("Cannot write to file:", op.basename(filename))
        return


if __name__ == '__main__':
    main()
