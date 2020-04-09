import os.path as op
from time import time
from tkinter import Tk, filedialog

import cvxpy as cv
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from scipy.fftpack import dct

N = 90
S = 128
M = 1600
K = 5
B = N // K
Z = np.matrix(np.zeros(S))
gam = 0.01

@jit('void()')
def main():
    data = readfile()
    start = time()
    data = to2D(data)
    dwm = makeDWM(S)

    reconstruct(data,dwm)

    t = time() - start
    print(t, '[sec]')
    plt.show()

@jit('void(f8[:,:,:],f8[:,:])')
def reconstruct(data,obm):
    dmat = np.zeros([S,S,3])
    dmat[:N,:N] = data
    dataVec = dmat.reshape(S**2,3)
    pvec = culcP()
    print('n =',M,'における最適化問題を解いています...')
    usm = makeUSM(M, pvec)
    usdata = usm.dot(dataVec)

    redctBRDF = DR(obm,usm,usdata)
    reBRDF = obm.T.dot(redctBRDF).reshape(S,S,3)

    # plt.plot(Ind,Psnr) 
        # print('データ数:', n)
        # print('圧縮率:', n/N**2)
    print('RMSE:', RMSE(data, reBRDF))

    makefigure(221, data)
    makefigure(222, reBRDF)
    makefigure(223, redctBRDF.reshape(S,S,3))
    makefigure(224, usdata.reshape(S,S,3))

# Romeiroの手法で2次元に投影
@jit('f8[:,:,:](f8[:,:,:,:])')
def to2D(data):
    w = (data >= 0).astype(int)
    out = np.average(data, axis = 0, weights = w)
    return out

@jit('f8[:]()')
def culcP():
    p = np.zeros([S,S])
    x = y = np.linspace(0,np.pi/2,N)
    p[:N,:N] = np.cos(x) * np.sin(y.reshape(-1,1))
    p /= np.sum(p)
    np.random.seed(N**2)
    return p.reshape(S**2)

# L1最適化(ダグラス・ラシュフォードアルゴリズム)
# @jit('f8[:,:](f8[:,:],i1[:,:],f8[:,:])')
def DR(obm,usm,s):
    L = usm.dot(obm.T)
    print(3)
    out = np.zeros([S**2,3])
    P = np.matrix(L)
    for c in range(3):
        print('チャンネル',c+1,'を計算中')
        y = np.matrix(s[:,c])
        z = Z
        for i in range(10):
            x = S_g(z)
            z = z + Pi_C(2*x - z, P, y) - x
        out[:,c] = z.A
        # Wp = cv.Variable(S**2)
        # objective = cv.Minimize(cv.norm1(Wp))
        # constraints = [cv.pnorm(s[:,c] - L*Wp) < 0.02]
        # prob = cv.Problem(objective, constraints)
        # result = prob.solve()
        # out[:,c] = Wp.value.reshape(S**2)
    print('終了')
    return out

@jit
def S_g(v):
    val = np.zeros_like(v)
    for i, vi in enumerate(v):
        if vi >= gam:
            val[i] = vi - gam
        elif vi > -gam:
            val[i] = 0
        else:
            val[i] = vi + gam
    return val

@jit
def Pi_C(v, P, y):
    return v + P.T * (P*P.T).I * (y - P*v)

# ウェーブレット変換行列を作成
@jit('f8[:,:](i4)')
def makeDWM(size):
    mat = np.zeros([size,size])
    mat[0] = np.sqrt(size) / size
    axis = np.array([1, -1])
    for i in range(int(np.log2(size))):
        for j in range(2**i):
            ind = 2**i + j
            l = j * size // 2**i
            r = (j+1)*size // 2**i
            mat[ind,l:r] = np.sqrt(2**i/size) * axis.repeat(size // 2**(i + 1))
    return mat.repeat(S,0).repeat(S,1) * np.tile(mat,[S,S])

# ブロック2次元DCT行列を作成
@jit('f8[:,:]()')
def makeBDM():
    # ブロックDCT行列
    dctmtx = dct(np.identity(B), axis=0, norm='ortho')
    bdm = np.identity(K).repeat(B,0).repeat(B,1) * np.tile(dctmtx,[K,K])
    # 2次元ブロックDCT行列
    return bdm.repeat(N,0).repeat(N,1) * np.tile(bdm,[N,N])

# アンダーサンプリング行列を作成
@jit('i1[:,:](i1, f8[:])')
def makeUSM(m, p):
    out = np.zeros([S**2])
    index = np.random.choice(S**2, m, p = p, replace=False)
    out[index] = 1
    return np.diag(out)

@jit('f4(f8[:,:,:],f8[:,:,:])')
def RMSE(org, trans):
    return np.sqrt(np.mean((org - trans)**2))

@jit('f4(f8[:,:,:],f8[:,:,:])')
def PSNR(org,trans):
	MSE = np.mean((trans - org)**2)
	return 100 if MSE == 0 else 10 * np.log10(org.max()**2 / MSE)

@jit('void(i4, f8[:,:,:])')
def makefigure(args, data):
    plt.subplot(args)
    plt.tick_params(labelbottom=False,labelleft=False,left=False,bottom=False)
    plt.imshow(np.clip(data,0,1))

# ファイル読み込み
def readfile():
    root = Tk()
    root.withdraw()
    fTyp = [('BRDFファイル','*.binary'), ('すべてのファイル','*')]
    try:
        iDir = op.abspath(op.dirname(__file__))
        filename = filedialog.askopenfilename(filetypes = fTyp, initialdir = iDir)
        f = open(filename, 'rb')
        dims = np.fromfile(f,np.int32,3)
        vals = np.fromfile(f,np.float64,-1).reshape(dims[2], dims[1], dims[0], 3, order='F')
        f.close()
    except IOError:
        print('Cannot read file:', op.basename(filename))
        exit()

    vals *= (1.00/1500,1.15/1500,1.66/1500) #Colorscaling
    vals[vals<0] = -1

    return vals

def saveMERLBRDF(filename,BRDFVals,shape=(180,90,90),toneMap=True):
    "Saves a BRDF to a MERL-type .binary file"
    print("Saving MERL-BRDF: ",filename)
    BRDFVals = np.array(BRDFVals)   #Make a copy
    if(BRDFVals.shape != (np.prod(shape),3) and BRDFVals.shape != shape+(3,)):
        print("Shape of BRDFVals incorrect")
        return
        
    #Do MERL tonemapping if needed
    if(toneMap):
        BRDFVals /= (1.00/1500,1.15/1500,1.66/1500) #Colorscaling
    
    #Are the values not mapped in a cube?
    if(BRDFVals.shape[1] == 3):
        BRDFVals = BRDFVals.reshape(shape+(3,))
        
    #Vectorize:
    vec = BRDFVals.reshape(-1,order='F')
    shape = [shape[2],shape[1],shape[0]]
    
    try: 
        f = open(filename, "wb")
        np.array(shape).astype(np.int32).tofile(f)
        vec.astype(np.float64).tofile(f)
        f.close()
    except IOError:
        print("Cannot write to file:", op.basename(filename))
        return

if __name__=='__main__':
    main()
