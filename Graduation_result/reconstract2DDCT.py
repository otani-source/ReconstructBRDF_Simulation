import os.path as op
from time import time
from tkinter import Tk, filedialog

import cvxpy as cp
import numpy as np
from matplotlib import pyplot as plt
from numba import jit
from scipy.fftpack import dct
import cv2

N = 90
M = 1600
K = 5
B = N // K
gam = 0.01
lam1 = 0.001
lam2 = 0.001

def main():
    start = time()
    data = plt.imread('C:/Users/otani/Desktop/data.png')[:8, :8, :3]
    # ブロックDCT行列
    BDM = makeBDM()
    # dctData = np.einsum('ij, jkc, kl->ilc', BDM, data, BDM.T)
    # makefigure(111, dctData)
    # plt.show()

    # アンダーサンプリングされたデータベクトル
    USDataVec = US(data)
    USM = makeUSM(M)

    RData = np.zeros_like(data)
    for c in range(3):
        print('チャンネル', c, 'を計算中')
        y = USDataVec[..., c].reshape(N**2, 1)
        W = cp.Variable((N, N))
        objective = cp.Minimize(cp.pnorm(y - USM*cp.reshape(W, (N**2, 1))) + 0.1*cp.pnorm(BDM*W*BDM.T, 1))
        constraints = []
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        RData[..., c] = W.value
        

    makefigure(111, RData)
    t = time() - start
    print(t, '[sec]')
    plt.show()

def US(data):
    USM = makeUSM(M)
    return USM.dot(data.reshape(N**2, 3))

def TV(data, size=3):
    val = np.zeros(3)
    for c in range(3):
        dxW = cv2.Sobel(data[..., c], cv2.CV_64F, 0, 1) / 2**size
        dyW = cv2.Sobel(data[..., c], cv2.CV_64F, 1, 0) / 2**size
        vec = np.sqrt(dxW**2 + dyW**2)
        val[c] = np.sum(vec)
    return val
    

def reconstruct(data,bdm):
    makefigure(131, data)
    dct = np.einsum('ij, jkc, kl->ilc', bdm, data, bdm.T)
    usm = makeUSM(M)
    usdata2D = usm.dot(data)
    makefigure(132, usdata2D)

    # plt.imsave('C:/Users/otani/Desktop/Data/us.png', convertImg(usdata2D))

    redctBRDF = np.zeros_like(data)
    for col in range(N):
        redctBRDF[:, col] = l1min(bdm, usm, usdata2D[:, col])

    makefigure(133, redctBRDF)
    plt.show()
    reBRDF = bdm.T.dot(redctBRDF).reshape(N,N,3)

    print('RMSE:', RMSE(data, reBRDF))

    makefigure(231, data)
    makefigure(234, reBRDF)
    makefigure(235, redctBRDF.reshape(N,N,3))
    makefigure(236, usdata2D)

# Romeiroの手法で2次元に投影
def to2D(data):
    w = (data >= 0).astype(int)
    out = np.average(data, axis = 0, weights = w)
    return out

def RMSE(org, trans):
    return np.sqrt(np.mean((org - trans)**2))

# L1ノルム最小化

def l1min(bdm,usm,s):
    L = usm.dot(bdm.T)
    out = np.zeros([N,3])
    for c in range(3):
        print('チャンネル',c+1,'を計算中')
        Wp = cp.Variable(N)
        objective = cp.Minimize(cp.pnorm(s[:,c] - L*Wp) + 0.001*cp.norm1(Wp))
        constraints = []
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        out[:, c] = Wp.value.reshape(N)
    print('終了')
    return out

def Pi_C(v, P, y):
    return v + P.T * (P*P.T).I * (y - P*v)

# ブロック2次元DCT行列を作成
def makeBDM():
    # ブロックDCT行列
    dctmtx = dct(np.identity(B), axis=0, norm='ortho')
    bdm = np.identity(K).repeat(B,0).repeat(B,1) * np.tile(dctmtx,[K,K])
    # 2次元ブロックDCT行列
    return bdm #.repeat(N,0).repeat(N,1) * np.tile(bdm,[N,N])

# アンダーサンプリング行列を作成
def makeUSM(m):
    out = np.zeros([N**2],dtype='u1')
    index = np.random.choice(N**2, m, replace=False)
    out[index] = 1
    return np.diag(out)

def PSNR(org,trans):
	MSE = np.mean((trans - org)**2)
	return 100 if MSE == 0 else 10 * np.log10(org.max()**2 / MSE)

def makefigure(args, data):
    plt.subplot(args)
    plt.tick_params(labelbottom=False,labelleft=False,left=False,bottom=False)
    plt.imshow(data)

def convertImg(a):
    rimg = np.zeros_like(a)
    for th in range(90):
        rimg[:,th] = a[:,np.floor((th / 90)**0.5 * 90).astype(int)]
    return rimg

# ファイル読み込み
# @jit('f8[:,:,:,:]()')
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
