import numpy as np
from scipy import linalg as LA

def getTDE(Q, m=20, tao=1, hop=5, nbins=121):
    A = np.zeros((1, nbins*m//tao))
    QT = np.transpose(Q)
    for i in range(0,Q.shape[1]-(m*tao),hop):
        temp = np.reshape(QT[i:i+m:tao,:], (1, -1))
        A = np.append(A, np.reshape(QT[i:i+m:tao,:], (1, -1)), axis = 0)
    return A[1:]
