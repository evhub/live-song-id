import numpy as np
from scipy import linalg as LA

def getTDE(Q, m=20, tao=1, hop=5, nbins=121):
    A = np.zeros((1, nbins*m//tao))
    QT = np.transpose(Q)
    for i in range(0,Q.shape[1]-(m*tao),hop):
        temp = np.reshape(QT[i:i+m:tao,:], (1, -1))
        A = np.append(A, np.reshape(QT[i:i+m:tao,:], (1, -1)), axis = 0)
    return A[1:]

# cqtdir - where cqt and list of cqt (.txt) are stored
cqtdir = "/data1/mint/test"
artist = "taylorswift"
f = open(cqtdir+artist+"_cqtList.txt", 'r')

# Compute covariance matrix
numFeatures = 64
nbins = 121
m = 20
AccumCov = np.zeros((nbins*m, nbins*m))
count = 0
for line in f:
    Q = np.loadtxt(line[:-1])
    A = getTDE(Q)
    if A.shape[1] > 1:
        AccumCov = AccumCov + np.cov(np.transpose(A))
    count = count + 1

f.close()

# Do PCA
evals, evecs = LA.eig(AccumCov/count)
ind = np.argsort(-evals) # Sort eigenvalues in decreasing order
evecs = (evecs[:,ind])[:,:numFeatures]
#kernels = np.transpose(np.reshape(evecs, (m, nbins, numFeatures)), (1, 0, 2))
# Save to file
np.savetxt(cqtdir+artist+'_model.dat', evecs)