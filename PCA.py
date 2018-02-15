import numpy as np

def getTDE(Q, m=20, tao=1, hop=5, nbins=121):
    numFrames = Q.shape[1]
    numPitches = Q.shape[0]
    tdespan = tao * (m - 1) + 1
    endIndex = numFrames - tdespan + 1
    offsets = np.arange(0, endIndex, hop)
    A = np.zeros((len(offsets), numPitches * m))
    for i in range(m):
        frameIdx = offsets + i * tao
        A[:, i*numPitches:(i+1)*numPitches] = Q[:,frameIdx].T
    return A
