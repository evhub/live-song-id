"""
Code for testing model.py
"""

import numpy as np
import scipy.io
from scipy.stats import descrive
from utils import *
from model import *

# Load eigenvectors
mat = scipy.io.loadmat('/data1/mint/public/taylorswift_out/model.mat', squeeze_me=True)
evecs = mat['eigvecs'][::-1][:64]
assert evecs.shape == (64, 2420)
print("evecs.shape == (64, 2420")

# Load preprocessed CQT
mat = scipy.io.loadmat('/data1/mint/public/taylorswift_out/taylorswift_preprocessed_ref1.mat', squeeze_me=True)
Q = np.abs(mat['B']).T

# Generate PCA matrix
m = 20
pca_matrix = np.array([vec.reshape((m, -1)) for vec in evecs])
delta = 16
max_pitch_shift = 4

# Generate pitch-shifting
pitch_shift_Qs = np.empty((2 * max_pitch_shift + 1, ) + Q.shape)
pitch_shift_Qs[0, :, :] = Q
for i in range(1, max_pitch_shift + 1):
    pitch_shift_Qs[i, :, :] = pitch_shift_CQT(Q.T, i).T
for i in range(1, max_pitch_shift + 1):
    pitch_shift_Qs[i + max_pitch_shift, :, :] = pitch_shift_CQT(Q.T, -i).T

# Our output
conv_1d_net = build_model(pca_matrix, Q.shape, delta=delta)
fpseqs = run_model(conv_1d_net, pitch_shift_Qs)

# Expected fpseqs
mat = scipy.io.loadmat('/data1/mint/public/taylorswift_out/db.mat', squeeze_me=True)
expected_fpseqs = np.array([mat["beforeDeltas"][0][:,:,k] for k in range(9)])

assert fpseqs.shape == expected_fpseqs.shape, (fpseqs.shape, expected_fpseqs.shape)
print("""Got fpseqs: {}
Expected fpseqs: {}""".format(
    describe(fpseqs),
    describe(expected_fpseqs)
))
