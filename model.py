import numpy as np

from keras.models import Sequential
from keras.layers import Conv1D

def build_model(pca_matrix, query_shape, batch_size, delta=16):
    """Generate the convolution model.
        pca_matrix: 3D array
        query_shape: tuple
        batch_size: int
        delta: int
    """
    pca_kernels = [k for k in pca_matrix]
    pca_kernel_iter = iter(pca_kernels)
    def pca_kernels_init(shape, dtype=None):
        next_pca_kernel = next(pca_kernel_iter)
        assert next_pca_kernel.shape == shape
        return next_pca_kernel

    ker1D = np.zeros(delta+1)
    ker1D[0] = 1
    ker1D[-1] = -1
    def ker1D_init(shape, dtype=None):
        assert ker1D.shape == shape
        return ker1D

    return Sequential([
        Conv1D(
            input_shape=(batch_size,) + query_shape,
            filters=len(pca_kernels),
            kernel_size=pca_kernels[0].shape,
            kernel_initializer=pca_kernels_init,
            use_bias=False,
        ),
        Conv1D(
            filters=1,
            kernel_size=ker1D.shape,
            kernel_initializer=ker1D_init,
            use_bias=False,
        ),
    ])

def run_model(model, queries_matrix, batch_size, threshold=0):
    """Run the convolution model.
        model: result of build_model
        queries_matrix: 3D array of 2D queries
        batch_size: int
        threshold: float
    """
     conv_result = model.predict(queries_matrix, batch_size=batch_size)
     return np.where(conv_result > threshold, 1, 0)
