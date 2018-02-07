import numpy as np

def pitch_shift_CQT(M, shiftBins):
	'''
		pitchshift equivalent of Prof Tsai's code in matlab

		M: a 2-D CQT matrix
		shiftBins: An integer that indicates the pitch to shift to

		return: a pitchshifted matrix
	'''
	shifted = np.roll(M, shiftBins, axis=0)
	if shiftBins > 0:
		shifted[:shiftBins, :] = 0.
	else:
		shifted[shiftBins:, :] = 0.
	return shifted
