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

def get_querytoref(artist, listdir):
	'''
		artist: a string representing the artist
		listdir: a string representing the directory of the .list file

		returns an array of integer, where each entry corresponds to the
			reference index
	'''
    ref_idxs = []
    f = open(listdir+artist+'_querytoref.list', 'r')
    for line in f:
        ref_idx = (line.split(' '))[1]
        ref_idxs.append(int(ref_idx))
    f.close()
    return ref_idxs
