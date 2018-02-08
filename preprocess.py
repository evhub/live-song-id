import librosa
import numpy as np

def get_allpaths(artist, listdir, file_type='ref'):
    # Get names of all wav files
    file_paths = []
    f = open(listdir+artist+'_%stoname.txt'%file_type, 'r')
    for line in f:
        filename = (line.split('_'))[0]
        file_paths.append(artist+'_'+file_type+line.split('_')[0])
    f.close()
    return file_paths

def preprocess(Q, ds):
    """ Preprocess CQT matrix of a song. """
    absQ = np.absolute(Q)
    smoothQ = np.zeros((absQ.shape[0], absQ.shape[1]//ds))
    for row in range(absQ.shape[0]):
        smoothQ[row] = np.convolve(absQ[row], [1/ds]*ds, 'valid')[0:absQ.shape[1]-ds+1:ds]
    logQ = np.log(1+1000000*smoothQ)
    return logQ
