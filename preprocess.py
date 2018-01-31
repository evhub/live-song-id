import librosa
import numpy as np

def get_allpaths(artist, listdir):
    # Get names of all wav files
    file_paths = []
    f = open(listdir+artist+"_reftoname.txt", 'r')
    for line in f:
        filename = (line.split('_'))[0]
        file_paths.append(artist+'_'+line.split('_')[0])
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

# audiopath - where audio files are stored
# listdir - where list of softlinks are stored
# cqtdir - where results will be stored
audiopath = "/home/nbanerjee/SoftLinks/"
listdir = "/home/nbanerjee/SoftLinks/Lists/"
cqtdir = "/data1/mint/test"
artist = "taylorswift"
file_paths = get_allpaths(artist, listdir)

# Open file that will contain the paths to all preprocessed CQT files
f = open(cqtdir+artist+"_cqtList.txt", 'w')

for curFile in file_paths:
    y, sr = librosa.load(audiopath + curFile + '.wav')
    Q = librosa.cqt(y, sr=sr, fmin = 130.81, n_bins = 121, bins_per_octave = 24)
    logQ = preprocess(Q, 3)
    
    # Save array
    np.savetxt(cqtdir+curFile+'.dat', logQ)

    # Write .dat file path to cqtList file
    f.write(cqtdir+curFile+'.dat\n')

f.close()