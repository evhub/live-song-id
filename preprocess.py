import librosa
import numpy as np

from keras.models import Sequential
from keras.layers import Reshape

from kapre.utils import Normalization2D
from kapre.time_frequency import Spectrogram


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

def normalization_model(audio_len):
    """Build a normalization model."""
    return Sequential([
        Reshape((1, 1, audio_len), input_shape=(1, audio_len)),
        Normalization2D(str_axis="data_sample"),
    ])

def stft_model(audio_len, normalize=True, **kwargs):
    """Build an STFT preprocessing model.
    
    Pass normalize=False to disable the normalization layer.
    Pass arguments to https://github.com/keunwoochoi/kapre/blob/master/kapre/time_frequency.py#L11."""
    return Sequential([
        Spectrogram(input_shape=(1, audio_len), **kwargs),
    ] + ([
        Normalization2D(str_axis='freq'),
    ] if normalize else []))

def run_preprocessing(model, audio_arr):
    """Run the given preprocessing model on the given audio array."""
    batch_size, audio_len = audio_arr.shape
    return np.squeeze(model.predict(
        audio_arr.reshape((batch_size, 1, audio_len)),
        batch_size=batch_size,
    ))
