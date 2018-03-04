# Version check
import sys
assert sys.version_info >= (3,), "Python 3 is required"

# Imports
import os.path

import numpy as np
from scipy.io import wavfile

# Constants
DB_DIR = os.path.join("/data1/mint/Data/", "artists")
ARTISTS = (
    "taylorswift",
)
SONG_NAMES = {
    "ourref1": ("ourquery1", "ourquery2"),
    "ourref2": ("ourquery3", "ourquery4"),
}
BEAT_TIME = 1

# Utilities
def get_beats(artist, song_name):
    """Returns an array of beats and a list of their corresponding names."""
    base_name = os.path.join(DB_DIR, artist, "{}_{}".format(artist, song_name))
    sample_rate, audio = wavfile.read(base_name+".wav")
    annotations = np.genfromtxt(base_name+".csv", delimiter=",")

    beats = annotations.shape[0]
    beat_width = int(BEAT_TIME*sample_rate)

    beat_names = []
    beat_array = np.zeros((beats, beat_width), dtype=audio.dtype)
    for i, (time, beat_name) in enumerate(annotations):
        start = int((time - BEAT_TIME/2)*sample_rate)
        if start < 0:
            start = 0
        stop = start + beat_width
        beat_array[i] = audio[start:stop]
        beat_names.append(float(beat_name))

    return beat_array, beat_names

def align_ref(ref_beat_array, ref_beat_names, query_beat_array, query_beat_names):
    """Snips the beats from the given reference to match the given query."""
    aligned_ref = np.zeros(query_beat_array.shape, dtype=ref_beat_array.dtype)
    for i, (query_beat_name, query_beat) in enumerate(zip(query_beat_names, query_beat_array)):
        aligned_ref[i] = ref_beat_array[ref_beat_names.index(query_beat_name)]
    return aligned_ref

def get_ref_query_pairs(artist):
    """Returns an iterator of aligned (ref_beat_array, query_beat_array) pairs.
    Since this function is implemented as a generator, no work is done until it is iterated over."""
    for ref_name, query_names in SONG_NAMES.items():
        ref_beat_array, ref_beat_names = get_beats(artist, ref_name)

        for query_name in query_names:
            query_beat_array, query_beat_names = get_beats(artist, query_name)

            aligned_ref = align_ref(ref_beat_array, ref_beat_names, query_beat_array, query_beat_names)
            assert aligned_ref.shape == query_beat_array.shape, (aligned_ref.shape, query_beat_array.shape)
            yield aligned_ref, query_beat_array

def data_dict():
    """Returns a dictionary mapping artists to iterators over their ref_query_pairs.
    Since each dictionary value is an iterator, no work is done until that value is iterated over."""
    return {artist: get_ref_query_pairs(artist) for artist in ARTISTS}

# Main
if __name__ == "__main__":
    for artist, ref_query_pairs in data_dict().items():
        print("artist {}".format(artist))
        for ref, query in ref_query_pairs:
            beats, beat_width = query.shape
            print("\t{} beats of width {}".format(beats, beat_width))