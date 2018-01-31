import numpy as np
import scipy.spatial

def compare(queryFeatures, refFeatures):
    """
    Return a hamming distance between a short query feature
    matrix and a long reference feature matrix.
    """
    # Initialization
    _, n = refFeatures.shape
    _, k = queryFeatures.shape
    bestDistance, bestIndex = np.inf, 0

    # Iterate on all context frames.
    for i in range(n - k):
        startCol = i
        endCol   = startCol + k - 1
        distance = scipy.spatial.distance.hamming(queryFeatures, refFeatures[:,startCol:endCol+1])
        if distance < bestDistance:
            bestDistance = distance
            bestIndex = i

    return bestDistance, bestIndex

def search(query, refs):
    """
    query: np.array() of shape (64,k)
    refs: an array containing an array of pitch-shifted features from each artist;
          each pitch-shifted feature is a form of np.array() of shape (64,n) where
          n resembles the length of the song
    """
    # Initialization
    bestDistance, bestRefIdx, bestPitch = np.inf, 0, 0

    for refIdx, ref in enumerate(refs):
        distances = []

        # Find the best score across multiple pitch-shifted versions
        for pitch, refPitch in enumerate(ref):
            distance, _ = compare(query, refPitch)
            distances.append((pitch, distance))

        # Obtain the minimum distance from multiple versions
        bestRefPitch, bestRefDistance = min(distances, key=lambda x: x[1])

        # Update the distance
        if bestRefDistance < bestDistance:
            bestDistance = bestRefDistance
            bestRefIdx = refIdx
            bestPitch = bestRefPitch

    return bestDistance, bestRefIdx, bestPitch

