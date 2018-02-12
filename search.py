import numpy as np
import scipy.spatial
import time

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
        distance = np.count_nonzero(queryFeatures != refFeatures[:,startCol:endCol+1])
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

    output: an array containing sorted by the score.
    """
    # Initialization
    output = []
    
    for refIdx, ref in enumerate(refs):
        distances = []

        # Find the best score across multiple pitch-shifted versions
        for pitch, refPitch in enumerate(ref):
            distance, _ = compare(query, refPitch)
            distances.append((pitch, distance))
        
        # Obtain the minimum distance from multiple versions
        bestRefPitch, bestRefDistance = min(distances, key=lambda x: x[1])

        # Add score for each ref to the output array
        output.append((refIdx, bestRefDistance, bestRefPitch))

    # Sort an output array based on distance
    return sorted(output, key=lambda x: x[1]) 

def calculateMRR(queries, refs, groundTruth):
    """
    queries: array of query
    refs: an array of refs similar to search
    groundTruth: a column vector specifying the ground truth
    """
    MRR = 0
    for id_query, query in enumerate(queries):
        print("==> Searching query {:}".format(id_query))    
        t_start = time.time()
        searchResult = search(query, refs)
        rank = 0
        for id_result, result in enumerate(searchResult):
            # Note that groundTruth[id_query] starts from 1
            if result[0] + 1 == groundTruth[id_query]:
                rank = id_result + 1
                break
        MRR += 1.0 / rank
        t_end = time.time()
        print('\t-- Time elapsed: %d seconds'%(t_end - t_start))
    return MRR / len(queries)
