import numpy as np
from getRawScores import getScores 
from getEmbeddings import getAlibiModifiedEmeddings
sequence = np.array([
    [1, 0, 2],
    [0, 1, 2],
    [1, 1, 1],
    [2, 0, 1],
    [0, 2, 1]
], dtype=float)

seq_len , dim = sequence.shape

slopes = [0.1, 0.3, 0.6] 

distance = np.arange(seq_len)[None,:] - np.arange(seq_len)[:,None]
distance = -np.abs(distance)


raw_scores = getScores(sequence,dim)
final_embeddings = getAlibiModifiedEmeddings(slopes,raw_scores,sequence)
