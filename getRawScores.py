import numpy as np

def getScores(seqeunce,dim):
    scores = seqeunce @ seqeunce.T #dot produt of the elemnts of the seq
    scores = scores/np.sqrt(dim) #scale the scores
    return scores