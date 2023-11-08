"""
Python implementation of `hubalign`

Kapil Devkota
"""

from hubalign_net import Network
import numpy as np

def compute_alignment_scores(net1, net2, E = None, _lambda = 1, alpha = 0.6):
    """
    Here the A1, A2 variables implement the HubAlign Network
    E is the matrix that represents the blast scores
    _lambda and alpha are the hubalign parameters
    """
    nodeweight1 = net1.nodeweight.copy()
    nodeweight2 = net2.nodeweight.copy()
    
    skA1 = net1.skA.copy()
    skA2 = net2.skA.copy()
    
    nscore1 = (1-_lambda) * nodeweight1 + _lambda * np.sum(skA1, axis = 1)
    nscore2 = (1-_lambda) * nodeweight2 + _lambda * np.sum(skA2, axis = 1)
    
    # get the maximum score
    maxscore = np.max(np.concatenate([nscore1, nscore2], axis = 0))
    nscore1 = nscore1 / maxscore
    nscore2 = nscore2 / maxscore

    # alignment matrix
    align = np.zeros([nscore1.shape[0], nscore2.shape[0]])
    for i in range(nscore1.shape[0]):
        align[i] = np.where(nscore2 > nscore1[i], nscore2, nscore1[i])
    
    if E is None:
        E = np.zeros([d1.shape, d2.shape])

    align = alpha * align + (1-alpha) * E
    return align, maxscore


def compute_hubalign_assignment(net1, net2, E = None, _lambda = 1, alpha = 0.6, no_pairs = 2000):
    """
    Compute hubalign assignment
    """
    R1, maxscore = compute_alignment_scores(net1, net2, E, _lambda, alpha)
    
    aligned = []
    R = R1.copy()
    
    n_align = min(no_pairs, *R.shape)
    
    itr = 1
    
    # Used by hubalign
    updateRcoeff = (net1.no_edges / net2.no_edges) if (net1.no_edges > net2.no_edges) else (net2.no_edges / net1.no_edges) 
    updateRcoeff /= maxscore
    
    while(len(aligned) < n_align):
        itr   += 1
        maxcols = np.argmax(R, axis = 1) # best y ids
        maxid = np.argmax(np.max(R, axis = 1)) # best x id
        maxcol = maxcols[maxid]
        aligned.append((maxid, maxcol))
        R[:, maxcol] = -1
        R[maxid, :]  = -1
        # hubalign update the neighbors of maxcol and maxid
        rneighbors = net1.neighbors[maxid]
        cneighbors = net2.neighbors[maxcol]
        R[rneighbors, maxcol] += updateRcoeff
        R[maxid, cneighbors]  += updateRcoeff
    return aligned

