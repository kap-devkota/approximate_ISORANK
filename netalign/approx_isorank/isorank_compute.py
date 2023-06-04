import numpy as np
import pandas as pd
import numpy as np
import scipy.io as sio
import argparse
import pandas as pd

def pair_acc(pair1, pair2):
    p1map = set(pair1)
    simcount = 0
    for i, (p, q) in enumerate(pair2):
        if (p, q) in p1map or (q, p) in p1map:
            simcount += 1
    return simcount / len(p1map)



def compute_greedy_assignment(R1, n_align):
    """
    Compute greedy assignment
    """
    aligned = []
    R = R1.copy()
    
    n_align = min(n_align, *R.shape)
    
    itr = 1
    while(len(aligned) < n_align):
        itr   += 1
        maxcols = np.argmax(R, axis = 1) # best y ids
        maxid = np.argmax(np.max(R, axis = 1)) # best x id
        maxcol = maxcols[maxid]
        aligned.append((maxid, maxcol))
        R[:, maxcol] = -1
        R[maxid, :]  = -1
    return aligned



def compute_isorank(A1, A2, E, alpha, maxiter = 20, get_R0 = False, get_R1 = False):
    """
    Compute the isorank using the eigendecomposition
    """

    d1 = np.sum(A1, axis = 1).reshape(-1, 1)
    d2 = np.sum(A2, axis = 1).reshape(-1, 1)
    
    P1 = A1 / d1.T
    P2 = A2 / d2.T
    
    E = E / np.sum(E)
    
    d = d1 @ d2.T 
    d = d / (np.sum(d1) * np.sum(d2))
    
    R = (1-alpha) * d + alpha * E
    
    if maxiter <= 0:
        return R
    
    if get_R0:
        R0 = R.copy()
    
    # Reshape R and E
    R = R.T
    E = E.T
    
    for i in range(maxiter):
        R = (1-alpha) * (P2 @ R @ P1.T) + alpha * E
        if get_R1 and i == 0:
            R1 = R.T.copy()
            
    payload = [R.T]
    if get_R1:
        payload = [R1] + payload
    if get_R0:
        payload = [R0] + payload
    return payload
    
    
    