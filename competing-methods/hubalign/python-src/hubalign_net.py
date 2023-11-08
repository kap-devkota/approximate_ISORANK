"""
Python implementation of `hubalign`

Kapil Devkota
"""


import numpy as np
import pickle as pkl

class Network:
    def __init__(self, A, no_edges):
        assert A.shape[0] == A.shape[1]
        self.skA = A.copy()
        self.no_edges = no_edges
        self.dim = A.shape[0]
        self.nodeweight  = np.zeros((self.dim))
        
        # nodes to ignore while creating skeleton 
        self.ignorenodes = np.zeros((self.dim)) 
        
        self.currdegree = np.sum(A, axis = 1)
        
        # neighbors
        self.neighbors  = [np.argwhere(A[i] > 0).ravel() for i in range(self.dim)]
        
    
    def create_skeleton(self, deg):
        self._remove_deg1nodes()
        
        for i in range(2, deg):
            self._remove_deg(i)
        
    def _remove_deg1nodes(self):
        while(True):
            for i in range(self.dim):
                if self.currdegree[i] != 1:
                    continue
                for j in self.neighbors[i]:
                    if self.ignorenodes[j] == 0:
                        self.nodeweight[j] += self.nodeweight[i] + self.skA[i][j]
                        self.currdegree[j] -= 1
                        self.currdegree[i] -= 1
                self.ignorenodes[i] = 1
                
            # run until no node with degree 1
            if not (self.currdegree == 1).any():
                break
        self.ignorenodes[(self.currdegree < 1)] = 1
        return
    
    def _remove_deg(self, deg):
        curr_d = deg
        while(True):
            # For all the nodes with `degree == d`
            for i in range(self.dim):
                if self.currdegree[i] != curr_d:
                    continue
                currweight = self.nodeweight[i]
                neighborstoupdate = []
                for j in self.neighbors[i]:
                    # get non-removed neighbors of i, calculate the weights to update
                    if self.ignorenodes[j] == 0:
                        currweight += self.skA[i, j]
                        neighborstoupdate += [j]
                # set the currdegree to 0
                self.currdegree[i] = 0
                n_ntoupdate = len(neighborstoupdate)
                # for these neighbors of `i`, decrease its degree by 1,
                # and apply weight update
                for j in range(n_ntoupdate):
                    p = neighborstoupdate[j]
                    self.currdegree[p] -= 1
                    for k in range(j+1, n_ntoupdate):
                        q = neighborstoupdate[k]
                        self.skA[p, q] += currweight / (curr_d * (curr_d - 1) / 2)
                        self.skA[q, p]  = self.skA[p, q]
                self.ignorenodes[i] = 1
                
            # Now check if the update process has caused some nodes to have degree <= curr_d        
            endloop = True
            for d in range(deg, 0, -1):
                if d != 1:
                    if (self.currdegree == d).any():
                        curr_d = d
                        endloop = False
                        break
                else:
                    if (self.currdegree == 1).any():
                        curr_d = deg
                        endloop = False
                        self._remove_deg1nodes()
            # Break if no nodes found with degree < deg
            if endloop:
                break
                
        self.ignorenodes[(self.currdegree < 1)] = 1
        return
    
    def save(self, filename):
        f = open(filename, "wb")
        pkl.dump(self, f)
        f.close()