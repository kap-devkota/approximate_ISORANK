# coding: utf-8
from hubalign_net import Network
import numpy as np
import pandas as pd
dfh = pd.read_csv("../../../data/intact/human.s.tsv", sep = "\t", header = None)
nodes = set(dfh[0]).union(dfh[1])
nmap = {k: i for i, k in enumerate(nodes)}
dfh[0] = dfh[0].apply(lambda x:nmap[x])
dfh[1] = dfh[1].apply(lambda x:nmap[x])
A = np.zeros((len(nodes), len(nodes))
)
for p, q in dfh.values:
    A[p, q] = 1
    A[q, p] = 1
    
A.shape
netA = Network(A)
netA.create_skeleton(10)
