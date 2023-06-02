# coding: utf-8
import numpy as np
from isorank_compute import  compute_isorank

import pandas as pd

db = pd.read_csv("intact/bakers.a.tsv", sep = "\t", header = None)

df = pd.read_csv("intact/fly.a.tsv", sep = "\t", header = None)

dpairs = pd.read_csv("intact/fly-bakers.tsv", sep = "\t", header = None)


def computeA(df):
    nodeset = set(df[0]).union(set(df[1]))
    nodemap = {k: i for i, k in enumerate(nodeset)}
    n = len(nodeset)
    A = np.zeros((n, n))
    for p, q in df.values:
        p_, q_ = [nodemap[p], nodemap[q]]
        A[p_, q_] = 1
        A[q_, p_] = 1
    return A, nodemap
Af, nmapF = computeA(df)
Ab, nmapB = computeA(db)
def computeP(df, nmapA, nmapB):
    m, n = len(nmapA), len(nmapB)
    A = np.zeros((m, n))
    for p, q, v in df.values:
        p_, q_ = [nmapA[int(p)], nmapB[int(q)]]
        A[p_, q_] = v
    return A
E = computeP(dpairs, nmapF, nmapB)
CI = compute_isorank(Af, Ab, E, 0.5, get_R0 = True, get_R1= True)
