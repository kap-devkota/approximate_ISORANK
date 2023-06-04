import sys
from .io_utils import compute_adjacency, compute_pairs
from .isorank_compute import compute_isorank, compute_greedy_assignment
import pandas as pd
from numpy.linalg import norm
import argparse
import os
import json

class ApproxIsorankArgs(NamedTuple):
    cmd: str
    net1: str
    net2: str
    rblast: str
    alpha: Optional[float]
    niter: Optional[int]
    npairs: Optional[int]
    output: Optional[int]
    func: Callable[[ApproxIsorankArgs], None]


def getargs(parser):
    parser.add_argument("--net1", required = True, help = "Network 1")
    parser.add_argument("--net2", required = True, help = "Network 2")
    parser.add_argument("--rblast", required = True, help  = "Reciprocal Blast results")
    parser.add_argument("--alpha", default = 0.6, type = float, help  = "Alpha parameter")
    parser.add_argument("--niter", default = 1, type = int, 
                       help = "If --niter is set to 0, it returns the R_0 approximation. If set to 1, runs the R_1 approximation. To get the original R, set --niter to a value >= 10.")
    
    parser.add_argument("--npairs", default = 1000, type = int)
    parser.add_argument("--output", default = "output.tsv")
                        
    return parser.parse_args()

def main(args):
    df1 = pd.read_csv(args.net1, sep = "\t", header = None)
    df2 = pd.read_csv(args.net2, sep = "\t", header = None)
    dpairs = pd.read_csv(args.rblast, sep = "\t")

    org1 = args.net1.split("/")[-1].split(".")[0]
    org2 = args.net2.split("/")[-1].split(".")[0]
    
    Af1, nA1 = compute_adjacency(df1)
    Af2, nA2 = compute_adjacency(df2)

    df1[0] = df1[0].apply(lambda x : nA1[x])
    df1[1] = df1[1].apply(lambda x : nA1[x])
    
    df2[0] = df2[0].apply(lambda x : nA2[x])
    df2[1] = df2[1].apply(lambda x : nA2[x])
    
    E = compute_pairs(dpairs, nA1, nA2, org1, org2)
    
    print("Computing Isorank similarity matrices...")
    R = compute_isorank(Af1, 
                        Af2, 
                        E, 
                        alpha=args.alpha, 
                        maxiter = args.niter)

    
    print("Computing pairs...")
    pairs = compute_greedy_assignment(R, args.npairs)
    
    results = pd.read_csv(pairs, columns = ["Sp-A", "Sp-B"])
    results.loc[:, "Sp-A"] = results.loc[:, "Sp-A"].apply(lambda x : nA1[x])
    results.loc[:, "Sp-B"] = results.loc[:, "Sp-B"].apply(lambda x : nA2[x])
    results.to_csv(args.output, sep = "\t", index = None)
    
    return
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    main(getargs(parser))
        
        