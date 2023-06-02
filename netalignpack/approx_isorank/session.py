#!/cluster/tufts/cowenlab/.envs/denoise/bin/python
import sys
sys.path.append("/cluster/home/kdevko01/Approx_ISORANK/src")
from io_utils import compute_adjacency, compute_pairs
from isorank_compute import compute_isorank, compute_greedy_assignment, pair_acc
from pair_evaluations import compute_edge_correctness, semantic_sim, symmetric_substructure, lccs
import pandas as pd
from numpy.linalg import norm
import argparse
import os
import json


def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net1", required = True, help = "Network 1")
    parser.add_argument("--net2", required = True, help = "Network 2")
    parser.add_argument("--js1", default = None, help = "JSON for Network 1")
    parser.add_argument("--js2", default = None, help = "JSON for Network 2")
    parser.add_argument("--rblast", required = True, help  = "Reciprocal Blast results")
    parser.add_argument("--alpha", default = 0.6, type = float, help  = "Alpha parameter")
    parser.add_argument("--niter", default = 20, type = int)
    parser.add_argument("--npairs", default = 1000, type = int)
    parser.add_argument("--output", default = "output.tsv")
    parser.add_argument("--oboloc", help = "Location of OBO file", required = True)
    parser.add_argument("--annot1", help = "Location of the annotation for the first species", required = True)
    parser.add_argument("--annot2", help = "Location of the annotation for the second species", required = True)

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
    R0, R1, R2 = compute_isorank(Af1, 
                                 Af2, 
                                 E, 
                                 alpha=args.alpha, 
                                 maxiter = args.niter, 
                                 get_R0 = True, 
                                 get_R1 = True)

    
    norm0 = norm(R0 - R2)
    norm1 = norm(R1 - R2)

    
    print("Computing pairs...")
    pairs0 = compute_greedy_assignment(R0, args.npairs)
    pairs1 = compute_greedy_assignment(R1, args.npairs)
    pairs2 = compute_greedy_assignment(R2, args.npairs)
    

    ecorrectness0 = compute_edge_correctness(pairs0, df1, df2)
    ecorrectness1 = compute_edge_correctness(pairs1, df1, df2)
    ecorrectness2 = compute_edge_correctness(pairs2, df1, df2)
    
    
    sstructure0 = symmetric_substructure(pairs0, df1, df2)
    sstructure1 = symmetric_substructure(pairs1, df1, df2)
    sstructure2 = symmetric_substructure(pairs2, df1, df2)
    
    
    lc0 = lccs(pairs0, df1, df2)
    lc1 = lccs(pairs1, df1, df2)
    lc2 = lccs(pairs2, df1, df2)
    
    
    p0sim = pair_acc(pairs0, pairs2)
    p1sim = pair_acc(pairs1, pairs2)
    
    if args.js1 is not None and args.js2 is not None:
        nA1 = json.load(open(args.js1, "r"))
        nA2 = json.load(open(args.js2, "r"))
    
    rnA1 = {val:key for key, val in nA1.items()}
    rnA2 = {val:key for key, val in nA2.items()}
    

    df1.iloc[:, 0] = df1.iloc[:, 0].apply(lambda x: rnA1[x])
    df1.iloc[:, 1] = df1.iloc[:, 1].apply(lambda x: rnA1[x])
    

    df2.iloc[:, 0] = df2.iloc[:, 0].apply(lambda x: rnA2[x])
    df2.iloc[:, 1] = df2.iloc[:, 1].apply(lambda x: rnA2[x])

    FC = {}

    for name, pair in [ ("R0", pairs0), ("R1", pairs1), ("R2", pairs2)]:
        pair_ = [(rnA1[p], rnA2[q]) for (p, q) in pair]
        for gotype in ["molecular_function", "biological_process", "cellular_component"]:
            FC[f"FC-{gotype}({name})"] = semantic_sim(pair_, df1, df2, 
                                                    obofile = args.oboloc, annot1file = args.annot1, annot2file = args.annot2, type = gotype)

    
    
    columns = ["org1", 
               "org2", 
               "norm(R-R0)", 
               "norm(R-R1)", 
               "sim_with_isorank-R(R0)", 
               "sim_with_isorank-R(R1)",
               "Edge Correctness (R0)",
               "Edge Correctness (R1)",
               "Edge Correctness (R)",
               "Symmetric substructure (S^3-R0)",
               "Symmetric substructure (S^3-R1)",
               "Symmetric substructure (S^3-R)",
               "LCCS (R0)",
               "LCCS (R1)",
               "LCCS (R)"
              ]
    
    FC_Keys = list(FC.keys())

    columns += FC_Keys
    
    data = [org1, org2, 
            norm0, norm1, 
            p0sim, p1sim, 
            ecorrectness0, ecorrectness1, ecorrectness2,
            sstructure1, sstructure1, sstructure2, 
            lc0, lc1, lc2]
    data += [FC[key] for key in FC_Keys]
    
    results = None
    if os.path.exists(args.output):
        results = pd.read_csv(args.output, sep = "\t")
        results = pd.concat([results, 
                            pd.DataFrame([results], columns = columns)])
    else:
        results = pd.DataFrame([data], columns = columns)
    results.to_csv(args.output, sep = "\t", index = None)
    
    return
    
if __name__ == "__main__":
    main(getargs())
        
        