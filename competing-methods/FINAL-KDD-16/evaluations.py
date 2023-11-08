#!/cluster/tufts/cowenlab/.envs/netalign/bin/python
import argparse
import pandas as pd
import numpy as np
from netalign.approx_isorank.io_utils import compute_adjacency, compute_pairs
from netalign.approx_isorank.isorank_compute import compute_isorank, compute_greedy_assignment, pair_acc
from netalign.approx_isorank.pair_evaluations import compute_edge_correctness, semantic_sim, symmetric_substructure, lccs
from numpy.linalg import norm
import json
from scipy.io import loadmat
import os

DATALOC="../../data/intact/"
GOLOC="../../data/go/"
FINALLOC="temp-data/"

def add_args(parser):
    parser.add_argument("--org1", required = True, help = "First organism")
    parser.add_argument("--org2", required = True, help = "Second organism")
    parser.add_argument("--pairs", 
                        type = int, 
                        default = 2000,
                        help = "How many pairs to get using greedy search?")
    parser.add_argument("--output", default = "output.tsv", help = "Output file")
    return parser




def main(args):
    global DATALOC, GOLOC, FINALLOC

    spA = args.org1
    spB = args.org2

    annotA = f"{GOLOC}/{spA}.output.mapping.gaf"
    annotB = f"{GOLOC}/{spB}.output.mapping.gaf"
    obo = f"{GOLOC}/go-basic.obo"

    df1 = pd.read_csv(f"{DATALOC}/{spA}.s.tsv", sep = "\t", header = None)
    df2 = pd.read_csv(f"{DATALOC}/{spB}.s.tsv", sep = "\t", header = None)

    with open(f"{FINALLOC}/{spA}.json", "r") as jA, open(f"{FINALLOC}/{spB}.json", "r") as jB:
        nA = json.load(jA)
        nB = json.load(jB)
        rnA = {v: k for k, v in nA.items()}
        rnB = {v: k for k, v in nB.items()}

    dfA = df1.copy()
    dfB = df2.copy()

    dfA[0] = dfA[0].apply(lambda x: nA[x])
    dfA[1] = dfA[1].apply(lambda x: nA[x])

    dfB[0] = dfB[0].apply(lambda x: nB[x])
    dfB[1] = dfB[1].apply(lambda x: nB[x])

    comb = f"{spA}-{spB}"
    comb_ = f"{spA}_{spB}"
    if not os.path.exists(f"{FINALLOC}/{spA}-{spB}.mat"):
        comb = f"{spB}-{spA}"
        comb_ = f"{spB}_{spA}"
    
    E = loadmat(f"{FINALLOC}/{comb}")[f"{comb_}"].todense()
    if not os.path.exists(f"{FINALLOC}/{spA}-{spB}.mat"):
        print("Transposing the matrix...")
        E = E.T
    
    
    print("Matrix Loaded. Computing greedy alignment...")
    pairs = compute_greedy_assignment(E, args.pairs)

    # slight formatting issue here
    pairs = [(i, j[0, 0]) for i, j in pairs]
    pairs_ = [(rnA[p], rnB[q]) for p, q in pairs]


    columns = ["Species A",
               "Species B",
               "edge correctness",
               "lccs",
               "symmetric_substructure",
               "FC(mf)",
               "FC(bp)",
               "FC(cc)"]
    
    results = [(spA,
               spB,
               compute_edge_correctness(pairs, dfA, dfB),
               lccs(pairs, dfA, dfB),
               symmetric_substructure(pairs, dfA, dfB),
               *[semantic_sim(pairs_, 
                              df1, 
                              df2, 
                              obofile = obo,
                              annot1file = annotA,
                              annot2file = annotB,
                              type = go)
                            for go in ["molecular_function", 
                                       "biological_process", 
                                       "cellular_component"
                                       ]
                            ]
               )]
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(args.output, mode = "a", index= None, header = not os.path.exists(args.output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    main(add_args(parser).parse_args())