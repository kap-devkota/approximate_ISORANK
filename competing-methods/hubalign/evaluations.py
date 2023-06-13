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
    parser.add_argument("--output", default = "output_tail.tsv", help = "Output file")
    return parser




def main(args):
    global DATALOC, GOLOC, FINALLOC

    spA = args.org1
    spB = args.org2

    annotA = f"{GOLOC}/{spA}.output.mapping.gaf"
    annotB = f"{GOLOC}/{spA}.output.mapping.gaf"
    obo = f"{GOLOC}/go-basic.obo"

    dfA = pd.read_csv(f"{spA}.tab", sep = "\t", header = None)
    dfB = pd.read_csv(f"{spB}.tab", sep = "\t", header = None)

    comb = f"{spA}.tab-{spB}.tab.alignment"
    if not os.path.exists(comb):
        comb = f"{spB}.tab-{spA}.tab.alignment"
    
    pairs = pd.read_csv(comb, sep = " ", header = None).tail(args.pairs)
    pairs = pairs.values
    print(pairs)

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
               *[semantic_sim(pairs, 
                              dfA, 
                              dfB, 
                              obofile = obo,
                              annot1file = annotA,
                              annot2file = annotB,
                              type = go)
                            for go in ["molecular_function", 
                                       "biological_process", 
                                       "cellular_component"
                                       ]
                            ])]
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(args.output, mode = "a", sep="\t", index= None, header = not os.path.exists(args.output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    main(add_args(parser).parse_args())