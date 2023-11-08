"""
Python implementation of `hubalign`

Kapil Devkota
"""

import numpy as np
from netalign.approx_isorank.io_utils import compute_adjacency, compute_pairs
from netalign.approx_isorank.isorank_compute import compute_isorank, compute_greedy_assignment, pair_acc
from netalign.approx_isorank.pair_evaluations import compute_edge_correctness, semantic_sim, symmetric_substructure, lccs
from hubalign_net import Network
from hubalign_align import compute_hubalign_assignment
import pandas as pd
import os
import argparse

DATALOC="../../../data/intact/"
GOLOC="../../../data/go/"
OUTPUTLOC="../outputs/"

def add_args(parser):
    parser.add_argument("--org1", required = True, help = "First organism network file")
    parser.add_argument("--org2", required = True, help = "Second organism network file")
    parser.add_argument("--pairfile", default = None, help = "The one-to-one mappings")
    parser.add_argument("--lam", type = float, default = 0.1, help="Hubalign lambda parameter")
    parser.add_argument("--alpha", type = float, default = 0.7, help = "Hubalign alpha parameter")
    parser.add_argument("--deg", type = int, default = 10, help = "Hubalign alpha parameter")
    parser.add_argument("--pairs", 
                        type = int, 
                        default = 2000,
                        help = "How many pairs to get using greedy search?")
    parser.add_argument("--output", default = "../outputs/output.tsv", help = "Output file")
    return parser

def main(args):
    global DATALOC, GOLOC, OUTPUTLOC
    org1 = args.org1
    org2 = args.org2
    
    # GO files
    annotA = f"{GOLOC}/{org1}.output.mapping.gaf"
    annotB = f"{GOLOC}/{org2}.output.mapping.gaf"
    obo = f"{GOLOC}/go-basic.obo"
    
    # Dataframes 
    sim  = f"{DATALOC}/{org1}-{org2}.tsv"
    if not os.path.exists(sim):
        sim  = f"{DATALOC}/{org2}-{org1}.tsv"
    
    net1 = f"{DATALOC}/{org1}.s.tsv"
    net2 = f"{DATALOC}/{org2}.s.tsv"
    df1  = pd.read_csv(net1, sep = "\t", header = None)
    df2  = pd.read_csv(net2, sep = "\t", header = None)
    print("Networks loaded.")
    
    # Check if the pairs have already been computed
    pairs_computed = False
    
    if args.pairfile != None:
        if os.path.exists(args.pairfile):
            pairfile = args.pairfile
            pairs_computed = True
    else:
        outs = [f"{OUTPUTLOC}/{p}-{q}-{args.pairs}.tsv" for p, q in [(org1, org2), (org2, org1)]]
        for o in outs:
            if os.path.exists(o):
                pairfile = o
                pairs_computed = True
                break
            
            
    if pairs_computed:
        alignment = pd.read_csv(pairfile, sep = "\t")
    else:
        A1, n1 = compute_adjacency(df1)
        A2, n2 = compute_adjacency(df2)
        
        netA = Network(A1, len(df1))
        netB = Network(A2, len(df2))
        
        # compute the hubalign skeleton
        netA.create_skeleton(args.deg)
        netB.create_skeleton(args.deg)
        
        print("Skeleton network constructed")
        
        rn1 = {v:k for k,v in n1.items()}
        rn2 = {v:k for k,v in n2.items()}

        bfile = pd.read_csv(sim, sep = "\t")
        bfile = bfile.loc[(bfile[org1].apply(lambda x : x in n1) & bfile[org2].apply(lambda x : x in n2)), :]
        
        E = compute_pairs(bfile, n1, n2, org1, org2)
        
        pairs = compute_hubalign_assignment(netA, netB, E, args.lam, args.alpha, args.pairs)
        
        alignment = pd.DataFrame(pairs, columns = [org1, org2])
        alignment.loc[:, org1] = alignment.loc[:, org1].apply(lambda x : rn1[x])
        alignment.loc[:, org2] = alignment.loc[:, org2].apply(lambda x : rn2[x])
        alignment.to_csv(f"{OUTPUTLOC}/{org1}-{org2}-{args.pairs}.tsv", sep = "\t", index = None)
        print("Alignments computed and saved")
        
    ### Evaluations
    pairs = alignment.loc[:, [org1, org2]].values
    columns = ["Species A",
               "Species B",
               "edge correctness",
               "lccs",
               "symmetric_substructure",
               "FC(mf)",
               "FC(bp)",
               "FC(cc)"]
    print("Evaluating...")
    results = [(org1,
               org2,
               compute_edge_correctness(pairs, df1, df2),
               lccs(pairs, df1, df2),
               symmetric_substructure(pairs, df1, df2),
               *[semantic_sim(pairs, 
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
    if args.output != None:
        df.to_csv(args.output, mode = "a", index= None, header = not os.path.exists(args.output))
    else:
        print(df)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    main(add_args(parser).parse_args())