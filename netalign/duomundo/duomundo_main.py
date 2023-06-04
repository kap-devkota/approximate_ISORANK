#!/cluster/tufts/cowenlab/.envs/denoise/bin/python
from __future__ import annotations
import sys
import numpy
import glidetools.algorithm.dsd as dsd
import numpy as np
import sys
import json
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.linalg import pinv
import argparse
import os
from .io_utils import compute_adjacency, compute_pairs
from .data import Data, CuratedData, PredictData
from .model import AttentionModel, AttentionModel2, AttentionModel3
from .isorank import compute_isorank_and_save
from .predict_score import topk_accs, compute_metric, dsd_func, dsd_func_mundo, scoring_fcn
from .linalg import compute_k_svd, compute_k_svd_uv
import re
import pandas as pd
from sklearn.manifold import Isomap
import torch
import sklearn 
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable, NamedTuple, Optional


class DuoMundoArgs(NamedTuple):
    cmd: str
    ppiA: str
    nameA: str
    nameB: str
    dsd_A_dist: Optional[str]
    dsd_B_dist: Optional[str]
    thres_dsd_dist: Optional[str]
    json_A: Optional[str]
    json_B: Optional[str]
    svd_AU: Optional[str]
    svd_AV: Optional[str]
    svd_BU: Optional[str]
    svd_BV: Optional[str]
    svd_r: Optional[int]
    landmarks_a_b: Optional[str]
    compute_isorank: Optional[bool]
    isorank_alpha: Optional[float]
    no_landmarks: Optional[int]
    model: Optional[str]
    svd_dist_a_b: Optional[str]
    weight_decay: Optional[float]
    no_epoch: Optional[int]
    compute_go_eval: Optional[bool]
    kA: Optional[str]
    kB: Optional[str]
    metrics: Optional[str]
    wB: Optional[str]
    output_file: Optional[str]
    compute_dsd: Optional[str]
    go_A: Optional[str]
    go_B: Optional[str]
    go_h: Optional[str]
    munkonly: Optional[bool]
    seed: Optional[int]
    func: Callable[[DuoMundoArgs], None]
        
        
def getargs(parser):
    
    # PPI
    parser.add_argument("--ppiA", help = "PPI for species A: (target)")
    parser.add_argument("--ppiB", help = "PPI for species B: (source)")
    parser.add_argument("--nameA", help = "Name of species A")
    parser.add_argument("--nameB", help = "Name of species B")
    
    # DSD
    parser.add_argument("--dsd_A_dist", default = None, help = "Precomputed DSD distance for Species A. If not present, the matrix will be computed at this location")
    parser.add_argument("--dsd_B_dist", default = None, help = "Precomputed DSD distance for Species B. If not present, the matrix will be computed at this location")
    parser.add_argument("--thres_dsd_dist", type = float, default = 100, help = "If the DSD distance computed is > this threshold, replace that with the threshold")
    
    # JSON
    parser.add_argument("--json_A", default = None, help = "Protein annotation to Matrix index in json format for DSD and SVD matrix of A. If not present, the JSON file will be computed at this location")
    parser.add_argument("--json_B", default = None, help = "Protein annotation to Matrix index in json format for DSD and SVD matrix of B. If not present, the JSON file will be computed at this location")
    
    # SVD
    parser.add_argument("--svd_AU", default = None, help = "Left SVD representation of the species A. If not present, the SVD representation will be produced at this location")
    parser.add_argument("--svd_BU", default = None, help = "Left SVD representation of the species B. If not present, the SVD representation will be produced at this location")
    parser.add_argument("--svd_AV", default = None, help = "Right SVD representation of the species A. If not present, the SVD representation will be produced at this location")
    parser.add_argument("--svd_BV", default = None, help = "Right SVD representation of the species B. If not present, the SVD representation will be produced at this location")
    
    parser.add_argument("--svd_r", default = 100, type = int, help = "SVD default dimension")
    
    # ISORANK
    parser.add_argument("--landmarks_a_b", default = None, help = "the tsv file with format: `protA  protB  [score]`. Where protA is from speciesA and protB from speciesB")
    parser.add_argument("--compute_isorank", action = "store_true", default = False, help = "If true, then the landmarks tsv file is a sequence map obtained from Reciprocal BLAST and we need to compute isorank. If false, then the landmark file is a precomputed isorank mapping. This computed isorank matrix is then saved to the file {landmarks_a_b}_{alpha}.tsv")
    parser.add_argument("--isorank_alpha", type = float, default = 0.7)
    parser.add_argument("--no_landmarks", type = int, default = 500, help = "How many ISORANK mappings to use for MUNK?")
    # MODEL
    parser.add_argument("--model", default = None, help = "Location of the trained model. If not provided, the trained model is saved at this location")
    parser.add_argument("--svd_dist_a_b", default = None, help = "Location of the SVD COSINE distance between A and transformed B. If not present, the distance will be created at this location")
    parser.add_argument("--weight_decay", default=0, type = float, help = "Weight decay for the model")
    parser.add_argument("--no_epoch", default = 20, type = int, help = "The number of epochs")
    
    # Evaluation
    parser.add_argument("--compute_go_eval", default = False, action = "store_true", help = "Compute GO functional evaluation on the model?")
    parser.add_argument("--kA", default = "10,15,20,25,30", help = "Comma seperated values of kA to test")
    parser.add_argument("--kB", default = "10,15,20,25,30", help = "Comma seperated values of kB to test")
    parser.add_argument("--metrics", default="top-1-acc,aupr,f1max", help = "Comma separated metrics to test on")
    parser.add_argument("--wB", default = 0.4, type = float)
    parser.add_argument("--output_file", help = "Output tsv scores file")
    parser.add_argument("--compute_dsd", help = "Compute the DSD file", action = "store_true", default = False)
    
    # GO files
    parser.add_argument("--go_A", default = None, help = "GO files for species A in TSV format")
    parser.add_argument("--go_B", default = None, help = "GO files for species B in TSV format")
    parser.add_argument("--go_h", default = "molecular_function,biological_process,cellular_component", help = "Which of the three GO hierarchies to use, in comma separated format?")
    parser.add_argument("--munkonly", default = False, action = "store_true")
    
    parser.add_argument("--seed", default = 121, type = int)
    return parser.parse_args()


def get_scoring(metric, all_go_labels = None, **kwargs):
    acc = re.compile(r'top-([0-9]+)-acc')
    match_acc = acc.match(metric)
    if match_acc:
        k = int(match_acc.group(1))
        def score(prots, pred_go_map, true_go_map):
            return topk_accs(prots, pred_go_map, true_go_map, k = k)
        return score
    else:
        if metric == "aupr":
            met = average_precision_score
        elif metric == "auc":
            met = roc_auc_score
        elif metric == "f1max":
            def f1max(true, pred):
                pre, rec, _ = precision_recall_curve(true, pred)
                f1 = (2 * pre * rec) / (pre + rec + 1e-7)
                return np.max(f1)
            met = f1max
        sfunc = scoring_fcn(all_go_labels, met, **kwargs)
    return sfunc



def compute_dsd_dist(ppifile, dsdfile, jsonfile, threshold = -1, **kwargs):
    assert (os.path.exists(ppifile)) or (os.path.exists(dsdfile) and os.path.exists(jsonfile))
    print(f"[!] {kwargs['msg']}")
    if dsdfile is not None and os.path.exists(dsdfile):
        print(f"[!!] \tAlready computed!")
        DSDdist = np.load(dsdfile)
        if threshold > 0:
            DSDdist = np.where(DSDdist > threshold, threshold, DSDdist)
        with open(jsonfile, "r") as jf:
            protmap = json.load(jf)
        return DSDdist, protmap
    else:
        ppdf = pd.read_csv(ppifile, sep = "\t", header = None)
        Ap, protmap = compute_adjacency(ppdf)
        if jsonfile is not None:
            with open(jsonfile, "w") as jf:
                json.dump(protmap, jf)                
        DSDemb = dsd.compute_dsd_embedding(Ap, is_normalized = False)
        DSDdist = squareform(pdist(DSDemb))
        if dsdfile is not None:
            np.save(dsdfile, DSDdist)
        if threshold > 0:
            DSDdist = np.where(DSDdist > threshold, threshold, DSDdist)
        return DSDdist, protmap
    
    
# SVD FILE
def compute_svd(svdufile, svdvfile, dsddist, svd_r = 100, **kwargs):
    assert (os.path.exists(svdufile) and os.path.exists(svdvfile)) or dsddist is not None
    print(f"[!] {kwargs['msg']}")
    if os.path.exists(svdufile):
        print(f"[!!] \tAlready computed!")
        SVDUemb = np.load(svdufile)
        SVDVemb = np.load(svdvfile)
        
        # Row sort
        UVrowsum  = np.absolute(np.sum(SVDUemb, axis = 0))
        UVrowsort = np.argsort(-UVrowsum)
        SVDUemb  = SVDUemb[:, UVrowsort[:svd_r]]
        SVDVemb  = SVDVemb[:, UVrowsort[:svd_r]]
        print(np.sort(-np.absolute(UVrowsum))[:500])
        
        return SVDUemb[:, :svd_r], SVDVemb[:, :svd_r]
    else:
        SVDUemb, SVDVemb = compute_k_svd_uv(dsddist)
        if svdufile is not None:
            np.save(svdufile, SVDUemb)
        if svdvfile is not None:
            np.save(svdvfile, SVDVemb)
        return SVDUemb[:, :svd_r], SVDVemb[:, :svd_r]

def linear_munk(svdAU, svdAV, svdBU, no_matches, isorankfile, mapA, mapB):
    """
    svdA = nA x d
    svdB = nB x d
    """
    df = pd.read_csv(isorankfile, sep = "\t")
    df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x : mapA[x])
    df.iloc[:, 1] = df.iloc[:, 1].apply(lambda x : mapB[x])
    
    # Matches for MUNK, matches_train for the model
    matches = df.loc[:no_matches, :].values
    
    svdAU_l  = svdAU[matches[:, 0]] # match x d
    svdBU_l  = svdBU[matches[:, 1]] # match x d
    tb_to_a  = svdBU @ pinv(svdBU_l) @ svdAU_l # [nB, d] @ [d , match] @ [match, d]
    munk     =  svdAV @ tb_to_a.T # [nA, d] @ [d, nB] = [nA, nB]
    # y = Ax  + nonlinear
    # y = mod(x)
    # y = Ax + mod(x)
    # input = x, expected = y - Ax
    loss = svdAU_l #- tb_to_a[matches[:, 1]]
    return munk, loss, svdBU_l
    
     
    
def train_model_and_project(modelfile, svdAU, svdAV, svdBU, isorankfile, no_matches, protAmap, protBmap, no_matches_train = None, **kwargs):
    # B is moved to A
    
    assert os.path.exists(modelfile) or (svdAU is not None and svdBU is not None 
                                         and svdAV is not None)
    if no_matches_train == None:
        no_matches_train = no_matches
        
    munk, loss, svdBU_l = linear_munk(svdAU, svdAV, 
                                    svdBU,
                                    no_matches,
                                    isorankfile,
                                    protAmap,
                                    protBmap)
    
    if "munkonly" in kwargs and kwargs["munkonly"]:
        return munk
    
    pdata = PredictData(svdBU)
    predictloader = DataLoader(pdata, shuffle = False, batch_size = 10)
    #svdBUtorch = torch.tensor(svdBU, dtype = torch.float32).unsqueeze(-1)
    print(f"[!] {kwargs['msg']}")
    if os.path.exists(modelfile):
        model = torch.load(modelfile, map_location = "cpu")
        model.eval()
        with torch.no_grad():
            svdBnls = []
            for j, data in enumerate(predictloader):
                segment = data # [batch, seq, 1]
                svdBnls.append(model(segment).squeeze(-1).detach().numpy())
            svdBnl = np.concatenate(svdBnls, axis = 0)
            addterm = svdAV @ svdBnl.T
            print(f"[!!!] MUNK contribution : {np.linalg.norm(munk)}, Deep nn contribution: {np.linalg.norm(addterm)}")
            return 0.0000001 * munk + addterm
    else:
        data = CuratedData(loss, svdBU_l) #y, x 
        trainloader = DataLoader(data, shuffle = True, batch_size = 10)
        
        loss_fn = nn.MSELoss()
        model = AttentionModel3(svd_dim = svdAU.shape[1])
        model.train()
        optim = torch.optim.Adam(model.parameters(), 
                                 lr = 0.0005,
                                weight_decay = kwargs["weight_decay"])
        ep = kwargs["no_epoch"]
        print(f"[!]\t Training the Attention Model:")
        for e in range(ep):
            loss = 0
            for i, data in enumerate(trainloader):
                y, x = data # y is the embedding for species A (target). x is the embedding for species B (source). Model here moves source to target.
                optim.zero_grad()
                yhat = model(x)
                closs = loss_fn(y, yhat)
                closs.backward()
                optim.step()
                loss += closs.item()
            loss = loss / (i+1)
            print(f"[!]\t\t Epoch {e+1}: Loss : {loss}")
        if modelfile is not None:
            torch.save(model, modelfile)
        model.eval()
        with torch.no_grad():
            # model here transforms the complete source to target
            svdBnls = [] 
            for j, data in enumerate(predictloader):
                segment = data # [batch, seq, 1]
                svdBnls.append(model(segment).squeeze(-1).detach().numpy())
            svdBnl = np.concatenate(svdBnls, axis = 0)
            addterm = svdAV @ svdBnl.T
            print(f"[!!!] MUNK contribution : {np.linalg.norm(munk)}, Deep nn contribution: {np.linalg.norm(addterm)}")
            return 0.000001 * munk + addterm

    
def get_go_maps(nmap, gofile, gotype):
    """
    Get the GO maps
    """
    df = pd.read_csv(gofile, sep = "\t")
    df = df.loc[df["type"] == gotype]
    gomaps = df.loc[:, ["GO", "swissprot"]].groupby("swissprot", as_index = False).aggregate(list)
    gomaps = gomaps.values
    go_outs = {}
    all_gos = set()
    for prot, gos in gomaps:
        if prot in nmap:
            all_gos.update(gos)
            go_outs[nmap[prot]] = set(gos)
    for i in range(len(nmap)):
        if i not in go_outs:
            go_outs[i] = {}
    return go_outs, all_gos
    
def main(args):
    """
    Main function
    """
    DSDA, nmapA = compute_dsd_dist(args.ppiA, args.dsd_A_dist, args.json_A, threshold = 10,
                                  msg = "Running DSD distance for Species A")
    DSDB, nmapB = compute_dsd_dist(args.ppiB, args.dsd_B_dist, args.json_B, threshold = 10,
                                  msg = "Running DSD distance for Species B")
    
    SVDAU, SVDAV = compute_svd(args.svd_AU, args.svd_AV, DSDA, svd_r = args.svd_r, 
                      msg = "Computing the SVD embeddings from the DSD distances for Species A")
    SVDBU, SVDBV = compute_svd(args.svd_BU, args.svd_BV, DSDB, svd_r = args.svd_r, 
                      msg = "Computing the SVD embeddings from the DSD distances for Species B")
    
    if args.svd_dist_a_b is not None and os.path.exists(args.svd_dist_a_b):
        print("[!] SVD transformed distances between species A and B already computed")
        DISTS = np.load(args.svd_dist_a_b)
    else:
        isorank_file = f"{args.nameA}_{args.nameB}_isorank_alpha_{args.isorank_alpha}_N_{args.no_landmarks}.tsv"
        print(f"[!] Looking for the file {isorank_file}: {os.path.exists(isorank_file)}")
        if args.compute_isorank and not os.path.exists(isorank_file):
            compute_isorank_and_save(args.ppiA, 
                                     args.ppiB, 
                                     args.nameA, 
                                     args.nameB,
                                     matchfile = args.landmarks_a_b,
                                     alpha = args.isorank_alpha,
                                     n_align = args.no_landmarks, 
                                     save_loc = isorank_file,
                                     msg = "Running ISORANK.")
        elif not args.compute_isorank:
            isorank_file = args.landmarks_a_b
        train_kwargs = {"no_epoch" : args.no_epoch, 
                        "weight_decay" : args.weight_decay}
        
        DISTS = train_model_and_project(args.model, 
                                         SVDAU, SVDAV, SVDBU, isorank_file, 
                                         args.no_landmarks, 
                                         nmapA, nmapB, 
                                         msg = "Training the Attention Model and Computing the distances",
                                        munkonly = args.munkonly,
                                        **train_kwargs)
        
        if args.svd_dist_a_b is not None:
            np.save(args.svd_dist_a_b, DISTS)
            
    results = []
    #settings: nameA, nameB, SVD_emb, landmark, gotype, topkacc, dsd/mundo?, kA, kB, 
    
    settings = [args.nameA, args.nameB, args.svd_r, args.no_landmarks] 
    if args.compute_go_eval:
        """
        Perform evaluations
        """
        kAs = [int(k) for k in args.kA.split(",")]
        kBs = [int(k) for k in args.kB.split(",")]
        gos = args.go_h.split(",")
        gomapsA = {}
        gomapsB = {}
        for go in gos:
            gomapsA[go], golabelsA = get_go_maps(nmapA, args.go_A, go)
            gomapsB[go], golabelsB = get_go_maps(nmapB, args.go_B, go)
            golabels = golabelsA.union(golabelsB)
            print(f"GO count: {go} ---- {len(golabels)}")
            for metric in args.metrics.split(","):
                score = get_scoring(metric, golabels)
                for kA in kAs:
                    if args.compute_dsd:
                        settings_dsd = settings + [go, metric, "dsd-knn", kA, -1]
                        scores, _ = compute_metric(dsd_func(DSDA, k=kA), score, list(range(len(nmapA))), gomapsA[go], kfold = 5)
                        print(f"GO: {go}, DSD, k: {kA} ===> {np.average(scores):0.3f} +- {np.std(scores):0.3f}")
                        settings_dsd += [np.average(scores), np.std(scores)]
                        results.append(settings_dsd)
                    for kB in kBs:
                        settings_mundo = settings + [go, metric, f"mundo4-knn-weight-{args.wB:0.3f}", kA, kB]
                        scores, _ = compute_metric(dsd_func_mundo(DSDA, DISTS, gomapsB[go], k=kA, k_other=kB, weight_other = args.wB),
                                                  score, list(range(len(nmapA))), gomapsA[go], kfold = 5, seed = args.seed)
                        settings_mundo += [np.average(scores), np.std(scores)]
                        
                        print(f"GO: {go}, MUNDO4, kA: {kA}, kB: {kB} ===> {np.average(scores):0.3f} +- {np.std(scores):0.3f}")
                        results.append(settings_mundo)
        columns = ["Species A", "Species B", "SVD embedding", "Landmark no", "GO type", "Scoring metric", "Prediction method",
                  "kA", "kB", "Average score", "Standard deviation"]
        resultsdf = pd.DataFrame(results, columns = columns)
        resultsdf.to_csv(args.output_file, sep = "\t", index = None, mode = "a", header = not os.path.exists(args.output_file))
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    main(getargs(parser))
      
    
    
    
