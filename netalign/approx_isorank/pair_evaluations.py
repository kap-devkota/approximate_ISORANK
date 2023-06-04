import numpy as np
import pandas as pd
import networkx as nx
from goatools.base import get_godag
from goatools.semantic import semantic_similarity

def compute_edge_correctness(pairs, Gdf1, Gdf2):
    """
    Given $G_1 = (V_1, E_1), G_2 = (V_2, E_2)$ and a mapping $g: V_1 \rightarrow V_2$, we can compute the EC metric as
    
    EC = \frac{|(u, v) \in E_1:(g(u), g(v)) \in E_2|}{|E_1|} * 100
    """
    pairmap = {v1: v2 for v1, v2 in pairs}
    EC    = 0
    niter = 0
    
    g2set = {(p, q) for p, q in Gdf2.values}
    
    for p, q in Gdf1.values:
        if (p not in pairmap) or (q not in pairmap):
            continue
        niter += 1
        p_, q_ = (pairmap[p], pairmap[q])
        if (p_, q_) in g2set or (q_, p_) in g2set:
            EC += 1
    return EC / niter

def symmetric_substructure(pairs, Gdf1, Gdf2):
    """
    Same setting as above:
    
    Let f(E_1) = {(g(u), g(v)) \in E_2 : (u, v) \in E_1} and 
    Let f(V_1) = {g(v) \in V_2 : v \in V_1}
    S^3 = \frac{|f(E_1)|}{|E_1| + |E(G_2(f(V_1))| + |f(E_1)|}
    """
    
    pairmap = {v1: v2 for v1, v2 in pairs}
    v2set   = set(list(pairmap.values()))
    
    Gdf1filter = Gdf1.loc[Gdf1[0].isin(pairmap) & Gdf1[1].isin(pairmap), :]
    normE1 = len(Gdf1filter)
    
    g2set = {(p, q) for p, q in Gdf2.values}
    niter = 0
    normfE1 = 0
    for p, q in Gdf1filter.values:
        niter += 1
        p_, q_ = (pairmap[p], pairmap[q])
        if (p_, q_) in g2set or (q_, p_) in g2set:
            normfE1 += 1
    
    normG2fV1 = len(Gdf2.loc[Gdf2[0].isin(v2set) & Gdf2[1].isin(v2set), :])
    assert normE1 + normG2fV1 + normfE1 > 0
    return normfE1 / (normE1 + normG2fV1 + normfE1)

def lccs(pairs, Gdf1, Gdf2):
    """
    Compute the subgraphs of Gdf1 and Gdf2 from the pairing, find the 
    largest connected component and return the corresponding edgecount
    """
    try:
        pairmap = {v1: v2 for v1, v2 in pairs}
        g1set = [(p, q) for p, q in Gdf1.values]
        g2set = {(p, q) for p, q in Gdf2.values}

        combinedset = []
        for p, q in g1set:
            if (p not in pairmap) or (q not in pairmap):
                continue
            p_, q_ = pairmap[p], pairmap[q]
            if (p_, q_) in g2set or (q_, p_) in g2set:
                combinedset.append((p, q))
        G = nx.Graph()
        G.add_edges_from(combinedset)
        LCC = max(nx.connected_components(G), key = len)
        return len(G.subgraph(LCC).edges())
    except Exception as e:
        return 0
    
def semantic_sim(pairs, Gdf1, Gdf2, **kwargs):
    """
    Compute the semantic similarity
    """
    assert "obofile" in kwargs and "type" in kwargs and "annot1file" in kwargs and "annot2file" in kwargs
    
    godag = get_godag(kwargs["obofile"], optional_attrs = "relationship")
    
    a1df = pd.read_csv(kwargs["annot1file"], sep = "\t")
    a2df = pd.read_csv(kwargs["annot2file"], sep = "\t")
    
    gotype = kwargs["type"]
    
    a1df = a1df[a1df["type"] == gotype]
    a2df = a2df[a2df["type"] == gotype]
    
    average_sem_sim = []
    
    for p, q in pairs:
        goterms_1 = a1df.loc[a1df["swissprot"] == p, "GO"].values
        goterms_2 = a2df.loc[a2df["swissprot"] == q, "GO"].values
        
        if len(goterms_1) == 0 or len(goterms_2) == 0:
            continue
            # average_sem_sim.append(0)
        else:
            pairwise = []
            for i in range(len(goterms_1)):
                for j in range(len(goterms_2)):
                    pairwise.append(semantic_similarity(goterms_1[i], 
                                                   goterms_2[j],
                                                   godag))
            pairwise = np.average(pairwise)
            average_sem_sim.append(pairwise)
    average_sem_sim = np.average(average_sem_sim)
    return average_sem_sim
    