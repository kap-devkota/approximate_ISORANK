import pandas as pd
from tqdm import tqdm 
import networkit as nit
import networkx as nx
from networkit.scd import PageRankNibble
from collections import OrderedDict
import random
import os
import numpy as np
from isorankn.utils import NodeInfo
from copy import copy

class Isorankn:
    def __init__(self, speciesmap, matchmap, beta, gamma, r_iter, isorank_alpha, 
                 species_ordering = None, eps = 1e-5):
        """
        self.bR0 = np.where(R0 > beta * R0_max_rows)
        """
        self.bR0 = None
        # working copy while running iterations
        self.wR0 = None
        
        # Required during the ISORANK-N loop
        self.exclusionset = set()
        self.membership   = {}
        self.annotated_membership = {}
        self.speciesrun   = {}
        
        self.node = None
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        
        self.adjacencies = {}
        self.matches = {}
        self.nodemap = {}
        self.isorankmats = {}
        
        self.speciesmap = speciesmap     # dictionary: species_name -> adjacency file
        self.species = list(speciesmap.keys())
        self.workingspecieslist = set(self.species)
        self.matchmap = matchmap         # dictionary: (sp1, sp2) -> match file
        self.species_ordering = species_ordering
        if self.species_ordering is not None:
            assert len(self.species_ordering) == len(self.species)
        self.working_species_ordering = copy(species_ordering)
        self.isorank_alpha = isorank_alpha
        self.r_iter = r_iter
        print("Initializing Isorank-N...")
        print(f"\t1. Constructing the Isorank matrices...")
        # compute the adjacency, map and isorank matrices
        for i, sp1 in enumerate(self.species):
            for j in range(i+1):
                sp2 = self.species[j]
                self.compute_isorank(sp1, sp2)
                print(f"\t\tComputed Isorank-N({sp1}, {sp2})")
        
        print(f"\t2. Initializing nodeinfo")
        self.initialize_nodeinfo()
        
        print(f"\t3. Constructing the BigR matrix")
        # construct the bigR matrix
        self.construct_bigR(self.node)
        
        # these temp matrices are no longer needed
        del self.adjacencies
        del self.matches
        del self.isorankmats
        
        self.reintialization_isorankn()
        print(f"Initialization complete.")
        return 
    
    def initialize_nodeinfo(self):
        self.node = NodeInfo(self.nodemap)
        for sp in self.species:
            self.speciesrun[sp] = self.node.getsize(sp)
        return
    
    def get_adjacency(self, sp, update_adjacency = False):
        if not update_adjacency and sp in self.adjacencies:
            return self.adjacencies[sp], self.nodemap[sp]
        
        # header should be none
        df = pd.read_csv(self.speciesmap[sp], sep = "\t", header = None)
        df.columns = [0, 1]
        
        nodes = list(set(df[0]).union(df[1]))
        nmap  = {k: i for i, k in enumerate(nodes)}
        
        # reannotate
        df.loc[:, 0] = df.loc[:, 0].apply(lambda x : nmap[x]).values
        df.loc[:, 1] = df.loc[:, 1].apply(lambda x : nmap[x]).values
        
        # construct adjacencies
        A = np.zeros((len(nodes), len(nodes)))
        
        for edge in df.values:
            assert len(edge) in {2, 3}
            if len(edge) == 2:
                p, q = edge
                w = 1
            else:
                p, q, w = edge
            A[p, q] = w
            A[q, p] = w
        
        self.adjacencies[sp] = A
        self.nodemap[sp] = nodes
        return A, nodes
    
    def get_matchmap(self, sp1, sp2, update_map = False):
        if not update_map: 
            if (sp1, sp2) in self.matches:
                return self.matches[(sp1, sp2)]
            if (sp2, sp1) in self.matches:
                return self.matches[(sp2, sp1)].T
            
        if sp1 not in self.adjacencies:
            self.get_adjacency(sp1)
        if sp2 not in self.adjacencies:
            self.get_adjacency(sp2)
        
        if (sp1, sp2) in self.matchmap:
            df = pd.read_csv(self.matchmap[(sp1, sp2)], sep = "\t")
            df = df.loc[:, [sp1, sp2, "score"]]
        elif (sp2, sp1) in self.matchmap:
            df = pd.read_csv(self.matchmap[(sp2, sp1)], sep = "\t")
            df = df.loc[:, [sp2, sp1, "score"]]
            
        nmap1 = {k: i for i, k in enumerate(self.nodemap[sp1])}
        nmap2 = {k: i for i, k in enumerate(self.nodemap[sp2])}
        if sp1 == sp2 and (sp1, sp2) not in self.matchmap:
            E = np.eye(len(nmap1))
            return E
        
        # ensure that the map file contains only the proteins in the network
        df = df.loc[
            df[sp1].apply(lambda x : x in nmap1) &
            df[sp2].apply(lambda x : x in nmap2)
            , :]
        
        df[sp1] = df[sp1].apply(lambda x : nmap1[x]).values
        df[sp2] = df[sp2].apply(lambda x : nmap2[x]).values
        E = np.zeros((len(nmap1), len(nmap2)))
        for p, q, w  in df.loc[:, [sp1, sp2, "score"]].values:
            p = int(p + 0.1)
            q = int(q + 0.1)
            E[p, q] = w
            if sp1 == sp2:
                E[q, p] = w
        self.matches[(sp1, sp2)] = E
        return E
    
    def compute_isorank(self, sp1, sp2):
        """
        Compute the isorank using the eigendecomposition
        """
        alpha = self.isorank_alpha
        maxiter = self.r_iter
        
        # get the relevant matrices
        A1, _ = self.get_adjacency(sp1)
        A2, _ = self.get_adjacency(sp2)
        E  = self.get_matchmap(sp1, sp2)
        d1 = np.sum(A1, axis = 1).reshape(-1, 1)
        d2 = np.sum(A2, axis = 1).reshape(-1, 1)
        P1 = A1 / d1.T
        P2 = A2 / d2.T
        E = E / np.sum(E)
        d = d1 @ d2.T 
        d = d / (np.sum(d1) * np.sum(d2))
        R = (1-alpha) * d + alpha * E
        if maxiter > 0:
            # Reshape R and E
            R = R.T
            E = E.T
            for i in range(maxiter):
                R = (1-alpha) * (P2 @ R @ P1.T) + alpha * E
            R = R.T # revert back
        self.isorankmats[(sp1, sp2)] = R
        return R
        
    def construct_bigR(self, nodes, add_self = True):
        finalsize = nodes.getfinalsize()
        bigR = np.zeros((finalsize, finalsize))
        for (sp1, sp2), R in self.isorankmats.items():
            if sp1 == sp2 and (not add_self):
                continue
            rowi, rowj = nodes.getid(sp1)
            coli, colj = nodes.getid(sp2)
            rowids = np.arange(rowi, rowj).tolist()
            colids = np.arange(coli, colj).tolist()
            bigR[np.ix_(rowids, colids)] = R
            if sp1 != sp2:
                bigR[np.ix_(colids, rowids)] = R.T
        # Discretization
        maxR = np.max(bigR, axis = 1)
        bR0  = bigR - self.beta * maxR[:, None]
        bR0  = np.where(bR0 >0, bR0, 0)
        self.bR0 = bR0
        return
    
    def reintialization_isorankn(self):
        self.wR0 = self.bR0.copy()
        self.exclusionset = set()
        if self.species_ordering is not None:
            self.working_species_ordering = copy(self.species_ordering)
        self.membership = {}
        self.annotated_membership = {}
    
    def compute_pagerank_nibble(self, R):
        ## Construct the networkx graph
        nxG = nx.from_numpy_array(R)
        ## Construct the networkit graph from nxG
        nitG = nit.nxadapter.nx2nk(nxG)
        ## construct the pagerank graph    
        Pnibble = PageRankNibble(nitG, self.gamma, self.eps)
        ## Return the low conductance graph
        return list(Pnibble.expandOneCommunity(0))[1:] # 0th index corresponds to s, and is removed
    
    def iteration(self):
        """
        Compute the Isorank-nibble iterations, step 3 and 4
        """
        # Get a random network
        if self.working_species_ordering is not None:
            while(True):
                net = random.sample(list(self.workingspecieslist), k = 1)[0]
                startid, endid = self.node.getid(net)
                # find the node with largest connectivity
                sumwrtclst = np.sum(self.wR0[startid:endid, :], axis = 1)
                s = np.argmax(sumwrtclst)
                if sumwrtclst[s] == 0:
                    self.workingspecieslist.remove(net)
                    print(f"Removed {net}")
                    if len(self.workingspecieslist) == 0:
                        return -1 # completion code
                else:
                    s = s + startid
                    break
        else: #use the order specified in the config file
            while(True):
                net = self.working_species_ordering[0]
                startid, endid = self.node.getid(net)
                sumwrtclst = np.sum(self.wR0[startid:endid, :], axis = 1)
                s = np.argmax(sumwrtclst)
                if sumwrtclst[s] == 0:
                    self.working_species_ordering.pop(0)
                    continue
                else:
                    s = s + startid
                    break
        # construct a sub-adjacency matrix
        nonzeroids = set(np.argwhere(self.wR0[s, :] > 0).flatten().tolist())
        # ensure that it does not contain "s"
        if s in nonzeroids:
            nonzeroids.remove(s)
        # construct the final set, placing "s" in index 0 on nonzeroids
        nonzeroids = np.array([s] + list(nonzeroids), dtype = int)
        
        Rv = self.wR0[np.ix_(nonzeroids, nonzeroids)] # 0th index corresponds to `s`
        
        # We ensure that the new set has no index "0"
        Svaset  = self.compute_pagerank_nibble(Rv)
        
        # map the rows of Rv to the actual indices
        Svatrue = nonzeroids[Svaset].tolist() # lacks s
        
        self.exclusionset.update(Svatrue + [s])
        
        # merging
        self.merge(Svatrue, s)
        
        self.wR0[Svatrue + [s], :] = 0
        self.wR0[:, Svatrue + [s]] = 0
        return len(set(Svatrue)) + 1
    
    def merge(self, s1members, s1):
        # s1 greater than beta max
        if not isinstance(s1members, set):
            s1members = set(s1members)
        
        s1gtbm = set(np.argwhere(self.bR0[s1, :] > 0).flatten().tolist())
        
        idtojoin = None
        for s2, s2members in self.membership.items():
            if not self.node.insameppi(s1, s2):
                continue
            s2gtbm = set(np.argwhere(self.bR0[s2, :] > 0).flatten().tolist())
            if s2members.issubset(s1gtbm) and s1members.issubset(s2gtbm):
                ## merge
                idtojoin = s2
                break
            
        if idtojoin == None:
            self.membership[s1] = s1members
            self.annotated_membership[self.node.completedict[s1]] = {self.node.completedict[p] for p in s1members}
        else:
            self.membership[s2].update(s1members.union([s1]))
            self.annotated_membership[self.node.completedict[s2]].update({self.node.completedict[p] for p in s1members.union([s1])})
        return
    
    def run(self):
        total_prot = self.bR0.shape[0]
        with tqdm(total=total_prot, desc = "Computing Clusters") as pbar:
            while True:
                csize = self.iteration()
                if len(self.workingspecieslist) == 0:
                    break
                pbar.update(csize)