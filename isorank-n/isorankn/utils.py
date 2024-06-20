import numpy as np
import pandas as pd

class NodeInfo:
    def __init__(self, multispeciesnodes):
        self.startid = {}
        self.endid = {}
        self.allnodes = multispeciesnodes
        self.species = []
        self.completedict = {}
        i = 0
        for species, nodes in multispeciesnodes.items():
            self.startid[species] = i
            sizef = len(nodes)
            for j, name in enumerate(nodes):
                self.completedict[j+i] = name
            i += sizef
            self.endid[species] = i
            self.species.append(species)
        self.size = i
        return
    
    def getid(self, species):
        return self.startid[species], self.endid[species]
    
    def get_curr_species(self, idx):
        for species, sid in self.startid.items():
            eid = self.endid[species]
            if idx >= sid and idx < eid:
                return species
        
    def getfinalsize(self):
        return self.size
    
    def insameppi(self, idx1, idx2):
        if self.get_curr_species(idx1) == self.get_curr_species(idx2):
            return True
        return False
    
    def getsize(self, species):
        return self.endid[species] - self.startid[species]