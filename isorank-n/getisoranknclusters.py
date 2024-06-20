import os
import yaml
import random
import numpy as np
from argparse import ArgumentParser
from isorankn.utils import NodeInfo
from isorankn.isorankn import Isorankn
import pickle as pkl
import numpy as np

class Config:
    def __init__(self, filename):
        with open(filename, 'r') as file:
            configdict = yaml.safe_load(file)
        self.__dict__.update(configdict)
        self.speciesmap = {
            sp: f"{self.networkfolder}/{sp}.tsv" for 
            sp in self.species
        }
        self.matchmap = {}
        for sp1 in self.species:
            for sp2 in self.species:
                if os.path.exists(f"{self.networkfolder}/{sp1}-{sp2}.tsv"):
                    self.matchmap[(sp1, sp2)] = f"{self.networkfolder}/{sp1}-{sp2}.tsv"
                elif os.path.exists(f"{self.networkfolder}/{sp2}-{sp1}.tsv"):
                    self.matchmap[(sp2, sp1)] = f"{self.networkfolder}/{sp2}-{sp1}.tsv"
        return
        
def main(config):
    isorankn = Isorankn(config.speciesmap, config.matchmap, config.beta, 
                       config.gamma, config.r_iter, config.isorank_alpha)
    isorankn.run()
    with open(config.output_cluster, "wb") as obf:
        pkl.dump(isorankn.annotated_membership, obf)
    return
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", help = "Configuration file")
    config = Config(parser.parse_args().config)
    main(config)