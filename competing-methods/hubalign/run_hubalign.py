import argparse
import pandas as pd
import shlex
import subprocess as sp
import os

NETFILE="../../data/intact"

def add_args(parser):
    parser.add_argument("--spA", required = True, help = "First species.")
    parser.add_argument("--spB", required = True, help = "Second species.")
    parser.add_argument("--alpha", type = float, default = 0.7, help = "Alpha parameter")
    parser.add_argument("--lambdap", type = float, default = 0.1, help = "Hubalign lambda parameter")
    parser.add_argument("--tempfolder", default = "temp-data", help = "Temporary data")
    parser.add_argument("--outpref", default = "output", help = "Output prefix")
    return parser


def main(args):
    global NETFILE
    
    spA   = args.spA
    spB   = args.spB
    netA, netB = [f"{sp}.tab" for sp in [spA, spB]]
    
    alph  = args.alpha
    lam   = args.lambdap
    simf  = f"{NETFILE}/{spA}-{spB}.tsv"
    tempf = args.tempfolder
    
    if not os.path.exists(tempf):
        os.mkdir(tempf)
    
    assert os.path.exists(netA) and os.path.exists(netB) and os.path.exists(simf)
    
    # Remove the head from csv file
    df = pd.read_csv(simf, sep = "\t")
    df = df.loc[:, [spA, spB, "score"]]
    df.to_csv(f"{spA}-{spB}.txt", sep = "\t", header = None, index = None)
    _simf = f"{spA}-{spB}.txt"
    
    out = shlex.split(f"build/hubalign {netA} {netB} -b {_simf} -l {lam} -a {alph}")
    print(out)
    
    p = sp.Popen(out)
    stdout, stderr = p.communicate()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    main(add_args(parser).parse_args())