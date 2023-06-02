#!/cluster/tufts/cowenlab/.envs/denoise/bin/python
import numpy as np
import scipy.io as sio
import argparse
import pandas as pd

def compute_greedy_assignment(R1, n_align):
    """
    Compute greedy assignment
    """
    aligned = []
    R = R1.copy()
    
    n_align = min(n_align, *R.shape)
    
    itr = 1
    while(len(aligned) < n_align):
        print(f"\tRunning iter {itr}")
        itr   += 1
        #print(R)
        maxcols = np.argmax(R, axis = 1) # best y ids
        #print(maxcols)
        maxid = np.argmax(np.max(R, axis = 1)) # best x id
        print(maxid)
        maxcol = maxcols[maxid]
        #print(maxcol)
        aligned.append((maxid, maxcol))
        R[:, maxcol] = -1
        R[maxid, :]  = -1
    return aligned


def compute_accuracy(R0, R1, R, max_align, steps):
    print("Matching R0:")
    R0match = compute_greedy_assignment(R0, max_align)
    print("Matching R1:")
    R1match = compute_greedy_assignment(R1, max_align)
    print("Matching R:")
    Rmatch = compute_greedy_assignment(R, max_align)
    
    batch = max_align // steps
    Rset  = set()
    R0set = set()
    R1set = set()
    acc0 = []
    acc1 = []
    for i in range(batch):
        print(R0match[i * steps: (i+1) * steps])
        R0set.update(R0match[i * steps: (i+1) * steps])
        R1set.update(R1match[i * steps: (i+1) * steps])
        Rset.update(Rmatch[i * steps: (i+1) * steps])
        acc0.append(len(R0set.intersection(Rset)) / len(R0set.union(Rset)))
        acc1.append(len(R1set.intersection(Rset)) / len(R0set.union(Rset)))
    return acc0, acc1
    
def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required = True, help = "Prefix of the files. The program will add _1.mat, _0.mat and _20.mat for R1, R0 and R respectively")
    parser.add_argument("--max_align", default = 1000, type = int)
    parser.add_argument("--steps", default = 20, type = int)
    parser.add_argument("--output", default = "ouput.tsv")
    return parser.parse_args()
    # ./run_matching.py --prefix ../data/intact_output/working/outputs/fly_bakers_0.200000 --max_align 2 --steps 1 --output output_test.tsv

def main(args):
    R0 = sio.loadmat(f"{args.prefix}_0.mat")["R0"]
    R1 = sio.loadmat(f"{args.prefix}_1.mat")["R1"]
    R = sio.loadmat(f"{args.prefix}_20.mat")["R"]
    
    assert (R0.shape == R.shape) and (R.shape == R1.shape)
    max_align = min(args.max_align, *R.shape)
    
    
    a0, a1 = compute_accuracy(R0, R1, R, max_align, args.steps)
    x = list(range(args.steps, max_align, args.steps))
    pd.DataFrame(zip(x, a0, a1), columns = ["max-k-match", "Accuracy(R0)", "Accuracy(R1)"]).to_csv(args.output, index = None, sep = "\t")

if __name__ == "__main__":
    main(getargs())
    