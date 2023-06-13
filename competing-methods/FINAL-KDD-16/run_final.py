import shlex
import subprocess as sp
import argparse
import os


def add_args(parser):
    parser.add_argument("--alpha", type = float, default = 0.5, help = "Final parameter. default value = 0.5")
    parser.add_argument("--max_iter", type = int, default = 30, help = "How many iterations to run?")
    parser.add_argument("--tol", type = float, default = 1e-4, help = "Tolerance")
    return parser


def main(args):

    species = ["fly", "bakers", "rat", "mouse"] # human
    
    for i in range(1):
        for j in range(4):
            sp1 = "human"
            sp2 = species[j]

            net1 = f"temp-data/{sp1}.mat"
            net2 = f"temp-data/{sp2}.mat"
            order = f"{sp1}-{sp2}"
            E = f"temp-data/{order}.mat"

            if not os.path.exists(E):
                order = f"{sp2}-{sp1}"
                E = f"temp-data/{order}.mat"
            sub = "_".join(order.split("-"))

            out = shlex.split(f"matlab -nodisplay -r \"disp('Running {sp1} v. {sp2}...'); addpath('.'); A1=load('{net1}').{sp1}; A2=load('{net2}').{sp2}; H=load('{E}').{sub}; disp('\t Loaded files. Running FINAL iterations...'); out=FINAL(A1, A2, [], [], [], [], H, {args.alpha}, {args.max_iter}, {args.tol}); save('temp-data/FINAL-{order}.mat', 'out'); disp('Run complete!'); exit;\"")
            print(out)

            p = sp.Popen(out)
            stdout, stderr = p.communicate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    main(add_args(parser).parse_args())