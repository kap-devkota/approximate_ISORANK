# Network Alignment using Approximate-Isorank 

This repo contains a faster, scalable implementation of Approximate-Isorank.

Citation: 

*** Devkota, K., Cowen, L. J., Blumer, A., & Hu, X. (2023). Fast Approximate IsoRank for Scalable Global Alignment of Biological Networks. bioRxiv, 2023-03. ***

# Table of contents
1. [Installation](#installation)
2. [Running Approximate Isorank](#running-approximate-isorank)

## Installation

You can install the `netalign` package in your local python environment using the command 

```
python setup.py build; python setup.py install;
```

All the packages dependencies required to run approximate isorank is provided in the `environment.yml` YAML file.


## Running Approximate Isorank
After you install the `netalign` package, you can run the approximate isorank code by running the command:

```
netalign isorank --net1 <net1-filename> --net2 <net2-filename> --rblast <rblast-filename> --alpha <alpha-value-float> --niter <no-iterations-int> --npairs <no-of-one-to-one-pairs-int> --output <output-filename>
```

### Parameters
1. --net1: The PPI network of the first species. Should be tab delimited without a header. Example name: *mouse.tsv, fly-string.tsv, mouse-intact.filtered.tsv*
2. --net2: The PPI network of the second species. Should be tab delimited without a header. Follows the same naming conventions as the argument provided by --net1
3. --rblast: A file that holds the sequence similarity information between the two species. Should be a tab-delimited file and should contain three columns.
             For example, if the PPI files provided are `--net1 fly-string.tsv --net2 mouse-biogrid.tsv`, the header of the reciprocal blast file should be
             ***fly-mouse.tsv:***
             <center>
             | fly-string | mouse-biogrid | score | 
             |------------|:-------------:|------:|
             | Q9VPH0     |   Q8R2G6      | xxx   |
             </center>
             The score should be a floating point number.
4. --alpha: Isorank parameter, with value between 0 and 1. The recommended value is 0.7
5. --niter: This parameter decides how close the approximation should be to the true Isorank. If set to 0, this returns the `R0` approximation. If set to 1, this       
returns the `R1` approximation. If we want to get the true Isorank alignment, set --niter to some value > 10.
6. --npairs: This parameter decides the number of top one-to-one protein pairs to be outputted by the alignment
7. --output: The final one-to-one alignment is saved to this output file.