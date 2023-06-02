#!/usr/bin/env python

from setuptools import setup, find_packages
import netalignpack

setup(
    name="NetworkAlignmentTools",
    version=netalignpack.__version__,
    description="Tools for Network Alignment",
    author="Kapil Devkota, Grigorii Sterin",
    author_email="kapil.devkota@tufts.edu",
    # url="",
    license="MIT",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "netalign = netalign.__main__:main",
        ],
    },
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "biopython",
        "matplotlib",
        "seaborn",
        "tqdm",
        "scikit-learn",
        "networkx"
    ],
)