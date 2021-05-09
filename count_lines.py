#!/usr/bin/env python

# adapted from https://stackoverflow.com/a/46132145/2580489
from json import load
from sys import argv
import glob, os

def loc(nb):
    cells = load(open(nb))['cells']
    return sum(len([l for l in c['source'] if l.strip()]) for c in cells if c['cell_type'] == 'code')

def run(ipynb_files):
    return sum(loc(nb) for nb in ipynb_files)

if __name__ == '__main__':
    #print(glob.glob("**/*.ipynb", recursive=True))
    print("ipynb notebooks: ", run(glob.glob("notebooks/**/*.ipynb", recursive=True)))
    print("ipynb plotting: ", run(glob.glob("plotting/**/*.ipynb", recursive=True)))
    tot = 0
    #print(glob.glob("*.py", recursive=True))
    for f in glob.glob("envs/**/*.py", recursive=True):
        tot += sum(1 for line in open(f) if line.strip())
    for f in glob.glob("notebooks/**/*.py", recursive=True):
        tot += sum(1 for line in open(f) if line.strip())
    print("py: ", tot)