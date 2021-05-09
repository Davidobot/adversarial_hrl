from os import system, listdir
from shutil import move
from os import mkdir, getcwd
import glob, os


def jup2pdf(ipynb_files):
    for f in ipynb_files:
        if f.endswith("ipynb"):
            system("jupyter nbconvert --to pdf " + f)

    return None


'''

'''
print("ipynb notebooks: ", jup2pdf(glob.glob("notebooks/**/*.ipynb", recursive=True)))
print("ipynb plotting: ", jup2pdf(glob.glob("plotting/**/*.ipynb", recursive=True)))
check_pres_of = [f.endswith("ipynb") for f in listdir(".")]
