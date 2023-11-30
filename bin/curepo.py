# -*- coding: utf-8 -*-
import os
import sys
import argparse
import pathlib

STEM = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(STEM))
REPO = STEM/"repo"
REPO.mkdir(parents=True, exist_ok=True)

from CuAsm.CuInsAssemblerRepos import CuInsAssemblerRepos
from CuAsm.CuInsFeeder import CuInsFeeder

newRepoTemplate = "InsAsmRepos.{0}.txt"
defaultRepoTemplate = "CuAsm/InsAsmRepos/DefaultInsAsmRepos.{0}.txt"

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sass', required=True)
parser.add_argument('-r', '--repo', default=None)
parser.add_argument('-a', '--arch', required=False, default="sm_86")
args = parser.parse_args()

if args.repo:
    repo = REPO/args.repo
else:
    repo = REPO/newRepoTemplate.format(args.arch)

new_default_repo = REPO/defaultRepoTemplate.format(args.arch)
new_default_repo.parent.mkdir(parents=True, exist_ok=True)

def constructReposFromFile(sassname, savname=None, arch='sm_86'):
    # initialize a feeder with sass
    feeder = CuInsFeeder(sassname, archfilter=arch)

    # initialize an empty repos
    repos = CuInsAssemblerRepos(arch=arch)#

    # Update the repos with instructions from feeder
    repos.update(feeder)

    # reset the feeder back to start
    feeder.restart()

    # verify the repos
    # actually the codes is already verifed during repos construction
    repos.verify(feeder)

    if savname is not None:
        repos.save2file(savname)

    return repos

def verifyReposFromFile(sassname, reposfile, arch='sm_86'):

    # initialize a feeder with sass
    feeder = CuInsFeeder(sassname, archfilter=arch)

    # initialize an empty repos
    repos = CuInsAssemblerRepos(reposfile, arch=arch)#

    # verify the repos
    repos.verify(feeder)

if __name__ == '__main__':
    print(f"Reading SASS from {args.sass} and saving to {repo}...")

    constructReposFromFile(args.sass, str(repo), arch=args.arch)
    print('### Construction done!')
    verifyReposFromFile(args.sass, str(repo), arch=args.arch)
    print('### Verification done!')

    default_repo = CuInsAssemblerRepos.getDefaultRepos(args.arch)
    #default_repo = CuInsAssemblerRepos(str(default_repo))
    default_repo.merge(str(repo))

    default_repo.completePredCodes()
    default_repo.save2file(str(new_default_repo))
