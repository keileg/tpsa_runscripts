# Runscripts for exploring the performance of two-point stress approximation methods
This repository contains runscripts for numerical experiments that are reported in 

J. M. Nordbotten, E. Keilegavlen: Two-point stress approximation: A simple and robust finite volume method for linearized (poro-)mechanics and Stokes flow, [arXiv:2405.10390](https://arxiv.org/abs/2405.10390).

## Docker image
A fully runnable docker image of the runscript and the necessary requirements is available at [Zenodo](https://dx.doi.org/10.5281/zenodo.14615444). This is the recommended way to access the runscript and reproduce the results from the paper.

Log into the Docker image and run 

```bash
python run_3d.py
```

to generate the figures.

## Manual installation
The experiments are based on the simulation tool [PorePy](https://github.com/pmgbergen/porepy), see that repository for how to install PorePy.
Note that as PorePy is further developed, the runscript contained in this repository may not be updated, hence no guarantees can be given for their future applicability. If you want a sure way to run the scripts, please use the Docker image.

The modifications of PorePy needed to run the numerical experiments are not yet included in the `develop` branch of PorePy. To access these, run the following commands in a local installation of PorePy:

```bash
git git remote add tpsa_repo https://github.com/keileg/porepy.git
git fetch
git switch -c tpsa tpsa_repo/tpsa_no_cosserat
```

In addition, it may be necessary to install `PyAMG` and related packages to solve the
linear systems resulting from the discretization.

## How to run
The main script is `run_3d.py`; running this will generate the convergence plots reported in Section 6 of the arXiv paper.
