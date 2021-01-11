#!/bin/bash

conda env create -f .azureml/pytorch-env.yml    # create conda environment

# $ python --version
# Python 3.8.5
# (aml) (base) [~/github/cj-azure/aml]$ which python
# /Users/cjoakim/opt/anaconda3/bin/python

conda activate pytorch-env                      # activate conda environment

python src/train.py                             # train model