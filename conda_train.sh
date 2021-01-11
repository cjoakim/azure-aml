#!/bin/bash

conda env create -f .azureml/pytorch-env.yml    # create conda environment

conda activate pytorch-env                      # activate conda environment

python src/train.py                             # train model