#!/bin/bash

echo 'python before:'
python --version
which python

echo 'prepending anaconda3 to $PATH...'
PATH=/Users/cjoakim/opt/anaconda3/bin:/Users/cjoakim/opt/anaconda3/condabin:$PATH

echo 'python after:'
python --version
which python

echo 'done'
