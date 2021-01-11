#!/bin/bash

# Chris Joakim, 2020/11/25

# python -m ipykernel install --user --name=localml

# Output:
# Installed kernelspec localml in /Users/cjoakim/Library/Jupyter/kernels/localml

echo 'installing ipykernel in pyenv: localml'
python -m ipykernel install --user --name=localml

echo '/////////////////////////////////////////////////////////////////////////'
echo 'NOTE: In the Jypyter UI, change the kernel to: localml'
echo '      Kernel -> Change Kernel -> localml'
echo ''
echo 'URL:  http://localhost:8888/notebooks/Untitled.ipynb?kernel_name=localml'
echo '/////////////////////////////////////////////////////////////////////////'

sleep 3

jupyter notebook
