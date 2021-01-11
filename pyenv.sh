#!/bin/bash

# Bash script to create, populate, and activate the python virtual environment
# for this project with pyenv.
# Chris Joakim, 2021/01/11

echo '=== removing env directory, .python-version, requirements.txt'
rm .python-version
venv_dir=$HOME/.pyenv/versions/3.8.6/envs/aml/
rm -rf $venv_dir
rm requirements.txt

# These are the only two values that need to change between projects:
venv_name="aml"
python_version="3.8.6"  # 3.7.9

echo '=== creating virtualenv '$venv_name

pyenv virtualenv -f $python_version $venv_name

echo '=== python version'
python --version 

echo '=== setting pyenv local ...'
pyenv local $venv_name

echo '=== upgrade pip ...'
pip install --upgrade pip

echo '=== install pip-tools ...'
pip install pip-tools

echo '=== pip compile ...'
pip-compile

echo '=== pip install ...'
pip install -r requirements.txt

echo '=== pip list ...'
pip list

echo '=== .python-version ...'
cat .python-version

echo 'done'
