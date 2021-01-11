#!/bin/bash

# Connect to an existing Azure Machine Learning service.
# Chris Joakim, Microsoft, 2020/12/21
#
# Usage
# ./connect_to_workspace.sh <rg> <name>
# ./connect_to_workspace.sh cjoakimaml2 cjoakimaml2

rm -rf .azureml/

python aml_client.py connect_to_workspace $AZURE_SUBSCRIPTION_ID $1 $2
