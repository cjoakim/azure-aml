#!/bin/bash

# Create a new Azure Machine Learning service, with assotiated other services, 
# in a RG of the same name, with the Python SDK.
# Chris Joakim, Microsoft, 2021/01/11

# We can do this with either the az CLI, or python SDK.  
# With the az CLI, like this:
# az ml workspace create -w 'aml-workspace' -g 'aml-resources'

rm -rf .azureml/

python aml_client.py create_workspace cjoakimaml eastus2

echo 'done'
