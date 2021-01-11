#!/bin/bash

# Register a model created as part of of an AML Experiment run.
# Chris Joakim, Microsoft, 2021/01/11

# See https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where?tabs=azcli#registermodel

# az extension add -n azure-cli-ml
# az login
# az account set -s $AZURE_SUBSCRIPTION_ID

workspace='cjoakimaml'
resource_group='cjoakimaml'
model_name='skl-knn-us-states-geo'
model_path='outputs/skl-knn-us-states-geo.joblib'
model_framework='ScikitLearn'
model_framework_version='0.22.2.post1'  # same as pip version?
experiment_name='skl-knn-us-states-geo'
run_id='skl-knn-us-states-geo_1607981189_753c1911'

az ml model deploy -m mymodel:1 --ic inferenceconfig.json --dc deployment_config_aci.json
