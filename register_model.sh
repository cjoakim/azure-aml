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
run_id='skl-knn-us-states-geo_1607883224_23b1eb2d'

az ml model register \
    --workspace-name $workspace \
    --resource-group $resource_group \
    --name $model_name \
    --asset-path $model_path \
    --model-framework $model_framework \
    --model-framework-version $model_framework_version \
    --experiment-name $experiment_name \
    --run-id $run_id \
    --tag author=chris

echo 'done'

# Output:
# {
#   "cpu": "",
#   "createdTime": "2020-12-13T18:32:55.719758+00:00",
#   "description": "",
#   "experimentName": "skl-knn-us-states-geo",
#   "framework": "ScikitLearn",
#   "frameworkVersion": "0.22.2.post1",
#   "gpu": "",
#   "id": "skl-knn-us-states-geo:1",
#   "memoryInGB": "",
#   "name": "skl-knn-us-states-geo",
#   "properties": "",
#   "runId": "skl-knn-us-states-geo_1607883224_23b1eb2d",
#   "sampleInputDatasetId": "",
#   "sampleOutputDatasetId": "",
#   "tags": {
#     "author": "chris"
#   },
#   "version": 1
# }

# Help Content:
# $ az ml model register --help
# This command is from the following extension: azure-cli-ml

# Command
#     az ml model register : Register a model to the workspace.

# Arguments
#     --name -n       [Required] : Name of model to register.
#     --asset-path               : The cloud path where the experiement run stores the model file.
#     --cc --cpu-cores           : The default number of CPU cores to allocate for this model. Can be
#                                  a decimal.
#     --description -d           : Description of the model.
#     --experiment-name          : The name of the experiment.
#     --gb --memory-gb           : The default amount of memory (in GB) to allocate for this model.
#                                  Can be a decimal.
#     --gc --gpu-cores           : The default number of GPUs to allocate for this model.
#     --model-framework          : Framework of the model to register. Currently supported frameworks:
#                                  TensorFlow, ScikitLearn, Onnx, Custom.
#     --model-framework-version  : Framework version of the model to register (e.g. 1.0.0, 2.4.1).
#     --model-path -p            : Full path of the model file to register.
#     --output-metadata-file -t  : Path to a JSON file where model registration metadata will be
#                                  written. Used as input for model deployment.
#     --path                     : Path to a project folder. Default: current directory.
#     --property                 : Key/value property to add (e.g. key=value ). Multiple properties
#                                  can be specified with multiple --property options.
#     --resource-group -g        : Resource group corresponding to the provided workspace.
#     --run-id -r                : The ID for the experiment run where model is registered from.
#     --run-metadata-file -f     : Path to a JSON file containing experiement run metadata.
#     --sample-input-dataset-id  : The ID for the sample input dataset.
#     --sample-output-dataset-id : The ID for the sample output dataset.
#     --subscription-id          : Specifies the subscription Id.
#     --tag                      : Key/value tag to add (e.g. key=value ). Multiple tags can be
#                                  specified with multiple --tag options.
#     --workspace-name -w        : Name of the workspace to register this model with.
#     -v                         : Verbosity flag.

# Global Arguments
#     --debug                    : Increase logging verbosity to show all debug logs.
#     --help -h                  : Show this help message and exit.
#     --only-show-errors         : Only show errors, suppressing warnings.
#     --output -o                : Output format.  Allowed values: json, jsonc, none, table, tsv,
#                                  yaml, yamlc.  Default: json.
#     --query                    : JMESPath query string. See http://jmespath.org/ for more
#                                  information and examples.
#     --subscription             : Name or ID of subscription. You can configure the default
#                                  subscription using `az account set -s NAME_OR_ID`.
#     --verbose                  : Increase logging verbosity. Use --debug for full debug logs.

# For more specific examples, use: az find "az ml model register"
