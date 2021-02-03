#!/bin/bash

# Create a new Azure Machine Learning service, with assotiated other services, 
# in a RG of the same name, with the Python SDK.  Also create compute instance,
# compute target, blob statstore, datasets, set secrets, and submit two simple
# experiments.
# Chris Joakim, Microsoft, 2021/02/03

rm -rf .azureml/

mkdir outputs

echo '---'
echo 'create_workspace...'
python aml_client.py create_workspace cjoakimaml eastus

sleep 30

echo '---'
echo 'create_compute_instance (for notebooks)...'
python aml_client.py create_compute_instance nb3

echo '---'
echo 'create_compute_target (for experiments)...'
python aml_client.py create_compute_target compute3

echo '---'
echo 'create_blob_datastore...'
python aml_client.py create_blob_datastore cjoakimstorage_aml aml

echo '---'
echo 'create_datasets_from_datastore'
python aml_client.py create_dataset_from_datastore cjoakimstorage_aml postal_codes_us.csv
python aml_client.py create_dataset_from_datastore cjoakimstorage_aml batch_locations.csv
python aml_client.py create_dataset_from_datastore cjoakimstorage_aml iris/iris_data.csv
python aml_client.py create_dataset_from_datastore cjoakimstorage_aml postal_codes_us_processed_training.csv
python aml_client.py create_dataset_from_datastore cjoakimstorage_aml postal_codes_us_processed_testing.csv

echo '---'
echo 'list_datastores...'
python aml_client.py list_datastores

echo '---'
echo 'list_datasets...'
python aml_client.py list_datasets

echo '---'
echo 'set_secrets (in keyvault)...'
python aml_client.py set_secrets

sleep 30

echo '---'
echo 'submit_experiment hello'
python aml_client.py submit_experiment hello.py --wait

echo '---'
echo 'submit_experiment explore_env'
python aml_client.py submit_experiment explore_env.py --wait

echo 'done'
