"""
The purpose of this module is to interact with an Azure Machine Service
via the Python SDK (azureml-sdk).

Usage:
  python aml_client.py create_workspace cjoakimaml eastus2
  python aml_client.py get_workspace_details
  python aml_client.py connect_to_workspace subscription_id, resource_group, name
  -
  python aml_client.py create_compute_config
  python aml_client.py create_experiments_config
  -
  python aml_client.py create_compute_target compute11
  python aml_client.py delete_compute_target compute2
  python aml_client.py list_compute_vmsizes compute2
  -
  python aml_client.py create_compute_instance nb3
  python aml_client.py delete_compute_instance nb3
  python aml_client.py start_compute_instance nb3
  python aml_client.py stop_compute_instance nb3
  -
  python aml_client.py create_blob_datastore cjoakimstorage_aml aml
  python aml_client.py create_dataset_from_datastore default openflights_airlines.csv
  python aml_client.py create_dataset_from_datastore workspaceblobstore busiest_airports_2017.csv
  python aml_client.py create_dataset_from_datastore cjoakimstorage_aml postal_codes_us.csv
  python aml_client.py create_dataset_from_datastore cjoakimstorage_aml batch_locations.csv
  python aml_client.py create_dataset_from_datastore cjoakimstorage_aml iris/iris_data.csv
  python aml_client.py create_dataset_from_datastore cjoakimstorage_aml postal_codes_us_processed_training.csv
  python aml_client.py create_dataset_from_datastore cjoakimstorage_aml postal_codes_us_processed_testing.csv
  python aml_client.py create_dataset_from_url <url>
  -
  python aml_client.py list_datasets
  python aml_client.py list_datastores
  python aml_client.py list_environments > config/environments.txt
  python aml_client.py list_models
  -
  python aml_client.py set_secret AZURE_STORAGE_ACCOUNT
  python aml_client.py set_secrets
  -
  python aml_client.py submit_experiment hello.py --wait 
  python aml_client.py submit_experiment explore_env.py --wait
  python aml_client.py submit_experiment train_iris.py --wait
  python aml_client.py submit_experiment delete-blobs.py --wait
  python aml_client.py submit_experiment skl-knn-us-states-geo.py --wait --register-model
  -
  python aml_client.py submit_automl_experiment skl-knn-us-states-geo.py --wait
  -
  python aml_client.py submit_example_pipeline1 skl-knn-us-states-geo-pipeline1 --wait
  python aml_client.py submit_example_pipeline2 skl-knn-us-states-geo-pipeline2 --wait
  -
  python aml_client.py create_conda_dependencies_yml skl-knn-us-states-geo

  python aml_client.py deploy_model_to_aci skl-knn-us-states-geo 1
  python aml_client.py deploy_model_to_aci_v2 skl-knn-us-states-geo 3
  python aml_client.py deploy_model_to_aks skl-knn-us-states-geo 3 
  python aml_client.py deploy_model_to_local skl-knn-us-states-geo 3 
  -
  python aml_client.py list_published_pipelines
  python aml_client.py submit_published_pipeline Prepare-Postal-Code-Data-Pipeline
  -
  python aml_client.py batch_scoring_example
  -
  python aml_client.py submit_pytorch_example compute2
Options:
  -h --help     Show this screen.
  --version     Show version.
"""

__author__  = 'Chris Joakim'
__email__   = "chjoakim@microsoft.com,christopher.joakim@gmail.com"
__license__ = "None"
__version__ = "2021/02/03"

import json
import logging
import os
import time
import traceback
import sys

import numpy as np
import pandas as pd

from docopt import docopt

from sklearn.model_selection import train_test_split

from azureml.core import Dataset
from azureml.core import Datastore
from azureml.core import Environment
from azureml.core import Experiment
from azureml.core import Keyvault
from azureml.core import Model
from azureml.core import ScriptRunConfig
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import RunConfiguration

from azureml.core import Workspace
from azureml.core.environment import Environment as EEnvironment
from azureml.core.compute import AksCompute
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeInstance
from azureml.core.compute import ComputeTarget

from azureml.core.compute_target import ComputeTargetException

from azureml.core.resource_configuration import ResourceConfiguration

from azureml.core.model import InferenceConfig

from azureml.pipeline.core.schedule import Schedule
from azureml.pipeline.core.schedule import ScheduleRecurrence

from azureml.core.webservice import AksWebservice
from azureml.core.webservice import AciWebservice
from azureml.core.webservice import LocalWebservice
from azureml.core.webservice import Webservice
from azureml.data.datapath import DataPath

from azureml.pipeline.core import Pipeline
from azureml.pipeline.core import PipelineData
from azureml.pipeline.core import PublishedPipeline
from azureml.pipeline.core import ScheduleRecurrence
from azureml.pipeline.core import Schedule

from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.steps import EstimatorStep
from azureml.pipeline.steps import ParallelRunConfig
from azureml.pipeline.steps import ParallelRunStep

from azureml.train.automl import AutoMLConfig

from aml_helpers import AmlComputeConfig
from aml_helpers import AmlExperimentConfig

# https://docs.python.org/3/library/logging.html
# levels: CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET
logging.basicConfig(level=logging.WARNING)   


def create_workspace(rg_ws_name, region):
    subs = os.environ['AZURE_SUBSCRIPTION_ID']
    name = rg_ws_name
    rg   = rg_ws_name
    loc  = region
    print('creating aml workspace: {} in rg: {} in region: {} ...'.format(name, rg, loc))

    ws = Workspace.create(
        name=name,             # provide a name for your workspace
        subscription_id=subs,  # provide your subscription ID
        resource_group=rg,     # provide a resource group name
        location=loc,          # for example: 'westeurope' or 'eastus2' or 'westus2'
        create_resource_group=True)

    ws.write_config(path='.azureml')  # write to .azureml/config.json
    print('done')

def get_workspace_details():
    print('get_workspace_details')
    try:
        ws = get_workspace(True)
        envs = Environment.list(workspace=ws)
        env_strings = list()
        for env in envs:
            env_strings.append('')
            env_strings.append('--- environment-name: {}'.format(env))
            env_strings.append('')
            env_strings.append( envs[env].python.conda_dependencies.serialize_to_string())
        write('config/environments.txt', "\n".join(env_strings), verbose=True)
    except:
        traceback.print_exc()

def list_compute_vmsizes(name):
    print('list_compute_vmsizes: {}'.format(name))
    try:
        ws = get_workspace()
        ci = get_compute_instance(ws, name)
        vm_list = ci.supported_vmsizes(ws)
        write_json('config/supported_compute_vmsizes.json', vm_list)
    except:
        traceback.print_exc()

def create_compute_target(name):
    print('create_compute_target: {}'.format(name))
    compute_conf = AmlComputeConfig(name)
    print('using compute_conf: {}'.format(compute_conf.data))
    ws = get_workspace()
    try:
        cpu_cluster = ComputeTarget(workspace=ws, name=name)
        print('found existing compute target: {}'.format(name))
    except ComputeTargetException:
        print('creating compute target: {}'.format(name))
        compute_config = AmlCompute.provisioning_configuration(
            vm_size = compute_conf.vm_size(),
            min_nodes = compute_conf.min_nodes(),
            max_nodes = compute_conf.max_nodes(),
            idle_seconds_before_scaledown = compute_conf.idle_secs())
        print(compute_config)
        cpu_cluster = ComputeTarget.create(ws, name, compute_config)
        cpu_cluster.wait_for_completion(show_output=True)

def delete_compute_target(name):
    print('delete_compute_target: {}'.format(name))
    try:
        ws = get_workspace()
        c = AmlCompute(ws, name)
        print('c: {} {}'.format(c, str(type(c))))
        result = c.delete()
        print('delete result: {}'.format(result))
    except:
        traceback.print_exc()

def create_compute_instance(name):
    print('create_compute_instance: {}'.format(name))
    compute_conf = AmlComputeConfig(name)
    print('using compute_conf: {}'.format(compute_conf.data))
    ws = get_workspace()
    try:
        instance = ComputeInstance(workspace=ws, name=name)
        print('found existing compute instance: {}'.format(name))
    except ComputeTargetException:
        print('creating compute instance: {}'.format(name))
        compute_config = ComputeInstance.provisioning_configuration(
            vm_size = compute_conf.vm_size(),
            ssh_public_access = False
        )
        instance = ComputeInstance.create(ws, name, compute_config)
        instance.wait_for_completion(show_output=True)

def delete_compute_instance(name):
    print('delete_compute_instance: {}'.format(name))
    delete_compute_target(name)

def start_compute_instance(name):
    print('start_compute_instance: {}'.format(name))
    try:
        ws = get_workspace()
        ci = get_compute_instance(ws, name)
        print(json.dumps(ci.get_status().serialize(), sort_keys=True, indent=2))
        ci.start(wait_for_completion=True, show_output=True)
    except:
        traceback.print_exc()

def stop_compute_instance(name):
    print('stop_compute_instance: {}'.format(name))
    try:
        ws = get_workspace()
        ci = get_compute_instance(ws, name)
        print(json.dumps(ci.get_status().serialize(), sort_keys=True, indent=2))
        ci.stop(wait_for_completion=True, show_output=True)
    except:
        traceback.print_exc()

def create_blob_datastore(blob_datastore_name, container_name):
    print('create_blob_datastore: {} {}'.format(blob_datastore_name, container_name))
    ws = get_workspace()
    try:
        blob_datastore = Datastore.get(ws, blob_datastore_name)
        print("Found Blob Datastore with name: %s" % blob_datastore_name)
    except:
        account_name = os.getenv('AZURE_STORAGE_ACCOUNT')
        account_key  = os.getenv('AZURE_STORAGE_KEY')
        print('using storage account: {} key: {}'.format(account_name, account_key))

        blob_datastore = Datastore.register_azure_blob_container(
            workspace=ws,
            datastore_name=blob_datastore_name,
            account_name=account_name,
            container_name=container_name,
            account_key=account_key)
        print("Registered blob datastore with name: %s" % blob_datastore_name)

def create_dataset_from_datastore(datastore_name, file):
    print('create_dataset_from_file: {} {}'.format(datastore_name, file))
    try:
        ws = get_workspace()
        if datastore_name == 'default':
            datastore = ws.get_default_datastore()
        else:
            datastore = Datastore.get(ws, datastore_name)
        print('datastore: {}'.format(datastore))

        datastore_path = [(datastore, file)]
        ds = None
        print('datastore_path: {}'.format(datastore_path))
        if is_tabular_file(file):
            ds = Dataset.Tabular.from_delimited_files(path=datastore_path)
            ds.register(workspace=ws, name=file, description=file, create_new_version=False)
            print('ds: {}'.format(ds))
            df = ds.take(3).to_pandas_dataframe()
            print(df.head())
        else:
            ds = Dataset.File.from_files(path=datastore_path)
            ds.register(workspace=ws, name=file, description=file, create_new_version=False)
            print('ds: {}'.format(ds))
    except:
        traceback.print_exc()

def is_tabular_file(filename):
    if '.csv' in file.lower():
        return True 
    elif '.tsv' in file.lower():
        return True 
    elif '.parquet' in file.lower():
        return True 
    else:
        return False 

def create_dataset_from_url(url):
    print('create_dataset_from_url: {} '.format(url))
    try:
        ws = get_workspace()
        ds = None
        web_path = [ url ]
        if '.csv' in url:
            ds = Dataset.Tabular.from_delimited_files(path=web_path)
            ds.register(workspace=ws, name=url, description=url)
            print('ds: {}'.format(ds))
            df = ds.take(3).to_pandas_dataframe()
            print(df.head())
        else:
            ds = Dataset.File.from_files(path=web_path)
            ds.register(workspace=ws, name=url, description=url)
            print('ds: {}'.format(ds))
    except:
        traceback.print_exc()

def create_conda_dependencies_yml(name):
    pass 
    # instead; see src/azure-ml-tutorial.yml

    # # Add the dependencies for your model
    # myenv = CondaDependencies()
    # myenv.add_conda_package("scikit-learn")

    # # Save the environment config as a .yml file
    # env_file = 'service_files/env.yml'
    # with open(env_file,"w") as f:
    #     f.write(myenv.serialize_to_string())
    # print("Saved dependency info in", env_file)

def list_datasets():
    print('list_datasets')
    try:
        ws = get_workspace()
        dataset_dict = Dataset.get_all(ws)
        dataset_list = list()
        for name in sorted(dataset_dict.keys()):
            ds = dataset_dict[name]
            # ds is an instance of AbstractDataset; the parent of TabularDatasetFactory and FileDatasetFactory
            # https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.data.abstract_dataset.abstractdataset?view=azure-ml-py#attributes
            obj = dict()
            obj['name'] = name
            obj['id']   = ds.id
            obj['version'] = ds.version
            obj['description'] = ds.description
            obj['data_changed_time'] = str(ds.data_changed_time) # .strftime("%m/%d/%Y, %H:%M:%S")
            obj['tags'] = ds.tags
            print(obj)
            dataset_list.append(obj)
        write_json('datasets/dataset_list.json', dataset_list)
    except:
        traceback.print_exc()

def list_datastores():
    print('list_datastores')
    try:
        ws = get_workspace()
        for ds in sorted(ws.datastores):
            print(ds)
    except:
        traceback.print_exc()

def list_environments():
    print('list_environments')
    try:
        ws = get_workspace()
        envs = Environment.list(workspace=ws)
        for env in envs:
            print('---')
            print("name: ",env)
            print("packages: ", envs[env].python.conda_dependencies.serialize_to_string())
    except:
        traceback.print_exc()

def list_models():
    print('list_models')
    try:
        ws = get_workspace()
        models_list = list()
        for model in Model.list(ws):
            print('model - name: {}  version: {}'.format(model.name, model.version))
            models_list.append([model.name, model.version])
        write_json('logs/models_list.json', models_list)
    except:
        traceback.print_exc()

def list_published_pipelines():
    print('list_published_pipelines')
    try:
        ws = get_workspace()
        published_pipelines = PublishedPipeline.list(ws)
        for pp in  published_pipelines:
            print(pp)
    except:
        traceback.print_exc()

def submit_published_pipeline(name_or_id):
    try:
        ws = get_workspace()
        published_pipelines = PublishedPipeline.list(ws)
        for pp in  published_pipelines:
            if (pp.name.startswith(name_or_id)) or (pp.id == name_or_id):
                print('located published pipeline {}'.format(pp))
                sub_pp = PublishedPipeline.get(ws, pp.id)
                print(sub_pp)
                xname = sub_pp.name.split()[0]
                print('experiment name: {}'.format(xname))
                experiment = Experiment(workspace=ws, name=xname)
                pipeline_run = experiment.submit(
                    sub_pp,
                    continue_on_step_failure=True,
                    pipeline_parameters={"param1": "value1"})
    except:
        traceback.print_exc()

def submit_experiment(py_name):
    experiment_name = py_name.split('.')[0]
    experiment_config = AmlExperimentConfig(experiment_name)
    print('submit_experiment -> {} with {}'.format(experiment_name, experiment_config.data))
    compute_name = experiment_config.compute_name()
    env_name = experiment_config.env_name()
    args = experiment_config.args()
    max_seconds = experiment_config.max_run_duration_seconds()

    ws = get_workspace()
    experiment = Experiment(workspace=ws, name=experiment_name)

    config = ScriptRunConfig(
        source_directory='./src',
        script=py_name,
        compute_target=compute_name,
        arguments=args,
        max_run_duration_seconds=max_seconds)

    env = Environment.get(workspace=ws, name=env_name)
    config.run_config.environment = env

    run = experiment.submit(config)
    print('run submitted:')
    print('  py_name:      {}'.format(py_name))
    print('  compute_name: {}'.format(compute_name))
    print('  env_name:     {}'.format(env_name))
    print('  aml_url:      {}'.format(run.get_portal_url()))

    if boolean_flag_arg('--wait'):
        print('')
        print('////////////////////////////////////////////////////////////////////////////////')
        print('waiting for run completion ...')
        run.wait_for_completion(show_output=True)

        print('run completed:')
        status = run.get_status()
        print('  run id:     {}'.format(str(run.id)))
        print('  run name:   {}'.format(str(run.name)))
        print('  run number: {}'.format(str(run.number)))
        print('  run status: {}'.format(str(run.status)))
        print('  run type:   {}'.format(str(run.type)))

        print('getting run logs ...')
        run.get_all_logs(destination='logs/')

        print('getting run details ...')
        details = run.get_details_with_logs()
        outfile = 'logs/run_details_{}_{}.json'.format(experiment_name, run.number)
        write(outfile, str(details))

        print('getting run properties ...')
        artifact_name = 'properties_{}_{}'.format(experiment_name, run.number)
        save_run_json_artifact(artifact_name, run.get_properties())

        print('getting run metrics ...')
        artifact_name = 'metrics_{}_{}'.format(experiment_name, run.number)
        save_run_json_artifact(artifact_name, run.get_metrics())

        print('downloading run results.json ...')
        outfile = 'logs/run_results_{}_{}.json'.format(experiment_name, run.number)
        run.download_file('outputs/results.json', outfile)

        print('getting run file names ...')
        artifact_name = 'file_names_{}_{}'.format(experiment_name, run.number)
        save_run_json_artifact(artifact_name, run.get_file_names())

        if boolean_flag_arg('--register-model'):
            print('registering model ...')
            model = run.register_model(
                model_name='skl-knn-us-states-geo', 
                model_path='outputs/skl-knn-us-states-geo.joblib',
                model_framework=Model.Framework.SCIKITLEARN,
                model_framework_version='0.22.2.post1',
                resource_configuration=ResourceConfiguration(cpu=1, memory_in_gb=0.5))
            print('model: {}'.format(model))

def submit_automl_experiment(py_name):
    py_basename = py_name.split('.')[0]
    experiment_name = 'automl-{}'.format(py_basename)
    experiment_config = AmlExperimentConfig(experiment_name)
    print('submit_experiment -> {} with {}'.format(experiment_name, experiment_config.data))
    compute_name = experiment_config.compute_name()
    env_name = experiment_config.env_name()
    args = experiment_config.args()
    max_seconds = experiment_config.max_run_duration_seconds()

    if False:
        # create the training and testing dataframes, upload manually to blob, create Dataset 
        infile = 'datasets/postal_codes/postal_codes_us_processed.csv'
        df = read_pandas_df(infile)
        print('df type and shape: {} {}'.format(type(df), df.shape))
        print('df columns: {} '.format(df.columns.values))
        print_df_info(df, 'df')
        train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
        print(train_data.shape)
        print(test_data.shape)
        print(train_data.head(3))
        print(test_data.head(3))
        train_data.to_csv('datasets/postal_codes/postal_codes_us_processed_training.csv')
        test_data.to_csv('datasets/postal_codes/postal_codes_us_processed_testing.csv')

    ws = get_workspace()
    experiment = Experiment(workspace=ws, name=experiment_name)
    train_data = Dataset.get_by_name(ws, 'postal_codes_us_processed_training.csv')
    test_data  = Dataset.get_by_name(ws, 'postal_codes_us_processed_testing.csv')

    # SDK docs: https://docs.microsoft.com/en-us/python/api/azureml-train-automl-client/azureml.train.automl.automlconfig.automlconfig?view=azure-ml-py
    # primary_metric: 'AUC_weighted', 'accuracy'
    # featurization: 'off' or 'auto'
    automl_config=AutoMLConfig(
        compute_target = compute_name,
        task = 'classification',
        training_data = train_data,
        validation_data = test_data,
        label_column_name = 'state',
        primary_metric = 'accuracy',
        featurization = 'off',
        experiment_timeout_minutes = 15.0,
        enable_early_stopping = True)

    run = experiment.submit(automl_config, show_output=True)
    print('run submitted:')
    print('  experiment_name: {}'.format(experiment_name))
    print('  compute_name:    {}'.format(compute_name))
    print('  aml_url:         {}'.format(run.get_portal_url()))

    if boolean_flag_arg('--wait'):
        print('')
        print('////////////////////////////////////////////////////////////////////////////////')
        print('waiting for run completion ...')
        run.wait_for_completion(show_output=True)

        print('run completed:')
        status = run.get_status()
        print('  run id:     {}'.format(str(run.id)))
        print('  run name:   {}'.format(str(run.name)))
        print('  run number: {}'.format(str(run.number)))
        print('  run status: {}'.format(str(run.status)))
        print('  run type:   {}'.format(str(run.type)))

        print('getting run logs ...')
        run.get_all_logs(destination='logs/')

        print('getting run details ...')
        details = run.get_details_with_logs()
        outfile = 'logs/run_details_{}_{}.json'.format(experiment_name, run.number)
        write(outfile, str(details))

        print('getting run properties ...')
        artifact_name = 'properties_{}_{}'.format(experiment_name, run.number)
        save_run_json_artifact(artifact_name, run.get_properties())

        print('getting run metrics ...')
        artifact_name = 'metrics_{}_{}'.format(experiment_name, run.number)
        save_run_json_artifact(artifact_name, run.get_metrics())


        print('getting run file names ...')
        artifact_name = 'file_names_{}_{}'.format(experiment_name, run.number)
        save_run_json_artifact(artifact_name, run.get_file_names())

        # env = best_run.get_context().get_environment()
        # inference_config = InferenceConfig(entry_script='score.py', environment=env)

        print('getting run output ...')
        best_run, fitted_model = run.get_output()
        print('best run ...')
        print(best_run)
        print('fitted model ...')
        print(fitted_model)

        deploy_name = 'cjoakim-automl-aci-{}'.format(epoch())
        deployment_config = AciWebservice.deploy_configuration(
            cpu_cores=1, memory_gb=0.5, auth_enabled=True)

        web_service = Model.deploy(
            ws, deploy_name, [fitted_model], deployment_config=deployment_config)
        print('web_service: {}'.format(web_service))
        web_service.wait_for_deployment(show_output = True)
        print('web_service scoring_uri: {}'.format(web_service.scoring_uri))
        print('web_service swagger_uri: {}'.format(web_service.swagger_uri))
        primary, secondary = web_service.get_keys()
        print('primary   key: {}'.format(primary))
        print('secondary key: {}'.format(secondary))

def submit_example_pipeline1(pipeline_name):
    # See https://docs.microsoft.com/en-us/learn/modules/create-pipelines-in-aml/2-pipelines
    # See https://docs.microsoft.com/en-us/azure/machine-learning/how-to-move-data-in-out-of-pipelines
    # See https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-machine-learning-pipelines

    pipeline_config = AmlExperimentConfig(pipeline_name)
    print('pipeline_config {} -> {}'.format(pipeline_name, pipeline_config.data))
    compute_name = pipeline_config.compute_name()
    env_name = pipeline_config.env_name()
    max_seconds = pipeline_config.max_run_duration_seconds()

    ws = get_workspace()
    experiment = Experiment(workspace=ws, name=pipeline_name)

    data_store = ws.get_default_datastore()  # workspaceblobstore (Default)

    # Create two PipelineData objects simply to sequentially link
    # the three PythonScriptSteps in the Pipeline.

    pipeline_data_0_1 = PipelineData(
        name = "pipeline_data_0_1", 
        datastore=data_store,
        output_name="pipeline_data_0_1")

    pipeline_data_1_2 = PipelineData(
        name = "pipeline_data_1_2", 
        datastore=data_store,
        output_name="pipeline_data_1_2")

    run_config = RunConfiguration()
    run_config.target = compute_name
    run_config.environment = Environment.get(workspace=ws, name=env_name)

    step0 = PythonScriptStep(
        name = 'delete_blobs',
        source_directory = './src',
        script_name = 'delete-blobs.py',
        compute_target = compute_name,
        runconfig=run_config,
        arguments = [
            'postal_codes_us_prepared.csv', 
            'i.csv',
            'iris.csv',
            'not-there.csv'],
        outputs=[pipeline_data_0_1],
        allow_reuse = False)

    step1 = PythonScriptStep(
        name = 'data prep',
        source_directory = './src',
        script_name = 'data-prep-states-geo1.py',
        compute_target = compute_name,
        runconfig=run_config,
        arguments = [
            '--input-dataset', 'postal_codes_us.csv', 
            '--output-blobname', 'postal_codes_us_prepared.csv'],
        inputs=[pipeline_data_0_1],
        outputs=[pipeline_data_1_2],
        allow_reuse = False)

    step2 = PythonScriptStep(
        name = 'train model',
        source_directory = './src',
        script_name = 'skl-knn-us-states-geo1.py',
        compute_target = compute_name,
        runconfig=run_config,
        arguments = [
            '--input-blobname', 'postal_codes_us_prepared.csv'],
        inputs=[pipeline_data_1_2],
        allow_reuse = False)

    pipeline = Pipeline(workspace=ws, steps=[step0, step1, step2])
    
    run = experiment.submit(pipeline, regenerate_outputs=True)

    if boolean_flag_arg('--wait'):
        print('')
        print('////////////////////////////////////////////////////////////////////////////////')
        print('waiting for pipeline completion ...')
        run.wait_for_completion(show_output=True)

        print('run completed:')
        status = run.get_status()
        print('  run id:     {}'.format(str(run.id)))
        print('  run name:   {}'.format(str(run.name)))
        print('  run number: {}'.format(str(run.number)))
        print('  run status: {}'.format(str(run.status)))
        print('  run type:   {}'.format(str(run.type)))

        print('getting run logs ...')
        run.get_all_logs(destination='logs/')

        print('getting run details ...')
        details = run.get_details_with_logs()
        outfile = 'logs/run_details_{}_{}.json'.format(pipeline_name, run.number)
        write(outfile, str(details))

        print('getting run properties ...')
        artifact_name = 'properties_{}_{}'.format(pipeline_name, run.number)
        save_run_json_artifact(artifact_name, run.get_properties())

        print('getting run metrics ...')
        artifact_name = 'metrics_{}_{}'.format(pipeline_name, run.number)
        save_run_json_artifact(artifact_name, run.get_metrics())

        print('getting run file names ...')
        artifact_name = 'file_names_{}_{}'.format(pipeline_name, run.number)
        save_run_json_artifact(artifact_name, run.get_file_names())

def submit_example_pipeline2(pipeline_name):
    pipeline_config = AmlExperimentConfig(pipeline_name)
    print('pipeline_config {} -> {}'.format(pipeline_name, pipeline_config.data))
    compute_name = pipeline_config.compute_name()
    env_name = pipeline_config.env_name()
    max_seconds = pipeline_config.max_run_duration_seconds()

    ws = get_workspace()
    experiment = Experiment(workspace=ws, name=pipeline_name)

    data_store = ws.get_default_datastore()

    # Create two PipelineData objects simply to sequentially link
    # the three PythonScriptSteps in the Pipeline.

    pipeline_data_0_1 = PipelineData(
        name = "pipeline_data_0_1", 
        datastore=data_store,
        output_name="pipeline_data_0_1")

    pipeline_data_1_2 = PipelineData(
        name = "pipeline_data_1_2", 
        datastore=data_store,
        output_name="pipeline_data_1_2",
        output_mode="upload", 
        output_path_on_compute="data/prepped.csv")

    run_config = RunConfiguration()
    run_config.target = compute_name
    run_config.environment = Environment.get(workspace=ws, name=env_name)

    step0 = PythonScriptStep(
        name = 'delete_blobs',
        source_directory = './src',
        script_name = 'delete-blobs.py',
        compute_target = compute_name,
        runconfig=run_config,
        arguments = [
            'postal_codes_us_prepared.csv', 
            'i.csv',
            'iris.csv',
            'not-there.csv'],
        outputs=[pipeline_data_0_1],
        allow_reuse = False)

    step1 = PythonScriptStep(
        name = 'data prep',
        source_directory = './src',
        script_name = 'data-prep-states-geo2.py',
        compute_target = compute_name,
        runconfig=run_config,
        arguments = [
            '--input-dataset', 'postal_codes_us.csv', 
            '--output-path', pipeline_data_1_2],
        inputs=[pipeline_data_0_1],
        outputs=[pipeline_data_1_2],
        allow_reuse = False)

    step2 = PythonScriptStep(
        name = 'train model',
        source_directory = './src',
        script_name = 'skl-knn-us-states-geo2.py',
        compute_target = compute_name,
        runconfig=run_config,
        arguments = [
            '--input', pipeline_data_1_2],
        inputs=[pipeline_data_1_2.as_download()],
        allow_reuse = False)

    pipeline = Pipeline(workspace=ws, steps=[step0, step1, step2])
    
    run = experiment.submit(pipeline, regenerate_outputs=True)

    if boolean_flag_arg('--wait'):
        print('')
        print('////////////////////////////////////////////////////////////////////////////////')
        print('waiting for pipeline completion ...')
        run.wait_for_completion(show_output=True)

        print('run completed:')
        status = run.get_status()
        print('  run id:     {}'.format(str(run.id)))
        print('  run name:   {}'.format(str(run.name)))
        print('  run number: {}'.format(str(run.number)))
        print('  run status: {}'.format(str(run.status)))
        print('  run type:   {}'.format(str(run.type)))

        print('getting run logs ...')
        run.get_all_logs(destination='logs/')

        print('getting run details ...')
        details = run.get_details_with_logs()
        outfile = 'logs/run_details_{}_{}.json'.format(pipeline_name, run.number)
        write(outfile, str(details))

        print('getting run properties ...')
        artifact_name = 'properties_{}_{}'.format(pipeline_name, run.number)
        save_run_json_artifact(artifact_name, run.get_properties())

        print('getting run metrics ...')
        artifact_name = 'metrics_{}_{}'.format(pipeline_name, run.number)
        save_run_json_artifact(artifact_name, run.get_metrics())

        print('getting run file names ...')
        artifact_name = 'file_names_{}_{}'.format(pipeline_name, run.number)
        save_run_json_artifact(artifact_name, run.get_file_names())

def deploy_model_to_aci(name, version):
    print('deploy_model_to_aci; name: {}, version: {}'.format(name, version))
    try:
        ws = get_workspace()
        model = get_registered_model(ws, name, version)
        print('model: {} {}'.format(model, str(type(model))))
        deploy_name = 'cjoakim-aml-aci-{}'.format(epoch())
        print('deploying model to ACI: {}'.format(deploy_name))
        deployment_config = AciWebservice.deploy_configuration(
            cpu_cores=1, memory_gb=0.5, auth_enabled=True)

        web_service = Model.deploy(ws, deploy_name, [model], deployment_config=deployment_config)
        print('web_service: {}'.format(web_service))
        web_service.wait_for_deployment(show_output = True)
        print('web_service scoring_uri: {}'.format(web_service.scoring_uri))
        print('web_service swagger_uri: {}'.format(web_service.swagger_uri))
        primary, secondary = web_service.get_keys()
        print('primary   key: {}'.format(primary))
        print('secondary key: {}'.format(secondary))
    except:
        traceback.print_exc()
        return None

def deploy_model_to_aci_v2(name, version):
    print('deploy_model_to_aci_v2; name: {}, version: {}'.format(name, version))
    try:
        ws = get_workspace()
        model = get_registered_model(ws, name, version)
        print('model: {} {}'.format(model, str(type(model))))
        deploy_name = 'cjoakim-aml-aci-{}'.format(epoch())
        print('deploying model to ACI v2: {}'.format(deploy_name))

        deployment_config = AciWebservice.deploy_configuration(
            cpu_cores=1, memory_gb=0.5, auth_enabled=True)

        inference_config = InferenceConfig(
            runtime= "python",
            source_directory = 'src',
            entry_script="score_skl_knn_us_states_geo.py",
            conda_file='azure-ml-tutorial.yml')  # <-- sliced out of environments.txt

        web_service = Model.deploy(
            workspace=ws,
            name = deploy_name,
            models = [model],
            deployment_config = deployment_config,
            inference_config = inference_config)
        print('web_service: {}'.format(web_service))

        web_service.wait_for_deployment(show_output = True)
        print(web_service.get_logs())
        print('web_service scoring_uri: {}'.format(web_service.scoring_uri))
        print('web_service swagger_uri: {}'.format(web_service.swagger_uri))
        primary, secondary = web_service.get_keys()
        print('primary   key: {}'.format(primary))
        print('secondary key: {}'.format(secondary))
        print(web_service.get_logs())

    except:
        traceback.print_exc()
        return None

def deploy_model_to_aks(name, version):
    print('deploy_model_to_aks; name: {}, version: {}'.format(name, version))
    try:
        ws = get_workspace()

        # Create an AKS cluster
        cluster_name = 'cjoakimamlaks2'
        print('creating AKS cluster: {}'.format(cluster_name))
        compute_config = AksCompute.provisioning_configuration(location='eastus2')
        aks_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
        aks_cluster.wait_for_completion(show_output=True)

        model = get_registered_model(ws, name, version)
        print('model: {} {}'.format(model, str(type(model))))
        deploy_name = 'cjoakim-aml-aci-{}'.format(epoch())
        print('deploying model to AKS: {}'.format(deploy_name))

        deployment_config = AksWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

        web_service = Model.deploy(
            ws,
            deploy_name, 
            [model],
            deployment_config=deployment_config,
            deployment_target = aks_cluster)

        print('web_service: {}'.format(web_service))
        web_service.wait_for_deployment(show_output = True)
        print('web_service scoring_uri: {}'.format(web_service.scoring_uri))
        print('web_service swagger_uri: {}'.format(web_service.swagger_uri))
        primary, secondary = web_service.get_keys()
        print('primary   key: {}'.format(primary))
        print('secondary key: {}'.format(secondary))
        print(web_service.state)
        print(web_service.get_logs())
    except:
        traceback.print_exc()
        return None

def deploy_model_to_local(name, version):
    print('deploy_model_to_local; name: {}, version: {}'.format(name, version))
    try:
        ws = get_workspace()
        model = get_registered_model(ws, name, version)
        print('model: {} {}'.format(model, str(type(model))))
        deploy_name = 'local-cjoakim-aml-aci-{}'.format(epoch())
        print('deploying model to ACI: {}'.format(deploy_name))

        deployment_config = LocalWebservice.deploy_configuration(port=8890)

        web_service = Model.deploy(ws, deploy_name, [model], deployment_config=deployment_config)
        print('web_service: {}'.format(web_service))
        web_service.wait_for_deployment(show_output = True)
        print('web_service scoring_uri: {}'.format(web_service.scoring_uri))
        print('web_service swagger_uri: {}'.format(web_service.swagger_uri))
        primary, secondary = web_service.get_keys()
        print('primary   key: {}'.format(primary))
        print('secondary key: {}'.format(secondary))

        # $ docker ps
        # CONTAINER ID   IMAGE                              COMMAND                 CREATED         STATUS         PORTS                                                           NAMES
        # 074d165380ca   local-cjoakim-aml-aci-1608756512   "runsvdir /var/runit"   3 minutes ago   Up 3 minutes   8888/tcp, 127.0.0.1:8890->5001/tcp, 127.0.0.1:55000->8883/tcp   silly_villani
        # $ docker stop -t 1 silly_villani 
    except:
        traceback.print_exc()
        return None

def batch_scoring_example():
    # this isn't working yet, 12/23
    # https://docs.microsoft.com/en-us/learn/modules/deploy-batch-inference-pipelines-with-azure-machine-learning/2-batch-inference-pipelines
    try:
        ws = get_workspace()
        # Get the batch dataset for input
        batch_data_set = ws.datasets['batch_locations.csv']

        # Set the output location
        default_ds = ws.get_default_datastore()
        output_dir = PipelineData(
            name='batch_inferences',
            datastore=default_ds,
            output_path_on_compute='results')

        #env = Environment.get(workspace=ws, name="AzureML-Minimal")
        env = Environment.from_conda_specification(
            name='batch-env',
            file_path='src/azure-ml-tutorial.yml')

        # Define the parallel run step step configuration
        parallel_run_config = ParallelRunConfig(
            source_directory='src',
            entry_script='score_skl_knn_us_states_geo_batch.py',
            mini_batch_size='14',
            error_threshold=10,
            output_action="append_row",
            environment=env,
            compute_target='compute3',
            node_count=1)

        # Create the parallel run step
        parallelrun_step = ParallelRunStep(
            name='batch-score',
            parallel_run_config=parallel_run_config,
            inputs=[batch_data_set.as_named_input('batch_data')],
            output=output_dir,
            arguments=[],
            allow_reuse=True
        )
        # Create the pipeline
        pipeline = Pipeline(workspace=ws, steps=[parallelrun_step])

        # Run the pipeline as an experiment
        pipeline_run = Experiment(ws, 'batch_prediction_pipeline').submit(pipeline)
        pipeline_run.wait_for_completion(show_output=True)

        # Get the outputs from the first (and only) step
        prediction_run = next(pipeline_run.get_children())
        prediction_output = prediction_run.get_output_data('inferences')
        prediction_output.download(local_path='results')

        # Find the parallel_run_step.txt file
        for root, dirs, files in os.walk('results'):
            for file in files:
                if file.endswith('parallel_run_step.txt'):
                    result_file = os.path.join(root,file)

        # Load and display the results
        df = pd.read_csv(result_file, delimiter=":", header=None)
        df.columns = ["File", "Prediction"]
        print(df)
    except:
        traceback.print_exc()
        return None


def submit_pytorch_example(compute_name):
    print('submit_pytorch_example on {}'.format(compute_name))
    ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name='day1-experiment-train')
    config = ScriptRunConfig(
        source_directory='src', 
        script='day1-experiment-train.py',
        compute_target=compute_name)

    # set up pytorch environment
    env = Environment.from_conda_specification(
        name='pytorch-env',
        file_path='.azureml/pytorch-env.yml')
    config.run_config.environment = env

    run = experiment.submit(config)
    aml_url = run.get_portal_url()
    print('aml_url: {}'.format(aml_url))

def connect_to_workspace(subscription_id, resource_group, name):
    print('connect_to_workspace; subs: {} rg: {} name: {}'.format(
        subscription_id, resource_group, name))

    ws = get_workspace_from_params(name, subscription_id, resource_group)
    print('ws: {} {}'.format(ws.name, ws))

    ws.write_config(path='.azureml')  # write to .azureml/config.json
    print('done')

# AML Helper methods

def get_workspace(capture=False):
    ws = Workspace.from_config()  # This automatically looks in directory .azureml
    #print('get_workspace: {} {}'.format(ws, str(type(ws))))  # <class 'azureml.core.workspace.Workspace'>
    print('get_workspace: {}'.format(ws.name))
    if capture:
        write_json('tmp/workspace_details.json', ws.get_details())
        env = Environment.get(workspace=ws, name="AzureML-Minimal")
        print(env)
    return ws

def get_workspace_from_params(name, subscription_id, resource_group):
    ws = Workspace.get(name=name,
                   subscription_id=subscription_id,
                   resource_group=resource_group)
    print('get_workspace_from_params: {}'.format(ws.name))
    return ws 

def get_compute_instance(ws, name):
    ci = ComputeInstance(workspace=ws, name=name)  # this returns <class 'azureml.core.compute.amlcompute.AmlCompute'> instance !?
    print('get_compute_instance: {} {}'.format(ci, str(type(ci))))
    return ci

def get_registered_model(ws, name, version):
    try:
        name_version = '{}:{}'.format(name, version)
        print('get_registered_model; name: {}, version: {}'.format(name, version))
        return Model(ws, name=name, version=version)
    except:
        traceback.print_exc()
        return None

def set_secrets():
    set_secret('AZURE_SUBSCRIPTION_ID')
    set_secret('AZURE_STORAGE_ACCOUNT')
    set_secret('AZURE_STORAGE_CONNECTION_STRING')
    set_secret('AZURE_STORAGE_KEY')
    set_secret('AZURE_ML_NAME')
    set_secret('AZURE_ML_RG')
    set_secret('AZURE_ML_REGION')

def set_secret(envvar_name):
    print('set_secret: {}'.format(envvar_name))
    try:
        ws = get_workspace()
        value = os.environ.get(envvar_name)
        print('value: {}'.format(value))
        if value:
            scrubbed_name = envvar_name.replace('_','-')
            keyvault = ws.get_default_keyvault()
            keyvault.set_secret(name=scrubbed_name, value=value)
            print('secret set: {}'.format(scrubbed_name))
    except:
        traceback.print_exc()

def prune_this_code():
    # python aml_client.py prune_this_code
    omit_tokens = 'print(,xxx,xxx,xxx'.split(',')
    lines = read_text_file('aml_client.py', False)
    for line in lines:
        stripped = line.strip()
        keep_line = True
        for token in omit_tokens:
            if token in line:
                keep_line = False
        if stripped.startswith('def '):
            print('')
        if stripped.startswith('#'):
            keep_line = False 
        if len(stripped) < 1:
            keep_line = False 
        if keep_line:
            print(line)

# IO Helper methods

def read_text_file(infile, do_strip=True):
    lines = list()
    with open(infile, 'rt') as f:
        for idx, line in enumerate(f):
            if do_strip:
                lines.append(line.strip())
            else:
                lines.append(line.rstrip())
    return lines

def read_json(infile):
    with open(infile, 'rt') as f:
        return json.loads(f.read())


def write_json(outfile, obj, verbose=True):
    jstr = json.dumps(obj, sort_keys=True, indent=2)
    write(outfile, jstr, verbose)

def write(outfile, s, verbose=True):
    with open(outfile, 'w') as f:
        f.write(s)
        if verbose:
            print('file written: {}'.format(outfile))

def save_run_json_artifact(name, obj):
    try:
        outfile = 'logs/run_{}.json'.format(name)
        write_json(outfile, obj, verbose=True)
    except:
        traceback.print_exc()

def read_pandas_df(infile, delim=',', colnames=None):
    if colnames:
        return pd.read_csv(infile, delimiter=delim, names=colnames)
    else:
        return pd.read_csv(infile, delimiter=delim, header=0)

def print_df_info(df, msg=''):
    print('')
    print('print_df_info: {}'.format(msg))
    try:
        print(df.head(3))
        print(df.tail(3))
        print(df.dtypes)
        print(df.shape)
        #print(df.columns)
    except:
        print('df is None or invalid')

# Other Helper methods
   
def epoch():
    return int(time.time())

def keyword_arg(keyword, default_value):
    for idx, arg in enumerate(sys.argv):
        if arg == keyword:
            next_idx = idx + 1
            if next_idx < len(sys.argv):
                return sys.argv[next_idx]
    return default_value

def boolean_flag_arg(flag):
    for arg in sys.argv:
        if arg == flag:
            return True
    return False

def print_options(msg):
    print(msg)
    arguments = docopt(__doc__, version=__version__)
    print(arguments)


if __name__ == "__main__":
    #print('aml_client.py args: {}'.format(sys.argv))

    if len(sys.argv) > 1:
        func = sys.argv[1].lower()

        if func == 'create_compute_config':
            AmlComputeConfig.create_config_file()

        elif func == 'create_experiments_config':
            AmlExperimentConfig.create_config_file()

        elif func == 'create_workspace':
            rg_ws_name = sys.argv[2]
            region = sys.argv[3]
            create_workspace(rg_ws_name, region)

        elif func == 'connect_to_workspace':
            subscription_id = sys.argv[2]
            resource_group = sys.argv[3]
            name = sys.argv[4]
            connect_to_workspace(subscription_id, resource_group, name)

        elif func == 'get_workspace_details':
            get_workspace_details()

        elif func == 'list_compute_vmsizes':
            name = sys.argv[2]
            list_compute_vmsizes(name)

        elif func == 'create_compute_target':
            name = sys.argv[2]
            create_compute_target(name)

        elif func == 'delete_compute_target':
            name = sys.argv[2]
            delete_compute_target(name)

        elif func == 'create_compute_instance':
            name = sys.argv[2]
            create_compute_instance(name)

        elif func == 'delete_compute_instance':
            name = sys.argv[2]
            delete_compute_instance(name)

        elif func == 'start_compute_instance':
            name = sys.argv[2]
            start_compute_instance(name)

        elif func == 'stop_compute_instance':
            name = sys.argv[2]
            stop_compute_instance(name)

        elif func == 'submit_experiment':
            py_name = sys.argv[2]
            submit_experiment(py_name)

        elif func == 'submit_automl_experiment':
            py_name = sys.argv[2]
            submit_automl_experiment(py_name)

        elif func == 'submit_example_pipeline1':
            name = sys.argv[2]
            submit_example_pipeline1(name)

        elif func == 'submit_example_pipeline2':
            name = sys.argv[2]
            submit_example_pipeline2(name)

        elif func == 'submit_pytorch_example':
            compute_name = sys.argv[2]
            submit_pytorch_example(compute_name)

        elif func == 'create_blob_datastore':
            blob_datastore_name = sys.argv[2]
            container_name = sys.argv[3]
            create_blob_datastore(blob_datastore_name, container_name)

        elif func == 'create_dataset_from_datastore':
            datastore_name = sys.argv[2]
            file = sys.argv[3]
            create_dataset_from_datastore(datastore_name, file)

        elif func == 'create_dataset_from_url':
            url = sys.argv[2]
            create_dataset_from_url(url)

        elif func == 'create_conda_dependencies_yml':
            name = sys.argv[2]
            create_conda_dependencies_yml(name)

        elif func == 'list_datasets':
            list_datasets()

        elif func == 'list_datastores':
            list_datastores()  

        elif func == 'list_environments':
            list_environments()  

        elif func == 'list_models':
            list_models() 

        elif func == 'list_published_pipelines':
            list_published_pipelines() 

        elif func == 'submit_published_pipeline':
            name = sys.argv[2]
            submit_published_pipeline(name) 

        elif func == 'set_secret':
            envvar_name = sys.argv[2]
            set_secret(envvar_name)

        elif func == 'set_secrets':
            set_secrets()

        elif func == 'deploy_model_to_aci':
            name, version = sys.argv[2], sys.argv[3]
            deploy_model_to_aci(name, version)

        elif func == 'deploy_model_to_aci_v2':
            name, version = sys.argv[2], sys.argv[3]
            deploy_model_to_aci_v2(name, version)

        elif func == 'deploy_model_to_aks':
            name, version = sys.argv[2], sys.argv[3]
            deploy_model_to_aks(name, version)

        elif func == 'deploy_model_to_local':
            name, version = sys.argv[2], sys.argv[3]
            deploy_model_to_local(name, version)

        elif func == 'prune_this_code':
            prune_this_code()

        elif func == 'batch_scoring_example':
            batch_scoring_example()
        else:
            print_options('Error: invalid function: {}'.format(func))

    else:
        print_options('Error: no function argument provided.')
