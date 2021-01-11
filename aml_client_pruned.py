
import json
import logging
import numpy as np
import os
import pandas as pd
import sys
import time
import traceback

from azureml.core import Dataset
from azureml.core import Datastore
from azureml.core import Environment
from azureml.core import Experiment
from azureml.core import Keyvault
from azureml.core import Model
from azureml.core import ScriptRunConfig
from azureml.core import Workspace

from azureml.core.compute import AksCompute
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeInstance
from azureml.core.compute import ComputeTarget

from azureml.core.compute_target import ComputeTargetException

from azureml.core.conda_dependencies import CondaDependencies

from azureml.core.environment import Environment as EEnvironment

from azureml.core.model import InferenceConfig

from azureml.core.resource_configuration import ResourceConfiguration

from azureml.core.runconfig import RunConfiguration

from azureml.core.webservice import AciWebservice
from azureml.core.webservice import AksWebservice
from azureml.core.webservice import LocalWebservice
from azureml.core.webservice import Webservice

from azureml.data.datapath import DataPath

from azureml.pipeline.core import Pipeline
from azureml.pipeline.core import PipelineData
from azureml.pipeline.core import Schedule
from azureml.pipeline.core import ScheduleRecurrence

from azureml.pipeline.steps import EstimatorStep
from azureml.pipeline.steps import ParallelRunConfig
from azureml.pipeline.steps import ParallelRunStep
from azureml.pipeline.steps import PythonScriptStep

from aml_helpers import AmlComputeConfig
from aml_helpers import AmlExperimentConfig

logging.basicConfig(level=logging.WARNING)

def create_workspace(rg_ws_name, region):
    subs = os.environ['AZURE_SUBSCRIPTION_ID']
    name = rg_ws_name
    rg   = rg_ws_name
    loc  = region
    ws = Workspace.create(
        name=name,             # provide a name for your workspace
        subscription_id=subs,  # provide your subscription ID
        resource_group=rg,     # provide a resource group name
        location=loc,          # for example: 'westeurope' or 'eastus2' or 'westus2'
        create_resource_group=True)
    ws.write_config(path='.azureml')  # write to .azureml/config.json

def get_workspace_details():
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
    try:
        ws = get_workspace()
        ci = get_compute_instance(ws, name)
        vm_list = ci.supported_vmsizes(ws)
        write_json('config/supported_compute_vmsizes.json', vm_list)
    except:
        traceback.print_exc()

def create_compute_target(name):
    compute_conf = AmlComputeConfig(name)
    ws = get_workspace()
    try:
        cpu_cluster = ComputeTarget(workspace=ws, name=name)
    except ComputeTargetException:
        compute_config = AmlCompute.provisioning_configuration(
            vm_size = compute_conf.vm_size(),
            min_nodes = compute_conf.min_nodes(),
            max_nodes = compute_conf.max_nodes(),
            idle_seconds_before_scaledown = compute_conf.idle_secs())
        cpu_cluster = ComputeTarget.create(ws, name, compute_config)
        cpu_cluster.wait_for_completion(show_output=True)

def delete_compute_target(name):
    try:
        ws = get_workspace()
        c = AmlCompute(ws, name)
        result = c.delete()
    except:
        traceback.print_exc()

def create_compute_instance(name):
    compute_conf = AmlComputeConfig(name)
    ws = get_workspace()
    try:
        instance = ComputeInstance(workspace=ws, name=name)
    except ComputeTargetException:
        compute_config = ComputeInstance.provisioning_configuration(
            vm_size = compute_conf.vm_size(),
            ssh_public_access = False
        )
        instance = ComputeInstance.create(ws, name, compute_config)
        instance.wait_for_completion(show_output=True)

def delete_compute_instance(name):
    delete_compute_target(name)

def start_compute_instance(name):
    try:
        ws = get_workspace()
        ci = get_compute_instance(ws, name)
        ci.start(wait_for_completion=True, show_output=True)
    except:
        traceback.print_exc()

def stop_compute_instance(name):
    try:
        ws = get_workspace()
        ci = get_compute_instance(ws, name)
        ci.stop(wait_for_completion=True, show_output=True)
    except:
        traceback.print_exc()

def create_blob_datastore(blob_datastore_name, container_name):
    ws = get_workspace()
    try:
        blob_datastore = Datastore.get(ws, blob_datastore_name)
    except:
        account_name = os.getenv('AZURE_STORAGE_ACCOUNT')
        account_key  = os.getenv('AZURE_STORAGE_KEY')
        blob_datastore = Datastore.register_azure_blob_container(
            workspace=ws,
            datastore_name=blob_datastore_name,
            account_name=account_name,
            container_name=container_name,
            account_key=account_key)

def create_dataset_from_datastore(datastore_name, file):
    try:
        ws = get_workspace()
        if datastore_name == 'default':
            datastore = ws.get_default_datastore()
        else:
            datastore = Datastore.get(ws, datastore_name)
        datastore_path = [(datastore, file)]
        ds = None
        if is_tabular_file(file):
            ds = Dataset.Tabular.from_delimited_files(path=datastore_path)
            ds.register(workspace=ws, name=file, description=file, create_new_version=False)
            df = ds.take(3).to_pandas_dataframe()
        else:
            ds = Dataset.File.from_files(path=datastore_path)
            ds.register(workspace=ws, name=file, description=file, create_new_version=False)
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
    try:
        ws = get_workspace()
        ds = None
        web_path = [ url ]
        if '.csv' in url:
            ds = Dataset.Tabular.from_delimited_files(path=web_path)
            ds.register(workspace=ws, name=url, description=url)
            df = ds.take(3).to_pandas_dataframe()
        else:
            ds = Dataset.File.from_files(path=web_path)
            ds.register(workspace=ws, name=url, description=url)
    except:
        traceback.print_exc()

def create_conda_dependencies_yml(name):
    pass

def list_datasets():
    try:
        ws = get_workspace()
        dataset_dict = Dataset.get_all(ws)
        dataset_list = list()
        for name in sorted(dataset_dict.keys()):
            ds = dataset_dict[name]
            obj = dict()
            obj['name'] = name
            obj['id']   = ds.id
            obj['version'] = ds.version
            obj['description'] = ds.description
            obj['data_changed_time'] = str(ds.data_changed_time) # .strftime("%m/%d/%Y, %H:%M:%S")
            obj['tags'] = ds.tags
            dataset_list.append(obj)
        write_json('datasets/dataset_list.json', dataset_list)
    except:
        traceback.print_exc()

def list_datastores():
    try:
        ws = get_workspace()
        for ds in sorted(ws.datastores):
    except:
        traceback.print_exc()

def list_environments():
    try:
        ws = get_workspace()
        envs = Environment.list(workspace=ws)
        for env in envs:
    except:
        traceback.print_exc()

def list_models():
    try:
        ws = get_workspace()
        models_list = list()
        for model in Model.list(ws):
            models_list.append([model.name, model.version])
        write_json('logs/models_list.json', models_list)
    except:
        traceback.print_exc()

def submit_experiment(py_name):
    experiment_name = py_name.split('.')[0]
    experiment_config = AmlExperimentConfig(experiment_name)
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
    if boolean_flag_arg('--wait'):
        run.wait_for_completion(show_output=True)
        status = run.get_status()
        run.get_all_logs(destination='logs/')
        details = run.get_details_with_logs()
        outfile = 'logs/run_details_{}_{}.json'.format(experiment_name, run.number)
        write(outfile, str(details))
        artifact_name = 'properties_{}_{}'.format(experiment_name, run.number)
        save_run_json_artifact(artifact_name, run.get_properties())
        artifact_name = 'metrics_{}_{}'.format(experiment_name, run.number)
        save_run_json_artifact(artifact_name, run.get_metrics())
        outfile = 'logs/run_results_{}_{}.json'.format(experiment_name, run.number)
        run.download_file('outputs/results.json', outfile)
        artifact_name = 'file_names_{}_{}'.format(experiment_name, run.number)
        save_run_json_artifact(artifact_name, run.get_file_names())
        if boolean_flag_arg('--register-model'):
            model = run.register_model(
                model_name='skl-knn-us-states-geo',
                model_path='outputs/skl-knn-us-states-geo.joblib',
                model_framework=Model.Framework.SCIKITLEARN,
                model_framework_version='0.22.2.post1',
                resource_configuration=ResourceConfiguration(cpu=1, memory_in_gb=0.5))

def submit_example_pipeline1(pipeline_name):
    pipeline_config = AmlExperimentConfig(pipeline_name)
    compute_name = pipeline_config.compute_name()
    env_name = pipeline_config.env_name()
    max_seconds = pipeline_config.max_run_duration_seconds()
    ws = get_workspace()
    experiment = Experiment(workspace=ws, name=pipeline_name)
    data_store = ws.get_default_datastore()  # workspaceblobstore (Default)
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
        run.wait_for_completion(show_output=True)
        status = run.get_status()
        run.get_all_logs(destination='logs/')
        details = run.get_details_with_logs()
        outfile = 'logs/run_details_{}_{}.json'.format(pipeline_name, run.number)
        write(outfile, str(details))
        artifact_name = 'properties_{}_{}'.format(pipeline_name, run.number)
        save_run_json_artifact(artifact_name, run.get_properties())
        artifact_name = 'metrics_{}_{}'.format(pipeline_name, run.number)
        save_run_json_artifact(artifact_name, run.get_metrics())
        artifact_name = 'file_names_{}_{}'.format(pipeline_name, run.number)
        save_run_json_artifact(artifact_name, run.get_file_names())

def submit_example_pipeline2(pipeline_name):
    pipeline_config = AmlExperimentConfig(pipeline_name)
    compute_name = pipeline_config.compute_name()
    env_name = pipeline_config.env_name()
    max_seconds = pipeline_config.max_run_duration_seconds()
    ws = get_workspace()
    experiment = Experiment(workspace=ws, name=pipeline_name)
    data_store = ws.get_default_datastore()
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
        run.wait_for_completion(show_output=True)
        status = run.get_status()
        run.get_all_logs(destination='logs/')
        details = run.get_details_with_logs()
        outfile = 'logs/run_details_{}_{}.json'.format(pipeline_name, run.number)
        write(outfile, str(details))
        artifact_name = 'properties_{}_{}'.format(pipeline_name, run.number)
        save_run_json_artifact(artifact_name, run.get_properties())
        artifact_name = 'metrics_{}_{}'.format(pipeline_name, run.number)
        save_run_json_artifact(artifact_name, run.get_metrics())
        artifact_name = 'file_names_{}_{}'.format(pipeline_name, run.number)
        save_run_json_artifact(artifact_name, run.get_file_names())

def deploy_model_to_aci(name, version):
    try:
        ws = get_workspace()
        model = get_registered_model(ws, name, version)
        deploy_name = 'cjoakim-aml-aci-{}'.format(epoch())
        deployment_config = AciWebservice.deploy_configuration(
            cpu_cores=1, memory_gb=0.5, auth_enabled=True)
        web_service = Model.deploy(ws, deploy_name, [model], deployment_config=deployment_config)
        web_service.wait_for_deployment(show_output = True)
        primary, secondary = web_service.get_keys()
    except:
        traceback.print_exc()
        return None

def deploy_model_to_aci_v2(name, version):
    try:
        ws = get_workspace()
        model = get_registered_model(ws, name, version)
        deploy_name = 'cjoakim-aml-aci-{}'.format(epoch())
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
        web_service.wait_for_deployment(show_output = True)
        primary, secondary = web_service.get_keys()
    except:
        traceback.print_exc()
        return None

def deploy_model_to_aks(name, version):
    try:
        ws = get_workspace()
        cluster_name = 'cjoakimaml2aks2'
        compute_config = AksCompute.provisioning_configuration(location='eastus2')
        aks_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
        aks_cluster.wait_for_completion(show_output=True)
        model = get_registered_model(ws, name, version)
        deploy_name = 'cjoakim-aml-aci-{}'.format(epoch())
        deployment_config = AksWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)
        web_service = Model.deploy(
            ws,
            deploy_name,
            [model],
            deployment_config=deployment_config,
            deployment_target = aks_cluster)
        web_service.wait_for_deployment(show_output = True)
        primary, secondary = web_service.get_keys()
    except:
        traceback.print_exc()
        return None

def deploy_model_to_local(name, version):
    try:
        ws = get_workspace()
        model = get_registered_model(ws, name, version)
        deploy_name = 'local-cjoakim-aml-aci-{}'.format(epoch())
        deployment_config = LocalWebservice.deploy_configuration(port=8890)
        web_service = Model.deploy(ws, deploy_name, [model], deployment_config=deployment_config)
        web_service.wait_for_deployment(show_output = True)
        primary, secondary = web_service.get_keys()
    except:
        traceback.print_exc()
        return None

def batch_scoring_example():
    try:
        ws = get_workspace()
        batch_data_set = ws.datasets['batch_locations.csv']
        default_ds = ws.get_default_datastore()
        output_dir = PipelineData(
            name='batch_inferences',
            datastore=default_ds,
            output_path_on_compute='results')
        env = Environment.from_conda_specification(
            name='batch-env',
            file_path='src/azure-ml-tutorial.yml')
        parallel_run_config = ParallelRunConfig(
            source_directory='src',
            entry_script='score_skl_knn_us_states_geo_batch.py',
            mini_batch_size='14',
            error_threshold=10,
            output_action="append_row",
            environment=env,
            compute_target='compute3',
            node_count=1)
        parallelrun_step = ParallelRunStep(
            name='batch-score',
            parallel_run_config=parallel_run_config,
            inputs=[batch_data_set.as_named_input('batch_data')],
            output=output_dir,
            arguments=[],
            allow_reuse=True
        )
        pipeline = Pipeline(workspace=ws, steps=[parallelrun_step])
        pipeline_run = Experiment(ws, 'batch_prediction_pipeline').submit(pipeline)
        pipeline_run.wait_for_completion(show_output=True)
        prediction_run = next(pipeline_run.get_children())
        prediction_output = prediction_run.get_output_data('inferences')
        prediction_output.download(local_path='results')
        for root, dirs, files in os.walk('results'):
            for file in files:
                if file.endswith('parallel_run_step.txt'):
                    result_file = os.path.join(root,file)
        df = pd.read_csv(result_file, delimiter=":", header=None)
        df.columns = ["File", "Prediction"]
    except:
        traceback.print_exc()
        return None

def submit_pytorch_example(compute_name):
    ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name='day1-experiment-train')
    config = ScriptRunConfig(
        source_directory='src',
        script='day1-experiment-train.py',
        compute_target=compute_name)
    env = Environment.from_conda_specification(
        name='pytorch-env',
        file_path='.azureml/pytorch-env.yml')
    config.run_config.environment = env
    run = experiment.submit(config)
    aml_url = run.get_portal_url()

def connect_to_workspace(subscription_id, resource_group, name):
        subscription_id, resource_group, name))
    ws = get_workspace_from_params(name, subscription_id, resource_group)
    ws.write_config(path='.azureml')  # write to .azureml/config.json

def get_workspace(capture=False):
    ws = Workspace.from_config()  # This automatically looks in directory .azureml
    if capture:
        write_json('tmp/workspace_details.json', ws.get_details())
        env = Environment.get(workspace=ws, name="AzureML-Minimal")
    return ws

def get_workspace_from_params(name, subscription_id, resource_group):
    ws = Workspace.get(name=name,
                   subscription_id=subscription_id,
                   resource_group=resource_group)
    return ws

def get_compute_instance(ws, name):
    ci = ComputeInstance(workspace=ws, name=name)  # this returns <class 'azureml.core.compute.amlcompute.AmlCompute'> instance !?
    return ci

def get_registered_model(ws, name, version):
    try:
        name_version = '{}:{}'.format(name, version)
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
    try:
        ws = get_workspace()
        value = os.environ.get(envvar_name)
        if value:
            scrubbed_name = envvar_name.replace('_','-')
            keyvault = ws.get_default_keyvault()
            keyvault.set_secret(name=scrubbed_name, value=value)
    except:
        traceback.print_exc()

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

def save_run_json_artifact(name, obj):
    try:
        outfile = 'logs/run_{}.json'.format(name)
        write_json(outfile, obj, verbose=True)
    except:
        traceback.print_exc()

def epoch():
    return int(time.time())
