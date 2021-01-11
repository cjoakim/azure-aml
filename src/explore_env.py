
import joblib
import json
import logging
import os
import time

# Import the AML SDK library
import azureml.core
from azureml.core import Dataset
from azureml.core import Datastore
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import Workspace
from azureml.core.run import Run

logging.basicConfig(level=logging.DEBUG)

# https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-secrets-in-runs

def main():
    os.makedirs('./outputs', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    os.system('date > logs/start_date.txt')

    print('AML SDK library loaded; version:', azureml.core.VERSION)
    print('epoch: {}'.format(epoch()))

    run = Run.get_context()
    print('run: {} {}'.format(run, str(type(run))))

    ws = run.experiment.workspace
    print('ws: {} {}'.format(ws, str(type(ws))))

    print('getting secrets...')
    storage_acct = run.get_secret(name='AZURE-STORAGE-ACCOUNT')
    storage_key  = run.get_secret(name='AZURE-STORAGE-KEY')
    storage_cstr = run.get_secret(name='AZURE-STORAGE-CONNECTION-STRING')
    print('storage_acct: {} {}'.format(storage_acct, len(storage_acct)))
    print('storage_key:  {} {}'.format(storage_key,  len(storage_key)))
    print('storage_cstr: {} {}'.format(storage_cstr, len(storage_cstr)))

    print('Dataset.get_by_name...')
    iris = Dataset.get_by_name(ws, name='iris/iris_data.csv').to_pandas_dataframe()
    iris.head()
    iris.describe()

    print('os.system commands...')
    os.system('python --version > outputs/python_version.txt')
    os.system('pip list  > outputs/pip_list.txt')
    os.system('whoami  > outputs/whoami.txt')
    os.system('who am i  >> outputs/whoami.txt')
    os.system('ls -alR > outputs/ls.txt')
    os.system('ps aux > outputs/ps.txt')
    os.system('env | sort > outputs/env.txt')

    run.log(name='hello', value='world')
    run.log(name='storage_acct', value=storage_acct)
    # run.log('hello', 'world')
    # run.log('storage_acct', storage_acct)
    os.system('date > logs/finish_date.txt')

def epoch():
    return time.time()


if __name__ == '__main__':
    main()
