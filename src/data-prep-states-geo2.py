
__author__  = 'Chris Joakim'
__email__   = "chjoakim@microsoft.com,christopher.joakim@gmail.com"
__license__ = "None"
__version__ = "2021/01/11"

from azureml.core import Run

import logging
import os

# Import the AML SDK library
import azureml.core
from azureml.core import Dataset
from azureml.core import Datastore
from azureml.core import Environment
from azureml.core import Experiment
from azureml.core import Model
from azureml.core import Workspace
from azureml.core.run import Run
from azureml.core.resource_configuration import ResourceConfiguration

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

logging.basicConfig(level=logging.WARNING)


def main():
    print('beginning of main(), args: {}'.format(sys.argv))
    os.makedirs('./outputs', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./data', exist_ok=True)

    run = Run.get_context()
    print('run: {} {}'.format(run, str(type(run))))

    ws = run.experiment.workspace
    print('ws: {} {}'.format(ws, str(type(ws))))

    experiment_name = run.experiment.name
    print('experiment_name: {} {}'.format(experiment_name, str(type(experiment_name))))

    input_dataset_name = keyword_arg('--input-dataset', None)
    print('input_dataset_name: {}'.format(input_dataset_name))
    #print('output_path:        {}'.format(output_path))

    # Use Dataset objects for pre-existing data
    print('Dataset.get_by_name: {} ...'.format(input_dataset_name))
    df = Dataset.get_by_name(ws, name=input_dataset_name).to_pandas_dataframe()

    print_df_info(df, 'input dataframe')

    # select just these 3 columns
    lls_df = df[['lat', 'lng', 'state']]  

    # remove bad rows
    lls_df = lls_df[lls_df['lat'].notnull()].copy()
    lls_df = lls_df[lls_df['lng'].notnull()].copy()
    lls_df = lls_df[lls_df['state'].notnull()].copy()

    # ensure state code is uppercase
    lls_df['state'] = lls_df['state'].str.upper()

    print_df_info(lls_df, 'output dataframe')

    lls_df.to_csv('data/prepped.csv')
    os.system('ls -alR > outputs/ls.txt')
    os.system('cat outputs/ls.txt')

def print_df_info(df, msg=''):
    print('')
    print('print_df_info: {}'.format(msg))
    print(df.head(3))
    print(df.tail(3))
    print(df.dtypes)
    print(df.shape)
    print(df.columns)

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


if __name__ == '__main__':
    main()
