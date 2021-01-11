
__author__  = 'Chris Joakim'
__email__   = "chjoakim@microsoft.com,christopher.joakim@gmail.com"
__license__ = "None"
__version__ = "2020/12/21"

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

from helpers import BlobUtil

logging.basicConfig(level=logging.WARNING)


def main():
    run = Run.get_context()
    print('run: {} {}'.format(run, str(type(run))))

    bu = BlobUtil(run=run)

    for idx, arg in enumerate(sys.argv):
        if idx > 0:
            bu.delete(arg)


if __name__ == '__main__':
    main()
