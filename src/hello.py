
import joblib
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import neighbors

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

logging.basicConfig(level=logging.DEBUG)

def main():
    run = Run.get_context()
    print('run: {} {}'.format(run, str(type(run))))
    print('sys.argv: {}'.format(sys.argv))

    print('hello {}!'.format(keyword_arg('--person', 'world')))
    print('verbose: {}'.format(boolean_flag_arg('--verbose')))
    print('debug: {}'.format(boolean_flag_arg('--debug')))

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
