
# This script is intended to be submitted as a script inside a Pipeline.

__author__  = 'Chris Joakim'
__email__   = "chjoakim@microsoft.com,christopher.joakim@gmail.com"
__license__ = "None"
__version__ = "2021/01/11"

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

from helpers import BlobUtil

logging.basicConfig(level=logging.WARNING)

# https://www.dataquest.io/blog/k-nearest-neighbors-in-python/


def main():
    print('beginning of main(), args: {}'.format(sys.argv))
    #beginning of main(), args: ['skl-knn-us-states-geo.py', '--some-sample-arg', '1', '0']

    start_time = time.time()
    os.makedirs('./outputs', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./tmp', exist_ok=True)

    print('AML SDK library loaded; version:', azureml.core.VERSION)
    print('start_time: {}'.format(start_time))

    run = Run.get_context()
    print('run: {} {}'.format(run, str(type(run))))

    ws = run.experiment.workspace
    print('ws: {} {}'.format(ws, str(type(ws))))

    experiment_name = run.experiment.name
    print('experiment_name: {} {}'.format(experiment_name, str(type(experiment_name))))

    input_blobname = keyword_arg('--input-blobname', None)
    print('input_blobname: {}'.format(input_blobname))

    # Download the prepared csv file to Azure Storage
    local_blob_path = 'tmp/{}'.format(input_blobname)
    BlobUtil(run=run).download(input_blobname, local_blob_path)

    # Read the downloaded blob into a pandas dataframe
    df = pd.read_csv(local_blob_path, delimiter=',', header=0)

    print('df type and shape: {} {}'.format(type(df), df.shape))
    print('df columns: {} '.format(df.columns.values))
    df.head()
    df.describe()

    data = df[['lat', 'lng']]  # select just these two features
    target = df['state']
    print_info(data, 'data')
    print_info(target, 'target')

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.15)  # random_state=0, test_size=0.33
    print_info(X_train, 'X_train')
    print_info(X_test, 'X_test')
    print_info(y_train, 'y_train')
    print_info(y_test, 'y_test')

    classifier = neighbors.KNeighborsClassifier(n_neighbors=1)

    print('fitting...')
    classifier.fit(X_train, y_train)

    print('scoring...')
    score = classifier.score(X_test, y_test)
    print('score: {} {}'.format(type(score), str(score)))  # <class 'numpy.float64'> 0.9143029080736452

    print('predict X_test...')
    y_pred = classifier.predict(X_test)

    print('classification_report:')
    print(classification_report(y_test, y_pred))

    print('confusion_matrix:')
    print(confusion_matrix(y_test, y_pred))

    # Accuracy score
    accuracy = float(accuracy_score(y_pred, y_test))
    print('accuracy_score is: {}'.format(accuracy))

    # files saved in the "outputs" folder are automatically uploaded into run history
    joblib_path = 'outputs/{}.joblib'.format(experiment_name)
    print('saving model to: {}'.format(joblib_path))
    joblib.dump(classifier, joblib_path)

    results = dict()
    results['experiment_name'] = experiment_name
    results['accuracy_score'] = accuracy
    results['aml_core_version'] = azureml.core.VERSION
    results['run_id'] = run.id
    results['run_number'] = run.number
    results['run_status'] = run.get_status()
    results['run_properties'] = run.get_properties()
    results['run_name'] = run.name
    results['epoch'] = time.time()
    write_json_output('results.json', results)

    end_time = time.time()
    print('end_time: {}'.format(end_time))
    print('elapsed time: {}'.format(end_time - start_time))

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

def print_info(obj, name):
    print('{} -> type: {}, shape: {}'.format(name, type(obj), obj.shape))

def write_json_output(basename, obj, verbose=True):
    jstr = json.dumps(obj, sort_keys=True, indent=2)
    write_text_output(basename, jstr, verbose)

def write_text_output(basename, s, verbose=True):
    outfile = 'outputs/{}'.format(basename)
    with open(outfile, 'w') as f:
        f.write(s)
        if verbose:
            print('file written: {}'.format(outfile))


if __name__ == '__main__':
    main()
