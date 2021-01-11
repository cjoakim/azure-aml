
import os
import logging
import numpy as np
import joblib
import traceback

from azureml.core import Model

logging.basicConfig(level=logging.DEBUG)


def init():
    global model
    model_path = Model.get_model_path(model_name='skl-knn-us-states-geo')
    print('model_path: {}'.format(model_path))
    model = joblib.load(model_path)

def run(mini_batch):
    # This runs for each batch
    resultList = []

    # process each file in the batch
    for f in mini_batch:
        # Read comma-delimited data into an array
        data = np.genfromtxt(f, delimiter=',')
        # Reshape into a 2-dimensional array for model input
        prediction = model.predict(data.reshape(1, -1))
        # Append prediction to results
        resultList.append("{}: {}".format(os.path.basename(f), prediction[0]))
    return resultList
