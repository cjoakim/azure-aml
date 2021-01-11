import json
import numpy as np
import os
from sklearn.externals import joblib

from azureml.core.model import Model
import logging
logging.basicConfig(level=logging.DEBUG)


def init():
    global model
    print('init() ...')
    #model_path = Model.get_model_path(model_name='skl-knn-us-states-geo')
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'skl-knn-us-states-geo.joblib')
    print('model_path: {}'.format(model_path))
    model = joblib.load(model_path)

def run(data):
    try:
        data = np.array(json.loads(data))
        result = model.predict(data)
        # You can return any data type, as long as it is JSON serializable.
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
