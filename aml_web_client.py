"""
Usage:
    source exports.sh ; python aml_web_client.py get_swagger_definition
    source exports.sh ; python aml_web_client.py score_one <lat> <lng>
    source exports.sh ; python aml_web_client.py score_one 35.22718156801215 -80.84309935569763 --proba
    source exports.sh ; python aml_web_client.py score_one 36.37149605216152 -121.90174877643585
    -
    source exports.sh ; python aml_web_client.py score_one_v2 35.22718156801215 -80.84309935569763
    source exports.sh ; python aml_web_client.py score_one_v2 36.37149605216152 -121.90174877643585
    -
    source exports.sh ; python aml_web_client.py score_one_v3 35.22718156801215 -80.84309935569763
    source exports.sh ; python aml_web_client.py score_one_v3 36.37149605216152 -121.90174877643585
    -
    source exports.sh ; python aml_web_client.py score_csv_file <infile>
    source exports.sh ; python aml_web_client.py score_csv_file datasets/postal_codes/test_locations.csv
    source exports.sh ; python aml_web_client.py score_csv_file datasets/postal_codes/test_locations.csv --proba
"""

__author__  = 'Chris Joakim'
__email__   = "chjoakim@microsoft.com,christopher.joakim@gmail.com"
__license__ = "None"
__version__ = "2020/12/23"

import json
import os
import sys
import time
import requests

from docopt import docopt


class PostData():

    def __init__(self):
        self.data = dict()
        self.data['data'] = list()
        self.data['method'] = 'predict'

    def predict(self):
        self.data['method'] = 'predict'

    def proba(self):
        self.data['method'] = 'predict_proba'

    def add_score_item(self, array):
        self.data['data'].append(array)


class AmlWebserviceClient():

    def __init__(self):
        self.base_url    = os.environ['AML_WEB_BASE_URL']
        self.auth_key    = os.environ['AML_WEB_AUTH_KEY']

        if self.cli_flag_arg_present('--local'):
            self.base_url = 'http://localhost:8890'
            self.auth_key = 'none'

        self.swagger_url = '{}/swagger.json'.format(self.base_url)
        self.score_url   = '{}/score'.format(self.base_url)

    def get_swagger_definition(self):
        print('get_swagger_definition')
        self.invoke('swagger', 'get', self.swagger_url, self.headers())

    def score_one(self, lat, lng):
        print('score_one: {} {}'.format(lat, lng))
        function_name = 'score_one_predict'
        pd = PostData()
        if self.cli_flag_arg_present('--proba'):
            pd.proba()
            function_name = 'score_one_proba'
        pd.add_score_item([lat, lng])
        print(json.dumps(pd.data, sort_keys=False, indent=2))
        r = self.invoke(function_name, 'post', self.score_url, self.headers(), pd.data)
        print("response: " + r.text)

    def score_one_v2(self, lat, lng):
        # This invokes the ACI web service deployed with:
        # python aml_client.py deploy_model_to_aci_v2 skl-knn-us-states-geo 3
        # which uses this InferenceConfig:
        # inference_config = InferenceConfig(
        #     runtime= "python",
        #     source_directory = 'src',
        #     entry_script="score_skl_knn_us_states_geo.py",
        #     conda_file='azure-ml-tutorial.yml')  # <-- sliced out of environments.txt

        print('score_one_v2: {} {}'.format(lat, lng))
        function_name = 'score_one_v2_predict'
        data = [[lat, lng]]
        print(json.dumps(data, sort_keys=False, indent=2))
        r = self.invoke(function_name, 'post', self.score_url, self.headers(), data)
        print("response: " + r.text)

    def score_one_v3(self, lat, lng):
        print('score_one_v3: {} {}'.format(lat, lng))
        function_name = 'score_one_v3'
        # see the generated swagger.json from the deployed web service for the POST-data format
        data = {
            "data": [
                {
                    "Column1": 0.0,
                    "lat": lat,
                    "lng": lng
                }
            ]
        }
        print(json.dumps(data, sort_keys=False, indent=2))
        r = self.invoke(function_name, 'post', self.score_url, self.headers(), data)
        jstr = json.loads(r.text)
        print("response: " + jstr)

    def score_csv_file(self, infile):
        print('score_csv_file: {}'.format(infile))
        function_name = 'score_csv_file_predict'
        pd = PostData()
        if self.cli_flag_arg_present('--proba'):
            pd.proba()
            function_name = 'score_csv_file_proba'

        for idx, line in enumerate(self.read_lines(infile)):
            if idx > 0:
                tokens = line.strip().split(',')
                if len(tokens) > 3:
                    lat, lng, loc, other = tokens
                    pd.add_score_item([lat, lng])

        print(json.dumps(pd.data, sort_keys=False, indent=2))
        self.invoke(function_name, 'post', self.score_url, self.headers(), pd.data)

    def invoke(self, function_name, method, url, headers={}, json_body={}):
        # This is a generic method which invokes all HTTP Requests to the Azure AML WebService
        print('===')
        print("invoke: {} {} {}\nheaders: {}\nbody: {}".format(function_name, method.upper(), url, headers, json_body))
        print('---')
        if method == 'get':
            r = requests.get(url=url, headers=headers)
        elif method == 'post':
            r = requests.post(url=url, headers=headers, json=json_body)
        elif method == 'put':
            r = requests.put(url=url, headers=headers, json=json_body)
        elif method == 'delete':
            r = requests.delete(url=url, headers=headers)
        else:
            print('error; unexpected method value passed to invoke: {}'.format(method))

        print('response: {}'.format(r))
        if r.status_code < 300:
            try:
                resp_obj = json.loads(r.text)
                outfile = 'tmp/{}.json'.format(function_name)
                self.write_json_file(resp_obj, outfile)
            except Exception as e:
                print("exception processing http response".format(e))
                print(r.text)
        else:
            print(r.text)
        return r

    def cli_flag_arg_present(self, flag_arg):
        for arg in sys.argv:
            if arg == flag_arg:
                return True
        return False

    def headers(self):
        obj = dict()
        obj['Content-Type'] = 'application/json'
        obj['Authorization'] = 'Bearer {}'.format(self.auth_key)
        return obj 

    def epoch(self):
        return time.time()
    
    def read_lines(self, infile):
        lines = list()
        with open(infile, 'rt') as f:
            for line in f:
                lines.append(line)
        return lines

    def load_json_file(self, infile):
        with open(infile, 'rt') as json_file:
            return json.loads(str(json_file.read()))

    def write_json_file(self, obj, outfile):
        with open(outfile, 'wt') as f:
            f.write(json.dumps(obj, sort_keys=False, indent=2))
            print('file written: {}'.format(outfile))


def print_options(msg):
    print(msg)
    arguments = docopt(__doc__, version=__version__)
    print(arguments)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        func = sys.argv[1].lower()
        print('func: {}'.format(func))
        client = AmlWebserviceClient()

        if func == 'get_swagger_definition':
            client.get_swagger_definition()

        elif func == 'score_one':
            lat = sys.argv[2]
            lng = sys.argv[3]
            client.score_one(lat, lng)

        elif func == 'score_one_v2':
            lat = sys.argv[2]
            lng = sys.argv[3]
            client.score_one_v2(lat, lng)

        elif func == 'score_one_v3':
            lat = sys.argv[2]
            lng = sys.argv[3]
            client.score_one_v3(lat, lng)

        elif func == 'score_csv_file':
            infile = sys.argv[2]
            client.score_csv_file(infile)

        else:
            print_options('Error: invalid function: {}'.format(func))
    else:
        print_options('Error: no function argument provided.')
