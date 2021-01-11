
__author__  = 'Chris Joakim'
__email__   = "chjoakim@microsoft.com,christopher.joakim@gmail.com"
__license__ = "None"
__version__ = "2021/01/11"

import json
import os
import sys
import time
import traceback

# This file defines helper classes which simplify and de-cluttter file aml_client.py,
# which is focused on interacting with the Azure Machine Learning service via the SDK.

# Configuration details for AML Compute and Experiments can be more easily
# understood, edited, and used via these non-code JSON files.
AML_COMPUTE_CONFIG_FILE     = 'config/aml_compute.json'
AML_EXPERIMENTS_CONFIG_FILE = 'config/aml_experiments.json'

class AbstractBaseConfig():

    def __init__(self):
        pass

    def read_json(self, infile):
        with open(infile, 'rt') as f:
            return json.loads(f.read())

    @classmethod
    def write_json(cls, outfile, obj):
        jstr = json.dumps(obj, sort_keys=True, indent=2)
        cls.write(outfile, jstr)

    @classmethod
    def write(cls, outfile, s):
        with open(outfile, 'w') as f:
            f.write(s)
            print('file written: {}'.format(outfile))


class AmlComputeConfig(AbstractBaseConfig):

    def __init__(self, compute_name):
        AbstractBaseConfig.__init__(self)
        try:
            all_data = self.read_json(AML_COMPUTE_CONFIG_FILE)
            self.name = compute_name
            self.data = all_data[compute_name]
        except:
            traceback.print_exc()
 
    def vm_size(self):
        return self.data['vm_size']

    def min_nodes(self):
        return self.data['min_nodes']

    def max_nodes(self):
        return self.data['max_nodes']

    def idle_secs(self):
        return self.data['idle_secs']

    @classmethod
    def create_config_file(cls):
        # AmlComputeConfig.create_config_file()
        conf = dict()

        nb2 = dict()
        nb2['vm_size'] = 'STANDARD_D2_V2'
        nb2['type']    = 'compute instance (for notebooks)'
        conf['nb2'] = nb2

        nb11 = dict()
        nb11['vm_size'] = 'Standard_DS11_v2'
        nb11['type']    = 'compute instance (for notebooks)'
        conf['nb11'] = nb11

        compute2 = dict()
        compute2['vm_size']   = 'STANDARD_D2_V2'
        compute2['idle_secs'] = 3600
        compute2['min_nodes'] = 0
        compute2['max_nodes'] = 4
        conf['compute2'] = compute2

        compute3 = dict()
        compute3['vm_size']   = 'Standard_DS3_v2'
        compute3['idle_secs'] = 36000
        compute3['min_nodes'] = 0
        compute3['max_nodes'] = 4
        conf['compute3'] = compute3

        compute11 = dict()
        compute11['vm_size']   = 'Standard_DS11_v2'
        compute11['idle_secs'] = 3600
        compute11['min_nodes'] = 0
        compute11['max_nodes'] = 4
        conf['compute11'] = compute11

        cls.write_json(AML_COMPUTE_CONFIG_FILE, conf)


class AmlExperimentConfig(AbstractBaseConfig):

    def __init__(self, experiment_name):
        AbstractBaseConfig.__init__(self)
        try:
            all_data = self.read_json(AML_EXPERIMENTS_CONFIG_FILE)
            self.name = experiment_name
            self.data = all_data[experiment_name]
        except:
            traceback.print_exc()

    def compute_name(self):
        return self.data['compute_name']

    def env_name(self):
        return self.data['env_name']

    def args(self):
        return self.data['args']

    def max_run_duration_seconds(self):
        return self.data['max_run_duration_seconds']

    @classmethod
    def create_config_file(cls):
        # AmlExperimentConfig.create_config_file()
        conf = dict()

        hello = dict()
        hello['compute_name'] = 'compute2'
        hello['env_name'] = 'AzureML-Tutorial'
        hello['args'] = '--person,Azure,--verbose'.split(',')
        hello['max_run_duration_seconds'] = 3600
        conf['hello'] = hello 

        states_geo = dict()
        states_geo['compute_name'] = 'compute2'
        states_geo['env_name'] = 'AzureML-Tutorial'
        states_geo['args'] = '--display-secrets,--verbose'.split(',')
        states_geo['max_run_duration_seconds'] = 3600
        conf['skl-knn-us-states-geo'] = states_geo

        states_geo_pipeline = dict()
        states_geo_pipeline['compute_name'] = 'compute2'
        states_geo_pipeline['env_name'] = 'AzureML-Tutorial'
        states_geo_pipeline['args'] = '--display-secrets,--verbose'.split(',')
        states_geo_pipeline['max_run_duration_seconds'] = 3600
        steps = list()
        steps.append(
            {'name': 'prepare_data', 
            'script': 'data-prep-states-geo.py',
            'arguments': ['--input_dataset', 'postal_codes_us.csv'],
            'outputs':   ['ppp'],
            })
        steps.append(
            {'name': 'train_model', 
            'script': 'skl-knn-us-states-geo2.py',
            'arguments': ['--input_dataset', 'postal_codes_us.csv']
            })
        states_geo_pipeline['steps'] = steps
        conf['skl-knn-us-states-geo-pipeline'] = states_geo_pipeline

        cls.write_json(AML_EXPERIMENTS_CONFIG_FILE, conf)
