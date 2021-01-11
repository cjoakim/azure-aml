
# Chris Joakim, 2021/01/11

import arrow
import json
import os
import pickle
import sys
import time
import uuid

import numpy as np
import pandas as pd

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.exceptions import ResourceExistsError
from azure.core.exceptions import ResourceNotFoundError

from helpers import BlobUtil

def prep_states_geo():
    infile = 'datasets/postal_codes/postal_codes_us.csv'
    df = read_pandas_dataframe(infile)
    print_df_info(df, 'raw input')

    # select just these 3 columns
    lls_df = df[['lat', 'lng', 'state']]  

    # remove bad rows
    lls_df = lls_df[lls_df['lat'].notnull()].copy()
    lls_df = lls_df[lls_df['lng'].notnull()].copy()
    lls_df = lls_df[lls_df['state'].notnull()].copy()

    # ensure state code is uppercase
    lls_df['state'] = lls_df['state'].str.upper()

    print_df_info(lls_df, 'prepared output')

def designer_script1():
    infile = 'datasets/postal_codes/postal_codes_us.csv'
    dataframe1 = read_pandas_dataframe(infile)
    print_df_info(dataframe1, 'dataframe1')
    print_df_info(None, 'dataframe2')

    #df2 = dataframe1['state'].str.lower()
    dataframe1['state'] = dataframe1['state'].str.upper()

    print_df_info(dataframe1, 'dataframe1')

def blob_storage_container_client(container_name='pipelines'):
    conn_str = os.environ['AZURE_STORAGE_CONNECTION_STRING']
    blob_svc_client = BlobServiceClient.from_connection_string(conn_str)
    return blob_svc_client.get_container_client(container_name)

def storage_upload(input_path):
    BlobUtil().upload(input_path)

def storage_download(blobname, output_path):
    BlobUtil().download(blobname, output_path)


def invalid_function(f):
    print('invalid_function: {}'.format(f)) 

def print_df_info(df, msg=''):
    print('')
    print('print_df_info: {}'.format(msg))
    try:
        print(df.head(3))
        print(df.tail(3))
        print(df.dtypes)
        print(df.shape)
        print(df.columns)
    except:
        print('df is None or invalid')

def read_pandas_dataframe(infile, delim=',', colnames=None):
    if colnames:
        return pd.read_csv(infile, delimiter=delim, names=colnames)
    else:
        return pd.read_csv(infile, delimiter=delim, header=0)


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


if __name__ == "__main__":
    # Command-line samples:
    # python prototyping.py prep_states_geo 
    # python prototyping.py wrong --cat Miles --register --verbose
    # python prototyping.py storage_upload datasets/postal_codes/us_states.csv
    # python prototyping.py storage_download us_states.csv tmp/us_states.csv

    func = sys.argv[1].lower()
    cat  = keyword_arg('--cat', 'elsa')
    print('func:     {}'.format(func))
    print('cat:      {}'.format(cat))
    print('register: {}'.format(boolean_flag_arg('--register')))
    print('deploy:   {}'.format(boolean_flag_arg('--deploy')))

    print('hello {}!'.format(keyword_arg('--person', 'world')))
    print('verbose: {}'.format(boolean_flag_arg('--verbose')))
    print('debug: {}'.format(boolean_flag_arg('--debug')))

    if func == 'prep_states_geo':
        prep_states_geo()
    elif func == 'storage_upload':
        infile = sys.argv[2]
        storage_upload(infile)
    elif func == 'storage_download':
        blobname = sys.argv[2]
        output_path = sys.argv[3]
        storage_download(blobname, output_path)
    elif func == 'designer_script1':
        designer_script1()
    else:
        invalid_function(func)
