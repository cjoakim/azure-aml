
__author__  = 'Chris Joakim'
__email__   = "chjoakim@microsoft.com,christopher.joakim@gmail.com"
__license__ = "None"
__version__ = "2021/01/11"

import json
import os
import pickle
import sys
import time
import uuid

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.exceptions import ResourceExistsError
from azure.core.exceptions import ResourceNotFoundError


class BlobUtil():

    def __init__(self, run=None, container_name='pipelines'):
        #conn_str = os.environ['AZURE_STORAGE_CONNECTION_STRING']
        conn_str = run.get_secret(name='AZURE-STORAGE-CONNECTION-STRING')
        blob_svc_client = BlobServiceClient.from_connection_string(conn_str)
        self.container_client = blob_svc_client.get_container_client(container_name)

    def upload(self, input_path):
        basename = os.path.basename(input_path)
        blob_client = self.container_client.get_blob_client(basename)
        with open(input_path, 'rb') as data:
            blob_client.upload_blob(data, blob_type='BlockBlob', overwrite=True)
            print('uploaded: {} -> {}'.format(input_path, basename))

    def download(self, blobname, output_path):
        blob_client = self.container_client.get_blob_client(blobname)
        with open(output_path, 'wb') as file:
            data = blob_client.download_blob()
            file.write(data.readall())
            print('downloaded: {} -> {}'.format(blobname, output_path))

    def ls_blobs():
        items = list()
        for blob in self.container_client.list_blobs():
            item = dict()
            item['name'] = blob.name
            item['creation_time'] = blob.creation_time
            items.append(item)
        return items

    def delete(self, blobname):
        try:
            self.container_client.delete_blob(blobname)
            print('deleted: {}'.format(blobname))
        except:
            pass
