__author__  = 'Chris Joakim'
__email__   = "chjoakim@microsoft.com,christopher.joakim@gmail.com"
__license__ = "None"
__version__ = "2020/12/21"

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
        conn_str = None
        if run == None:  # run is an object in the Azure Machine Learning SDK
            conn_str = os.environ['AZURE_STORAGE_CONNECTION_STRING']
        else:
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

    def ls_blobs(self):
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


if __name__ == "__main__":
    bu = BlobUtil()
    bu.upload('requirements.in')
    bu.upload('requirements.txt')
    bu.upload('pyenv.sh')
    bu.delete('us_states.csv')
    bu.download('requirements.in', 'tmp/requirements.in')
    items = bu.ls_blobs()
    for item in items:
        print(item)

# $ python helpers.py
# uploaded: requirements.in -> requirements.in
# uploaded: requirements.txt -> requirements.txt
# uploaded: pyenv.sh -> pyenv.sh
# deleted: us_states.csv
# downloaded: requirements.in -> tmp/requirements.in
# {'name': 'pyenv.sh', 'creation_time': datetime.datetime(2020, 12, 21, 18, 20, 8, tzinfo=datetime.timezone.utc)}
# {'name': 'requirements.in', 'creation_time': datetime.datetime(2020, 12, 22, 19, 12, 15, tzinfo=datetime.timezone.utc)}
# {'name': 'requirements.txt', 'creation_time': datetime.datetime(2020, 12, 22, 19, 12, 15, tzinfo=datetime.timezone.utc)}
