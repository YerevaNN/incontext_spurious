import os
import zipfile
import tarfile
import requests
import tempfile

from tqdm import tqdm

def download_from_url(path_to_save: str, url: str, mode: str = "zip"):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    tmp = tempfile.NamedTemporaryFile()

    with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
        with open(tmp.name, 'wb') as temp_file:
            for chunk in response.iter_content(chunk_size=1024):
                pbar.update(len(chunk))
                temp_file.write(chunk)

        if mode == 'zip':
            with zipfile.ZipFile(tmp.name, 'r') as zip_ref:
                zip_ref.extractall(path_to_save)
        elif mode == 'tar':
            with tarfile.open(tmp.name, mode='r:gz') as tar_ref:
                tar_ref.extractall(path_to_save)
