import os
from glob import glob
from zipfile import ZipFile

import numpy as np
import requests
from PIL import Image


def _download_and_save():
    url = f'https://sites.google.com/site/catdatacollection/data/CatOpen.zip'
    r = requests.get(url)

    path = f'./data/cat_face/CatOpen.zip'

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    with open(path, mode='wb') as f:
        for chunk in r.iter_content(chunk_size=128):
            f.write(chunk)

    url = f'https://sites.google.com/site/catdatacollection/data/CatClose.zip'
    r = requests.get(url, stream=True)

    path = f'./data/cat_face/CatClose.zip'

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    with open(path, mode='wb') as f:
        for chunk in r.iter_content(chunk_size=128):
            f.write(chunk)


def _extract():
    path = f'./data/cat_face/CatOpen.zip'
    with ZipFile(path, 'r') as zipfile:
        path = f'./data/cat_face/'
        zipfile.extractall(path)

    path = f'./data/cat_face/CatClose.zip'
    with ZipFile(path, 'r') as zipfile:
        path = f'./data/cat_face/'
        zipfile.extractall(path)


def _jpg2npy():
    arrays = []
    for i in range(7):
        path = f'./data/cat_face/CatOpen/CAT_{i:02d}_Open/*jpg'
        for img in glob(path):
            array = np.array(Image.open(img))
            arrays.append(array)

        path = f'./data/cat_face/CatClose/CAT_{i:02d}_Close/*jpg'
        for img in glob(path):
            array = np.array(Image.open(img))
            arrays.append(array)
    arrays = np.array(arrays)
    np.save('./data/cat_face.npy', arrays)


def main():
    _download_and_save()
    _extract()
    _jpg2npy()


if __name__ == '__main__':
    main()
