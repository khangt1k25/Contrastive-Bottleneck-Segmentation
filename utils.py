import os
import errno
import requests
import tarfile
import torchvision.transforms as transforms


CHUNK_SIZE = 32768

def download_file_from_google_drive(id, destination):
    URL = "https://drive.google.com/u/1/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)

    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise



def get_base_transform():
    return transforms.RandomResizedCrop(size=224, scale=(0.2, 1.))

def get_train_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2)
    ])

def get_tensor_and_normalize_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010))
    ])

def get_tensor_transform():
    return transforms.ToTensor()

