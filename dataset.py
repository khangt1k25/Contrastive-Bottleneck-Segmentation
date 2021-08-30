import os 
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor 
from utils import mkdir_if_missing, download_file_from_google_drive
import tarfile
from PIL import Image
from utils import *
from copy import deepcopy
import numpy as np

class VOC(Dataset):
    
    GOOGLE_DRIVE_ID = '1pxhY5vsLwXuz6UHZVUKhtb7EJdCg2kuH'
    FILE = 'PASCAL_VOC.tgz'
    DB_NAME = 'VOCSegmentation'
    VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                    [0, 64, 128]]

    VOC_CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

    def __init__(self, root, split='train', res=224, transform=True, download=False):
        self.root  = root 
        self.split = split
        self.transform = transform
        self.base_transform = get_base_transform()
        self.train_transform = get_train_transform()
        self.to_tensor_normalize_transform = get_tensor_and_normalize_transform()
        self.to_tensor = get_tensor_transform()
        if download:
            self._download()
        
        self.imdb = self.load_imdb()

    def _download(self):
        
        _fpath = os.path.join(self.root, self.FILE)

        if os.path.isfile(_fpath):
            print('Files already downloaded')
            return
        else:
            print('Downloading dataset from google drive')
            mkdir_if_missing(os.path.dirname(_fpath))
            download_file_from_google_drive(self.GOOGLE_DRIVE_ID, _fpath)

        # extract file
        cwd = os.getcwd()
        print('\nExtracting tar file')
        tar = tarfile.open(_fpath)
        os.chdir(os.path.join(self.root))
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('Done!')

    def load_imdb(self):
        
        path_to_train = os.path.join(self.root, self.DB_NAME, 'sets', '{}.txt'.format(self.split))

        with open(path_to_train, 'r') as f:
            imdb = f.read().splitlines()
        return imdb


    def __getitem__(self, index):
        
        imgid = self.imdb[index]

        image, label = self.load_data(imgid)

        if self.transform:
            image_base = self.base_transform(image)
            image_da = self.train_transform(deepcopy(image_base))

        image_base = self.to_tensor_normalize_transform(image_base)
        image_da = self.to_tensor_normalize_transform(image_da)
        
        label = self.transform_label(label)


        return {"base":image_base, "aug": image_da, "label": label}

    def load_data(self, image_id):
        image_path = os.path.join(self.root, self.DB_NAME, 'images', '{}.jpg'.format(str(image_id))) 
        image = Image.open(image_path).convert('RGB')
    
        label_path = os.path.join(self.root, self.DB_NAME, 'SegmentationClassAug', '{}.png'.format(str(image_id))) 
        label = Image.open(label_path)
        
        
        return image, label
    

    
    def transform_label(self, label):
        label = np.array(label)
        mask = (label!=0) & (label !=255)
        label = label[mask]  # discard background

        counts = np.bincount(label)
        y = np.argmax(counts)
        y = np.array([y])
        return torch.from_numpy(y)

    def __len__(self):
        return len(self.imdb)
        

    