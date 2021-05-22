import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor, ToPILImage
from models import *
from utils import *
from tqdm import tqdm
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

maskgenerator = MaskGenerator("cifar10", out_channels=1)

try:
    path = './dumps/model1.pt'
    checkpoint = torch.load(path,map_location=device)
    maskgenerator.load_state_dict(checkpoint['maskgenerator_state_dict'])
    print("Load successful")
except:
    print("Load fail")

cifar10 = datasets.CIFAR10(root='./data/cifar10',
                                         transform=test_transform,
                                         download=False)
loader = DataLoader(cifar10, batch_size=1, shuffle=True, num_workers=2 )


maskgenerator.eval()
with torch.no_grad():
    for images, labels in tqdm(loader, leave=False):

        images = images.to(device)
        mask = maskgenerator(images)
        segmented = images*mask
        #print(mask)
        img = ToPILImage()(images.cpu().squeeze())
        img2 = ToPILImage()(segmented.cpu().squeeze())
        img.save('./img_origin.png')
        img2.save('./img_after.png')
        #print(labels)
        break

# load model

