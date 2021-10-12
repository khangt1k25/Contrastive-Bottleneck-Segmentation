import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.serialization import load
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, ToPILImage
from models import *
from utils import *
from dataset import *
from collate import collate_custom
from tqdm import tqdm
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




dset = VOC(root='./PASCAL_VOC', split='trainaug', res=224, transform=True, download=False)
loader = DataLoader(dset, batch_size=5,  shuffle=True, num_workers=2, collate_fn=collate_custom)


maskgenerator = MaskGenerator("voc2012", out_channels=1)
encoder = SupConResNet(name="resnet18", head="mlp", feat_dim=128)



try:
    path = './dumps/new_model.pt'
    checkpoint = torch.load(path, map_location=device)
    # print(checkpoint['maskgenerator_state_dict'])
    # encoder.load_state_dict(checkpoint['encoder_state_dict'])
    maskgenerator.load_state_dict(checkpoint['maskgenerator_state_dict'])
    print("Load successful")
except:
    print("Load fail")


maskgenerator.eval()
with torch.no_grad():
    for i, batch in tqdm(enumerate(loader), leave=False):
        images_base = batch['base']
        images_da = batch['aug']
        labels = batch['label']

        images = images_base.to(device)
        mask = maskgenerator(images)
        segmented = images*mask

        print(images.shape)
        print(segmented.shape)
        #print(mask)
        
        
        for k in range(0, 5):
            # img = ToPILImage()(images[k].cpu().squeeze()).show()
            img2 = ToPILImage()(segmented[k].cpu().squeeze())
            # img.save('./pics/img_origin{}.png'.format(k))
            img2.save('./pics/img_after{}.png'.format(k))
        break

# load model

