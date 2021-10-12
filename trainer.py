from torch._C import device
from torch.functional import norm
from models import SupConResNet
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.datasets as datasets
from tqdm import tqdm
from models import *
from losses import * 
from utils import *




class Trainer():
    def __init__(self, maskgenerator, encoder, optimizer, criterion, device):
        self.device = device
        self.maskgenerator = maskgenerator.to(self.device)
        self.encoder = encoder.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion

    def saving(self):
        path = './dumps/model.pt'
        torch.save({
            'maskgenerator_state_dict': self.maskgenerator.state_dict(),
            'encoder_state_dict': self.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        print("Saving successful")

    def load(self):
        try:
            path = './dumps/model.pt'
            checkpoint = torch.load(path, map_location=self.device)
            self.maskgenerator.load_state_dict(checkpoint['maskgenerator_state_dict'])
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Load successful")
        except:
            print("Load fail")
    
    def train(self, trainloader, start_epoch, end_epoch):
        for epoch in range(start_epoch, end_epoch+1):
            epoch_loss = 0.
            step = 0
            for i, batch in tqdm(enumerate(trainloader), leave=False):
                images_base = batch['base']
                images_da = batch['aug']
                labels = batch['label']

                bsz = labels.shape[0]
                images_base, images_da, labels = images_base.to(self.device),images_da.to(self.device),labels.to(self.device)
                
                mask = self.maskgenerator(images_base)
                images_base = images_base * mask
               
                # with torch.no_grad():
                #     images2 = images.clone().detach()

                images = torch.cat([images_base, images_da], dim=0)



                features = self.encoder(images)
                f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

                contrastive_loss = self.criterion(features, labels)

                mask = mask.view(bsz, -1)

                
                norm_mask_loss = torch.mean(mask)

                loss = contrastive_loss + norm_mask_loss
                
            
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                step += 1
            print(f"\nepoch {epoch} with loss {epoch_loss/step}\n") 
            if epoch % 5 == 0:
                self.saving()





    