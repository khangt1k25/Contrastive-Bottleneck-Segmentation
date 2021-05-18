from torch._C import device
from models import SupConResNet
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.datasets as datasets
from tqdm import tqdm
from models import *
from SupConLoss import *
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
        
    def train(self, trainloader, start_epoch, end_epoch):
        for epoch in range(start_epoch, end_epoch+1):
            epoch_loss = 0.
            for batch, (images, labels) in tqdm(enumerate(trainloader), leave=False):
                bsz = labels.shape[0]
                mask = self.maskgenerator(images)
                images1 = images*mask
                with torch.no_grad():
                    images2 = images.clone().detach()

                images = torch.cat([images1, images2], dim=0)
                features = self.encoder(images)
                f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

                contrastive_loss = self.criterion(features, labels)

                mask = mask.view(bsz, -1)

                norm_mask_loss = torch.norm(mask, p=1)/(mask.shape[0]*mask.shape[1])

                loss = contrastive_loss + 0.01*norm_mask_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            print(f"\nepoch {epoch} with loss {epoch_loss/batch}\n") 
            if epoch % 5 == 0:
                self.saving()





    