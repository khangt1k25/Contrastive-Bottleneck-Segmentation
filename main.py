import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models import *
from losses import *
from trainer import *
from utils import *
from dataset import *
from collate import collate_custom
import torchvision
toPIL = torchvision.transforms.ToPILImage()

if __name__ == '__main__':
    criterion = SupConLoss(temperature=0.07, contrast_mode="all", base_temperature=0.07)

    train_dataset = VOC(root='./PASCAL_VOC', split='trainaug', res=224, transform=True)
    train_loader = DataLoader(train_dataset, batch_size=64,  shuffle=True, num_workers=2, collate_fn=collate_custom)


    maskgenerator = MaskGenerator("voc2012", out_channels=1)
    encoder = SupConResNet(name="resnet18", head="mlp", feat_dim=128)

    optimizer = optim.Adam(params=list(maskgenerator.parameters())+list(encoder.parameters()))
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(maskgenerator, encoder, optimizer, criterion, device)
    trainer.train(train_loader, 0, 10)
