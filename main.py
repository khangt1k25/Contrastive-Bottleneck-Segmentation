import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from models import *
from SupConLoss import *
from trainer import *
from utils import *

if __name__ == '__main__':
    criterion = SupConLoss(temperature=0.07, contrast_mode="all", base_temperature=0.07)

    train_dataset = datasets.CIFAR10(root='./data/cifar10',
                                         transform=train_transform,
                                         download=False)
    train_loader = DataLoader(train_dataset, batch_size=10,  shuffle=True, num_workers=2)

    maskgenerator = MaskGenerator("cifar10", out_channels=1)
    encoder = SupConResNet(name="resnet18", head="mlp", feat_dim=64)

    optimizer = optim.Adam(params=list(maskgenerator.parameters())+list(encoder.parameters()), lr=0.001)
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(maskgenerator, encoder, optimizer, criterion, device)
    trainer.train(train_loader, 0, 10)
