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
    train_loader = DataLoader(train_dataset, batch_size=10,  shuffle=True, num_workers=2, collate_fn=collate_custom)

    sample = train_dataset[0]
    for i, batch in tqdm(enumerate(train_loader), leave=False):
        image_base = batch['base']
        image_aug = batch['aug']
        print(image_base.shape)
        print(image_aug.shape)

        toPIL(image_base[0]).show()
        toPIL(image_aug[0]).show()
        
        break

    # maskgenerator = MaskGenerator("cifar10", out_channels=1)
    # encoder = SupConResNet(name="resnet18", head="mlp", feat_dim=64)

    # optimizer = optim.Adam(params=list(maskgenerator.parameters())+list(encoder.parameters()))
    
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # trainer = Trainer(maskgenerator, encoder, optimizer, criterion, device)
    # trainer.train(train_loader, 0, 10)
