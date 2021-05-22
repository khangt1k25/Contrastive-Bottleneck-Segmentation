import torchvision.transforms as transforms

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, base_transform, transform):
        self.base_transform = base_transform
        self.transform = transform

    def __call__(self, x):
        return [self.base_transform(x), self.transform(x)]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    #transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010)),
])

base_transfrom  = transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
    transforms.ToTensor()
])



test_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    #transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010)),  
])
# test_transform = transforms.Compose([
#     # transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
#     # transforms.RandomHorizontalFlip(),
#     # transforms.RandomApply([
#     #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
#     # ], p=0.8),
#     # transforms.RandomGrayscale(p=0.2),
#     transforms.ToTensor(),
#     #transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010)),  
# ])
