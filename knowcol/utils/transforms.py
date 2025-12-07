
from torchvision import transforms

size = (32, 32)
batch_size = 1024
n_cpu = 8

s = 1
train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
            transforms.Resize(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
               transforms.ColorJitter(0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s)
             ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            # normalize,
        ])

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

# for SSL with transform
train_transform_aug = TwoCropTransform(transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
            transforms.Resize(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            # normalize,
        ]))

val_transform = transforms.Compose([transforms.Resize(size=size),
                                    transforms.ToTensor(),
                                    # normalize,
                                    ])